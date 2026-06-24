import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from routing_benchmark.models import ModelTarget
from routing_benchmark.provider import ProviderUnavailableError
from routing_benchmark.providers.anthropic_cloud import AnthropicCloudProvider


@dataclass
class FakeContentBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    content: list
    stop_reason: str
    usage: FakeUsage


class FakeUnavailableError(Exception):
    pass


class FakeMessagesEndpoint:
    def __init__(self, response=None, raise_error: Exception | None = None):
        self.response = response
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


class FakeAnthropicClient:
    def __init__(self, response=None, raise_error: Exception | None = None):
        self.messages = FakeMessagesEndpoint(response=response, raise_error=raise_error)


def make_provider(model_id="claude-sonnet-4-6", response=None, raise_error=None, **kwargs) -> AnthropicCloudProvider:
    client = FakeAnthropicClient(response=response, raise_error=raise_error)
    return AnthropicCloudProvider(
        model_id=model_id,
        client=client,
        unavailable_error_types=(FakeUnavailableError,),
        **kwargs,
    )


def test_constructor_validation():
    with pytest.raises(ValueError):
        AnthropicCloudProvider(model_id="", client=FakeAnthropicClient())
    with pytest.raises(ValueError):
        AnthropicCloudProvider(model_id="claude-sonnet-4-6", max_tokens=0, client=FakeAnthropicClient())


def test_constructor_without_client_requires_anthropic_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError):
        AnthropicCloudProvider(model_id="claude-sonnet-4-6")


def test_target_class_is_cloud():
    provider = make_provider(response=FakeMessage(content=[], stop_reason="end_turn", usage=FakeUsage(0, 0)))
    assert provider.target_class is ModelTarget.CLOUD


def test_generate_parses_text_response_and_computes_cost():
    response = FakeMessage(
        content=[FakeContentBlock(type="text", text="the answer is 42")],
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=1000, output_tokens=2000),
    )
    provider = make_provider(response=response)

    result = provider.generate("what is the answer?", {})

    assert result.text == "the answer is 42"
    assert result.tool_call is None
    assert result.finish_reason == "stop"
    assert result.token_usage.prompt_tokens == 1000
    assert result.token_usage.completion_tokens == 2000
    # claude-sonnet-4-6 pricing: $3/1M input, $15/1M output
    assert result.token_usage.cost_usd == pytest.approx(1000 * 3.0 / 1e6 + 2000 * 15.0 / 1e6)
    assert result.provider_latency_ms >= 0.0


def test_generate_parses_tool_use_response():
    response = FakeMessage(
        content=[FakeContentBlock(type="tool_use", name="lookup", input={"q": "revenue"})],
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    provider = make_provider(response=response)

    result = provider.generate("look it up", {})

    assert result.tool_call is not None
    assert result.tool_call.tool_name == "lookup"
    assert result.tool_call.arguments == {"q": "revenue"}
    assert result.finish_reason == "tool_calls"


def test_generate_maps_max_tokens_stop_reason_to_length():
    response = FakeMessage(content=[FakeContentBlock(type="text", text="cut off")], stop_reason="max_tokens", usage=FakeUsage(5, 5))
    provider = make_provider(response=response)

    result = provider.generate("go on", {})
    assert result.finish_reason == "length"


def test_unknown_model_id_falls_back_to_zero_cost():
    response = FakeMessage(content=[FakeContentBlock(type="text", text="hi")], stop_reason="end_turn", usage=FakeUsage(100, 100))
    provider = make_provider(model_id="some-future-model", response=response)

    result = provider.generate("hi", {})
    assert result.token_usage.cost_usd == 0.0


def test_generate_propagates_provider_unavailable_error():
    provider = make_provider(raise_error=FakeUnavailableError("network down"))

    with pytest.raises(ProviderUnavailableError):
        provider.generate("hello", {})


def test_request_kwargs_include_tools_max_tokens_and_temperature():
    response = FakeMessage(content=[FakeContentBlock(type="text", text="ok")], stop_reason="end_turn", usage=FakeUsage(1, 1))
    tools = [{"name": "lookup"}]
    provider = make_provider(response=response, tools=tools, max_tokens=256)
    client: FakeAnthropicClient = provider._client

    provider.generate("hello world", {"temperature": 0.3})

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 256
    assert kwargs["messages"] == [{"role": "user", "content": "hello world"}]
    assert kwargs["tools"] == tools
    assert kwargs["temperature"] == 0.3


def test_model_params_max_tokens_overrides_constructor_default():
    response = FakeMessage(content=[FakeContentBlock(type="text", text="ok")], stop_reason="end_turn", usage=FakeUsage(1, 1))
    provider = make_provider(response=response, max_tokens=256)
    client: FakeAnthropicClient = provider._client

    provider.generate("hello", {"max_tokens": 64})

    assert client.messages.calls[0]["max_tokens"] == 64
