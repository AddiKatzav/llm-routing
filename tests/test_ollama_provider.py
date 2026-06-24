import pytest

from routing_benchmark.models import ModelTarget
from routing_benchmark.provider import ProviderUnavailableError
from routing_benchmark.providers.ollama import OllamaProvider


class RecordingPostJson:
    """Stub transport: records the request, returns a preset response."""

    def __init__(self, response: dict | None = None, raise_unavailable: bool = False):
        self.response = response or {"message": {"content": "the answer is 42"}, "done": True}
        self.raise_unavailable = raise_unavailable
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url, payload, timeout_s):
        self.calls.append((url, payload, timeout_s))
        if self.raise_unavailable:
            raise ProviderUnavailableError("ollama daemon down")
        return self.response


def test_constructor_validation():
    with pytest.raises(ValueError):
        OllamaProvider(model_id="")
    with pytest.raises(ValueError):
        OllamaProvider(model_id="llama3.1:8b", timeout_s=0.0)


def test_target_class_is_local():
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=RecordingPostJson())
    assert provider.target_class is ModelTarget.LOCAL


def test_generate_parses_plain_text_response():
    transport = RecordingPostJson(response={
        "message": {"content": "the answer is 42"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 6,
        "total_duration": 250_000_000,  # ns -> 250ms
    })
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    result = provider.generate("what is the answer?", {})

    assert result.text == "the answer is 42"
    assert result.tool_call is None
    assert result.finish_reason == "stop"
    assert result.token_usage.prompt_tokens == 12
    assert result.token_usage.completion_tokens == 6
    assert result.token_usage.cost_usd == 0.0
    assert result.provider_latency_ms == pytest.approx(250.0)


def test_generate_parses_tool_call_response():
    transport = RecordingPostJson(response={
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"q": "revenue"}}}
            ],
        },
        "done": True,
    })
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    result = provider.generate("look it up", {})

    assert result.tool_call is not None
    assert result.tool_call.tool_name == "lookup"
    assert result.tool_call.arguments == {"q": "revenue"}
    assert result.finish_reason == "tool_calls"


def test_generate_infers_length_finish_reason_when_not_done():
    transport = RecordingPostJson(response={"message": {"content": "partial..."}, "done": False})
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    result = provider.generate("go on", {})
    assert result.finish_reason == "length"


def test_generate_falls_back_to_measured_latency_when_total_duration_missing():
    transport = RecordingPostJson(response={"message": {"content": "ok"}, "done": True})
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    result = provider.generate("hello", {})
    assert result.provider_latency_ms >= 0.0


def test_generate_propagates_provider_unavailable_error():
    transport = RecordingPostJson(raise_unavailable=True)
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    with pytest.raises(ProviderUnavailableError):
        provider.generate("hello", {})


def test_payload_includes_model_messages_and_tools():
    transport = RecordingPostJson()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    provider = OllamaProvider(model_id="llama3.1:8b", base_url="http://example:11434", tools=tools, post_json=transport)

    provider.generate("hello world", {"temperature": 0.2})

    assert len(transport.calls) == 1
    url, payload, timeout_s = transport.calls[0]
    assert url == "http://example:11434/api/chat"
    assert payload["model"] == "llama3.1:8b"
    assert payload["messages"] == [{"role": "user", "content": "hello world"}]
    assert payload["tools"] == tools
    assert payload["options"] == {"temperature": 0.2}


def test_payload_omits_tools_key_when_no_tools_configured():
    transport = RecordingPostJson()
    provider = OllamaProvider(model_id="llama3.1:8b", post_json=transport)

    provider.generate("hello", {})

    _, payload, _ = transport.calls[0]
    assert "tools" not in payload


def test_default_options_merge_with_per_call_model_params():
    transport = RecordingPostJson()
    provider = OllamaProvider(
        model_id="llama3.1:8b",
        default_options={"num_predict": 256, "temperature": 0.8},
        post_json=transport,
    )

    provider.generate("hello", {"temperature": 0.1})

    _, payload, _ = transport.calls[0]
    # Per-call model_params override the constructor default on conflict,
    # but defaults not overridden are still present.
    assert payload["options"] == {"num_predict": 256, "temperature": 0.1}
