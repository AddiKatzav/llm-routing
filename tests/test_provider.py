import pytest

from routing_benchmark.models import CompletionResult, ModelTarget, TokenUsage
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError


def test_base_model_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModelProvider()  # type: ignore[abstract]


class StubLocalProvider(BaseModelProvider):
    """Minimal concrete provider used to exercise the BaseModelProvider contract."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls: list[str] = []

    def generate(self, prompt, model_params):
        self.calls.append(prompt)
        if not self.available:
            raise ProviderUnavailableError("ollama daemon unreachable")
        return CompletionResult(
            text="stub response",
            tool_call=None,
            finish_reason="stop",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.0),
            provider_latency_ms=42.0,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.LOCAL


class StubCloudProvider(BaseModelProvider):
    def generate(self, prompt, model_params):
        return CompletionResult(
            text="cloud response",
            tool_call=None,
            finish_reason="stop",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.002),
            provider_latency_ms=300.0,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.CLOUD


def test_concrete_provider_satisfies_contract():
    provider = StubLocalProvider()
    result = provider.generate("hello", {})

    assert result.text == "stub response"
    assert provider.target_class is ModelTarget.LOCAL
    assert provider.calls == ["hello"]


def test_provider_unavailable_error_propagates():
    provider = StubLocalProvider(available=False)
    with pytest.raises(ProviderUnavailableError):
        provider.generate("hello", {})


def test_target_class_distinguishes_providers():
    assert StubLocalProvider().target_class is ModelTarget.LOCAL
    assert StubCloudProvider().target_class is ModelTarget.CLOUD


def test_missing_abstract_method_blocks_instantiation():
    class IncompleteProvider(BaseModelProvider):
        def generate(self, prompt, model_params):
            return None

        # target_class intentionally omitted

    with pytest.raises(TypeError):
        IncompleteProvider()  # type: ignore[abstract]
