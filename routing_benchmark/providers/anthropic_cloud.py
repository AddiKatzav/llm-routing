"""Anthropic Claude-backed BaseModelProvider.

Wraps the Anthropic Messages API behind the BaseModelProvider contract,
for the benchmark's CLOUD side. The ``anthropic`` SDK is imported lazily
(only when no ``client`` is injected) so the rest of the suite -- and even
this module's own tests -- can run without it installed; see the
``[cloud]`` extra in ``pyproject.toml`` for production use.

Pricing is a static per-model table rather than a live lookup, per spec
section 8's open question ("pricing source of truth ... static config
table vs live API pricing lookup; static table recommended for benchmark
reproducibility"). Update ``_PRICING_PER_MILLION_TOKENS_USD`` if list
pricing changes.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from routing_benchmark.models import CompletionResult, ModelTarget, TokenUsage, ToolCall
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError

__all__ = ["AnthropicCloudProvider"]

_PRICING_PER_MILLION_TOKENS_USD: dict[str, tuple[float, float]] = {
    # model_id -> (input $/1M tokens, output $/1M tokens)
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
}

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


class AnthropicCloudProvider(BaseModelProvider):
    """Calls the Anthropic Messages API for one Claude model.

    Pass ``client`` to inject a pre-built (or fake/test) client and skip
    constructing a real ``anthropic.Anthropic`` instance; in that case
    also pass ``unavailable_error_types`` so this provider knows which
    exceptions from your client mean "unreachable" and should be
    translated to ``ProviderUnavailableError``.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        tools: Optional[list[dict[str, Any]]] = None,
        timeout_s: float = 60.0,
        client: Optional[Any] = None,
        unavailable_error_types: Optional[tuple[type[BaseException], ...]] = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.tools = tools or []

        if client is not None:
            self._client = client
            self._unavailable_error_types: tuple[type[BaseException], ...] = unavailable_error_types or ()
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicCloudProvider requires the 'anthropic' package when no "
                    "client is injected; install it with `pip install anthropic` "
                    "(or the routing-benchmark[cloud] extra)."
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
            self._unavailable_error_types = unavailable_error_types or (
                anthropic.APIConnectionError,
                anthropic.APIStatusError,
                anthropic.APITimeoutError,
            )

    def generate(self, prompt: str, model_params: dict[str, Any]) -> CompletionResult:
        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": model_params.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.tools:
            request_kwargs["tools"] = self.tools
        if "temperature" in model_params:
            request_kwargs["temperature"] = model_params["temperature"]

        start = time.monotonic()
        try:
            response = self._client.messages.create(**request_kwargs)
        except self._unavailable_error_types as exc:
            raise ProviderUnavailableError(f"Anthropic API unreachable: {exc}") from exc
        elapsed_ms = (time.monotonic() - start) * 1000.0

        return self._parse_response(response, elapsed_ms)

    def _parse_response(self, response: Any, elapsed_ms: float) -> CompletionResult:
        text_parts: list[str] = []
        tool_call = None
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use" and tool_call is None:
                tool_call = ToolCall(
                    tool_name=block.name,
                    arguments=dict(block.input),
                    raw_text=str(block.input),
                )

        text = "\n".join(text_parts) if text_parts else None
        finish_reason = _FINISH_REASON_MAP.get(response.stop_reason, response.stop_reason or "stop")

        input_price, output_price = _PRICING_PER_MILLION_TOKENS_USD.get(self.model_id, (0.0, 0.0))
        cost_usd = (
            response.usage.input_tokens * input_price + response.usage.output_tokens * output_price
        ) / 1_000_000

        token_usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cost_usd=cost_usd,
        )

        return CompletionResult(
            text=text,
            tool_call=tool_call,
            finish_reason=finish_reason,
            token_usage=token_usage,
            provider_latency_ms=elapsed_ms,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.CLOUD
