"""Local Ollama-backed BaseModelProvider.

Wraps Ollama's HTTP chat API (``/api/chat``) behind the BaseModelProvider
contract, so the benchmark can route turns to an actual local model rather
than a stub. Uses only the standard library (``urllib``) -- no extra
dependency is needed for the LOCAL side of the benchmark, unlike
``AnthropicCloudProvider``, which talks to a billed external API and needs
its vendor SDK.

The HTTP call itself is isolated behind an injectable ``post_json``
callable so this module's logic (payload construction, response parsing,
error translation) can be exercised in tests without a real Ollama server
running -- see ``tests/test_ollama_provider.py``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from routing_benchmark.models import CompletionResult, ModelTarget, TokenUsage, ToolCall
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError

__all__ = ["OllamaProvider"]

PostJsonFn = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _default_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ProviderUnavailableError(f"ollama request to {url} failed: {exc}") from exc


class OllamaProvider(BaseModelProvider):
    """Calls a local Ollama server's ``/api/chat`` endpoint for one model.

    Token usage: Ollama reports ``prompt_eval_count``/``eval_count``
    (prompt and completion token counts) in its response; ``cost_usd`` is
    always 0.0 since local inference has no per-token API cost in this
    benchmark's pricing model.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        tools: Optional[list[dict[str, Any]]] = None,
        default_options: Optional[dict[str, Any]] = None,
        post_json: PostJsonFn = _default_post_json,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.tools = tools or []
        # Constructor-level defaults (e.g. num_predict/temperature to bound
        # generation length on a real local model); a router's per-call
        # model_params override these on key conflicts.
        self.default_options = default_options or {}
        self._post_json = post_json

    def generate(self, prompt: str, model_params: dict[str, Any]) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        options = {**self.default_options, **{k: v for k, v in model_params.items() if k != "tools"}}
        if options:
            payload["options"] = options
        if self.tools:
            payload["tools"] = self.tools

        start = time.monotonic()
        body = self._post_json(f"{self.base_url}/api/chat", payload, self.timeout_s)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        return self._parse_response(body, elapsed_ms)

    def _parse_response(self, body: dict[str, Any], elapsed_ms: float) -> CompletionResult:
        message = body.get("message", {}) or {}
        text = message.get("content") or None

        tool_call = None
        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            function = raw_tool_calls[0].get("function", {}) or {}
            tool_call = ToolCall(
                tool_name=function.get("name", ""),
                arguments=function.get("arguments") or {},
                raw_text=json.dumps(raw_tool_calls[0]),
            )

        done_reason = body.get("done_reason")
        if done_reason:
            finish_reason = done_reason
        elif tool_call is not None:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop" if body.get("done", True) else "length"

        token_usage = TokenUsage(
            prompt_tokens=body.get("prompt_eval_count", 0) or 0,
            completion_tokens=body.get("eval_count", 0) or 0,
            cost_usd=0.0,
        )

        # Ollama reports its own server-side total_duration in nanoseconds;
        # prefer that when present, fall back to our own wall-clock
        # measurement around the request otherwise.
        reported_duration_ms = (body.get("total_duration") or 0) / 1_000_000
        provider_latency_ms = reported_duration_ms if reported_duration_ms > 0 else elapsed_ms

        return CompletionResult(
            text=text,
            tool_call=tool_call,
            finish_reason=finish_reason,
            token_usage=token_usage,
            provider_latency_ms=provider_latency_ms,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.LOCAL
