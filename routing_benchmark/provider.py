"""Model Provider interface for the LLM Routing Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: the
``BaseModelProvider`` abstract base class that wraps any concrete backend
(local Ollama model, cloud API model) behind one uniform call signature, so
the Agent Environment and Router Interface never need to know which kind of
backend they are talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from routing_benchmark.models import CompletionResult, ModelTarget

# Re-exported for convenience: BaseModelProvider.generate() raises this on
# backend unavailability, not RouterTimeoutError (which is a router-side
# decision-latency concern, not a provider-side connectivity concern).
from routing_benchmark.router import ProviderUnavailableError

__all__ = ["BaseModelProvider", "ProviderUnavailableError"]


class BaseModelProvider(ABC):
    """Abstract base for a callable backend (local or cloud).

    A single ModelProvider instance corresponds to one concrete model id;
    the Agent Environment resolves ``RoutingDecision.model_id`` to a
    provider instance via a registry maintained outside this module.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model_params: dict[str, Any],
    ) -> CompletionResult:
        """Produce a single completion for the given prompt.

        Args:
            prompt: Fully-rendered prompt string for this turn.
            model_params: Generation parameter overrides from the
                RoutingDecision (temperature, max_tokens, etc.).

        Returns:
            A normalized CompletionResult, regardless of underlying
            provider SDK shape.

        Raises:
            ProviderUnavailableError: If the backend cannot be reached
                (e.g. local Ollama daemon down, cloud API auth failure).
                This must be distinguished from a completion-wall event,
                which is a successful response that is unusable, not a
                failed call.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_class(self) -> ModelTarget:
        """Whether this provider counts as LOCAL or CLOUD for KPI bucketing."""
        raise NotImplementedError
