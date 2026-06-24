"""Router interface for the LLM Routing Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: the ``BaseRouter``
abstract base class that all three routing paradigms (static semantic,
context-aware dynamic, commercial cloud) must implement, plus the error
types routers are expected to raise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from routing_benchmark.models import RoutingDecision, RoutingRequest


class RouterTimeoutError(Exception):
    """Raised by BaseRouter.route() when the routing decision itself times out.

    Implementations must raise this rather than silently defaulting to a
    fallback target, so timeout events are visible in metrics rather than
    masked as routing accuracy.
    """


class ProviderUnavailableError(Exception):
    """Raised by BaseModelProvider.generate() when the backend cannot be reached.

    Distinguished from a completion-wall event: this is a failed call,
    whereas a wall hit is a successful-but-unusable response.
    """


class BaseRouter(ABC):
    """Abstract base for all routing paradigms under benchmark.

    Implementations must be stateless across runs: any internal state an
    implementation accumulates during a run (e.g. a context-aware router's
    escalation counters) must be cleared by :meth:`reset` at the start of
    each new run, so the Benchmark Driver can reuse one router instance
    across the full evaluation matrix without cross-run contamination.
    """

    @abstractmethod
    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Decide which model/provider should handle the current turn.

        Args:
            request: Current routing request, including state-derived
                features and task metadata.

        Returns:
            A RoutingDecision indicating the selected target, concrete
            model id, and the reasoning trace for that decision.

        Raises:
            RouterTimeoutError: If the router's own decision process
                (e.g. a local judge model call) exceeds its configured
                timeout budget.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-run internal state before starting a new run."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this router, used as a metrics dimension."""
        raise NotImplementedError
