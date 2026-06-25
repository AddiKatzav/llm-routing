"""Benchmark Driver for the LLM Routing Benchmark Suite.

Implements the ``BenchmarkDriverProtocol`` and orchestrator described in
section 6 of ``routing_benchmark_spec.md`` and the "Benchmark Driver"
component of section 2.2: it iterates the (task x router x repeat)
evaluation matrix, invoking the Agent Environment (``environment.run_task``)
once per cell.

One deliberate addition beyond the spec's literal protocol: ``run_task``
can raise several *expected* runtime errors -- a context-aware router's
judge timing out (``RouterTimeoutError``), a router selecting a model with
no registered provider (``UnknownModelError``), or a provider/cloud-router
being unreachable (``ProviderUnavailableError``, ``CloudRouterUnavailableError``).
These are not bugs; a judge timeout in particular is exactly the kind of
phenomenon this benchmark exists to measure. Letting any one of them abort
the entire matrix sweep (section 2.2: "ensures isolation between runs")
would silently drop the rest of the dataset. ``run_matrix`` instead catches
this specific set, records a ``RunFailure`` for later inspection, and
continues with a degenerate ``success=False`` ``RunResult`` standing in for
that cell. Any other exception is treated as a real bug and propagates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from routing_benchmark.environment import DEFAULT_CONTEXT_WINDOW_LIMIT, UnknownModelError, run_task
from routing_benchmark.metrics import BaseMetricCollector, RunResult
from routing_benchmark.models import TaskCase
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError
from routing_benchmark.router import BaseRouter, RouterTimeoutError
from routing_benchmark.routers.commercial_cloud import CloudRouterUnavailableError
from routing_benchmark.tooling import BaseMockToolingLayer

__all__ = ["BenchmarkDriverProtocol", "RunFailure", "ShadowConfig", "BenchmarkDriver"]

_EXPECTED_RUN_ERRORS = (
    RouterTimeoutError,
    UnknownModelError,
    ProviderUnavailableError,
    CloudRouterUnavailableError,
)


class BenchmarkDriverProtocol(Protocol):
    """Structural type for the top-level orchestrator, for type-checking
    call sites without forcing a concrete base class dependency.
    """

    def run_matrix(
        self,
        tasks: list[TaskCase],
        routers: list[BaseRouter],
        n_repeats: int,
    ) -> list[RunResult]:
        """Execute every (task, router) pair n_repeats times and collect results."""
        ...


@dataclass(frozen=True)
class RunFailure:
    """Record of one (task, router, repeat) cell that raised an expected
    runtime error instead of completing normally.

    The matrix sweep does not abort when this happens -- see module
    docstring -- so the failure detail (lost from the degenerate RunResult
    placeholder, which has no error field of its own) lives here instead,
    keyed by run_id.
    """

    run_id: str
    task_id: str
    router_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ShadowConfig:
    """Per-router shadow-evaluation config backing spec section 5.3's
    static-vs-dynamic comparative metrics.

    Attributes:
        static_router: Evaluated (not executed) every turn to compute
            Decision Divergence Rate against the live router's choice.
        local_model_id: Looked up in the driver's ``providers`` registry
            to get the LOCAL provider for Escalation Precision/Recall's
            shadow probe, made only on turns the live router escalates.
    """

    static_router: BaseRouter
    local_model_id: str


class BenchmarkDriver:
    """Sweeps a (tasks x routers x repeats) matrix through the Agent Environment."""

    def __init__(
        self,
        providers: dict[str, BaseModelProvider],
        mock_tooling: BaseMockToolingLayer,
        metric_collector: BaseMetricCollector | None = None,
        context_window_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT,
        shadow_configs: dict[str, ShadowConfig] | None = None,
    ) -> None:
        """
        Args:
            shadow_configs: Maps a *live* router's ``name`` to the shadow
                config that should run alongside it -- e.g.
                ``{"context_aware": ShadowConfig(static_router=..., local_model_id=...)}``.
                Routers with no matching entry (typically
                static_semantic itself, and commercial_cloud) get no
                shadow evaluation, matching spec section 7.3's "present
                only on the context_aware entry."
        """
        self.providers = providers
        self.mock_tooling = mock_tooling
        self.metric_collector = metric_collector
        self.context_window_limit = context_window_limit
        self.shadow_configs = shadow_configs or {}
        self.failures: list[RunFailure] = []

    def run_matrix(
        self,
        tasks: list[TaskCase],
        routers: list[BaseRouter],
        n_repeats: int = 1,
    ) -> list[RunResult]:
        """Execute every (task, router) pair n_repeats times and collect results.

        Raises:
            ValueError: If n_repeats < 1.
        """
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")

        results: list[RunResult] = []
        for task in tasks:
            for router in routers:
                for repeat in range(n_repeats):
                    run_id = f"{task.id}:{router.name}:repeat{repeat}"
                    results.append(self._run_one(task, router, run_id))
        return results

    def _run_one(self, task: TaskCase, router: BaseRouter, run_id: str) -> RunResult:
        shadow_config = self.shadow_configs.get(router.name)
        shadow_static_router = shadow_config.static_router if shadow_config else None
        shadow_local_provider = self.providers.get(shadow_config.local_model_id) if shadow_config else None

        try:
            result = run_task(
                task,
                router,
                self.providers,
                self.mock_tooling,
                run_id=run_id,
                context_window_limit=self.context_window_limit,
                shadow_static_router=shadow_static_router,
                shadow_local_provider=shadow_local_provider,
            )
        except _EXPECTED_RUN_ERRORS as exc:
            self.failures.append(
                RunFailure(
                    run_id=run_id,
                    task_id=task.id,
                    router_name=router.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            result = RunResult(
                run_id=run_id,
                task=task,
                router_name=router.name,
                success=False,
                total_turns=0,
                wall_events=0,
                silent_failures_injected=0,
                silent_failures_recovered=0,
                total_cost_usd=0.0,
                turn_metrics=[],
            )

        if self.metric_collector is not None:
            for turn_metric in result.turn_metrics:
                self.metric_collector.record_turn(run_id, turn_metric)
            self.metric_collector.record_run(result)

        return result
