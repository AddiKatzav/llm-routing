"""Metric Collector interface and result structures for the LLM Routing
Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: ``TurnMetric`` and
``RunResult`` (with real ``from_state``/``finalize`` logic, not ``...``
placeholders), ``KPISummary``, and the ``BaseMetricCollector`` abstract
interface.

Two deliberate deviations from the spec's literal method signatures, made
because the draft left them underspecified:

1. ``TurnMetric.from_state`` and ``RunResult.finalize`` take explicit
   ``run_id`` and ``router_name`` arguments. ``AgentState`` only carries a
   ``task_id`` (shared across repeated runs of the same TaskCase), and a
   router instance is never threaded through ``AgentState`` -- so neither
   value is otherwise recoverable at the point these classmethods are
   called.
2. ``RunResult.finalize``'s success/recovery criteria are not defined
   anywhere in section 5 beyond "task-specific completion criterion met
   AND no unresolved silent failure". This module implements a concrete,
   documented default: a silent failure is considered *recovered* if a
   later turn in the same run makes a clean (non-malformed, successful)
   call to the same tool; the run is *successful* if it made at least
   ``task.expected_tool_calls`` tool calls and every silent failure that
   occurred was recovered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from routing_benchmark.models import AgentState, RoutingDecision, TaskCase, TokenUsage

__all__ = ["TurnMetric", "RunResult", "KPISummary", "BaseMetricCollector"]


@dataclass(frozen=True)
class TurnMetric:
    """Per-turn metric record, the atomic unit persisted to the Run Store."""

    run_id: str
    turn_index: int
    router_name: str
    routing_decision: RoutingDecision
    routing_latency_ms: float
    inference_latency_ms: float
    wall_hit: bool
    silent_failure_detected: bool
    token_usage: TokenUsage

    @classmethod
    def from_state(
        cls,
        run_id: str,
        router_name: str,
        state: AgentState,
        decision: RoutingDecision,
        wall_hit: bool,
        silent_failure: bool,
    ) -> "TurnMetric":
        """Construct a TurnMetric from the most recently appended turn.

        Must be called after ``state.append_turn(...)`` for the turn being
        recorded -- it reads ``state.history[-1]`` for latencies and token
        usage, matching the order of operations in the benchmark's per-turn
        loop (append_turn happens before metric capture).

        Raises:
            ValueError: If ``state.history`` is empty.
        """
        if not state.history:
            raise ValueError(
                "AgentState has no turns recorded; call state.append_turn() "
                "before TurnMetric.from_state()"
            )
        last_record = state.history[-1]
        return cls(
            run_id=run_id,
            turn_index=state.turn_count - 1,
            router_name=router_name,
            routing_decision=decision,
            routing_latency_ms=last_record.routing_latency_ms,
            inference_latency_ms=last_record.inference_latency_ms,
            wall_hit=wall_hit,
            silent_failure_detected=silent_failure,
            token_usage=last_record.completion.token_usage,
        )


def _tool_name_at(history, index: int) -> str | None:
    tool_call = history[index].completion.tool_call
    return tool_call.tool_name if tool_call is not None else None


def _is_recovered(history, index: int) -> bool:
    """Whether the silent failure at ``history[index]`` was later recovered.

    Recovery heuristic: a later turn makes a clean (successful,
    non-malformed) call to the same tool.
    """
    failed_tool_name = _tool_name_at(history, index)
    for later in history[index + 1 :]:
        result = later.tool_result
        if result is None or result.is_silently_malformed or not result.success:
            continue
        later_tool_name = later.completion.tool_call.tool_name if later.completion.tool_call else None
        if later_tool_name == failed_tool_name:
            return True
    return False


@dataclass(frozen=True)
class RunResult:
    """Finalized outcome of a single benchmark run."""

    run_id: str
    task: TaskCase
    router_name: str
    success: bool
    total_turns: int
    wall_events: int
    silent_failures_injected: int
    silent_failures_recovered: int
    total_cost_usd: float
    turn_metrics: list[TurnMetric]

    @classmethod
    def finalize(
        cls,
        run_id: str,
        router_name: str,
        task: TaskCase,
        state: AgentState,
        metrics: list[TurnMetric],
    ) -> "RunResult":
        """Compute the final success/cost/failure-recovery summary for a run.

        See module docstring for the concrete success and recovery
        criteria used here.
        """
        tool_calls_made = sum(1 for record in state.history if record.completion.requests_tool_call)

        silent_failure_indices = [
            i
            for i, record in enumerate(state.history)
            if record.tool_result is not None and record.tool_result.is_silently_malformed
        ]
        silent_failures_injected = len(silent_failure_indices)
        silent_failures_recovered = sum(
            1 for i in silent_failure_indices if _is_recovered(state.history, i)
        )

        success = (
            tool_calls_made >= task.expected_tool_calls
            and silent_failures_recovered == silent_failures_injected
        )

        total_cost_usd = sum(metric.token_usage.cost_usd for metric in metrics)

        return cls(
            run_id=run_id,
            task=task,
            router_name=router_name,
            success=success,
            total_turns=state.turn_count,
            wall_events=state.wall_events,
            silent_failures_injected=silent_failures_injected,
            silent_failures_recovered=silent_failures_recovered,
            total_cost_usd=total_cost_usd,
            turn_metrics=list(metrics),
        )


@dataclass(frozen=True)
class KPISummary:
    """Aggregate KPI values for one router across all matrix runs."""

    router_name: str
    wall_avoidance_rate: float
    routing_overhead_p50_ms: float
    routing_overhead_p95_ms: float
    task_success_rate: float
    silent_failure_recovery_rate: float
    cost_efficiency: float
    effective_cost_per_success_usd: float
    sample_size: int


class BaseMetricCollector(ABC):
    """Abstract base for per-turn and per-run metric capture and persistence."""

    @abstractmethod
    def record_turn(self, run_id: str, turn_metric: TurnMetric) -> None:
        """Persist a single turn's metrics, append-only."""
        raise NotImplementedError

    @abstractmethod
    def record_run(self, run_result: RunResult) -> None:
        """Persist the finalized aggregate metrics for a completed run."""
        raise NotImplementedError

    @abstractmethod
    def compute_kpis(self, router_name: str) -> KPISummary:
        """Aggregate persisted runs into the KPI set defined in spec section 5.1.

        Args:
            router_name: Restrict aggregation to runs using this router.

        Returns:
            A KPISummary with WAR, routing overhead, TSR, SFRR, and CE
            computed per the formulas in section 5.1.
        """
        raise NotImplementedError
