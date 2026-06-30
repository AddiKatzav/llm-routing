import pytest

from conftest import make_completion, make_decision, make_task
from routing_benchmark.metrics import BaseMetricCollector, KPISummary, RunResult, TurnMetric, compute_kpis
from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    TaskCase,
    ToolCall,
    ToolFailureProfile,
    ToolResult,
    TokenUsage,
)


def append_tool_turn(state: AgentState, tool_name: str, tool_result: ToolResult, cost=0.001) -> None:
    decision = make_decision()
    tool_call = ToolCall(tool_name=tool_name, arguments={}, raw_text="...")
    completion = make_completion(tool_call=tool_call, cost=cost)
    state.append_turn(decision, completion, routing_latency_ms=2.0, inference_latency_ms=80.0, tool_result=tool_result)


# ---------------------------------------------------------------------------
# TurnMetric.from_state
# ---------------------------------------------------------------------------

def test_from_state_raises_on_empty_history():
    state = AgentState.initial(make_task())
    with pytest.raises(ValueError):
        TurnMetric.from_state(
            run_id="run-1",
            router_name="static_semantic",
            state=state,
            decision=make_decision(),
            wall_hit=False,
            silent_failure=False,
        )


def test_from_state_pulls_latencies_and_tokens_from_last_turn():
    state = AgentState.initial(make_task())
    decision = make_decision()
    completion = make_completion()
    state.append_turn(decision, completion, routing_latency_ms=3.5, inference_latency_ms=120.0)

    metric = TurnMetric.from_state(
        run_id="run-1",
        router_name="static_semantic",
        state=state,
        decision=decision,
        wall_hit=True,
        silent_failure=False,
    )

    assert metric.turn_index == 0
    assert metric.routing_latency_ms == 3.5
    assert metric.inference_latency_ms == 120.0
    assert metric.token_usage == completion.token_usage
    assert metric.wall_hit is True
    assert metric.silent_failure_detected is False


# ---------------------------------------------------------------------------
# RunResult.finalize
# ---------------------------------------------------------------------------

def test_finalize_success_with_no_failures():
    task = make_task(expected_tool_calls=1)
    state = AgentState.initial(task)
    clean_result = ToolResult(success=True, output="ok", is_silently_malformed=False, latency_ms=5.0)
    append_tool_turn(state, "lookup", clean_result)

    metric = TurnMetric.from_state(
        run_id="run-1", router_name="static_semantic", state=state,
        decision=make_decision(), wall_hit=False, silent_failure=False,
    )
    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=[metric])

    assert result.success is True
    assert result.silent_failures_injected == 0
    assert result.silent_failures_recovered == 0
    assert result.total_turns == 1
    assert result.total_cost_usd == pytest.approx(0.001)


def test_finalize_unrecovered_silent_failure_is_not_success():
    task = make_task(expected_tool_calls=1)
    state = AgentState.initial(task)
    malformed = ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=5.0)
    append_tool_turn(state, "lookup", malformed)

    metric = TurnMetric.from_state(
        run_id="run-1", router_name="static_semantic", state=state,
        decision=make_decision(), wall_hit=False, silent_failure=True,
    )
    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=[metric])

    assert result.silent_failures_injected == 1
    assert result.silent_failures_recovered == 0
    assert result.success is False


def test_finalize_recovered_silent_failure_via_later_clean_call():
    task = make_task(expected_tool_calls=1)
    state = AgentState.initial(task)
    malformed = ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=5.0)
    clean = ToolResult(success=True, output="actually fine", is_silently_malformed=False, latency_ms=5.0)

    append_tool_turn(state, "lookup", malformed)
    append_tool_turn(state, "lookup", clean)

    metrics = [
        TurnMetric.from_state(run_id="run-1", router_name="static_semantic", state=state, decision=make_decision(), wall_hit=False, silent_failure=False)
    ]
    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=metrics)

    assert result.silent_failures_injected == 1
    assert result.silent_failures_recovered == 1
    assert result.success is True


def test_finalize_recovery_requires_same_tool_name():
    task = make_task(expected_tool_calls=1)
    state = AgentState.initial(task)
    malformed = ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=5.0)
    clean_other_tool = ToolResult(success=True, output="unrelated", is_silently_malformed=False, latency_ms=5.0)

    append_tool_turn(state, "lookup", malformed)
    append_tool_turn(state, "other_tool", clean_other_tool)

    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=[])

    assert result.silent_failures_recovered == 0
    assert result.success is False


def test_finalize_insufficient_tool_calls_is_not_success():
    task = make_task(expected_tool_calls=2)
    state = AgentState.initial(task)
    clean_result = ToolResult(success=True, output="ok", is_silently_malformed=False, latency_ms=5.0)
    append_tool_turn(state, "lookup", clean_result)

    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=[])

    assert result.success is False


def test_finalize_sums_cost_across_metrics():
    task = make_task(expected_tool_calls=0)
    state = AgentState.initial(task)
    state.append_turn(make_decision(), make_completion(cost=0.01), routing_latency_ms=1.0, inference_latency_ms=10.0)
    state.append_turn(make_decision(), make_completion(cost=0.02), routing_latency_ms=1.0, inference_latency_ms=10.0)

    # Built directly (rather than via from_state, which only reads the
    # latest turn) so both turns' costs are reflected.
    metrics = [
        TurnMetric(
            run_id="run-1", turn_index=0, router_name="static_semantic",
            routing_decision=make_decision(), routing_latency_ms=1.0, inference_latency_ms=10.0,
            wall_hit=False, silent_failure_detected=False,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.01),
        ),
        TurnMetric(
            run_id="run-1", turn_index=1, router_name="static_semantic",
            routing_decision=make_decision(), routing_latency_ms=1.0, inference_latency_ms=10.0,
            wall_hit=False, silent_failure_detected=False,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.02),
        ),
    ]

    result = RunResult.finalize(run_id="run-1", router_name="static_semantic", task=task, state=state, metrics=metrics)
    assert result.total_cost_usd == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# BaseMetricCollector
# ---------------------------------------------------------------------------

def test_base_metric_collector_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMetricCollector()  # type: ignore[abstract]


class InMemoryMetricCollector(BaseMetricCollector):
    """Minimal concrete collector used to exercise the BaseMetricCollector contract."""

    def __init__(self) -> None:
        self.turns: list[TurnMetric] = []
        self.runs: list[RunResult] = []

    def record_turn(self, run_id: str, turn_metric: TurnMetric) -> None:
        self.turns.append(turn_metric)

    def record_run(self, run_result: RunResult) -> None:
        self.runs.append(run_result)

    def compute_kpis(self, router_name: str) -> KPISummary:
        runs = [r for r in self.runs if r.router_name == router_name]
        turns = [t for t in self.turns if t.router_name == router_name]
        return compute_kpis(router_name, runs, turns)


def test_concrete_collector_satisfies_contract_and_computes_kpis():
    collector = InMemoryMetricCollector()
    task = make_task(expected_tool_calls=0)

    # Two independent runs: a cheap successful one and an expensive failed
    # one (failure is asserted directly via RunResult, since this task's
    # expected_tool_calls=0 makes every run trivially "successful" through
    # RunResult.finalize -- here we want to exercise the failure-aggregation
    # path in compute_kpis, not re-derive a failing finalize() scenario
    # already covered above).
    success_state = AgentState.initial(task)
    success_state.append_turn(make_decision(), make_completion(cost=0.01), routing_latency_ms=5.0, inference_latency_ms=50.0)
    success_metric = TurnMetric.from_state(
        run_id="run-success", router_name="static_semantic", state=success_state,
        decision=make_decision(), wall_hit=False, silent_failure=False,
    )
    collector.record_turn("run-success", success_metric)
    collector.record_run(RunResult.finalize(
        run_id="run-success", router_name="static_semantic", task=task, state=success_state, metrics=[success_metric],
    ))

    failed_state = AgentState.initial(task)
    failed_state.append_turn(make_decision(), make_completion(cost=0.02), routing_latency_ms=5.0, inference_latency_ms=50.0)
    failed_state.record_wall_event(ModelTarget.LOCAL)
    failed_metric = TurnMetric.from_state(
        run_id="run-failed", router_name="static_semantic", state=failed_state,
        decision=make_decision(), wall_hit=True, silent_failure=False,
    )
    collector.record_turn("run-failed", failed_metric)
    collector.record_run(RunResult(
        run_id="run-failed", task=task, router_name="static_semantic", success=False,
        total_turns=failed_state.turn_count, wall_events=failed_state.wall_events,
        silent_failures_injected=0, silent_failures_recovered=0,
        total_cost_usd=0.02, turn_metrics=[failed_metric],
    ))

    summary = collector.compute_kpis("static_semantic")

    assert summary.sample_size == 2
    assert summary.wall_avoidance_rate == pytest.approx(0.5)
    assert summary.task_success_rate == pytest.approx(0.5)
    assert summary.silent_failure_recovery_rate == 1.0  # no silent failures injected
    # total_cost_usd is summed across ALL runs (0.01 + 0.02), divided by
    # the single successful run, per the spec's "total_cost_actual /
    # successful_runs" formula.
    assert summary.effective_cost_per_success_usd == pytest.approx(0.03)


def test_compute_kpis_raises_for_unknown_router():
    collector = InMemoryMetricCollector()
    with pytest.raises(ValueError):
        collector.compute_kpis("never_recorded")


def test_missing_abstract_method_blocks_instantiation():
    class IncompleteCollector(BaseMetricCollector):
        def record_turn(self, run_id, turn_metric):
            pass

        def record_run(self, run_result):
            pass

        # compute_kpis intentionally omitted

    with pytest.raises(TypeError):
        IncompleteCollector()  # type: ignore[abstract]
