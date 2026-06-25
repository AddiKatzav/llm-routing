"""Tests for spec section 5.3's static-vs-dynamic comparative metrics:
TurnMetric's shadow fields, environment.run_task's shadow evaluation,
BenchmarkDriver's ShadowConfig wiring, and persistence's
compute_comparative_metrics/write_kpi_summary.
"""

import pytest

from routing_benchmark.driver import BenchmarkDriver, ShadowConfig
from routing_benchmark.metrics import RunResult, TurnMetric
from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
    ToolCall,
    ToolFailureProfile,
    TokenUsage,
)
from routing_benchmark.persistence import JsonlCsvMetricCollector
from routing_benchmark.provider import BaseModelProvider
from routing_benchmark.router import BaseRouter
from routing_benchmark.tooling import BaseMockToolingLayer
from routing_benchmark import environment


# ---------------------------------------------------------------------------
# TurnMetric shadow fields
# ---------------------------------------------------------------------------

def make_task(max_turns: int = 5) -> TaskCase:
    return TaskCase(
        id="task-1",
        domain="data_lookup",
        complexity=IntentComplexity.MODERATE,
        initial_prompt="Find the Q3 revenue figure.",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=max_turns,
        expected_tool_calls=0,
    )


def make_decision(target=ModelTarget.LOCAL, model_id="local-model") -> RoutingDecision:
    return RoutingDecision(target=target, model_id=model_id, reason="r")


def make_completion(text="ok", finish_reason="stop", cost=0.001) -> CompletionResult:
    return CompletionResult(
        text=text,
        tool_call=None,
        finish_reason=finish_reason,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=cost),
        provider_latency_ms=5.0,
    )


def make_features(occupancy_tokens=100, window=1000) -> RoutingFeatures:
    return RoutingFeatures(
        context_tokens_used=occupancy_tokens,
        context_window_limit=window,
        turn_count=0,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.5,
    )


def test_turn_metric_shadow_fields_default_to_none():
    task = make_task()
    state = AgentState.initial(task)
    decision = make_decision()
    state.append_turn(decision, make_completion(), 1.0, 10.0)

    metric = TurnMetric.from_state(
        run_id="run-1", router_name="static_semantic", state=state,
        decision=decision, wall_hit=False, silent_failure=False,
    )

    assert metric.context_occupancy_ratio is None
    assert metric.shadow_static_target is None
    assert metric.shadow_local_wall_hit is None
    assert metric.shadow_call_cost_usd is None


def test_turn_metric_records_shadow_fields_when_provided():
    task = make_task()
    state = AgentState.initial(task)
    decision = make_decision(target=ModelTarget.CLOUD)
    state.append_turn(decision, make_completion(), 1.0, 10.0)
    features = make_features(occupancy_tokens=300, window=1000)
    shadow_decision = make_decision(target=ModelTarget.LOCAL)

    metric = TurnMetric.from_state(
        run_id="run-1", router_name="context_aware", state=state,
        decision=decision, wall_hit=False, silent_failure=False,
        features=features,
        shadow_static_decision=shadow_decision,
        shadow_local_wall_hit=True,
        shadow_call_cost_usd=0.0005,
    )

    assert metric.context_occupancy_ratio == pytest.approx(0.3)
    assert metric.shadow_static_target is ModelTarget.LOCAL
    assert metric.shadow_local_wall_hit is True
    assert metric.shadow_call_cost_usd == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# environment.run_task shadow evaluation
# ---------------------------------------------------------------------------

class RecordingRouter(BaseRouter):
    """Routes per a fixed script of decisions, one per call; records requests."""

    def __init__(self, decisions: list[RoutingDecision], name: str = "scripted"):
        self._decisions = list(decisions)
        self._name = name
        self.call_count = 0
        self.reset_calls = 0
        self.seen_requests: list[RoutingRequest] = []

    def route(self, request: RoutingRequest) -> RoutingDecision:
        self.seen_requests.append(request)
        decision = self._decisions[min(self.call_count, len(self._decisions) - 1)]
        self.call_count += 1
        return decision

    def reset(self) -> None:
        self.reset_calls += 1

    @property
    def name(self) -> str:
        return self._name


class ScriptedProvider(BaseModelProvider):
    def __init__(self, completions: list[CompletionResult], target_class=ModelTarget.LOCAL):
        self._completions = list(completions)
        self.call_count = 0
        self.seen_prompts: list[str] = []
        self._target_class = target_class

    def generate(self, prompt, model_params):
        self.seen_prompts.append(prompt)
        completion = self._completions[min(self.call_count, len(self._completions) - 1)]
        self.call_count += 1
        return completion

    @property
    def target_class(self):
        return self._target_class


class NoOpMockTooling(BaseMockToolingLayer):
    def invoke(self, tool_call, failure_profile, turn_index):
        raise AssertionError("no tool calls expected in these tests")


def test_run_task_records_decision_divergence_when_shadow_static_router_differs():
    task = make_task(max_turns=1)
    live_router = RecordingRouter([make_decision(target=ModelTarget.CLOUD, model_id="cloud-model")], name="context_aware")
    shadow_static_router = RecordingRouter([make_decision(target=ModelTarget.LOCAL, model_id="local-model")], name="static_semantic")

    providers = {
        "cloud-model": ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.CLOUD),
    }

    result = environment.run_task(
        task, live_router, providers, NoOpMockTooling(),
        shadow_static_router=shadow_static_router,
    )

    assert shadow_static_router.reset_calls == 1
    metric = result.turn_metrics[0]
    assert metric.shadow_static_target is ModelTarget.LOCAL
    assert metric.routing_decision.target is ModelTarget.CLOUD


def test_run_task_makes_shadow_local_call_only_when_escalated():
    task = make_task(max_turns=1)
    live_router = RecordingRouter([make_decision(target=ModelTarget.CLOUD, model_id="cloud-model")], name="context_aware")
    shadow_local = ScriptedProvider([make_completion(text="local would say this", finish_reason="stop", cost=0.0002)])

    providers = {
        "cloud-model": ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.CLOUD),
    }

    result = environment.run_task(
        task, live_router, providers, NoOpMockTooling(),
        shadow_local_provider=shadow_local,
    )

    assert shadow_local.call_count == 1
    metric = result.turn_metrics[0]
    assert metric.shadow_local_wall_hit is False  # natural "stop" completion, not a wall hit
    assert metric.shadow_call_cost_usd == pytest.approx(0.0002)


def test_run_task_skips_shadow_local_call_when_not_escalated():
    task = make_task(max_turns=1)
    live_router = RecordingRouter([make_decision(target=ModelTarget.LOCAL, model_id="local-model")], name="context_aware")
    shadow_local = ScriptedProvider([make_completion(finish_reason="stop")])

    providers = {
        "local-model": ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.LOCAL),
    }

    result = environment.run_task(
        task, live_router, providers, NoOpMockTooling(),
        shadow_local_provider=shadow_local,
    )

    assert shadow_local.call_count == 0
    metric = result.turn_metrics[0]
    assert metric.shadow_local_wall_hit is None
    assert metric.shadow_call_cost_usd is None


def test_run_task_shadow_local_uses_same_prompt_as_live_call():
    task = make_task(max_turns=1)
    live_router = RecordingRouter([make_decision(target=ModelTarget.CLOUD, model_id="cloud-model")], name="context_aware")
    live_provider = ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.CLOUD)
    shadow_local = ScriptedProvider([make_completion(finish_reason="stop")])

    providers = {"cloud-model": live_provider}

    environment.run_task(
        task, live_router, providers, NoOpMockTooling(),
        shadow_local_provider=shadow_local,
    )

    assert live_provider.seen_prompts == shadow_local.seen_prompts


def test_run_task_records_context_occupancy_ratio():
    task = make_task(max_turns=1)
    live_router = RecordingRouter([make_decision(target=ModelTarget.LOCAL, model_id="local-model")], name="static_semantic")
    providers = {"local-model": ScriptedProvider([make_completion(finish_reason="stop")])}

    result = environment.run_task(task, live_router, providers, NoOpMockTooling(), context_window_limit=1000)

    metric = result.turn_metrics[0]
    assert metric.context_occupancy_ratio is not None
    assert metric.context_occupancy_ratio == pytest.approx(0.0)  # no history yet at turn 0


# ---------------------------------------------------------------------------
# BenchmarkDriver ShadowConfig wiring
# ---------------------------------------------------------------------------

def test_driver_wires_shadow_config_only_for_matching_router_name():
    task = make_task(max_turns=1)

    context_aware_router = RecordingRouter([make_decision(target=ModelTarget.CLOUD, model_id="cloud-model")], name="context_aware")
    static_router = RecordingRouter([make_decision(target=ModelTarget.LOCAL, model_id="local-model")], name="static_semantic")
    shadow_static_router = RecordingRouter([make_decision(target=ModelTarget.LOCAL, model_id="local-model")], name="static_semantic")

    providers = {
        "cloud-model": ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.CLOUD),
        "local-model": ScriptedProvider([make_completion(finish_reason="stop")], target_class=ModelTarget.LOCAL),
    }

    driver = BenchmarkDriver(
        providers=providers,
        mock_tooling=NoOpMockTooling(),
        shadow_configs={
            "context_aware": ShadowConfig(static_router=shadow_static_router, local_model_id="local-model"),
        },
    )

    results = driver.run_matrix([task], [context_aware_router, static_router], n_repeats=1)

    context_aware_result = next(r for r in results if r.router_name == "context_aware")
    static_result = next(r for r in results if r.router_name == "static_semantic")

    # context_aware got shadow evaluation (divergence populated).
    assert context_aware_result.turn_metrics[0].shadow_static_target is ModelTarget.LOCAL
    # static_semantic has no matching ShadowConfig entry -- no shadow data.
    assert static_result.turn_metrics[0].shadow_static_target is None


# ---------------------------------------------------------------------------
# persistence: compute_comparative_metrics / write_kpi_summary
# ---------------------------------------------------------------------------

def _turn_metric(
    run_id, router_name="context_aware", target=ModelTarget.LOCAL,
    shadow_static_target=None, shadow_local_wall_hit=None, wall_hit=False,
    context_occupancy_ratio=0.5,
):
    return TurnMetric(
        run_id=run_id,
        turn_index=0,
        router_name=router_name,
        routing_decision=RoutingDecision(target=target, model_id="x", reason="r"),
        routing_latency_ms=1.0,
        inference_latency_ms=10.0,
        wall_hit=wall_hit,
        silent_failure_detected=False,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        context_occupancy_ratio=context_occupancy_ratio,
        shadow_static_target=shadow_static_target,
        shadow_local_wall_hit=shadow_local_wall_hit,
    )


def _run_result(run_id, router_name, task, turn_metrics):
    return RunResult(
        run_id=run_id, task=task, router_name=router_name, success=True,
        total_turns=len(turn_metrics), wall_events=0, silent_failures_injected=0,
        silent_failures_recovered=0, total_cost_usd=0.0, turn_metrics=turn_metrics,
    )


def test_compute_comparative_metrics_returns_none_without_shadow_data(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()
    metric = _turn_metric("run-1")
    collector.record_turn("run-1", metric)
    collector.record_run(_run_result("run-1", "context_aware", task, [metric]))

    assert collector.compute_comparative_metrics("context_aware") is None


def test_compute_comparative_metrics_divergence_and_precision_recall(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()

    # Turn 1: live=LOCAL, shadow_static=LOCAL -> agree (no divergence), real wall_hit=False.
    m1 = _turn_metric("run-1", target=ModelTarget.LOCAL, shadow_static_target=ModelTarget.LOCAL, wall_hit=False)
    # Turn 2: live=CLOUD (escalated), shadow_static=LOCAL -> divergence; shadow probe says local WOULD have wall-hit (true positive).
    m2 = _turn_metric("run-1", target=ModelTarget.CLOUD, shadow_static_target=ModelTarget.LOCAL, shadow_local_wall_hit=True, context_occupancy_ratio=0.8)
    # Turn 3: live=LOCAL, shadow_static=LOCAL -> agree, but real wall_hit=True (a false negative: should have escalated but didn't).
    m3 = _turn_metric("run-1", target=ModelTarget.LOCAL, shadow_static_target=ModelTarget.LOCAL, wall_hit=True)

    for m in (m1, m2, m3):
        collector.record_turn("run-1", m)
    collector.record_run(_run_result("run-1", "context_aware", task, [m1, m2, m3]))

    comparative = collector.compute_comparative_metrics("context_aware")

    assert comparative["decision_divergence_rate"] == pytest.approx(1 / 3)
    assert comparative["escalation_precision"] == pytest.approx(1.0)  # 1 true positive / 1 escalation
    assert comparative["escalation_recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)
    assert comparative["escalation_lead_time_headroom_mean"] == pytest.approx(0.2)  # 1 - 0.8


def test_write_kpi_summary_includes_comparative_block_only_when_present(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()

    shadowed = _turn_metric("run-1", router_name="context_aware", shadow_static_target=ModelTarget.LOCAL)
    collector.record_turn("run-1", shadowed)
    collector.record_run(_run_result("run-1", "context_aware", task, [shadowed]))

    plain = _turn_metric("run-2", router_name="static_semantic")
    collector.record_turn("run-2", plain)
    collector.record_run(_run_result("run-2", "static_semantic", task, [plain]))

    collector.write_kpi_summary("context_aware")
    collector.write_kpi_summary("static_semantic")

    import json
    persisted = json.loads(collector.kpi_summary_path.read_text(encoding="utf-8"))

    assert "comparative_static_vs_dynamic" in persisted["context_aware"]
    assert "comparative_static_vs_dynamic" not in persisted["static_semantic"]


def test_record_turn_persists_shadow_fields_to_jsonl(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    metric = _turn_metric("run-1", shadow_static_target=ModelTarget.CLOUD, shadow_local_wall_hit=True)
    collector.record_turn("run-1", metric)

    import json
    line = collector.turns_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)

    assert record["shadow_static_decision_target"] == "cloud"
    assert record["shadow_local_wall_hit"] is True
    assert record["context_occupancy_ratio"] == pytest.approx(0.5)
