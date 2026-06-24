import pytest

from routing_benchmark.driver import BenchmarkDriver, RunFailure
from routing_benchmark.environment import UnknownModelError
from routing_benchmark.metrics import BaseMetricCollector, RunResult, TurnMetric
from routing_benchmark.models import (
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    RoutingRequest,
    TaskCase,
    TokenUsage,
    ToolFailureProfile,
)
from routing_benchmark.provider import BaseModelProvider
from routing_benchmark.router import BaseRouter, RouterTimeoutError
from routing_benchmark.tooling import BaseMockToolingLayer


def make_task(task_id: str = "task-1", max_turns: int = 1) -> TaskCase:
    return TaskCase(
        id=task_id,
        domain="data_lookup",
        complexity=IntentComplexity.TRIVIAL,
        initial_prompt="What is the revenue?",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=max_turns,
        expected_tool_calls=0,
    )


class FixedRouter(BaseRouter):
    """Always routes to one model_id; counts reset() calls."""

    def __init__(self, model_id: str, router_name: str = "fixed"):
        self.model_id = model_id
        self._name = router_name
        self.reset_calls = 0
        self.route_calls = 0

    def route(self, request: RoutingRequest) -> RoutingDecision:
        self.route_calls += 1
        return RoutingDecision(target=ModelTarget.LOCAL, model_id=self.model_id, reason="fixed")

    def reset(self) -> None:
        self.reset_calls += 1

    @property
    def name(self) -> str:
        return self._name


class AlwaysTimeoutRouter(BaseRouter):
    def reset(self) -> None:
        pass

    def route(self, request: RoutingRequest) -> RoutingDecision:
        raise RouterTimeoutError("judge never responded")

    @property
    def name(self) -> str:
        return "always_timeout"


class BuggyRouter(BaseRouter):
    """Raises an exception type NOT in the driver's expected-error set."""

    def reset(self) -> None:
        pass

    def route(self, request: RoutingRequest) -> RoutingDecision:
        raise ZeroDivisionError("a real bug, not a benchmark signal")

    @property
    def name(self) -> str:
        return "buggy"


class StopImmediatelyProvider(BaseModelProvider):
    """Always returns a single final "stop" completion -- one turn per run."""

    def generate(self, prompt, model_params):
        return CompletionResult(
            text="done",
            tool_call=None,
            finish_reason="stop",
            token_usage=TokenUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.001),
            provider_latency_ms=1.0,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.LOCAL


class NoOpMockTooling(BaseMockToolingLayer):
    def invoke(self, tool_call, failure_profile, turn_index):
        raise AssertionError("no tool calls expected in these driver tests")


class InMemoryCollector(BaseMetricCollector):
    def __init__(self):
        self.turns: list[TurnMetric] = []
        self.runs: list[RunResult] = []

    def record_turn(self, run_id, turn_metric):
        self.turns.append(turn_metric)

    def record_run(self, run_result):
        self.runs.append(run_result)

    def compute_kpis(self, router_name):
        raise NotImplementedError


def make_driver(providers=None, metric_collector=None) -> BenchmarkDriver:
    return BenchmarkDriver(
        providers=providers if providers is not None else {"local-stub": StopImmediatelyProvider()},
        mock_tooling=NoOpMockTooling(),
        metric_collector=metric_collector,
    )


def test_run_matrix_rejects_n_repeats_below_one():
    driver = make_driver()
    with pytest.raises(ValueError):
        driver.run_matrix([make_task()], [FixedRouter("local-stub")], n_repeats=0)


def test_run_matrix_produces_one_result_per_cell_with_correct_run_ids():
    tasks = [make_task("task-a"), make_task("task-b")]
    router_a = FixedRouter("local-stub", router_name="router-a")
    router_b = FixedRouter("local-stub", router_name="router-b")
    driver = make_driver()

    results = driver.run_matrix(tasks, [router_a, router_b], n_repeats=2)

    assert len(results) == 2 * 2 * 2  # tasks x routers x repeats
    run_ids = {r.run_id for r in results}
    assert "task-a:router-a:repeat0" in run_ids
    assert "task-a:router-a:repeat1" in run_ids
    assert "task-b:router-b:repeat1" in run_ids


def test_run_matrix_resets_router_once_per_run():
    task = make_task(max_turns=1)
    router = FixedRouter("local-stub")
    driver = make_driver()

    driver.run_matrix([task], [router], n_repeats=3)

    assert router.reset_calls == 3
    assert router.route_calls == 3  # one route() call per single-turn run


def test_run_matrix_wires_metric_collector():
    task = make_task(max_turns=1)
    router = FixedRouter("local-stub")
    collector = InMemoryCollector()
    driver = make_driver(metric_collector=collector)

    driver.run_matrix([task], [router], n_repeats=1)

    assert len(collector.runs) == 1
    assert len(collector.turns) == 1
    assert collector.runs[0].run_id == "task-1:fixed:repeat0"


def test_run_matrix_survives_unknown_model_and_records_failure():
    task = make_task()
    router = FixedRouter("model-with-no-provider")
    driver = make_driver()  # only registers "local-stub"

    results = driver.run_matrix([task], [router], n_repeats=1)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].total_turns == 0
    assert len(driver.failures) == 1
    failure = driver.failures[0]
    assert isinstance(failure, RunFailure)
    assert failure.error_type == "UnknownModelError"
    assert failure.task_id == "task-1"


def test_run_matrix_survives_router_timeout_and_keeps_sweeping_other_cells():
    good_task = make_task("task-good", max_turns=1)
    bad_router = AlwaysTimeoutRouter()
    good_router = FixedRouter("local-stub", router_name="good")
    driver = make_driver()

    results = driver.run_matrix([good_task], [bad_router, good_router], n_repeats=1)

    assert len(results) == 2
    timeout_result = next(r for r in results if r.router_name == "always_timeout")
    good_result = next(r for r in results if r.router_name == "good")

    assert timeout_result.success is False
    assert good_result.success is True  # the sweep kept going after the timeout
    assert len(driver.failures) == 1
    assert driver.failures[0].error_type == "RouterTimeoutError"


def test_run_matrix_does_not_swallow_unexpected_exceptions():
    driver = make_driver()
    with pytest.raises(ZeroDivisionError):
        driver.run_matrix([make_task()], [BuggyRouter()], n_repeats=1)


def test_failed_run_does_not_record_to_metric_collector_as_turns():
    task = make_task()
    router = FixedRouter("ghost-model")
    collector = InMemoryCollector()
    driver = make_driver(metric_collector=collector)

    driver.run_matrix([task], [router], n_repeats=1)

    assert collector.turns == []
    assert len(collector.runs) == 1
    assert collector.runs[0].success is False
