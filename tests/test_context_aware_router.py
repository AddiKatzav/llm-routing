import pytest

from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
    TokenUsage,
    ToolFailureProfile,
)
from routing_benchmark.provider import BaseModelProvider, ProviderUnavailableError
from routing_benchmark.router import RouterTimeoutError
from routing_benchmark.routers.context_aware import ContextAwareRouter


class FakeJudgeProvider(BaseModelProvider):
    """Stub local judge model with configurable response/failure behavior."""

    def __init__(self, response_text: str | None = "LOCAL", raise_unavailable: bool = False) -> None:
        self.response_text = response_text
        self.raise_unavailable = raise_unavailable
        self.calls: list[str] = []

    def generate(self, prompt, model_params):
        self.calls.append(prompt)
        if self.raise_unavailable:
            raise ProviderUnavailableError("judge model unreachable")
        return CompletionResult(
            text=self.response_text,
            tool_call=None,
            finish_reason="stop",
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=1, cost_usd=0.0),
            provider_latency_ms=5.0,
        )

    @property
    def target_class(self) -> ModelTarget:
        return ModelTarget.LOCAL


class SequenceClock:
    """Returns successive values from a fixed list; used to fake elapsed time."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def make_request(
    consecutive_tool_failures: int = 0,
    context_tokens_used: int = 100,
    context_window_limit: int = 8000,
    cumulative_silent_failure_count: int = 0,
    rolling_wall_hit_rate: float = 0.0,
) -> RoutingRequest:
    task = TaskCase(
        id="task-1",
        domain="data_lookup",
        complexity=IntentComplexity.MODERATE,
        initial_prompt="find the revenue figure",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=10,
        expected_tool_calls=1,
    )
    state = AgentState.initial(task)
    features = RoutingFeatures(
        context_tokens_used=context_tokens_used,
        context_window_limit=context_window_limit,
        turn_count=3,
        consecutive_tool_failures=consecutive_tool_failures,
        cumulative_silent_failure_count=cumulative_silent_failure_count,
        rolling_wall_hit_rate=rolling_wall_hit_rate,
        intent_complexity_score=0.4,
    )
    return RoutingRequest(state=state, features=features, task=task)


def test_constructor_validation():
    judge = FakeJudgeProvider()
    with pytest.raises(ValueError):
        ContextAwareRouter(local_judge_model=judge, escalation_threshold=0)
    with pytest.raises(ValueError):
        ContextAwareRouter(local_judge_model=judge, wall_proximity_threshold=0.0)
    with pytest.raises(ValueError):
        ContextAwareRouter(local_judge_model=judge, judge_timeout_ms=0.0)


def test_escalates_on_consecutive_tool_failures_without_calling_judge():
    judge = FakeJudgeProvider()
    router = ContextAwareRouter(local_judge_model=judge, escalation_threshold=2)

    decision = router.route(make_request(consecutive_tool_failures=2))

    assert decision.target is ModelTarget.CLOUD
    assert decision.reason.startswith("failure_escalation:2")
    assert judge.calls == []


def test_escalates_on_context_proximity_to_wall_without_calling_judge():
    judge = FakeJudgeProvider()
    router = ContextAwareRouter(local_judge_model=judge, wall_proximity_threshold=0.85)

    decision = router.route(make_request(context_tokens_used=7000, context_window_limit=8000))

    assert decision.target is ModelTarget.CLOUD
    assert decision.reason.startswith("context_proximity_to_wall:0.8750")
    assert judge.calls == []


def test_consults_judge_when_below_both_thresholds_and_parses_local():
    judge = FakeJudgeProvider(response_text="LOCAL")
    router = ContextAwareRouter(local_judge_model=judge)

    decision = router.route(make_request())

    assert decision.target is ModelTarget.LOCAL
    assert decision.model_id == router.local_model_id
    assert decision.reason == "judge:local"
    assert len(judge.calls) == 1
    assert decision.raw_provider_metadata["judge_token_usage"].prompt_tokens == 30


def test_consults_judge_and_parses_cloud():
    judge = FakeJudgeProvider(response_text="Verdict: CLOUD please")
    router = ContextAwareRouter(local_judge_model=judge)

    decision = router.route(make_request())

    assert decision.target is ModelTarget.CLOUD
    assert decision.model_id == router.cloud_model_id
    assert decision.reason == "judge:cloud"


def test_ambiguous_judge_response_falls_back_to_configured_default():
    judge = FakeJudgeProvider(response_text="I am not sure what to do here")
    router = ContextAwareRouter(local_judge_model=judge, ambiguous_parse_fallback=ModelTarget.CLOUD)

    decision = router.route(make_request())

    assert decision.target is ModelTarget.CLOUD
    assert decision.reason == "judge:cloud:ambiguous_fallback"


def test_judge_provider_unavailable_falls_back_to_cloud_escalation():
    judge = FakeJudgeProvider(raise_unavailable=True)
    router = ContextAwareRouter(local_judge_model=judge)

    decision = router.route(make_request())

    assert decision.target is ModelTarget.CLOUD
    assert decision.reason.startswith("judge_unavailable_escalation:")


def test_judge_timeout_raises_router_timeout_error():
    judge = FakeJudgeProvider(response_text="LOCAL")
    clock = SequenceClock([0.0, 1.0])  # 1.0s elapsed = 1000ms, far over budget
    router = ContextAwareRouter(local_judge_model=judge, judge_timeout_ms=50.0, clock=clock)

    with pytest.raises(RouterTimeoutError):
        router.route(make_request())


def test_reset_is_a_safe_noop():
    judge = FakeJudgeProvider()
    router = ContextAwareRouter(local_judge_model=judge)
    router.reset()  # must not raise
    decision = router.route(make_request())
    assert decision.target is ModelTarget.LOCAL


def test_name_property():
    router = ContextAwareRouter(local_judge_model=FakeJudgeProvider())
    assert router.name == "context_aware"


def test_failure_escalation_takes_priority_over_context_proximity():
    judge = FakeJudgeProvider()
    router = ContextAwareRouter(local_judge_model=judge, escalation_threshold=1, wall_proximity_threshold=0.85)

    decision = router.route(
        make_request(consecutive_tool_failures=1, context_tokens_used=7500, context_window_limit=8000)
    )

    assert decision.reason.startswith("failure_escalation")
    assert judge.calls == []
