import pytest

from routing_benchmark.models import (
    AgentState,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
    ToolFailureProfile,
)
from routing_benchmark.router import BaseRouter, RouterTimeoutError


def make_request() -> RoutingRequest:
    task = TaskCase(
        id="task-1",
        domain="data_lookup",
        complexity=IntentComplexity.TRIVIAL,
        initial_prompt="hello",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=5,
        expected_tool_calls=0,
    )
    state = AgentState.initial(task)
    features = RoutingFeatures(
        context_tokens_used=100,
        context_window_limit=8000,
        turn_count=0,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.1,
    )
    return RoutingRequest(state=state, features=features, task=task)


def test_base_router_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRouter()  # type: ignore[abstract]


class AlwaysLocalRouter(BaseRouter):
    """Minimal concrete router used to exercise the BaseRouter contract."""

    def __init__(self) -> None:
        self._route_calls = 0

    def route(self, request: RoutingRequest) -> RoutingDecision:
        self._route_calls += 1
        return RoutingDecision(target=ModelTarget.LOCAL, model_id="llama3.1:8b", reason="always_local")

    def reset(self) -> None:
        self._route_calls = 0

    @property
    def name(self) -> str:
        return "always_local"


class FlakyJudgeRouter(BaseRouter):
    """Concrete router that raises RouterTimeoutError to verify error contract."""

    def route(self, request: RoutingRequest) -> RoutingDecision:
        raise RouterTimeoutError("local judge model did not respond in time")

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "flaky_judge"


def test_concrete_router_satisfies_contract():
    router = AlwaysLocalRouter()
    request = make_request()

    decision = router.route(request)

    assert decision.target is ModelTarget.LOCAL
    assert decision.model_id == "llama3.1:8b"
    assert router.name == "always_local"
    assert router._route_calls == 1

    router.reset()
    assert router._route_calls == 0


def test_router_timeout_error_propagates():
    router = FlakyJudgeRouter()
    request = make_request()

    with pytest.raises(RouterTimeoutError):
        router.route(request)


def test_missing_abstract_method_blocks_instantiation():
    class IncompleteRouter(BaseRouter):
        def route(self, request: RoutingRequest) -> RoutingDecision:
            return RoutingDecision(target=ModelTarget.CLOUD, model_id="x", reason="x")

        @property
        def name(self) -> str:
            return "incomplete"

        # reset() intentionally omitted

    with pytest.raises(TypeError):
        IncompleteRouter()  # type: ignore[abstract]
