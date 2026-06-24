import pytest

from routing_benchmark.models import (
    AgentState,
    IntentComplexity,
    ModelTarget,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
    ToolFailureProfile,
)
from routing_benchmark.routers.commercial_cloud import (
    CloudRouterResponse,
    CloudRouterUnavailableError,
    CommercialCloudRouter,
)


class FakeCloudRouterClient:
    """Stub external commercial router client with configurable behavior."""

    def __init__(self, selected_model: str = "anthropic/claude-3.5-sonnet", raw: dict | None = None, raise_unavailable: bool = False):
        self.selected_model = selected_model
        self.raw = raw or {}
        self.raise_unavailable = raise_unavailable
        self.received_payloads: list[dict] = []

    def submit(self, payload: dict) -> CloudRouterResponse:
        self.received_payloads.append(payload)
        if self.raise_unavailable:
            raise CloudRouterUnavailableError("openrouter unreachable")
        return CloudRouterResponse(selected_model=self.selected_model, raw=self.raw)


def make_request(task_id: str = "task-1") -> RoutingRequest:
    task = TaskCase(
        id=task_id,
        domain="multi_step_calculation",
        complexity=IntentComplexity.COMPLEX_MULTI_TOOL,
        initial_prompt="reconcile these three spreadsheets",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=10,
        expected_tool_calls=2,
    )
    state = AgentState.initial(task)
    features = RoutingFeatures(
        context_tokens_used=2000,
        context_window_limit=8000,
        turn_count=4,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.7,
    )
    return RoutingRequest(state=state, features=features, task=task)


def test_cloud_router_response_rejects_empty_model_id():
    with pytest.raises(ValueError):
        CloudRouterResponse(selected_model="")


def test_routes_using_external_service_selection():
    client = FakeCloudRouterClient(selected_model="anthropic/claude-3.5-sonnet")
    router = CommercialCloudRouter(cloud_router_client=client)

    decision = router.route(make_request())

    assert decision.target is ModelTarget.CLOUD  # default_target
    assert decision.model_id == "anthropic/claude-3.5-sonnet"
    assert decision.reason == "cloud_router:anthropic/claude-3.5-sonnet"


def test_model_target_map_overrides_default_for_known_models():
    client = FakeCloudRouterClient(selected_model="ollama/llama3.1:8b")
    router = CommercialCloudRouter(
        cloud_router_client=client,
        model_target_map={"ollama/llama3.1:8b": ModelTarget.LOCAL},
        default_target=ModelTarget.CLOUD,
    )

    decision = router.route(make_request())

    assert decision.target is ModelTarget.LOCAL
    assert decision.model_id == "ollama/llama3.1:8b"


def test_unmapped_model_falls_back_to_default_target():
    client = FakeCloudRouterClient(selected_model="some/unrecognized-model")
    router = CommercialCloudRouter(
        cloud_router_client=client,
        model_target_map={"ollama/llama3.1:8b": ModelTarget.LOCAL},
        default_target=ModelTarget.CLOUD,
    )

    decision = router.route(make_request())

    assert decision.target is ModelTarget.CLOUD


def test_raw_provider_metadata_is_passed_through_unmodified():
    raw_payload = {"latency_ms": 812, "candidates": ["model-a", "model-b"]}
    client = FakeCloudRouterClient(raw=raw_payload)
    router = CommercialCloudRouter(cloud_router_client=client)

    decision = router.route(make_request())

    assert decision.raw_provider_metadata == raw_payload


def test_payload_includes_task_and_feature_context():
    client = FakeCloudRouterClient()
    router = CommercialCloudRouter(cloud_router_client=client)

    router.route(make_request())

    assert len(client.received_payloads) == 1
    payload = client.received_payloads[0]
    assert payload["task_domain"] == "multi_step_calculation"
    assert payload["intent_complexity"] == "complex_multi_tool"
    assert payload["turn_count"] == 4
    assert payload["context_occupancy_ratio"] == pytest.approx(0.25)
    assert "reconcile these three spreadsheets" in payload["prompt"]


def test_unavailable_error_propagates_unmodified():
    client = FakeCloudRouterClient(raise_unavailable=True)
    router = CommercialCloudRouter(cloud_router_client=client)

    with pytest.raises(CloudRouterUnavailableError):
        router.route(make_request())


def test_reset_is_a_safe_noop():
    client = FakeCloudRouterClient()
    router = CommercialCloudRouter(cloud_router_client=client)
    router.reset()  # must not raise
    decision = router.route(make_request())
    assert decision.model_id == "anthropic/claude-3.5-sonnet"


def test_name_property():
    router = CommercialCloudRouter(cloud_router_client=FakeCloudRouterClient())
    assert router.name == "commercial_cloud"
