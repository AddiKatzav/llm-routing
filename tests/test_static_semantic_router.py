import pytest

from routing_benchmark.embedding import HashingEmbedder
from routing_benchmark.models import (
    AgentState,
    IntentComplexity,
    ModelTarget,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
    ToolFailureProfile,
)
from routing_benchmark.routers.static_semantic import IntentExemplar, StaticSemanticRouter


EXEMPLARS = [
    IntentExemplar(
        label="simple_lookup",
        example_text="what is the capital of france",
        target=ModelTarget.LOCAL,
        model_id="llama3.1:8b",
    ),
    IntentExemplar(
        label="complex_multi_tool",
        example_text="orchestrate a multi step financial analysis across several spreadsheets and tools",
        target=ModelTarget.CLOUD,
        model_id="claude-sonnet-4-6",
    ),
]


def make_request(prompt: str, task_id: str = "task-1", turn_count: int = 0) -> RoutingRequest:
    task = TaskCase(
        id=task_id,
        domain="data_lookup",
        complexity=IntentComplexity.TRIVIAL,
        initial_prompt=prompt,
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=10,
        expected_tool_calls=0,
    )
    state = AgentState.initial(task)
    state.turn_count = turn_count
    features = RoutingFeatures(
        context_tokens_used=100,
        context_window_limit=8000,
        turn_count=turn_count,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.0,
    )
    return RoutingRequest(state=state, features=features, task=task)


class CountingEmbedder:
    """Wraps HashingEmbedder to count embed() calls, for cache-hit assertions."""

    def __init__(self):
        self._inner = HashingEmbedder(n_dims=128)
        self.call_count = 0

    def embed(self, text: str):
        self.call_count += 1
        return self._inner.embed(text)


def test_requires_at_least_one_exemplar():
    with pytest.raises(ValueError):
        StaticSemanticRouter(intent_exemplars=[])


def test_routes_to_nearest_exemplar_by_intent():
    router = StaticSemanticRouter(intent_exemplars=EXEMPLARS)

    simple_decision = router.route(make_request("what is the capital of spain"))
    assert simple_decision.target is ModelTarget.LOCAL
    assert simple_decision.model_id == "llama3.1:8b"
    assert simple_decision.reason.startswith("semantic:simple_lookup")

    complex_decision = router.route(
        make_request(
            "orchestrate a multi step financial analysis across spreadsheets and tools",
            task_id="task-2",
        )
    )
    assert complex_decision.target is ModelTarget.CLOUD
    assert complex_decision.model_id == "claude-sonnet-4-6"


def test_empty_prompt_falls_back_to_default():
    router = StaticSemanticRouter(
        intent_exemplars=EXEMPLARS,
        default_target=ModelTarget.LOCAL,
        default_model_id="phi3:mini",
    )
    decision = router.route(make_request("   "))
    assert decision.target is ModelTarget.LOCAL
    assert decision.model_id == "phi3:mini"
    assert decision.reason == "semantic:empty_query_fallback"


def test_decision_is_cached_per_task_and_blind_to_later_state():
    embedder = CountingEmbedder()
    router = StaticSemanticRouter(intent_exemplars=EXEMPLARS, embedder=embedder)
    # Construction embeds every exemplar once.
    calls_after_construction = embedder.call_count
    assert calls_after_construction == len(EXEMPLARS)

    request_turn_0 = make_request("what is the capital of spain", task_id="task-1", turn_count=0)
    first = router.route(request_turn_0)
    assert embedder.call_count == calls_after_construction + 1

    # Same task, much later turn, completely different (cloud-leaning) text --
    # the static router must NOT re-embed or change its decision.
    request_turn_5 = make_request(
        "orchestrate a multi step financial analysis across spreadsheets and tools",
        task_id="task-1",
        turn_count=5,
    )
    second = router.route(request_turn_5)

    assert second == first
    assert embedder.call_count == calls_after_construction + 1


def test_reset_clears_cache_and_allows_recomputation():
    embedder = CountingEmbedder()
    router = StaticSemanticRouter(intent_exemplars=EXEMPLARS, embedder=embedder)
    calls_after_construction = embedder.call_count

    router.route(make_request("what is the capital of spain", task_id="task-1"))
    assert embedder.call_count == calls_after_construction + 1

    router.reset()
    router.route(make_request("what is the capital of spain", task_id="task-1"))
    assert embedder.call_count == calls_after_construction + 2


def test_different_tasks_are_classified_independently():
    router = StaticSemanticRouter(intent_exemplars=EXEMPLARS)

    decision_a = router.route(make_request("what is the capital of spain", task_id="task-a"))
    decision_b = router.route(
        make_request(
            "orchestrate a multi step financial analysis across spreadsheets and tools",
            task_id="task-b",
        )
    )

    assert decision_a.target is ModelTarget.LOCAL
    assert decision_b.target is ModelTarget.CLOUD


def test_name_property():
    router = StaticSemanticRouter(intent_exemplars=EXEMPLARS)
    assert router.name == "static_semantic"
