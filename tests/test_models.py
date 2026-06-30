import pytest

from conftest import make_task
from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    ContextDepthLevel,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    RoutingFeatures,
    TaskCase,
    ToolCall,
    ToolFailureProfile,
    ToolResult,
    TokenUsage,
)


def make_token_usage(prompt=10, completion=5, cost=0.001) -> TokenUsage:
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost)


def test_token_usage_total_and_validation():
    usage = make_token_usage()
    assert usage.total_tokens == 15
    with pytest.raises(ValueError):
        TokenUsage(prompt_tokens=-1, completion_tokens=0, cost_usd=0.0)
    with pytest.raises(ValueError):
        TokenUsage(prompt_tokens=0, completion_tokens=0, cost_usd=-0.01)


def test_completion_result_requests_tool_call():
    no_tool = CompletionResult(
        text="hello",
        tool_call=None,
        finish_reason="stop",
        token_usage=make_token_usage(),
        provider_latency_ms=12.0,
    )
    assert no_tool.requests_tool_call is False

    with_tool = CompletionResult(
        text=None,
        tool_call=ToolCall(tool_name="lookup", arguments={"q": "revenue"}, raw_text="..."),
        finish_reason="tool_calls",
        token_usage=make_token_usage(),
        provider_latency_ms=12.0,
    )
    assert with_tool.requests_tool_call is True


def test_routing_features_context_occupancy_ratio():
    features = RoutingFeatures(
        context_tokens_used=4000,
        context_window_limit=8000,
        turn_count=3,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.5,
    )
    assert features.context_occupancy_ratio == 0.5

    bad_features = RoutingFeatures(
        context_tokens_used=1,
        context_window_limit=0,
        turn_count=0,
        consecutive_tool_failures=0,
        cumulative_silent_failure_count=0,
        rolling_wall_hit_rate=0.0,
        intent_complexity_score=0.0,
    )
    with pytest.raises(ValueError):
        _ = bad_features.context_occupancy_ratio


def test_routing_decision_requires_model_id():
    with pytest.raises(ValueError):
        RoutingDecision(target=ModelTarget.LOCAL, model_id="", reason="x")

    decision = RoutingDecision(target=ModelTarget.CLOUD, model_id="claude-sonnet-4-6", reason="ok")
    assert decision.target is ModelTarget.CLOUD


def test_tool_result_rejects_negative_latency():
    with pytest.raises(ValueError):
        ToolResult(success=True, output="ok", is_silently_malformed=False, latency_ms=-1.0)


def test_tool_result_allows_success_and_silent_failure_together():
    result = ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=5.0)
    assert result.success is True
    assert result.is_silently_malformed is True


def test_task_case_validation():
    with pytest.raises(ValueError):
        TaskCase(
            id="bad",
            domain="data_lookup",
            complexity=IntentComplexity.TRIVIAL,
            initial_prompt="x",
            synthetic_history=[],
            failure_profile=ToolFailureProfile.NONE,
            max_turns=0,
            expected_tool_calls=0,
        )


def test_agent_state_initial_and_terminal_by_max_turns():
    task = make_task(max_turns=2)
    state = AgentState.initial(task)
    assert state.task_id == "task-1"
    assert state.turn_count == 0
    assert state.is_terminal() is False

    decision = RoutingDecision(target=ModelTarget.LOCAL, model_id="llama3.1:8b", reason="static")
    completion = CompletionResult(
        text="working on it",
        tool_call=None,
        finish_reason="stop",
        token_usage=make_token_usage(),
        provider_latency_ms=10.0,
    )

    state.append_turn(decision, completion, routing_latency_ms=1.0, inference_latency_ms=100.0)
    assert state.turn_count == 1
    assert state.is_terminal() is False

    state.append_turn(decision, completion, routing_latency_ms=1.0, inference_latency_ms=100.0)
    assert state.turn_count == 2
    assert state.is_terminal() is True


def test_agent_state_explicit_terminal_flag():
    state = AgentState.initial(make_task(max_turns=100))
    assert state.is_terminal() is False
    state.terminal = True
    assert state.is_terminal() is True


def test_agent_state_to_prompt_includes_tool_trace():
    task = make_task()
    state = AgentState.initial(task)
    decision = RoutingDecision(target=ModelTarget.LOCAL, model_id="llama3.1:8b", reason="static")
    tool_call = ToolCall(tool_name="lookup", arguments={"q": "revenue"}, raw_text="...")
    completion = CompletionResult(
        text=None,
        tool_call=tool_call,
        finish_reason="tool_calls",
        token_usage=make_token_usage(),
        provider_latency_ms=10.0,
    )
    tool_result = ToolResult(success=True, output="Q3 revenue: $4.2M", is_silently_malformed=False, latency_ms=2.0)

    state.append_turn(decision, completion, 1.0, 100.0, tool_result=tool_result)
    prompt = state.to_prompt()

    assert "Find the Q3 revenue figure." in prompt
    assert "ToolCall[lookup]" in prompt
    assert "Q3 revenue: $4.2M" in prompt


def test_agent_state_record_wall_event_and_silent_failure():
    state = AgentState.initial(make_task())
    state.record_wall_event(ModelTarget.LOCAL)
    state.record_wall_event(ModelTarget.LOCAL)
    assert state.wall_events == 2

    loud_result = ToolResult(success=False, output=None, is_silently_malformed=False, latency_ms=1.0)
    silent_result = ToolResult(success=True, output="fine?", is_silently_malformed=True, latency_ms=1.0)

    state.record_tool_result(loud_result, silent_failure=False)
    state.record_tool_result(silent_result, silent_failure=True)

    assert state.silent_failure_log == [silent_result]


@pytest.mark.parametrize("depth", list(ContextDepthLevel))
def test_context_depth_level_values_are_strings(depth):
    assert isinstance(depth.value, str)
