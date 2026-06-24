import pytest

from routing_benchmark.environment import extract_features
from routing_benchmark.models import (
    AgentState,
    IntentComplexity,
    SyntheticTurn,
    TaskCase,
    ToolFailureProfile,
)


def make_task(synthetic_history=None) -> TaskCase:
    return TaskCase(
        id="task-1",
        domain="data_lookup",
        complexity=IntentComplexity.MODERATE,
        initial_prompt="What is the revenue for Acme Corp?",
        synthetic_history=synthetic_history or [],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=5,
        expected_tool_calls=1,
    )


def test_synthetic_turn_rejects_negative_tokens():
    with pytest.raises(ValueError):
        SyntheticTurn(speaker="user", text="hi", approx_tokens=-1)


def test_agent_state_with_no_synthetic_history_has_zero_prefix():
    state = AgentState.initial(make_task())
    assert state.synthetic_prefix_tokens == 0
    assert "User: What is the revenue for Acme Corp?" in state.to_prompt()


def test_agent_state_folds_synthetic_history_into_prefix_tokens():
    history = [
        SyntheticTurn(speaker="user", text="earlier question", approx_tokens=10),
        SyntheticTurn(speaker="assistant", text="earlier answer", approx_tokens=15),
    ]
    state = AgentState.initial(make_task(synthetic_history=history))
    assert state.synthetic_prefix_tokens == 25


def test_agent_state_to_prompt_includes_synthetic_transcript_before_user_turn():
    history = [SyntheticTurn(speaker="user", text="earlier question", approx_tokens=10)]
    state = AgentState.initial(make_task(synthetic_history=history))

    prompt = state.to_prompt()
    lines = prompt.splitlines()

    assert "User: earlier question" in lines
    assert lines.index("User: earlier question") < lines.index(
        "User: What is the revenue for Acme Corp?"
    )


def test_extract_features_includes_synthetic_prefix_in_context_tokens_used():
    history = [SyntheticTurn(speaker="user", text="padding", approx_tokens=500)]
    task = make_task(synthetic_history=history)
    state = AgentState.initial(task)

    features = extract_features(state, task, context_window_limit=8000)

    assert features.context_tokens_used == 500
    assert features.context_occupancy_ratio == pytest.approx(0.0625)
