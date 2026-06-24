import pytest

from routing_benchmark.models import ToolCall, ToolFailureProfile
from routing_benchmark.tooling import DeterministicMockToolingLayer


@pytest.fixture
def tool_call() -> ToolCall:
    return ToolCall(tool_name="lookup", arguments={"q": "revenue"}, raw_text="...")


def test_constructor_validation():
    with pytest.raises(ValueError):
        DeterministicMockToolingLayer(single_failure_turn_index=-1)
    with pytest.raises(ValueError):
        DeterministicMockToolingLayer(cascade_start_turn_index=-1)
    with pytest.raises(ValueError):
        DeterministicMockToolingLayer(latency_ms=-1.0)


def test_none_profile_is_always_clean(tool_call):
    layer = DeterministicMockToolingLayer()
    for turn in range(4):
        result = layer.invoke(tool_call, ToolFailureProfile.NONE, turn)
        assert result.success is True
        assert result.is_silently_malformed is False


def test_loud_failure_is_detectable_on_every_turn(tool_call):
    layer = DeterministicMockToolingLayer()
    for turn in range(4):
        result = layer.invoke(tool_call, ToolFailureProfile.LOUD_FAILURE, turn)
        assert result.success is False
        assert result.output is None
        assert result.is_silently_malformed is False


def test_single_silent_failure_fires_only_at_configured_turn(tool_call):
    layer = DeterministicMockToolingLayer(single_failure_turn_index=2)
    results = {turn: layer.invoke(tool_call, ToolFailureProfile.SINGLE_SILENT_FAILURE, turn) for turn in range(4)}

    for turn, result in results.items():
        assert result.success is True
        assert result.is_silently_malformed == (turn == 2)


def test_cascading_silent_failures_fire_from_configured_turn_onward(tool_call):
    layer = DeterministicMockToolingLayer(cascade_start_turn_index=1)
    results = [layer.invoke(tool_call, ToolFailureProfile.CASCADING_SILENT_FAILURES, t) for t in range(4)]

    assert [r.is_silently_malformed for r in results] == [False, True, True, True]
    assert all(r.success for r in results)


def test_malformed_output_is_same_shape_regardless_of_arguments():
    layer = DeterministicMockToolingLayer(single_failure_turn_index=0)
    call_a = ToolCall(tool_name="lookup", arguments={"q": "revenue"}, raw_text="...")
    call_b = ToolCall(tool_name="lookup", arguments={"q": "churn rate"}, raw_text="...")

    result_a = layer.invoke(call_a, ToolFailureProfile.SINGLE_SILENT_FAILURE, 0)
    result_b = layer.invoke(call_b, ToolFailureProfile.SINGLE_SILENT_FAILURE, 0)

    # Different arguments, but the silently malformed output is identical --
    # it does not actually reflect what was asked for.
    assert result_a.output == result_b.output == "lookup_result:0"


def test_clean_output_varies_with_arguments(tool_call):
    layer = DeterministicMockToolingLayer()
    other_call = ToolCall(tool_name="lookup", arguments={"q": "churn rate"}, raw_text="...")

    result_a = layer.invoke(tool_call, ToolFailureProfile.NONE, 0)
    result_b = layer.invoke(other_call, ToolFailureProfile.NONE, 0)

    assert result_a.output != result_b.output
    assert result_a.output.startswith("lookup_result:")


def test_invoke_is_deterministic_across_repeated_calls(tool_call):
    layer = DeterministicMockToolingLayer()
    a = layer.invoke(tool_call, ToolFailureProfile.CASCADING_SILENT_FAILURES, 3)
    b = layer.invoke(tool_call, ToolFailureProfile.CASCADING_SILENT_FAILURES, 3)
    assert a == b


def test_invoke_is_deterministic_across_separate_instances(tool_call):
    layer_a = DeterministicMockToolingLayer()
    layer_b = DeterministicMockToolingLayer()
    assert layer_a.invoke(tool_call, ToolFailureProfile.NONE, 0) == layer_b.invoke(tool_call, ToolFailureProfile.NONE, 0)


def test_unhandled_failure_profile_raises_value_error(tool_call):
    layer = DeterministicMockToolingLayer()
    with pytest.raises(ValueError):
        layer.invoke(tool_call, "totally_unexpected_profile", 0)
