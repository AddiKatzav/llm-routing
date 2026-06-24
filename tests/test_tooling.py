import pytest

from routing_benchmark.models import ToolCall, ToolFailureProfile, ToolResult
from routing_benchmark.tooling import BaseMockToolingLayer


def test_base_mock_tooling_layer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMockToolingLayer()  # type: ignore[abstract]


class DeterministicToolingLayer(BaseMockToolingLayer):
    """Minimal concrete tooling layer exercising the failure-injection contract.

    - NONE: always a clean success.
    - SINGLE_SILENT_FAILURE: silently malformed on the first turn only.
    - CASCADING_SILENT_FAILURES: silently malformed on every turn from the
      second turn onward.
    - LOUD_FAILURE: an explicit, detectable failure (success=False).
    """

    def invoke(self, tool_call: ToolCall, failure_profile: ToolFailureProfile, turn_index: int) -> ToolResult:
        if failure_profile == ToolFailureProfile.NONE:
            return ToolResult(success=True, output="ok", is_silently_malformed=False, latency_ms=5.0)

        if failure_profile == ToolFailureProfile.LOUD_FAILURE:
            return ToolResult(success=False, output=None, is_silently_malformed=False, latency_ms=5.0)

        if failure_profile == ToolFailureProfile.SINGLE_SILENT_FAILURE:
            malformed = turn_index == 0
            return ToolResult(success=True, output="looks fine", is_silently_malformed=malformed, latency_ms=5.0)

        if failure_profile == ToolFailureProfile.CASCADING_SILENT_FAILURES:
            malformed = turn_index >= 1
            return ToolResult(success=True, output="looks fine", is_silently_malformed=malformed, latency_ms=5.0)

        raise ValueError(f"unhandled failure profile: {failure_profile}")


@pytest.fixture
def tool_call() -> ToolCall:
    return ToolCall(tool_name="lookup", arguments={"q": "revenue"}, raw_text="...")


def test_none_profile_is_always_clean(tool_call):
    layer = DeterministicToolingLayer()
    for turn in range(3):
        result = layer.invoke(tool_call, ToolFailureProfile.NONE, turn)
        assert result.success is True
        assert result.is_silently_malformed is False


def test_loud_failure_is_detectable_not_silent(tool_call):
    layer = DeterministicToolingLayer()
    result = layer.invoke(tool_call, ToolFailureProfile.LOUD_FAILURE, 0)
    assert result.success is False
    assert result.is_silently_malformed is False


def test_single_silent_failure_only_on_first_turn(tool_call):
    layer = DeterministicToolingLayer()
    first = layer.invoke(tool_call, ToolFailureProfile.SINGLE_SILENT_FAILURE, 0)
    second = layer.invoke(tool_call, ToolFailureProfile.SINGLE_SILENT_FAILURE, 1)

    assert first.success is True and first.is_silently_malformed is True
    assert second.success is True and second.is_silently_malformed is False


def test_cascading_silent_failures_from_second_turn_onward(tool_call):
    layer = DeterministicToolingLayer()
    results = [layer.invoke(tool_call, ToolFailureProfile.CASCADING_SILENT_FAILURES, t) for t in range(3)]

    assert [r.is_silently_malformed for r in results] == [False, True, True]
    assert all(r.success for r in results)


def test_invoke_is_deterministic_given_same_inputs(tool_call):
    layer = DeterministicToolingLayer()
    a = layer.invoke(tool_call, ToolFailureProfile.SINGLE_SILENT_FAILURE, 0)
    b = layer.invoke(tool_call, ToolFailureProfile.SINGLE_SILENT_FAILURE, 0)
    assert a == b


def test_missing_abstract_method_blocks_instantiation():
    class IncompleteToolingLayer(BaseMockToolingLayer):
        pass

    with pytest.raises(TypeError):
        IncompleteToolingLayer()  # type: ignore[abstract]
