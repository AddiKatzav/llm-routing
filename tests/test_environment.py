import pytest

from routing_benchmark.environment import (
    UnknownModelError,
    detect_completion_wall,
    extract_features,
    run_task,
)
from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    RoutingRequest,
    TaskCase,
    ToolCall,
    ToolFailureProfile,
    ToolResult,
    TokenUsage,
)
from routing_benchmark.provider import BaseModelProvider
from routing_benchmark.router import BaseRouter
from routing_benchmark.tooling import BaseMockToolingLayer


def make_task(expected_tool_calls: int = 1, max_turns: int = 5, complexity=IntentComplexity.MODERATE) -> TaskCase:
    return TaskCase(
        id="task-1",
        domain="data_lookup",
        complexity=complexity,
        initial_prompt="Find the Q3 revenue figure.",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=max_turns,
        expected_tool_calls=expected_tool_calls,
    )


def make_completion(text=None, tool_call=None, finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    return CompletionResult(
        text=text,
        tool_call=tool_call,
        finish_reason=finish_reason,
        token_usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=0.001),
        provider_latency_ms=10.0,
    )


# ---------------------------------------------------------------------------
# detect_completion_wall
# ---------------------------------------------------------------------------

def test_normal_stop_completion_is_not_a_wall_hit():
    completion = make_completion(text="here is the answer", finish_reason="stop")
    assert detect_completion_wall(completion) is False


def test_empty_completion_with_no_tool_call_is_a_wall_hit():
    completion = make_completion(text="", finish_reason="stop")
    assert detect_completion_wall(completion) is True


def test_truncated_length_without_tool_call_is_a_wall_hit():
    completion = make_completion(text="partial output that got cut off", finish_reason="length")
    assert detect_completion_wall(completion) is True


def test_truncated_length_with_tool_call_is_not_a_wall_hit():
    tool_call = ToolCall(tool_name="lookup", arguments={}, raw_text="...")
    completion = make_completion(text=None, tool_call=tool_call, finish_reason="length")
    assert detect_completion_wall(completion) is False


def test_timeout_is_always_a_wall_hit():
    completion = make_completion(text="some text", finish_reason="timeout")
    assert detect_completion_wall(completion) is True


def test_repeated_ngram_loop_is_a_wall_hit():
    looped_text = "I will try again " * 6
    completion = make_completion(text=looped_text, finish_reason="stop")
    assert detect_completion_wall(completion) is True


def test_short_non_repeating_text_is_not_a_wall_hit():
    completion = make_completion(text="a short reasonable answer", finish_reason="stop")
    assert detect_completion_wall(completion) is False


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

def test_extract_features_on_fresh_state():
    task = make_task(complexity=IntentComplexity.TRIVIAL)
    state = AgentState.initial(task)

    features = extract_features(state, task, context_window_limit=4096)

    assert features.context_tokens_used == 0
    assert features.context_window_limit == 4096
    assert features.turn_count == 0
    assert features.consecutive_tool_failures == 0
    assert features.cumulative_silent_failure_count == 0
    assert features.rolling_wall_hit_rate == 0.0
    assert features.intent_complexity_score == 0.1


def test_extract_features_tracks_consecutive_tool_failures_and_breaks_on_success():
    task = make_task()
    state = AgentState.initial(task)
    decision = RoutingDecision(target=ModelTarget.LOCAL, model_id="x", reason="r")

    failing = ToolResult(success=False, output=None, is_silently_malformed=False, latency_ms=1.0)
    succeeding = ToolResult(success=True, output="ok", is_silently_malformed=False, latency_ms=1.0)
    tool_call = ToolCall(tool_name="lookup", arguments={}, raw_text="...")

    state.append_turn(decision, make_completion(tool_call=tool_call, finish_reason="tool_calls"), 1.0, 1.0, tool_result=failing)
    state.append_turn(decision, make_completion(tool_call=tool_call, finish_reason="tool_calls"), 1.0, 1.0, tool_result=failing)
    features = extract_features(state, task)
    assert features.consecutive_tool_failures == 2

    state.append_turn(decision, make_completion(tool_call=tool_call, finish_reason="tool_calls"), 1.0, 1.0, tool_result=succeeding)
    features = extract_features(state, task)
    assert features.consecutive_tool_failures == 0


def test_extract_features_rolling_wall_hit_rate():
    task = make_task()
    state = AgentState.initial(task)
    decision = RoutingDecision(target=ModelTarget.LOCAL, model_id="x", reason="r")

    state.append_turn(decision, make_completion(text="ok"), 1.0, 1.0)
    state.record_wall_event(ModelTarget.LOCAL)
    state.append_turn(decision, make_completion(text="ok"), 1.0, 1.0)

    features = extract_features(state, task)
    assert features.rolling_wall_hit_rate == pytest.approx(0.5)


def test_extract_features_context_tokens_used_sums_history():
    task = make_task()
    state = AgentState.initial(task)
    decision = RoutingDecision(target=ModelTarget.LOCAL, model_id="x", reason="r")

    state.append_turn(decision, make_completion(text="ok", prompt_tokens=10, completion_tokens=5), 1.0, 1.0)
    state.append_turn(decision, make_completion(text="ok2", prompt_tokens=20, completion_tokens=8), 1.0, 1.0)

    features = extract_features(state, task)
    assert features.context_tokens_used == 43


# ---------------------------------------------------------------------------
# run_task: stubs
# ---------------------------------------------------------------------------

class ScriptedRouter(BaseRouter):
    """Always routes to one fixed model_id; records every request it saw."""

    def __init__(self, model_id: str, target: ModelTarget = ModelTarget.LOCAL):
        self.model_id = model_id
        self.target = target
        self.seen_requests: list[RoutingRequest] = []
        self.reset_calls = 0

    def route(self, request: RoutingRequest) -> RoutingDecision:
        self.seen_requests.append(request)
        return RoutingDecision(target=self.target, model_id=self.model_id, reason="scripted")

    def reset(self) -> None:
        self.reset_calls += 1

    @property
    def name(self) -> str:
        return "scripted"


class ScriptedProvider(BaseModelProvider):
    """Returns completions from a fixed script, one per call; errors if exhausted."""

    def __init__(self, completions: list[CompletionResult], target_class: ModelTarget = ModelTarget.LOCAL):
        self._completions = list(completions)
        self._target_class = target_class
        self.call_count = 0

    def generate(self, prompt, model_params):
        if self.call_count >= len(self._completions):
            raise AssertionError("ScriptedProvider exhausted: run_task made more calls than expected")
        completion = self._completions[self.call_count]
        self.call_count += 1
        return completion

    @property
    def target_class(self) -> ModelTarget:
        return self._target_class


class ScriptedMockTooling(BaseMockToolingLayer):
    """Returns tool results from a fixed script, one per invoke() call."""

    def __init__(self, results: list[ToolResult]):
        self._results = list(results)
        self.call_count = 0

    def invoke(self, tool_call, failure_profile, turn_index):
        result = self._results[self.call_count]
        self.call_count += 1
        return result


def make_tool_call(name="lookup"):
    return ToolCall(tool_name=name, arguments={"q": "revenue"}, raw_text="...")


# ---------------------------------------------------------------------------
# run_task: end-to-end scenarios
# ---------------------------------------------------------------------------

def test_run_task_clean_success_path_terminates_naturally():
    task = make_task(expected_tool_calls=1, max_turns=5)
    router = ScriptedRouter(model_id="local-stub")
    provider = ScriptedProvider([
        make_completion(tool_call=make_tool_call(), finish_reason="tool_calls"),
        make_completion(text="Here is the revenue figure.", finish_reason="stop"),
    ])
    tooling = ScriptedMockTooling([
        ToolResult(success=True, output="Q3 revenue: $4.2M", is_silently_malformed=False, latency_ms=2.0),
    ])

    result = run_task(task, router, {"local-stub": provider}, tooling, run_id="run-1")

    assert result.success is True
    assert result.total_turns == 2
    assert result.wall_events == 0
    assert result.silent_failures_injected == 0
    assert router.reset_calls == 1
    assert provider.call_count == 2
    assert tooling.call_count == 1


def test_run_task_raises_on_unknown_model_id():
    task = make_task(max_turns=3)
    router = ScriptedRouter(model_id="ghost-model")
    provider = ScriptedProvider([make_completion(text="ok")])
    tooling = ScriptedMockTooling([])

    with pytest.raises(UnknownModelError):
        run_task(task, router, {"local-stub": provider}, tooling)


def test_run_task_records_wall_hits_and_runs_to_max_turns_without_natural_stop():
    task = make_task(expected_tool_calls=0, max_turns=3)
    router = ScriptedRouter(model_id="local-stub")
    # finish_reason "length" with no tool_call on every turn: never naturally
    # terminates, so the loop must run exactly max_turns times.
    provider = ScriptedProvider([
        make_completion(text="cut off", finish_reason="length"),
        make_completion(text="cut off", finish_reason="length"),
        make_completion(text="cut off", finish_reason="length"),
    ])
    tooling = ScriptedMockTooling([])

    result = run_task(task, router, {"local-stub": provider}, tooling, run_id="run-2")

    assert result.total_turns == 3
    assert result.wall_events == 3
    assert provider.call_count == 3


def test_run_task_unrecovered_silent_failure_yields_failure():
    task = make_task(expected_tool_calls=1, max_turns=5)
    router = ScriptedRouter(model_id="local-stub")
    provider = ScriptedProvider([
        make_completion(tool_call=make_tool_call(), finish_reason="tool_calls"),
        make_completion(text="Done, I believe that's correct.", finish_reason="stop"),
    ])
    tooling = ScriptedMockTooling([
        ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=2.0),
    ])

    result = run_task(task, router, {"local-stub": provider}, tooling)

    assert result.silent_failures_injected == 1
    assert result.silent_failures_recovered == 0
    assert result.success is False


def test_run_task_recovered_silent_failure_yields_success():
    task = make_task(expected_tool_calls=1, max_turns=5)
    router = ScriptedRouter(model_id="local-stub")
    provider = ScriptedProvider([
        make_completion(tool_call=make_tool_call(), finish_reason="tool_calls"),
        make_completion(tool_call=make_tool_call(), finish_reason="tool_calls"),
        make_completion(text="Done.", finish_reason="stop"),
    ])
    tooling = ScriptedMockTooling([
        ToolResult(success=True, output="looks fine", is_silently_malformed=True, latency_ms=2.0),
        ToolResult(success=True, output="actually fine", is_silently_malformed=False, latency_ms=2.0),
    ])

    result = run_task(task, router, {"local-stub": provider}, tooling)

    assert result.silent_failures_injected == 1
    assert result.silent_failures_recovered == 1
    assert result.success is True


def test_run_task_passes_context_window_limit_into_routing_features():
    task = make_task(max_turns=1)
    router = ScriptedRouter(model_id="local-stub")
    provider = ScriptedProvider([make_completion(text="ok", finish_reason="stop")])
    tooling = ScriptedMockTooling([])

    run_task(task, router, {"local-stub": provider}, tooling, context_window_limit=2048)

    assert len(router.seen_requests) == 1
    assert router.seen_requests[0].features.context_window_limit == 2048


def test_run_task_generates_run_id_when_omitted():
    task = make_task(max_turns=1)
    router = ScriptedRouter(model_id="local-stub")
    provider = ScriptedProvider([make_completion(text="ok", finish_reason="stop")])
    tooling = ScriptedMockTooling([])

    result = run_task(task, router, {"local-stub": provider}, tooling)

    assert isinstance(result.run_id, str) and len(result.run_id) > 0
