import sys
from pathlib import Path

# Allow test files to do `from conftest import ...`
sys.path.insert(0, str(Path(__file__).parent))

from routing_benchmark.models import (
    CompletionResult,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    TaskCase,
    TokenUsage,
    ToolFailureProfile,
)


def make_task(
    task_id="task-1",
    domain="data_lookup",
    complexity=IntentComplexity.MODERATE,
    initial_prompt="Find the Q3 revenue figure.",
    synthetic_history=None,
    failure_profile=ToolFailureProfile.NONE,
    max_turns=5,
    expected_tool_calls=1,
    context_depth=None,
) -> TaskCase:
    kwargs = dict(
        id=task_id,
        domain=domain,
        complexity=complexity,
        initial_prompt=initial_prompt,
        synthetic_history=synthetic_history or [],
        failure_profile=failure_profile,
        max_turns=max_turns,
        expected_tool_calls=expected_tool_calls,
    )
    if context_depth is not None:
        kwargs["context_depth"] = context_depth
    return TaskCase(**kwargs)


def make_completion(
    text=None,
    tool_call=None,
    finish_reason=None,
    prompt_tokens=10,
    completion_tokens=5,
    cost=0.001,
    provider_latency_ms=10.0,
) -> CompletionResult:
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_call else "stop"
    if text is None and tool_call is None:
        text = "done"
    return CompletionResult(
        text=text,
        tool_call=tool_call,
        finish_reason=finish_reason,
        token_usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost),
        provider_latency_ms=provider_latency_ms,
    )


def make_decision(target=ModelTarget.LOCAL) -> RoutingDecision:
    return RoutingDecision(target=target, model_id="llama3.1:8b", reason="static")
