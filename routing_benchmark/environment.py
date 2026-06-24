"""Agent Environment turn loop for the LLM Routing Benchmark Suite.

Implements section 3.1 of ``routing_benchmark_spec.md``: the per-turn loop
that owns ``AgentState``, asks a ``BaseRouter`` for a routing decision each
turn, dispatches to the resolved ``BaseModelProvider``, runs any requested
tool call through a ``BaseMockToolingLayer``, and records ``TurnMetric``s
along the way, finishing with ``RunResult.finalize``.

Three deliberate decisions beyond the spec's literal pseudo-code, each
necessary to make the loop actually runnable and each documented because
the spec left it unspecified:

1. **Provider resolution.** The spec's pseudo-code calls
   ``resolve_provider(decision.target)`` -- but ``target`` is only LOCAL/
   CLOUD, too coarse to pick a specific model when several models share a
   target class. This module resolves by ``decision.model_id`` against an
   explicit ``providers: dict[str, BaseModelProvider]`` registry instead,
   raising ``UnknownModelError`` if a router selects a model_id with no
   registered provider.
2. **Natural completion.** ``AgentState.is_terminal()`` only knows about
   the explicit ``terminal`` flag and ``max_turns``; nothing in the spec
   says when an agent run reaches a natural finish before the turn cap.
   This loop sets ``state.terminal = True`` whenever a completion has
   ``finish_reason == "stop"`` and requests no tool call -- i.e. the model
   gave a final answer with nothing pending. Without this, every run would
   trivially burn all ``max_turns`` turns regardless of task completion.
3. **Feature extraction.** ``extract_features`` is implemented here (not
   inside any router) per the spec's own framing that routers receive
   already-computed ``RoutingFeatures`` -- see ``RoutingRequest.features``
   in ``models.py``.
"""

from __future__ import annotations

import time
import uuid

from routing_benchmark.metrics import RunResult, TurnMetric
from routing_benchmark.models import (
    AgentState,
    CompletionResult,
    IntentComplexity,
    RoutingFeatures,
    RoutingRequest,
    TaskCase,
)
from routing_benchmark.provider import BaseModelProvider
from routing_benchmark.router import BaseRouter
from routing_benchmark.tooling import BaseMockToolingLayer

__all__ = ["UnknownModelError", "detect_completion_wall", "extract_features", "run_task"]

DEFAULT_CONTEXT_WINDOW_LIMIT = 8192

_INTENT_COMPLEXITY_SCORES: dict[IntentComplexity, float] = {
    IntentComplexity.TRIVIAL: 0.1,
    IntentComplexity.MODERATE: 0.5,
    IntentComplexity.COMPLEX_MULTI_TOOL: 0.9,
}


class UnknownModelError(Exception):
    """Raised when a RoutingDecision selects a model_id with no registered provider."""


def _has_repeated_ngram_loop(
    text: str,
    max_period: int = 6,
    min_repeats: int = 3,
    min_total_words: int = 12,
) -> bool:
    """Detect exact-repeat loops: a word-sequence of some small period (up
    to ``max_period`` words) repeated back-to-back at least ``min_repeats``
    times, covering at least ``min_total_words`` words overall. The period
    is not fixed in advance -- a degenerate single-word loop ("no no no...")
    and a four-word phrase loop are both covered by scanning every period
    from 1 up to ``max_period``.
    """
    words = text.split()
    n = len(words)
    for period in range(1, max_period + 1):
        span = period * min_repeats
        if n < span:
            continue
        for start in range(n - span + 1):
            window = words[start : start + period]
            repeats = 1
            pos = start + period
            while pos + period <= n and words[pos : pos + period] == window:
                repeats += 1
                pos += period
            if repeats >= min_repeats and repeats * period >= min_total_words:
                return True
    return False


def detect_completion_wall(completion: CompletionResult) -> bool:
    """Deterministically classify a completion as a "hard completion wall" hit.

    A wall hit is a *successful provider call* that is nonetheless unusable:
    empty output with nothing pending, truncation without a closing tool
    call, an explicit provider timeout, or an exact-repeat generation loop.
    This is never a judgment call left to a router -- see spec section 3.1.
    """
    if completion.finish_reason == "timeout":
        return True
    if not completion.text and completion.tool_call is None:
        return True
    if completion.finish_reason == "length" and completion.tool_call is None:
        return True
    if completion.text and _has_repeated_ngram_loop(completion.text):
        return True
    return False


def _consecutive_tool_failures(state: AgentState) -> int:
    """Count trailing turns, most recent first, whose tool call failed or
    was silently malformed. Stops at the first turn that succeeded cleanly
    or made no tool call at all -- a non-tool turn breaks the streak just
    as a successful one does.
    """
    count = 0
    for record in reversed(state.history):
        tool_result = record.tool_result
        is_failure = tool_result is not None and (not tool_result.success or tool_result.is_silently_malformed)
        if not is_failure:
            break
        count += 1
    return count


def extract_features(
    state: AgentState,
    task: TaskCase,
    context_window_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT,
) -> RoutingFeatures:
    """Compute the state-derived signals every router receives this turn."""
    context_tokens_used = state.synthetic_prefix_tokens + sum(
        record.completion.token_usage.total_tokens for record in state.history
    )
    rolling_wall_hit_rate = (state.wall_events / state.turn_count) if state.turn_count else 0.0

    return RoutingFeatures(
        context_tokens_used=context_tokens_used,
        context_window_limit=context_window_limit,
        turn_count=state.turn_count,
        consecutive_tool_failures=_consecutive_tool_failures(state),
        cumulative_silent_failure_count=len(state.silent_failure_log),
        rolling_wall_hit_rate=rolling_wall_hit_rate,
        intent_complexity_score=_INTENT_COMPLEXITY_SCORES[task.complexity],
    )


def run_task(
    task: TaskCase,
    router: BaseRouter,
    providers: dict[str, BaseModelProvider],
    mock_tooling: BaseMockToolingLayer,
    run_id: str | None = None,
    context_window_limit: int = DEFAULT_CONTEXT_WINDOW_LIMIT,
) -> RunResult:
    """Run a single TaskCase to completion against one router, end to end.

    Args:
        task: The benchmark task to execute.
        router: The routing paradigm under test for this run.
        providers: Registry of model_id -> BaseModelProvider; every model_id
            the router can possibly return for this task must be present.
        mock_tooling: Deterministic, failure-injecting tool simulator.
        run_id: Unique identifier for this run; generated if omitted.
        context_window_limit: Token capacity used for wall-proximity
            features, representing the local model's context window.

    Returns:
        The finalized RunResult for this task/router pair.

    Raises:
        UnknownModelError: If the router selects a model_id with no
            registered provider.
    """
    run_id = run_id or str(uuid.uuid4())
    router.reset()
    state = AgentState.initial(task)
    metrics: list[TurnMetric] = []

    while not state.is_terminal():
        features = extract_features(state, task, context_window_limit)
        request = RoutingRequest(state=state, features=features, task=task)

        route_start = time.monotonic()
        decision = router.route(request)
        routing_latency_ms = (time.monotonic() - route_start) * 1000.0

        provider = providers.get(decision.model_id)
        if provider is None:
            raise UnknownModelError(f"no provider registered for model_id={decision.model_id!r}")

        infer_start = time.monotonic()
        completion = provider.generate(state.to_prompt(), decision.model_params)
        inference_latency_ms = (time.monotonic() - infer_start) * 1000.0

        wall_hit = detect_completion_wall(completion)
        if wall_hit:
            state.record_wall_event(decision.target)

        tool_result = None
        silent_failure = False
        if completion.requests_tool_call:
            tool_result = mock_tooling.invoke(completion.tool_call, task.failure_profile, state.turn_count)
            silent_failure = tool_result.is_silently_malformed
            state.record_tool_result(tool_result, silent_failure)

        state.append_turn(decision, completion, routing_latency_ms, inference_latency_ms, tool_result=tool_result)

        metrics.append(
            TurnMetric.from_state(
                run_id=run_id,
                router_name=router.name,
                state=state,
                decision=decision,
                wall_hit=wall_hit,
                silent_failure=silent_failure,
            )
        )

        if completion.finish_reason == "stop" and not completion.requests_tool_call:
            state.terminal = True

    return RunResult.finalize(run_id=run_id, router_name=router.name, task=task, state=state, metrics=metrics)
