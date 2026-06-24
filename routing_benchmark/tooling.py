"""Mock Tooling Layer interface for the LLM Routing Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: the
``BaseMockToolingLayer`` abstract base class that deterministically
simulates tool execution, including injected silent failures. This is the
benchmark's core stimulus for the "Silent Tool Failures" failure mode --
implementations must be able to report ``success=True`` while
``is_silently_malformed=True``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from routing_benchmark.models import ToolCall, ToolFailureProfile, ToolResult

__all__ = ["BaseMockToolingLayer", "DeterministicMockToolingLayer"]


class BaseMockToolingLayer(ABC):
    """Abstract base for the deterministic, failure-injecting tool simulator.

    Implementations must be deterministic given the same
    ``(tool_call, failure_profile, turn_index)`` triple, so failure
    injection is reproducible across repeated benchmark runs rather than
    randomly flaky.
    """

    @abstractmethod
    def invoke(
        self,
        tool_call: ToolCall,
        failure_profile: ToolFailureProfile,
        turn_index: int,
    ) -> ToolResult:
        """Simulate executing a tool call under a given failure profile.

        Args:
            tool_call: The parsed tool call to simulate.
            failure_profile: Which failure-injection mode is active for
                this TaskCase.
            turn_index: Current turn number, used by profiles like
                CASCADING_SILENT_FAILURES that depend on turn position.

        Returns:
            A ToolResult, which may have ``success=True`` while
            ``is_silently_malformed=True`` -- this combination is the
            benchmark's core stimulus for silent tool failures.
        """
        raise NotImplementedError


def _stable_digest(text: str, modulus: int) -> int:
    """SHA-256-based digest, stable across processes regardless of
    PYTHONHASHSEED -- unlike Python's built-in hash() for strings.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % modulus


class DeterministicMockToolingLayer(BaseMockToolingLayer):
    """Reference BaseMockToolingLayer covering all four ToolFailureProfile values.

    Holds no mutable state of its own -- every decision is a pure function
    of the ``(tool_call, failure_profile, turn_index)`` triple, as the base
    class requires. This matters because a single instance is typically
    shared across an entire BenchmarkDriver matrix sweep (many tasks,
    routers, and repeats); per-instance counters would leak failure state
    across runs that are supposed to be isolated from each other.

    Failure semantics:

    - NONE: always a clean, successful result.
    - LOUD_FAILURE: always an explicit, detectable failure (success=False)
      -- the control condition for isolating silent failures specifically.
    - SINGLE_SILENT_FAILURE: a successful-looking but malformed result at
      exactly ``single_failure_turn_index`` (default: the first turn that
      calls a tool), clean on every other turn.
    - CASCADING_SILENT_FAILURES: malformed from ``cascade_start_turn_index``
      onward (default: every turn after the first), clean before that.

    A malformed result is deliberately the same *shape* as a clean one
    (same string prefix, no error markers) but always carries the same
    placeholder value regardless of the tool's actual arguments -- as if
    the tool silently returned stale/cached data instead of a real
    answer. This is what makes it silent: there is nothing in the result
    itself that looks broken.
    """

    def __init__(
        self,
        single_failure_turn_index: int = 0,
        cascade_start_turn_index: int = 1,
        latency_ms: float = 5.0,
    ) -> None:
        if single_failure_turn_index < 0:
            raise ValueError("single_failure_turn_index must be non-negative")
        if cascade_start_turn_index < 0:
            raise ValueError("cascade_start_turn_index must be non-negative")
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        self.single_failure_turn_index = single_failure_turn_index
        self.cascade_start_turn_index = cascade_start_turn_index
        self.latency_ms = latency_ms

    def invoke(
        self,
        tool_call: ToolCall,
        failure_profile: ToolFailureProfile,
        turn_index: int,
    ) -> ToolResult:
        if failure_profile == ToolFailureProfile.NONE:
            return self._result(tool_call, malformed=False)
        if failure_profile == ToolFailureProfile.LOUD_FAILURE:
            return ToolResult(success=False, output=None, is_silently_malformed=False, latency_ms=self.latency_ms)
        if failure_profile == ToolFailureProfile.SINGLE_SILENT_FAILURE:
            return self._result(tool_call, malformed=turn_index == self.single_failure_turn_index)
        if failure_profile == ToolFailureProfile.CASCADING_SILENT_FAILURES:
            return self._result(tool_call, malformed=turn_index >= self.cascade_start_turn_index)
        raise ValueError(f"unhandled failure profile: {failure_profile!r}")

    def _result(self, tool_call: ToolCall, malformed: bool) -> ToolResult:
        return ToolResult(
            success=True,
            output=self._render_output(tool_call, malformed),
            is_silently_malformed=malformed,
            latency_ms=self.latency_ms,
        )

    def _render_output(self, tool_call: ToolCall, malformed: bool) -> str:
        if malformed:
            # Always the same placeholder, regardless of arguments -- a
            # stand-in for stale/cached data the tool silently returned.
            return f"{tool_call.tool_name}_result:0"
        arg_signature = repr(sorted(tool_call.arguments.items()))
        digest = _stable_digest(arg_signature, modulus=100_000)
        return f"{tool_call.tool_name}_result:{digest}"
