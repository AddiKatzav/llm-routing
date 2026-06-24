"""Mock Tooling Layer interface for the LLM Routing Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: the
``BaseMockToolingLayer`` abstract base class that deterministically
simulates tool execution, including injected silent failures. This is the
benchmark's core stimulus for the "Silent Tool Failures" failure mode --
implementations must be able to report ``success=True`` while
``is_silently_malformed=True``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from routing_benchmark.models import ToolCall, ToolFailureProfile, ToolResult

__all__ = ["BaseMockToolingLayer"]


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
