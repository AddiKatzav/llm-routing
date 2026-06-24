"""Core data structures for the LLM Routing Benchmark Suite.

Implements section 6 of ``routing_benchmark_spec.md``: the enums and typed
data structures shared by every router, model provider, and the agent
environment. Unlike the spec document, method bodies here are real
implementations, not ``...`` placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTarget(str, Enum):
    """Which backend class a routing decision points to."""
    LOCAL = "local"
    CLOUD = "cloud"


class ContextDepthLevel(str, Enum):
    """Synthetic context-window occupancy levels used in dataset generation."""
    SHALLOW = "shallow"
    MID = "mid"
    NEAR_WALL = "near_wall"
    OVER_WALL = "over_wall"


class ToolFailureProfile(str, Enum):
    """Injected tool-failure modes for a given TaskCase."""
    NONE = "none"
    SINGLE_SILENT_FAILURE = "single_silent_failure"
    CASCADING_SILENT_FAILURES = "cascading_silent_failures"
    LOUD_FAILURE = "loud_failure"


class IntentComplexity(str, Enum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX_MULTI_TOOL = "complex_multi_tool"


# ---------------------------------------------------------------------------
# Typed data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenUsage:
    """Token and cost accounting for a single Model Provider call."""
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by a model completion."""
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ToolResult:
    """Outcome of dispatching a ToolCall through the Mock Tooling Layer.

    Note ``success=True`` and ``is_silently_malformed=True`` can co-occur --
    that combination is exactly the silent-tool-failure condition under
    test, not a contradiction.
    """
    success: bool
    output: Any
    is_silently_malformed: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True)
class CompletionResult:
    """Normalized response from any Model Provider, local or cloud."""
    text: Optional[str]
    tool_call: Optional[ToolCall]
    finish_reason: str
    token_usage: TokenUsage
    provider_latency_ms: float

    @property
    def requests_tool_call(self) -> bool:
        """True if this completion includes a parsed tool call request."""
        return self.tool_call is not None


@dataclass(frozen=True)
class RoutingFeatures:
    """Snapshot of state-derived signals available to a router at decision time."""
    context_tokens_used: int
    context_window_limit: int
    turn_count: int
    consecutive_tool_failures: int
    cumulative_silent_failure_count: int
    rolling_wall_hit_rate: float
    intent_complexity_score: float

    @property
    def context_occupancy_ratio(self) -> float:
        """Fraction of the context window currently in use, in [0, +inf)."""
        if self.context_window_limit <= 0:
            raise ValueError("context_window_limit must be positive")
        return self.context_tokens_used / self.context_window_limit


@dataclass(frozen=True)
class TurnRecord:
    """One resolved turn appended to AgentState.history."""
    decision: "RoutingDecision"
    completion: CompletionResult
    routing_latency_ms: float
    inference_latency_ms: float
    tool_result: Optional[ToolResult] = None


@dataclass(frozen=True)
class TaskCase:
    """A single synthesized benchmark task definition."""
    id: str
    domain: str
    complexity: IntentComplexity
    initial_prompt: str
    synthetic_history: list[Any]
    failure_profile: ToolFailureProfile
    max_turns: int
    expected_tool_calls: int

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.expected_tool_calls < 0:
            raise ValueError("expected_tool_calls must be non-negative")


@dataclass
class AgentState:
    """Mutable conversation/turn state for a single benchmark run."""
    task_id: str
    turn_count: int = 0
    history: list[TurnRecord] = field(default_factory=list)
    wall_events: int = 0
    silent_failure_log: list[ToolResult] = field(default_factory=list)
    terminal: bool = False
    _initial_prompt: str = ""
    _max_turns: int = 0

    @classmethod
    def initial(cls, task: TaskCase) -> "AgentState":
        """Construct the starting state for a fresh run of the given task."""
        return cls(
            task_id=task.id,
            _initial_prompt=task.initial_prompt,
            _max_turns=task.max_turns,
        )

    def is_terminal(self) -> bool:
        """Whether the run loop should stop (explicit terminal flag or max turns)."""
        if self.terminal:
            return True
        return self._max_turns > 0 and self.turn_count >= self._max_turns

    def to_prompt(self) -> str:
        """Render the current state into a model-ready prompt string."""
        lines = [f"User: {self._initial_prompt}"]
        for record in self.history:
            if record.completion.text:
                lines.append(f"Assistant: {record.completion.text}")
            if record.completion.tool_call is not None:
                lines.append(
                    f"ToolCall[{record.completion.tool_call.tool_name}]: "
                    f"{record.completion.tool_call.arguments}"
                )
            if record.tool_result is not None:
                lines.append(f"ToolResult: {record.tool_result.output}")
        return "\n".join(lines)

    def record_wall_event(self, target: ModelTarget) -> None:
        """Mark that a completion-wall hit was detected on the given target."""
        self.wall_events += 1

    def record_tool_result(self, result: ToolResult, silent_failure: bool) -> None:
        """Track a tool result for later success/failure-recovery accounting."""
        if silent_failure:
            self.silent_failure_log.append(result)

    def append_turn(
        self,
        decision: "RoutingDecision",
        completion: CompletionResult,
        routing_latency_ms: float,
        inference_latency_ms: float,
        tool_result: Optional[ToolResult] = None,
    ) -> None:
        """Append a fully-resolved turn record to history and advance turn_count."""
        self.history.append(
            TurnRecord(
                decision=decision,
                completion=completion,
                routing_latency_ms=routing_latency_ms,
                inference_latency_ms=inference_latency_ms,
                tool_result=tool_result,
            )
        )
        self.turn_count += 1


@dataclass(frozen=True)
class RoutingRequest:
    """Input to BaseRouter.route() for a single turn."""
    state: AgentState
    features: RoutingFeatures
    task: TaskCase


@dataclass(frozen=True)
class RoutingDecision:
    """Output of BaseRouter.route()."""
    target: ModelTarget
    model_id: str
    reason: str
    model_params: dict[str, Any] = field(default_factory=dict)
    raw_provider_metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
