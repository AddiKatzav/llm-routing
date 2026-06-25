# OpenSpec Proposal: LLM Routing Paradigm Benchmark Suite

```yaml
spec_id: routing-benchmark-001
status: PROPOSED
type: capability
owner: addi katzav
created: 2026-06-24
target_release: v0.1.0 (benchmark-only, no production rollout)
```

## 0. Why

Local models (running on a single laptop, via Ollama or similar) exhibit two
failure modes in long-horizon agentic loops that cloud models do not:

1. **The Hard Completion Wall** — as context window occupancy grows (long
   tool histories, large retrieved documents), local models stop producing
   usable completions. This is not a graceful quality degradation; it is an
   abrupt failure to finish generation, finish a tool call, or terminate a
   turn at all.
2. **Silent Tool Failures** — local models frequently malform tool-call
   payloads, hallucinate tool names/arguments, or fail to recognize a tool
   error in the returned observation and simply continue as if it succeeded.
   Unlike an exception, this failure produces no signal the orchestrator can
   trap on.

Both failures are *state-dependent*, not just *prompt-dependent*. A request
that local handles fine at turn 1 may push the model past the wall at turn
12 of the same session. This implies routing decisions must be re-evaluated
per-turn, not just once at session start — which static semantic routers
cannot do by construction.

This spec defines a benchmark suite to measure, under controlled and
reproducible conditions, whether a **context-aware dynamic router** actually
outperforms (a) a **zero-overhead static semantic router** and (b) a
**commercial cloud router baseline**, on the combined objective of task
success rate, latency overhead, and cost.

## 1. Scope

### 1.1 In scope
- A harness that can run the same agent task against 3 interchangeable
  router implementations.
- Synthetic dataset generation for intent complexity, context depth, and
  injected tool failures.
- Metric collection, persistence (JSON/CSV), and a defined KPI computation
  layer.
- Abstract Python interfaces (typed, docstring-complete, **no implementation
  bodies**) for Router, Agent Environment, Mock Tooling, Model Provider, and
  Metric Collector.

### 1.2 Out of scope (deferred to follow-on specs)
- Production routing rollout / live traffic shadowing.
- Fine-tuning or training a custom router classifier.
- Multi-agent / multi-session concurrency benchmarking.
- UI/visualization layer (only the data schema it will consume is defined
  here).
- Actual implementation code for any method body.

## 2. System Architecture

### 2.1 Component Diagram (conceptual)

```
                         +---------------------------+
                         |      Benchmark Driver      |
                         |  (test matrix orchestrator) |
                         +-------------+---------------+
                                       |
                                       v
   +-----------------+      +---------------------+      +----------------------+
   |  Test Dataset    |----->|   Agent Environment  |<---->|   Mock Tooling Layer |
   |  (synthesized)   |      |  (turn loop, state)   |      |  (injects failures)  |
   +-----------------+      +----------+------------+      +----------------------+
                                       |
                                       | per-turn routing decision request
                                       v
                         +---------------------------+
                         |      Router Interface       |
                         |  (pluggable, 1-of-3)         |
                         +------+--------+--------------+
                                |        |
              +-----------------+        +------------------+
              v                                              v
   +----------------------+                      +----------------------+
   |  Model Provider:      |                      |  Model Provider:      |
   |  Local (Ollama)       |                      |  Cloud (API)          |
   +----------------------+                      +----------------------+
                                       |
                                       v
                         +---------------------------+
                         |      Metric Collector       |
                         |  (latency, tokens, success,  |
                         |   cost, wall-hit events)     |
                         +-------------+---------------+
                                       |
                                       v
                         +---------------------------+
                         |   Run Store (JSON/CSV)      |
                         +---------------------------+
```

### 2.2 Component Responsibilities

| Component | Responsibility | Key invariant |
|---|---|---|
| Benchmark Driver | Iterates the evaluation matrix (task x router x failure-injection profile), invokes Agent Environment per run, ensures isolation between runs | Each run starts from a clean `AgentState` |
| Test Dataset | Synthesizes `TaskCase` objects spanning intent complexity, context depth, and tool-failure injection points | Deterministic given a seed |
| Agent Environment | Owns the turn loop: builds prompt from state, asks Router for a routing decision, dispatches to the chosen Model Provider, applies Mock Tooling, updates `AgentState` | Never calls a Model Provider directly — always through Router |
| Mock Tooling Layer | Deterministically simulates tool execution, including injected silent failures (malformed result, wrong-looking-success, timeout-as-success) | Failure injection is config-driven and reproducible, not random-by-default |
| Router Interface | Given current `AgentState` + `RoutingRequest`, returns a `RoutingDecision` (which provider/model to use) | Must respond within a measurable, recorded latency budget |
| Model Provider | Wraps a concrete backend (local Ollama model, cloud API model) behind one uniform call signature | Reports token usage and wall-clock latency for every call |
| Metric Collector | Captures per-turn and per-run metrics, computes derived KPIs, persists to the Run Store | Append-only; never mutates past records |

## 3. Algorithmic Flow (Pseudo-code)

### 3.1 Common per-turn loop (Agent Environment)

```
function run_task(task: TaskCase, router: BaseRouter) -> RunResult:
    state = AgentState.initial(task)
    metrics = []

    while not state.is_terminal() and state.turn_count < task.max_turns:
        routing_request = build_routing_request(state, task)

        t0 = now()
        decision = router.route(routing_request)
        routing_latency_ms = now() - t0

        provider = resolve_provider(decision.target)
        t1 = now()
        completion = provider.generate(state.to_prompt(), decision.model_params)
        inference_latency_ms = now() - t1

        wall_hit = detect_completion_wall(completion)  # truncation / empty / repeated-loop
        if wall_hit:
            state.record_wall_event(decision.target)

        if completion.requests_tool_call:
            tool_result = mock_tooling.invoke(completion.tool_call, task.failure_profile, state.turn_count)
            silent_failure = tool_result.is_silently_malformed
            state.record_tool_result(tool_result, silent_failure)
        else:
            silent_failure = False

        state.append_turn(decision, completion, routing_latency_ms, inference_latency_ms)
        metrics.append(TurnMetric.from_state(state, decision, wall_hit, silent_failure))

    return RunResult.finalize(task, state, metrics)
```

`detect_completion_wall` is a deterministic classifier over the raw
completion object: empty content, finish_reason == "length" with no
tool-call closure, exact-repeat n-gram loops, or provider-reported timeout —
not a judgment call left to the router.

### 3.2 Router 1 — Static / Semantic Routing (Local)

Zero state-awareness by design. Routes once per request based purely on
embedding similarity to a fixed set of intent exemplars.

```
function StaticSemanticRouter.route(request) -> RoutingDecision:
    query_embedding = embed(request.task.initial_prompt)
    best_label, score = nearest_neighbor(query_embedding, self.intent_index)
    target = self.routing_table[best_label]   # static label -> provider mapping
    return RoutingDecision(target=target, reason=f"semantic:{best_label}:{score}")
```

Key property under test: this router computes its decision **once** (cached
after turn 1) since its inputs (the embedding of the original intent) do not
change turn-to-turn. This is the control condition — it should fail to
prevent wall-hits that emerge later in the conversation.

### 3.3 Router 2 — Context-Aware Dynamic Routing (Local LLM-as-Judge)

Re-evaluated **every turn**. Factors in state that the static router cannot
see.

```
function ContextAwareRouter.route(request) -> RoutingDecision:
    features = extract_features(request.state):
        - context_tokens_used / context_window_limit   # proximity to wall
        - turn_count
        - consecutive_tool_failures
        - cumulative_silent_failure_count
        - intent_complexity_score (from static embedding pass, reused)
        - rolling_wall_hit_rate (wall_events / turns so far, this run)

    if features.consecutive_tool_failures >= self.escalation_threshold:
        return RoutingDecision(target=CLOUD, reason="failure_escalation")

    if features.context_tokens_used / features.context_window_limit >= self.wall_proximity_threshold:
        return RoutingDecision(target=CLOUD, reason="context_proximity_to_wall")

    judge_prompt = render_judge_prompt(features, request.task.intent_summary)
    judge_response = self.local_judge_model.generate(judge_prompt)   # small, fast local model
    label = parse_judge_label(judge_response)   # {LOCAL, CLOUD}

    target = CLOUD if label == "CLOUD" else LOCAL
    return RoutingDecision(target=target, reason=f"judge:{label}", features=features)
```

Two design notes baked into the spec (not implementation, just constraints
the interfaces must support):

- The escalation thresholds (`escalation_threshold`,
  `wall_proximity_threshold`) must be exposed as constructor-level
  hyperparameters so the benchmark matrix can sweep them.
- The judge call's own latency and token cost **must** be recorded as
  routing overhead, not inference cost — this router's tax is the thing
  being measured against the static router's near-zero tax.

### 3.4 Router 3 — Commercial Cloud Router Baseline

```
function CommercialCloudRouter.route(request) -> RoutingDecision:
    payload = build_provider_payload(request.state, request.task)
    response = self.cloud_router_client.submit(payload)   # e.g. openrouter/auto, Not Diamond
    target = map_external_model_id(response.selected_model)
    return RoutingDecision(target=target, reason=f"cloud_router:{response.selected_model}",
                            raw_provider_metadata=response.raw)
```

This router is a black box by design — the spec only requires that its
decision latency, selected model, and cost be captured identically to the
other two, so it can be compared on equal footing despite being
non-introspectable.

## 4. Test Dataset & Evaluation Matrix

### 4.1 Dimensions of variation

| Dimension | Levels | Notes |
|---|---|---|
| Intent complexity | `trivial`, `moderate`, `complex_multi_tool` | Drives expected tool-call count and reasoning depth |
| Simulated context depth | `shallow` (~10% of local ctx window), `mid` (~50%), `near_wall` (~85%), `over_wall` (~110%, forces truncation) | Achieved by padding `AgentState` history with synthetic prior turns before the run starts, not by actually running N real turns |
| Tool failure injection | `none`, `single_silent_failure`, `cascading_silent_failures`, `loud_failure` (control) | `loud_failure` is the control case where the tool raises a real exception — used to confirm agents handle *detectable* failures fine, isolating the silent case as the interesting variable |
| Task domain | `data_lookup`, `multi_step_calculation`, `file_edit_simulation`, `ambiguous_intent` | Ensures the static router's embedding space is genuinely exercised |

Total matrix size = 3 intent levels x 4 context depths x 4 failure profiles x
4 domains x 3 routers = **576 run configurations**. Each configuration is
run with `N_REPEATS` (default 5) seeded variations to control for
stochastic decoding noise.

### 4.2 Dataset synthesis procedure

```
function synthesize_dataset(seed: int) -> list[TaskCase]:
    for domain in DOMAINS:
        for complexity in COMPLEXITY_LEVELS:
            base_prompt = template_bank[domain][complexity].render(seed)
            for depth in CONTEXT_DEPTHS:
                synthetic_history = generate_padding_turns(depth, domain, seed)
                for failure_profile in FAILURE_PROFILES:
                    yield TaskCase(
                        id=deterministic_id(domain, complexity, depth, failure_profile, seed),
                        domain=domain,
                        complexity=complexity,
                        initial_prompt=base_prompt,
                        synthetic_history=synthetic_history,
                        failure_profile=failure_profile,
                        max_turns=MAX_TURNS_BY_COMPLEXITY[complexity],
                        expected_tool_calls=expected_tool_count(domain, complexity),
                    )
```

`generate_padding_turns` synthesizes plausible prior turns (not lorem-ipsum
filler) so that the context-aware router's token-proximity feature is
realistic, not just numerically large.

## 5. Metrics & Target KPIs

### 5.1 Primary metrics (captured per turn, aggregated per run)

| Metric | Formula | Capture point |
|---|---|---|
| Wall Avoidance Rate (WAR) | `1 - (runs_with_wall_hit / total_runs)` per router | `Agent Environment.detect_completion_wall` |
| Routing Overhead (ms) | `mean(routing_latency_ms)` per router, per turn | Wrapped around every `router.route()` call |
| Task Success Rate (TSR) | `successful_runs / total_runs`, where success = task-specific completion criterion met AND no unresolved silent failure | `RunResult.finalize` |
| Silent Failure Recovery Rate (SFRR) | `runs_recovered_from_silent_failure / runs_with_injected_silent_failure` | Cross-reference injected failure profile against final task success |
| Cost Efficiency (CE) | `1 - (total_cost_actual / total_cost_all_cloud_baseline)` | `total_cost_actual` summed from `TokenUsage.cost_usd` per call; `all_cloud_baseline` computed by replaying the same run forcing `target=CLOUD` every turn |
| Effective Cost per Success | `total_cost_actual / successful_runs` | Derived metric, post-hoc |

### 5.2 Target KPIs (acceptance thresholds for this benchmark, not production SLAs)

- Context-Aware Router WAR ≥ Static Router WAR + 15 percentage points, on
  `near_wall` and `over_wall` depths.
- Context-Aware Router routing overhead ≤ 150ms p95 (it must stay
  "lightweight" by its own design claim — if it exceeds this, the dynamic
  router is disqualified as impractical regardless of accuracy gains).
- Context-Aware Router CE ≥ 40% relative to all-cloud baseline (it should
  still capture meaningful savings, not just escalate everything).
- Commercial Cloud Router Baseline TSR is the upper bound reference; report
  the gap, not a pass/fail threshold (its internal routing logic is opaque
  by definition, so it serves as ceiling not target).

### 5.3 Comparative Metrics: Static vs. Dynamic Routing

Section 5.1's aggregate metrics (WAR, TSR, SFRR, routing overhead) answer
"which router performed better overall," but not the more useful question
for a production routing decision: **when, and by how much, does
reconsidering the routing decision every turn actually pay for itself?**
This section defines three metrics specifically for the
StaticSemanticRouter-vs-ContextAwareRouter comparison (the
CommercialCloudRouter is intentionally set aside here — these metrics
assume both sides of the comparison are introspectable, which an opaque
commercial router is not).

| Metric | Formula | Capture point |
|---|---|---|
| Decision Divergence Rate (DDR) | `turns_where_dynamic_decision != shadow_static_decision / total_turns`, per run | Computed per turn during a ContextAwareRouter run by additionally evaluating (not executing) what StaticSemanticRouter.route() would have returned for the same RoutingRequest |
| Escalation Precision | `true_positive_escalations / total_escalations` | A turn is an escalation when the live decision is CLOUD. `true_positive_escalations` = escalated turns where a shadow call to the LOCAL provider on the same prompt shows `detect_completion_wall` would have fired |
| Escalation Recall | `true_positive_escalations / (true_positive_escalations + false_negative_local_wall_hits)` | `false_negative_local_wall_hits` = turns where the live decision was LOCAL (no shadow call needed -- it already *is* the ground truth) and the real `wall_hit` fired. No additional shadow calls beyond Escalation Precision's are needed: every wall-hit turn is either a real LOCAL wall hit or a shadow-probed CLOUD escalation |
| Escalation Lead Time | `1.0 - context_occupancy_ratio` at the escalation turn, averaged over true-positive escalations | A turn-index-based lead time would require shadow-probing LOCAL on *every* turn (not just escalations) to find the first turn the wall would have hit, which is the instrumentation cost this metric exists to avoid; occupancy headroom at the moment of a confirmed-correct escalation is a cheaper, escalation-turn-only proxy for "how early" it acted -- higher headroom means it escalated well before the window was actually full |

**Instrumentation cost.** Decision Divergence Rate is free — it only
evaluates StaticSemanticRouter's pure decision function, no extra model
call. Escalation Precision/Recall and Escalation Lead Time all derive
from the *same* shadow call: whenever the live decision escalates to
CLOUD, one additional call to the LOCAL provider on the same prompt
establishes ground truth for what would have happened locally. Turns the
live router did NOT escalate need no shadow call at all — the real
`wall_hit` already tells us what LOCAL did. This shadow call's
cost must be tracked separately from the run's real `total_cost_usd` — it
is benchmarking instrumentation, not something a production deployment of
ContextAwareRouter would ever actually pay for, and must not be allowed
to contaminate Cost Efficiency or Routing Overhead numbers.

**How to read the results.** Per spec section 1's framing, prefer the
section 7.3 per-context-depth and per-failure-profile breakdowns over a
single aggregate DDR/Precision/Recall number. The useful claim this
section is built to support is shaped like: *"DDR is near-zero and both
routers perform identically below `mid` depth; above `near_wall`, DDR
rises sharply, Escalation Recall stays high (the dynamic router reliably
sees the wall coming), and Escalation Lead Time (occupancy headroom) is
consistently well above zero (it escalates with room to spare, not at the
brink) — which is what justifies the judge's overhead at that depth and
not below it."*

## 6. Python Code Interface / Class Signatures

> No method body in this section contains implementation logic. Every
> non-trivial method body is `...`. This section is the contract the
> implementation phase must satisfy.

```python
"""routing_benchmark/interfaces.py

Abstract interfaces and typed data structures for the LLM Routing
Benchmark Suite. This module defines the contract only; concrete
implementations live in follow-on specs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTarget(str, Enum):
    """Which backend class a routing decision points to.

    Concrete model identifiers (e.g. "llama3.1:8b", "claude-sonnet-4-6")
    are resolved separately via RoutingDecision.model_id; this enum only
    distinguishes the provider class for cost/latency bucketing.
    """
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
    """Token and cost accounting for a single Model Provider call.

    Attributes:
        prompt_tokens: Tokens consumed by the input context.
        completion_tokens: Tokens generated in the response.
        cost_usd: Computed dollar cost of this call, using the provider's
            published or configured per-token pricing. Must be 0.0 for
            local providers unless an amortized hardware/energy cost model
            is explicitly configured.
    """
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by a model completion.

    Attributes:
        tool_name: Name of the tool the model asked to invoke.
        arguments: Parsed arguments for the tool call.
        raw_text: Unparsed text the model produced for this call, retained
            for malformed-call diagnostics.
    """
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ToolResult:
    """Outcome of dispatching a ToolCall through the Mock Tooling Layer.

    Attributes:
        success: Whether the tool call is reported as successful by the
            tool layer itself (note: this can be True even when
            is_silently_malformed is also True -- that combination is
            exactly the "silent tool failure" condition under test).
        output: The (possibly malformed) output payload returned to the
            agent's context.
        is_silently_malformed: Ground-truth label, known only to the
            benchmark harness (never exposed to the agent/model), marking
            whether this result is an injected silent failure.
        latency_ms: Simulated or actual tool execution latency.
    """
    success: bool
    output: Any
    is_silently_malformed: bool
    latency_ms: float


@dataclass(frozen=True)
class CompletionResult:
    """Normalized response from any Model Provider, local or cloud.

    Attributes:
        text: Raw generated text content, if any.
        tool_call: Parsed ToolCall if the model requested one, else None.
        finish_reason: Provider-reported stop reason (e.g. "stop",
            "length", "tool_calls", "timeout").
        token_usage: TokenUsage for this call.
        provider_latency_ms: Wall-clock time for the provider call itself,
            excluding routing overhead.
    """
    text: Optional[str]
    tool_call: Optional[ToolCall]
    finish_reason: str
    token_usage: TokenUsage
    provider_latency_ms: float

    @property
    def requests_tool_call(self) -> bool:
        """True if this completion includes a parsed tool call request."""
        ...


@dataclass(frozen=True)
class RoutingFeatures:
    """Snapshot of state-derived signals available to a router at decision time.

    Populated by the Agent Environment before every router.route() call so
    that all three router implementations receive the same feature surface,
    even though the Static router ignores most of it by design.

    Attributes:
        context_tokens_used: Tokens currently occupying the agent's context.
        context_window_limit: Token capacity of the active local model,
            used to compute proximity to the completion wall.
        turn_count: Number of turns elapsed in the current run.
        consecutive_tool_failures: Count of immediately-preceding turns
            flagged (by ground truth or detector heuristic) as failed.
        cumulative_silent_failure_count: Running total of detected/injected
            silent failures so far this run.
        rolling_wall_hit_rate: wall_events / turn_count so far, this run.
        intent_complexity_score: Numeric proxy for IntentComplexity, reused
            across routers to avoid redundant embedding calls.
    """
    context_tokens_used: int
    context_window_limit: int
    turn_count: int
    consecutive_tool_failures: int
    cumulative_silent_failure_count: int
    rolling_wall_hit_rate: float
    intent_complexity_score: float


@dataclass(frozen=True)
class RoutingRequest:
    """Input to BaseRouter.route() for a single turn.

    Attributes:
        state: Read-only view of the current AgentState (see AgentState).
        features: Precomputed RoutingFeatures for this turn.
        task: The TaskCase this run belongs to, for routers needing static
            task metadata (e.g. domain, original prompt).
    """
    state: "AgentState"
    features: RoutingFeatures
    task: "TaskCase"


@dataclass(frozen=True)
class RoutingDecision:
    """Output of BaseRouter.route().

    Attributes:
        target: Coarse provider class selected for this turn.
        model_id: Concrete model identifier to dispatch to (e.g.
            "llama3.1:8b-instruct", "claude-sonnet-4-6").
        reason: Human-readable explanation/trace of why this decision was
            made, for debugging and for the Data Output Schema's
            `routing_reason` field.
        model_params: Optional generation parameters override (temperature,
            max_tokens, etc.) the router wants applied for this call.
        raw_provider_metadata: Opaque passthrough for commercial router
            responses that include extra fields the benchmark should
            persist but not depend on.
    """
    target: ModelTarget
    model_id: str
    reason: str
    model_params: dict[str, Any] = field(default_factory=dict)
    raw_provider_metadata: Optional[dict[str, Any]] = None


@dataclass
class AgentState:
    """Mutable conversation/turn state for a single benchmark run.

    This is the single source of truth the Agent Environment mutates each
    turn and the Router Interface reads from (via RoutingFeatures, never
    directly, to keep routers decoupled from state internals).

    Attributes:
        task_id: Identifier of the originating TaskCase.
        turn_count: Number of completed turns so far.
        history: Ordered list of (decision, completion, tool_result)
            tuples-equivalent turn records; concrete element type defined
            by the implementation phase.
        wall_events: Count of detected completion-wall hits so far.
        silent_failure_log: Ordered record of detected/injected silent
            failures, for SFRR computation.
        terminal: Whether the run has reached a terminal state (success,
            max turns, or unrecoverable failure).
    """
    task_id: str
    turn_count: int = 0
    history: list[Any] = field(default_factory=list)
    wall_events: int = 0
    silent_failure_log: list[Any] = field(default_factory=list)
    terminal: bool = False

    @classmethod
    def initial(cls, task: "TaskCase") -> "AgentState":
        """Construct the starting state for a fresh run of the given task."""
        ...

    def is_terminal(self) -> bool:
        """Whether the run loop should stop (success, failure, or max turns)."""
        ...

    def to_prompt(self) -> str:
        """Render the current state into a model-ready prompt string."""
        ...

    def record_wall_event(self, target: ModelTarget) -> None:
        """Mark that a completion-wall hit was detected on the given target."""
        ...

    def record_tool_result(self, result: ToolResult, silent_failure: bool) -> None:
        """Append a tool result to history and update failure tracking."""
        ...

    def append_turn(
        self,
        decision: RoutingDecision,
        completion: CompletionResult,
        routing_latency_ms: float,
        inference_latency_ms: float,
    ) -> None:
        """Append a fully-resolved turn record to history."""
        ...


@dataclass(frozen=True)
class TaskCase:
    """A single synthesized benchmark task definition.

    Attributes:
        id: Deterministic identifier, reproducible from synthesis seed.
        domain: Task domain (see dataset synthesis section).
        complexity: IntentComplexity level.
        initial_prompt: The user-facing prompt that starts the run.
        synthetic_history: Pre-generated prior turns used to simulate
            ContextDepthLevel without actually running N real turns.
        failure_profile: ToolFailureProfile to apply during this run.
        max_turns: Hard cap on turns before the run is marked incomplete.
        expected_tool_calls: Expected count of tool calls for a "complete"
            run, used by the task-specific success criterion.
    """
    id: str
    domain: str
    complexity: IntentComplexity
    initial_prompt: str
    synthetic_history: list[Any]
    failure_profile: ToolFailureProfile
    max_turns: int
    expected_tool_calls: int


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class BaseRouter(ABC):
    """Abstract base for all routing paradigms under benchmark.

    Implementations must be stateless across runs (any internal state,
    e.g. a context-aware router's escalation counters, must be reset via
    reset() at the start of each run) so that the Benchmark Driver can
    reuse one router instance across the full evaluation matrix without
    cross-run contamination.
    """

    @abstractmethod
    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Decide which model/provider should handle the current turn.

        Args:
            request: Current routing request, including state-derived
                features and task metadata.

        Returns:
            A RoutingDecision indicating the selected target, concrete
            model id, and the reasoning trace for that decision.

        Raises:
            RouterTimeoutError: If the router's own decision process
                (e.g. a local judge model call) exceeds its configured
                timeout budget. Implementations must raise rather than
                silently defaulting, so timeout events are visible in
                metrics rather than masked as routing accuracy.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-run internal state before starting a new run."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this router, used as a metrics dimension."""
        ...


class BaseModelProvider(ABC):
    """Abstract base for a callable backend (local or cloud).

    A single ModelProvider instance corresponds to one concrete model id;
    the Agent Environment resolves RoutingDecision.model_id to a provider
    instance via a registry (out of scope for this interface file).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model_params: dict[str, Any],
    ) -> CompletionResult:
        """Produce a single completion for the given prompt.

        Args:
            prompt: Fully-rendered prompt string for this turn.
            model_params: Generation parameter overrides from the
                RoutingDecision (temperature, max_tokens, etc.).

        Returns:
            A normalized CompletionResult, regardless of underlying
            provider SDK shape.

        Raises:
            ProviderUnavailableError: If the backend cannot be reached
                (e.g. local Ollama daemon down, cloud API auth failure).
                This must be distinguished from a completion-wall event,
                which is a successful response that is unusable, not a
                failed call.
        """
        ...

    @property
    @abstractmethod
    def target_class(self) -> ModelTarget:
        """Whether this provider counts as LOCAL or CLOUD for KPI bucketing."""
        ...


class BaseMockToolingLayer(ABC):
    """Abstract base for the deterministic, failure-injecting tool simulator."""

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
            A ToolResult, which may have success=True while
            is_silently_malformed=True -- this combination is the
            benchmark's core stimulus for silent tool failures.
        """
        ...


class BaseMetricCollector(ABC):
    """Abstract base for per-turn and per-run metric capture and persistence."""

    @abstractmethod
    def record_turn(self, run_id: str, turn_metric: "TurnMetric") -> None:
        """Persist a single turn's metrics, append-only."""
        ...

    @abstractmethod
    def record_run(self, run_result: "RunResult") -> None:
        """Persist the finalized aggregate metrics for a completed run."""
        ...

    @abstractmethod
    def compute_kpis(self, router_name: str) -> "KPISummary":
        """Aggregate persisted runs into the KPI set defined in section 5.

        Args:
            router_name: Restrict aggregation to runs using this router.

        Returns:
            A KPISummary with WAR, routing overhead, TSR, SFRR, and CE
            computed per the formulas in section 5.1.
        """
        ...


class BenchmarkDriverProtocol(Protocol):
    """Structural type for the top-level orchestrator, for type-checking
    call sites without forcing a concrete base class dependency.
    """

    def run_matrix(
        self,
        tasks: list[TaskCase],
        routers: list[BaseRouter],
        n_repeats: int,
    ) -> list["RunResult"]:
        """Execute every (task, router) pair n_repeats times and collect results."""
        ...


# ---------------------------------------------------------------------------
# Result / metric structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnMetric:
    """Per-turn metric record, the atomic unit persisted to the Run Store.

    Attributes:
        run_id: Identifier of the parent run.
        turn_index: Position of this turn within the run.
        router_name: Name of the router used for this turn's decision.
        routing_decision: The decision made for this turn.
        routing_latency_ms: Time spent inside router.route().
        inference_latency_ms: Time spent inside provider.generate().
        wall_hit: Whether detect_completion_wall flagged this turn.
        silent_failure_detected: Whether this turn's tool result was a
            silent failure (ground truth from Mock Tooling Layer).
        token_usage: TokenUsage for this turn's model call.
    """
    run_id: str
    turn_index: int
    router_name: str
    routing_decision: RoutingDecision
    routing_latency_ms: float
    inference_latency_ms: float
    wall_hit: bool
    silent_failure_detected: bool
    token_usage: TokenUsage

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        decision: RoutingDecision,
        wall_hit: bool,
        silent_failure: bool,
    ) -> "TurnMetric":
        """Construct a TurnMetric from the current AgentState and turn outputs."""
        ...


@dataclass(frozen=True)
class RunResult:
    """Finalized outcome of a single benchmark run (one TaskCase x BaseRouter
    x repeat-seed combination).

    Attributes:
        run_id: Unique identifier for this run.
        task: The TaskCase that was executed.
        router_name: Name of the router used.
        success: Whether the task-specific completion criterion was met.
        total_turns: Final turn count when the run terminated.
        wall_events: Total completion-wall hits during this run.
        silent_failures_injected: Count of silent failures the Mock
            Tooling Layer injected during this run.
        silent_failures_recovered: Count of those injected failures the
            agent successfully detected/recovered from before termination.
        total_cost_usd: Sum of TokenUsage.cost_usd across all turns.
        turn_metrics: Ordered list of per-turn TurnMetric records.
    """
    run_id: str
    task: TaskCase
    router_name: str
    success: bool
    total_turns: int
    wall_events: int
    silent_failures_injected: int
    silent_failures_recovered: int
    total_cost_usd: float
    turn_metrics: list[TurnMetric]

    @classmethod
    def finalize(
        cls,
        task: TaskCase,
        state: AgentState,
        metrics: list[TurnMetric],
    ) -> "RunResult":
        """Compute the final success/cost/failure-recovery summary for a run."""
        ...


@dataclass(frozen=True)
class KPISummary:
    """Aggregate KPI values for one router across all matrix runs, per
    section 5.2's target thresholds.

    Attributes:
        router_name: Router this summary is computed for.
        wall_avoidance_rate: WAR, see section 5.1.
        routing_overhead_p50_ms: Median routing latency.
        routing_overhead_p95_ms: 95th percentile routing latency.
        task_success_rate: TSR, see section 5.1.
        silent_failure_recovery_rate: SFRR, see section 5.1.
        cost_efficiency: CE relative to all-cloud baseline, see section 5.1.
        effective_cost_per_success_usd: Derived metric, see section 5.1.
        sample_size: Number of runs this summary was computed over.
    """
    router_name: str
    wall_avoidance_rate: float
    routing_overhead_p50_ms: float
    routing_overhead_p95_ms: float
    task_success_rate: float
    silent_failure_recovery_rate: float
    cost_efficiency: float
    effective_cost_per_success_usd: float
    sample_size: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RouterTimeoutError(Exception):
    """Raised by BaseRouter.route() when the routing decision itself times out."""


class ProviderUnavailableError(Exception):
    """Raised by BaseModelProvider.generate() when the backend cannot be reached."""
```

## 7. Data Output Schema

All raw run data is persisted append-only, one record per turn plus one
summary record per run, so downstream plotting can reconstruct both
turn-level traces (e.g. "context occupancy vs. wall-hit over time") and
run-level aggregates (e.g. "TSR by router by context depth").

### 7.1 `turns.jsonl` (one JSON object per line, per turn)

```json
{
  "run_id": "string (uuid)",
  "task_id": "string",
  "router_name": "string (enum: static_semantic | context_aware | commercial_cloud)",
  "turn_index": "integer",
  "domain": "string",
  "intent_complexity": "string (enum: trivial | moderate | complex_multi_tool)",
  "context_depth": "string (enum: shallow | mid | near_wall | over_wall)",
  "failure_profile": "string (enum: none | single_silent_failure | cascading_silent_failures | loud_failure)",
  "routing_target": "string (enum: local | cloud)",
  "routing_model_id": "string",
  "routing_reason": "string",
  "routing_latency_ms": "float",
  "inference_latency_ms": "float",
  "prompt_tokens": "integer",
  "completion_tokens": "integer",
  "cost_usd": "float",
  "wall_hit": "boolean",
  "tool_called": "boolean",
  "tool_name": "string | null",
  "silent_failure_detected": "boolean",
  "finish_reason": "string",
  "timestamp_utc": "string (ISO-8601)",
  "shadow_static_decision_target": "string (enum: local | cloud) | null -- only populated on context_aware runs; the StaticSemanticRouter's decision for this same RoutingRequest, evaluated but not executed, per section 5.3's Decision Divergence Rate",
  "shadow_local_wall_hit": "boolean | null -- only populated on turns where the live decision escalated to cloud; result of a shadow call to the LOCAL provider on the same prompt, per section 5.3's Escalation Precision/Recall",
  "shadow_call_cost_usd": "float | null -- cost of the shadow_local_wall_hit call, if made; tracked separately from cost_usd so shadow-call cost never contaminates a run's real total_cost_usd",
  "context_occupancy_ratio": "float | null -- RoutingFeatures.context_occupancy_ratio at this turn's decision point; backs section 5.3's Escalation Lead Time (occupancy headroom)"
}
```

### 7.2 `runs.csv` (one row per completed run)

| column | type | notes |
|---|---|---|
| `run_id` | string | uuid, joins to `turns.jsonl` |
| `task_id` | string | joins to dataset definition |
| `router_name` | string | enum, see above |
| `domain` | string | |
| `intent_complexity` | string | enum |
| `context_depth` | string | enum |
| `failure_profile` | string | enum |
| `repeat_seed` | integer | which of `N_REPEATS` this row is |
| `success` | boolean | task-specific completion criterion |
| `total_turns` | integer | |
| `wall_events` | integer | |
| `silent_failures_injected` | integer | |
| `silent_failures_recovered` | integer | |
| `total_cost_usd` | float | |
| `total_routing_overhead_ms` | float | sum across turns |
| `total_inference_latency_ms` | float | sum across turns |
| `started_at_utc` | string (ISO-8601) | |
| `finished_at_utc` | string (ISO-8601) | |

### 7.3 `kpi_summary.json` (one record per router, regenerated each analysis pass)

```json
{
  "router_name": "string",
  "sample_size": "integer",
  "wall_avoidance_rate": "float",
  "routing_overhead_p50_ms": "float",
  "routing_overhead_p95_ms": "float",
  "task_success_rate": "float",
  "silent_failure_recovery_rate": "float",
  "cost_efficiency": "float",
  "effective_cost_per_success_usd": "float",
  "breakdown_by_context_depth": {
    "shallow": { "task_success_rate": "float", "wall_avoidance_rate": "float" },
    "mid": { "task_success_rate": "float", "wall_avoidance_rate": "float" },
    "near_wall": { "task_success_rate": "float", "wall_avoidance_rate": "float" },
    "over_wall": { "task_success_rate": "float", "wall_avoidance_rate": "float" }
  },
  "comparative_static_vs_dynamic": {
    "_comment": "Present only on the context_aware entry -- these compare it against StaticSemanticRouter using the shadow fields from turns.jsonl (section 7.1); absent/omitted entirely for static_semantic and commercial_cloud entries.",
    "decision_divergence_rate": "float",
    "escalation_precision": "float",
    "escalation_recall": "float",
    "escalation_lead_time_headroom_mean": "float",
    "breakdown_by_context_depth": {
      "shallow": { "decision_divergence_rate": "float", "escalation_precision": "float", "escalation_recall": "float" },
      "mid": { "decision_divergence_rate": "float", "escalation_precision": "float", "escalation_recall": "float" },
      "near_wall": { "decision_divergence_rate": "float", "escalation_precision": "float", "escalation_recall": "float" },
      "over_wall": { "decision_divergence_rate": "float", "escalation_precision": "float", "escalation_recall": "float" }
    }
  }
}
```

## 8. Open Questions for Implementation Phase

1. Local judge model choice for the Context-Aware Router (must itself be
   small enough not to contribute to the completion wall it is meant to
   route around) — candidate constraint, not a decision: parameter count
   should be materially smaller than the primary local model under test.
2. Pricing source of truth for Cost Efficiency calculations — static
   config table vs. live API pricing lookup; static table recommended for
   benchmark reproducibility.
3. Whether `over_wall` context depth should be achieved purely via synthetic
   history padding (current design) or also via a real long-running session
   variant, to validate the synthetic padding is representative.
4. Exact commercial cloud router(s) to integrate first (`openrouter/auto`
   vs. Not Diamond) — affects `CommercialCloudRouter`'s payload mapping but
   not the interface contract above.

### 8.1 Resolved: Reference Run Configuration (local-only substitution for cloud)

For the first real (non-mocked) execution of this benchmark, no
`ANTHROPIC_API_KEY` or commercial-router credentials are provisioned —
the operator's Claude Pro/Max subscription covers interactive Claude.ai
and Claude Code usage, but **not** the metered Anthropic Messages API
that `AnthropicCloudProvider` calls, and provisioning a separate
pay-as-you-go API key was explicitly declined to avoid additional billing
beyond the existing subscription. Resolving open question 4 above in the
process, the reference run instead uses three locally-hosted Ollama
models, each playing a distinct architectural role at zero marginal cost:

| Role | Model | Notes |
|---|---|---|
| LOCAL completion model | `llama3.2:3b` | Stands in for the resource-constrained laptop model whose completion wall this suite exists to characterize. |
| CLOUD completion model (stand-in) | `llama3.1:8b` | A larger *local* model standing in for a cloud model. This validates the full routing/escalation control flow end to end, but does **not** measure a real local-vs-cloud quality or latency tradeoff — both "providers" run on the same machine. |
| Context-aware router's local judge | `qwen2.5:1.5b` | Smallest available model, consistent with open question 1's constraint that the judge stay materially smaller than the primary local model. |

`CommercialCloudRouter`'s `CloudRouterClient` is likewise filled by a
local stand-in (`LocalStandInCloudRouterClient` in
`scripts/run_benchmark.py`) that deterministically selects the CLOUD
completion model above without making any network call — there is no
real OpenRouter/Not Diamond integration in this run. The KPI numbers this
run produces should be read as a **structural validation that the full
pipeline executes correctly end to end**, not as a measurement of actual
local-vs-cloud cost/quality tradeoffs; that requires a follow-up run with
a real cloud API key.

## 9. Acceptance Criteria for This Spec

- [ ] Reviewed and approved by repo owner before any implementation PR is opened.
- [ ] All class signatures in section 6 compile under `python -m py_compile`
      with no method bodies beyond `...` or docstrings.
- [ ] Section 4's matrix size and section 5's KPI formulas are referenced,
      not redefined, by the implementation phase.
- [ ] No network calls, no real Ollama/cloud invocations introduced by this
      spec — it is documentation only.
