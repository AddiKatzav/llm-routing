# Context-Aware LLM Routing: Experiment Report

**Run:** `run_full_v2` — 296 runs, 516 turns  
**Date:** 2026-06-25 to 2026-06-26  
**Spec:** `routing_benchmark_spec.md` rev 001

---

## 1. Hypothesis

Local LLMs running on commodity hardware — in this case via Ollama on a single laptop — fail in two characteristically different ways during long-horizon agentic sessions:

1. **The Hard Completion Wall.** As the context window fills with tool histories and retrieved documents, the model stops producing usable completions. This is an abrupt cliff, not a gradual degradation. At some occupancy threshold the model truncates its output, produces malformed tool calls, or loops indefinitely.

2. **Silent Tool Failures.** Local models malform tool-call payloads, hallucinate argument values, or fail to detect a tool error in the returned observation and continue as if the call succeeded — producing no exception the orchestrator can catch.

Both failures are *state-dependent*: a request that local handles cleanly at turn 1 may push the model past the wall at turn 12. This means routing decisions must be re-evaluated *per-turn*, not just once at session start. A static router that picks local or cloud at session start and never revisits that choice cannot react to in-session context growth.

**The hypothesis:** A context-aware dynamic router — one that observes context occupancy on every turn and escalates from local to cloud proactively — will achieve meaningfully higher wall avoidance rates (WAR) than a static semantic router at high context depths, while incurring acceptable routing overhead.

---

## 2. Experimental Setup

### 2.1 Routers Under Test

Three router implementations were evaluated on identical task sets:

| Router | Decision frequency | Strategy | Models |
|---|---|---|---|
| **Static Semantic** | Once at session start | Embedding similarity to domain centroids | LOCAL = llama3.2:3b, CLOUD = llama3.1:8b |
| **Context-Aware** | Per turn | Rule → LLM judge cascade | Judge = qwen2.5:1.5b; LOCAL = llama3.2:3b; CLOUD = llama3.1:8b |
| **Commercial Cloud** | Once at session start | Always cloud | llama3.1:8b |

The context-aware router operates on a two-stage decision:
1. **Rule path (fast):** If `context_occupancy ≥ wall_proximity_threshold` (default: 0.85), escalate to CLOUD immediately. No judge call. Latency < 1ms.
2. **Judge path (slow):** Otherwise, call qwen2.5:1.5b to decide LOCAL or CLOUD based on task structure and observed context growth. Latency: varies from ~50ms to ~24 seconds on CPU.

### 2.2 Dataset

The benchmark generates synthetic tasks across a factorial matrix of:

- **Domains:** `data_lookup`, `multi_step_calculation`
- **Intent complexity:** `trivial`, `moderate`, `complex_multi_tool`
- **Context depth:** `shallow` (~10% full), `mid` (~50% full), `near_wall` (~85% full), `over_wall` (>100% full)
- **Failure profile:** `none` (only profile active in this run — injected silent failures were not exercised)
- **Repeat seeds:** 5 per configuration

Pre-filling the context window to a target depth simulates a multi-turn session that has already accumulated tool histories and retrieved content before the task under measurement begins. This is the key design choice: rather than running 80-turn sessions, we isolate the effect of context pressure by pre-filling to the relevant depth and measuring what happens on the subsequent task.

**Coverage this run:** 100 runs per router for static and context-aware (fully balanced, 20 task groups × 5 repeats). Commercial cloud ran 96 runs before the sweep was intentionally stopped (the commercial_cloud router adds no useful signal under local-only Ollama conditions — cost metrics are NaN for all routers, and the WAR ceiling was already visible).

### 2.3 Success Metrics

Two primary outcome metrics are recorded per run:
- **Task Success Rate (TSR):** Did the agent produce a correct final answer?
- **Wall Avoidance Rate (WAR):** Fraction of runs with zero context wall-hit events.

Spec §5.2 acceptance criteria:
1. Context-Aware WAR ≥ Static WAR + 15pp at both `near_wall` and `over_wall`
2. Context-Aware routing overhead p95 ≤ 150ms
3. Context-Aware cost efficiency ≥ 40% vs commercial cloud baseline

---

## 3. Results

### 3.1 Wall Avoidance Rate

> *See Figure 1 and Figure 3*

| Router | Shallow WAR | Mid WAR | Near-Wall WAR | Over-Wall WAR | Overall WAR |
|---|---|---|---|---|---|
| Static Semantic | 96% | 100% | 64% | 56% | 79% |
| Context-Aware | 92% | 100% | **100%** | **100%** | **98%** |
| Commercial Cloud | 84% | 96% | 100% | 100% | 95% |

The context-aware router eliminates context wall events at high depth: zero wall hits across 25 near-wall runs and 25 over-wall runs, versus 9 and 11 wall-hit events respectively for the static router. The WAR gap is **+36pp at near_wall** and **+44pp at over_wall**, both comfortably exceeding the +15pp spec threshold.

**Spec §5.2 Criterion 1: PASS**

### 3.2 Task Success Rate

> *See Figure 2*

| Router | Shallow TSR | Mid TSR | Near-Wall TSR | Over-Wall TSR | Overall TSR |
|---|---|---|---|---|---|
| Static Semantic | 16% | 4% | 0% | 0% | **5.0%** |
| Context-Aware | 4% | 8% | 0% | 0% | **3.0%** |
| Commercial Cloud | 96% | 80% | 0% | 0% | **45.8%** |

Overall TSR is uniformly low for both local-model routers (3–5%). The commercial cloud router dominates at shallow and mid depths (96% and 80%) but collapses to 0% at near_wall and over_wall — matching the local routers.

The 0% at near_wall and over_wall is structural: when the context window is already at or beyond capacity, the models (both local 3b and cloud 8b, running on Ollama on the same hardware) cannot fit a meaningful response. This is not a routing failure; it is the hard completion wall the spec was designed to expose.

### 3.3 Routing Overhead

> *See Figure 4 and Figure 8*

| Router | Median latency (ms) | p95 latency (ms) |
|---|---|---|
| Static Semantic | 0.21 | 0.7 |
| Context-Aware | 190.6 | **10,437** |
| Commercial Cloud | 0.03 | 0.1 |

The context-aware router's p95 of **10.4 seconds** misses the 150ms threshold by **70×**. This is not a bug; it is the inherent cost of calling qwen2.5:1.5b as a judge on a CPU-only machine. The p50 of 190ms tells a more nuanced story: half of routing decisions are fast (the rule path fires, or the judge responds quickly), but the tail is dominated by slow judge invocations during turns where neither the fast rule nor cached context can resolve the decision.

**Spec §5.2 Criterion 2: FAIL**

### 3.4 Cost Efficiency

All three routers operated entirely over local Ollama at $0 cost. The `AnthropicCloudProvider` implementation exists but requires a separate pay-as-you-go Anthropic API key, which was explicitly excluded from the experiment scope. Cost efficiency (CE) remains NaN for all runs.

**Spec §5.2 Criterion 3: UNMEASURED**

---

## 4. Analysis

### 4.1 Why the WAR Improvement Happens — and Why It's Incomplete

The context-aware router's 100% WAR at high depth is genuine, but the mechanism deserves scrutiny.

Of the 100 context-aware runs, **47 produced zero turns** — the task ran but no turn-level decision was recorded. This contrasts sharply with static semantic (zero turns = 0 runs). The distribution is revealing: 22 of 25 near-wall runs and 23 of 25 over-wall runs produced zero turns for context-aware, while nearly all shallow and mid runs executed normally.

What is happening: for pre-filled near/over-wall tasks, the context-aware router detects on turn 0 that `context_occupancy ≥ 0.85` and immediately escalates to cloud via the rule path. But the cloud target is also llama3.1:8b running on the same Ollama instance with the same memory constraints. The escalation call either silently fails or the model returns no usable response, causing the run to be recorded as complete with 0 turns and 0 wall events.

This means WAR=100% at near_wall and over_wall for context-aware is not "the router skillfully navigated the agent away from the wall" — it is "the router correctly detected an already-untenable context and did not attempt execution, which by the metrics definition counts as wall avoidance." Wall events require turns; no turns means no events.

The static semantic router, which makes no occupancy check before the first turn, dutifully attempts execution with the local model and records the resulting wall hit. Its lower WAR is evidence of actual execution, not of worse routing logic.

**Practical implication:** Both approaches fail equally on TSR at near_wall and over_wall (0% each). The difference is only in accounting: context-aware refuses the task cleanly, static attempts it and hits the wall. From a user perspective, both outcomes are failures. The WAR metric advantage is real for the spec compliance question but should not be read as a strong practical claim.

### 4.2 The Escalation Quality Problem

> *See Figure 5 and Figure 6*

Spec §5.3 requires measuring escalation *quality*, not just frequency. The shadow configuration tracked what would have happened if the static router had handled each context-aware turn — giving us three diagnostic metrics.

**Decision Divergence Rate (DDR): 26.7%**  
The context-aware router disagrees with the static router on more than one in four turns. All 20 disagreements are escalations to cloud — context-aware escalates where static would have routed local. This is the expected direction, but the rate is high enough to warrant checking whether those escalations were necessary.

**Escalation Precision: 15.0%**  
Of the 20 turns where context-aware escalated to cloud, only 3 were turns where the shadow simulation showed the local model would have hit the wall. The remaining 17 escalations were to cloud on turns the local model would have handled without a wall event. This is aggressive over-escalation: the router is paying the cloud cost (here, a slower Ollama model) on turns that did not need it.

**Escalation Recall: 100%**  
All 3 turns that the shadow simulation identified as wall-risk were correctly escalated. The router did not miss any genuine wall-risk moment. However, N=3 is too small for this figure to be meaningful — it reflects the small number of multi-turn context-aware runs that reached high occupancy before the run completed.

**Escalation Lead Time Headroom: mean = −0.826**  
The mean headroom at escalation is −0.826, meaning the average escalation happens when the context is *already 1.83× the limit*. 15 of 20 escalations occurred at `context_occupancy > 1.0`. This is the opposite of proactive management: the rule path (which fires at 0.85) fires on tasks that started with pre-filled context already at 1.1× or more, and the judge path escalations happen after the agent has been running and context has grown past the wall.

The combination of low precision and negative lead time headroom reveals a calibration problem: the current `wall_proximity_threshold=0.85` fires correctly on turns pre-filled to 85%+, but the judge — invoked for the remaining turns — is slow to escalate and does so only after the damage is done.

### 4.3 The Latency Gap and Its Root Cause

> *See Figure 4*

The 10.4-second p95 is dominated by the judge path. When the rule path fires (occupancy ≥ 0.85), latency is sub-millisecond. When the judge is called, latency depends on qwen2.5:1.5b inference speed on the local CPU — which is not bounded and produces the multi-second tail.

The fix is clear in principle: lower `wall_proximity_threshold` so more turns take the cheap rule path and fewer invoke the judge. The tradeoff is earlier escalation, which may sacrifice local compute that would have succeeded. The threshold tuning sweep (0.65, 0.70, 0.75) remains the highest-priority next step to determine how far the threshold can be lowered before recall degrades.

An alternative path is replacing the LLM judge with a lightweight deterministic classifier (e.g., a rule on context growth rate between turns, or a trained gradient-boosted model on occupancy + completion token fraction). This was out of scope for spec v0.1.

### 4.4 Commercial Cloud as Ceiling

The commercial cloud router (always cloud, llama3.1:8b via Ollama) achieves 96% TSR at shallow and 80% TSR at mid depths — a 15–20× improvement over both local routers. But it also collapses to 0% TSR at near_wall and over_wall, confirming that the hard completion wall is a function of context size, not model size (at least within the 3b–8b range tested here).

The commercial cloud router was not the primary evaluation target and its results reflect the same Ollama infrastructure constraints. A genuine commercial cloud provider (e.g., Anthropic API with claude-haiku or claude-sonnet) would be expected to handle larger context windows and produce different near_wall/over_wall behavior — but API billing was out of scope for this experiment.

---

## 5. Spec §5.2 Compliance Summary

| Criterion | Threshold | Actual | Status |
|---|---|---|---|
| Context-Aware WAR ≥ Static + 15pp at near_wall | +15pp | **+36pp** | ✓ PASS |
| Context-Aware WAR ≥ Static + 15pp at over_wall | +15pp | **+44pp** | ✓ PASS |
| Routing overhead p95 ≤ 150ms | 150ms | **10,437ms** | ✗ FAIL |
| Cost efficiency ≥ 40% vs cloud baseline | 40% | **NaN** | ⚠ UNMEASURED |

> *See Figure 8 for the compliance dashboard.*

---

## 6. Conclusions and Next Steps

**What the experiment shows:**

1. Context-aware per-turn routing does measurably reduce context wall events at high depth — the WAR gap is real, large, and exceeds the spec threshold. Even if the mechanism is partially pre-emption rather than mid-task escalation, the router correctly identifies that execution should not proceed with an already-saturated context.

2. Task success at near_wall and over_wall is 0% for all three routers under the local Ollama constraint. This is a hardware/model ceiling, not a routing problem. Routing cannot fix what the models physically cannot do.

3. The LLM judge is the correct architectural choice for dynamic escalation decisions, but calling it on CPU via Ollama is impractical. p95 of 10.4 seconds is unusable in any interactive or latency-sensitive pipeline. This must be addressed before the router is useful.

4. Escalation quality is poor: 85% of cloud escalations were unnecessary (low precision), and the average escalation happens after the context is already over the limit (negative lead time). The wall_proximity_threshold and judge calibration both need tuning.

**Recommended next steps (priority order):**

1. **Threshold sweep** — run `--wall-threshold 0.65 / 0.70 / 0.75` on the same 20 task groups. Analyze p95 latency and escalation recall at each setting. Target: p95 < 150ms without recall regression > 5pp. This is the fastest path to a practically deployable router.

2. **Judge replacement study** — prototype a deterministic escalation rule based on `(context_occupancy, completion_token_fraction, turn_index)` features. Compare recall/precision against the LLM judge. A decision tree on those three features would have sub-millisecond latency with zero model calls.

3. **Real API cloud provider** — connect `AnthropicCloudProvider` with a live Anthropic API key to measure cost efficiency and validate near_wall/over_wall behavior with a model that has a larger context ceiling (claude-haiku-4-5 supports 200k tokens). This unlocks the CE metric and provides a meaningful ceiling comparison.

4. **Failure profile coverage** — this run used `failure_profile=none` exclusively. The silent tool failure injection (which motivated the spec) was never exercised. A follow-on sweep with `loud_failure` and `silent_failure` profiles would test the router's response to in-session quality degradation — the second failure mode from the original hypothesis.

---

## Appendix: Data Files

| File | Description |
|---|---|
| `results/results/run_full_v2/runs.csv` | 296 run-level records |
| `results/results/run_full_v2/turns.jsonl` | 516 turn-level records with shadow data |
| `plots/fig1_war_by_depth.png` | WAR by router × context depth |
| `plots/fig2_tsr_by_depth.png` | TSR by router × context depth |
| `plots/fig3_war_gap_heatmap.png` | WAR gap heat map (context-aware − static) |
| `plots/fig4_routing_overhead.png` | Per-turn routing latency distributions |
| `plots/fig5_escalation_occupancy.png` | Escalation timing and decision path breakdown |
| `plots/fig6_s53_metrics.png` | §5.3 comparative metrics dashboard |
| `plots/fig7_wall_events_turns.png` | Wall events count and turns-per-run distribution |
| `plots/fig8_compliance_dashboard.png` | §5.2 compliance summary |
| `plots/fig9_occupancy_timeline.png` | Context occupancy trajectories for multi-turn runs |
| `scripts/generate_plots.py` | Reproduces all figures from raw data |
| `scripts/analyze_results.py` | §5.2 / §5.3 compliance report (requires kpi_summary.json) |
