"""Reference benchmark runs against real local Ollama models.

Implements the configuration documented in routing_benchmark_spec.md
section 8.1: no Anthropic API key is provisioned (avoiding billing beyond
the operator's existing Pro/Max subscription), so every "CLOUD" slot --
the CommercialCloudRouter's selection, the ContextAwareRouter's escalation
target, and the StaticSemanticRouter's cloud-leaning exemplars -- is
filled by a second, larger local Ollama model instead of a real cloud API
call. This validates the full routing/escalation pipeline end to end at
zero marginal cost; it does not measure a real local-vs-cloud tradeoff.

Also implements section 8.1's context-window calibration fix: the real
llama3.2:3b model defaults to a 4096-token context window (confirmed via
`ollama ps`), but the benchmark's near_wall/over_wall padding math assumed
DEFAULT_CONTEXT_WINDOW_LIMIT=8192. The first instinct was to force
num_ctx=8192 on every Ollama request so the model would match the
benchmark's assumption -- but a direct timing test showed this makes
large-context calls dramatically slower (an 11k-word prompt at
num_ctx=8192 didn't finish within a 300s timeout), which would have made
the overnight run far less productive, not more. The cheaper, correct fix
goes the other way: recalibrate REAL_CONTEXT_WINDOW_LIMIT down to the
model's actual window and pass it to both dataset synthesis (so
"near_wall" approaches 4096 tokens, not 8192) and the driver (so
context_occupancy_ratio features are computed against the same number).
No num_ctx override needed -- it already matches Ollama's default.

Wires spec section 5.3's static-vs-dynamic comparative metrics
(Decision Divergence Rate, Escalation Precision/Recall, Escalation Lead
Time) via a ShadowConfig on the context_aware router: a second
StaticSemanticRouter instance is shadow-evaluated every turn (free), and
the LOCAL provider gets one extra shadow call per turn context_aware
escalates to CLOUD (the only part of this that costs real model-call
time/money, and only during benchmarking -- never in a real deployment).

Usage:
    python scripts/run_benchmark.py --subset demo
    python scripts/run_benchmark.py --subset overnight --n-repeats 1
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from routing_benchmark.dataset import synthesize_dataset
from routing_benchmark.driver import BenchmarkDriver, ShadowConfig
from routing_benchmark.models import (
    ContextDepthLevel,
    IntentComplexity,
    ModelTarget,
    TaskCase,
    ToolFailureProfile,
)
from routing_benchmark.persistence import JsonlCsvMetricCollector
from routing_benchmark.providers.ollama import OllamaProvider
from routing_benchmark.router import BaseRouter
from routing_benchmark.routers.commercial_cloud import CloudRouterResponse, CommercialCloudRouter
from routing_benchmark.routers.context_aware import ContextAwareRouter
from routing_benchmark.routers.static_semantic import IntentExemplar, StaticSemanticRouter
from routing_benchmark.tooling import DeterministicMockToolingLayer

# See spec section 8.1 for why these three roles are all local Ollama models.
LOCAL_MODEL_ID = "llama3.2:3b"
CLOUD_STANDIN_MODEL_ID = "llama3.1:8b"
JUDGE_MODEL_ID = "qwen2.5:1.5b"

# llama3.2:3b's real default context window (confirmed via `ollama ps` ->
# context_length: 4096). Dataset synthesis and the driver's
# context_occupancy_ratio feature are both calibrated against this number
# instead of the library's generic DEFAULT_CONTEXT_WINDOW_LIMIT (8192), so
# "near_wall"/"over_wall" tasks approach the model's *real* wall rather
# than starting past it. See module docstring for why num_ctx itself is
# left at Ollama's default rather than raised to match 8192.
REAL_CONTEXT_WINDOW_LIMIT = 4096

# Bounds real generation time/length; routers don't set model_params today,
# so these constructor-level defaults are the only generation knobs in
# play. num_predict trimmed down from an initial 300 after the first real
# run hit CPU-only generation times that exceeded a 60s request timeout on
# 12/18 calls (load average 8-10 on a CPU-only host, no GPU offload).
DEFAULT_GENERATION_OPTIONS = {"num_predict": 150, "temperature": 0.3}
JUDGE_GENERATION_OPTIONS = {"num_predict": 16, "temperature": 0.0}

# Default OllamaProvider.timeout_s (60s) was too short for CPU-only
# inference on this host -- bumped generously rather than risk a second
# round of timeouts eating most of the runs again. A direct timing test of
# a near_wall-sized prompt took 178.8s, close enough to the prior 240s
# budget that an unattended overnight run (no one watching to retry)
# warrants more headroom.
REQUEST_TIMEOUT_S = 300.0
JUDGE_TIMEOUT_MS = 90_000.0

# A generic, domain-agnostic tool so models that support function calling
# have something to call -- without this, no real model would ever
# populate CompletionResult.tool_call, and the Mock Tooling Layer's
# failure-injection logic would never actually run.
GENERIC_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_step",
            "description": (
                "Execute one concrete step toward completing the user's request "
                "(e.g. looking up a figure, performing a calculation, editing a "
                "file) and return its result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "A short description of the step to perform.",
                    }
                },
                "required": ["action"],
            },
        },
    }
]


class LocalStandInCloudRouterClient:
    """Stand-in for a real commercial router client (e.g. openrouter/auto).

    Makes no network call of its own -- it just always selects the local
    "cloud" stand-in model, so CommercialCloudRouter's wrapper logic
    (payload construction, model-target mapping) still runs for real.
    """

    def submit(self, payload: dict) -> CloudRouterResponse:
        return CloudRouterResponse(
            selected_model=CLOUD_STANDIN_MODEL_ID,
            raw={"note": "local stand-in -- no real commercial router called", "payload_domain": payload.get("task_domain")},
        )


def select_subset_demo(tasks: list[TaskCase]) -> list[TaskCase]:
    """A small, fast-to-run slice: one baseline task per domain plus two
    tasks chosen to exercise the completion-wall and silent-tool-failure
    mechanics this suite exists to measure. ~6 tasks.
    """
    by_domain_baseline = [
        t
        for t in tasks
        if t.complexity == IntentComplexity.MODERATE
        and t.context_depth == ContextDepthLevel.SHALLOW
        and t.failure_profile == ToolFailureProfile.NONE
    ]

    near_wall_stress = [
        t
        for t in tasks
        if t.domain == "multi_step_calculation"
        and t.complexity == IntentComplexity.MODERATE
        and t.context_depth == ContextDepthLevel.NEAR_WALL
        and t.failure_profile == ToolFailureProfile.NONE
    ]

    silent_failure_stress = [
        t
        for t in tasks
        if t.domain == "data_lookup"
        and t.complexity == IntentComplexity.MODERATE
        and t.context_depth == ContextDepthLevel.SHALLOW
        and t.failure_profile == ToolFailureProfile.SINGLE_SILENT_FAILURE
    ]

    return by_domain_baseline + near_wall_stress + silent_failure_stress


# A direct timing test against the recalibrated near_wall padding (3919
# words) took 178.8s for a single call on the 3B local model -- CPU-only
# prompt-processing time is dominated by context size itself, regardless
# of whether the call ends up hitting the wall. context_aware/
# commercial_cloud escalate near_wall/over_wall turns to the 8B cloud
# stand-in, which will be slower still. Several domain/complexity
# combinations at this depth have max_turns up to 14 by default, which
# would put a handful of (task, router) cells at an hour or more each and
# risk a multi-day run instead of an overnight one. Both near_wall and
# over_wall are turn-invariant for this purpose anyway -- the demo run
# showed the same depth pinned the model's context identically on every
# one of 8 turns -- so capping max_turns for those depths loses no real
# signal.
OVERNIGHT_MAX_TURNS_CAP = 2


def select_subset_overnight(tasks: list[TaskCase]) -> list[TaskCase]:
    """A bounded but much larger slice for an unattended overnight run.

    Every (domain x complexity x depth) combination at failure_profile=NONE
    (4 x 3 x 4 = 48 tasks) -- the full structural matrix minus the
    failure-injection dimension -- plus, for each domain at
    complexity=MODERATE/depth=SHALLOW, the two silent-failure profiles (4 x 2
    = 8 tasks). 56 tasks total; see select_subset_demo for the smaller,
    faster slice this was scoped down from. near_wall/over_wall tasks have
    their max_turns capped (see OVERNIGHT_MAX_TURNS_CAP) to bound worst-case
    run time.
    """
    baseline_full = [
        t
        for t in tasks
        if t.failure_profile == ToolFailureProfile.NONE
    ]

    failure_variants = [
        t
        for t in tasks
        if t.complexity == IntentComplexity.MODERATE
        and t.context_depth == ContextDepthLevel.SHALLOW
        and t.failure_profile
        in (ToolFailureProfile.SINGLE_SILENT_FAILURE, ToolFailureProfile.CASCADING_SILENT_FAILURES)
    ]

    selected = baseline_full + failure_variants
    return [
        dataclasses.replace(t, max_turns=min(t.max_turns, OVERNIGHT_MAX_TURNS_CAP))
        if t.context_depth in (ContextDepthLevel.NEAR_WALL, ContextDepthLevel.OVER_WALL)
        else t
        for t in selected
    ]


SUBSETS = {
    "demo": select_subset_demo,
    "overnight": select_subset_overnight,
}


def build_static_semantic_router() -> StaticSemanticRouter:
    exemplars = [
        IntentExemplar(
            label="data_lookup",
            example_text="look up a metric for a company in a given quarter",
            target=ModelTarget.LOCAL,
            model_id=LOCAL_MODEL_ID,
        ),
        IntentExemplar(
            label="multi_step_calculation",
            example_text="calculate compound interest and reconcile totals across files",
            target=ModelTarget.LOCAL,
            model_id=LOCAL_MODEL_ID,
        ),
        IntentExemplar(
            label="file_edit_simulation",
            example_text="open a file, update a column, and save the result",
            target=ModelTarget.LOCAL,
            model_id=LOCAL_MODEL_ID,
        ),
        IntentExemplar(
            label="ambiguous_complex",
            example_text="something doesn't add up across multiple files and a dashboard, figure out what's going on",
            target=ModelTarget.CLOUD,
            model_id=CLOUD_STANDIN_MODEL_ID,
        ),
    ]
    return StaticSemanticRouter(
        intent_exemplars=exemplars,
        default_target=ModelTarget.LOCAL,
        default_model_id=LOCAL_MODEL_ID,
    )


def build_context_aware_router() -> ContextAwareRouter:
    judge = OllamaProvider(
        model_id=JUDGE_MODEL_ID,
        default_options=JUDGE_GENERATION_OPTIONS,
        tools=None,
        timeout_s=REQUEST_TIMEOUT_S,
    )
    return ContextAwareRouter(
        local_judge_model=judge,
        escalation_threshold=2,
        wall_proximity_threshold=0.85,
        local_model_id=LOCAL_MODEL_ID,
        cloud_model_id=CLOUD_STANDIN_MODEL_ID,
        # Spec section 5.2's <=150ms p95 target is a production aspiration
        # for a dedicated lightweight classifier; a real local LLM judge
        # call over HTTP needs a much more generous budget to complete at
        # all. routing_latency_ms is still recorded accurately either way.
        judge_timeout_ms=JUDGE_TIMEOUT_MS,
    )


def build_commercial_cloud_router() -> CommercialCloudRouter:
    return CommercialCloudRouter(
        cloud_router_client=LocalStandInCloudRouterClient(),
        model_target_map={
            CLOUD_STANDIN_MODEL_ID: ModelTarget.CLOUD,
            LOCAL_MODEL_ID: ModelTarget.LOCAL,
        },
        default_target=ModelTarget.CLOUD,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="demo")
    parser.add_argument("--output-dir", default=None, help="Defaults to results/run_<subset>")
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir_name = args.output_dir or f"run_{args.subset}"
    output_dir = Path(__file__).resolve().parent.parent / "results" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = SUBSETS[args.subset](
        synthesize_dataset(seed=args.seed, context_window_limit=REAL_CONTEXT_WINDOW_LIMIT)
    )

    providers = {
        LOCAL_MODEL_ID: OllamaProvider(
            model_id=LOCAL_MODEL_ID,
            tools=GENERIC_TOOL_SCHEMA,
            default_options=DEFAULT_GENERATION_OPTIONS,
            timeout_s=REQUEST_TIMEOUT_S,
        ),
        CLOUD_STANDIN_MODEL_ID: OllamaProvider(
            model_id=CLOUD_STANDIN_MODEL_ID,
            tools=GENERIC_TOOL_SCHEMA,
            default_options=DEFAULT_GENERATION_OPTIONS,
            timeout_s=REQUEST_TIMEOUT_S,
        ),
    }
    mock_tooling = DeterministicMockToolingLayer()
    collector = JsonlCsvMetricCollector(output_dir=output_dir)

    # Spec section 5.3's comparative metrics need a *separate*
    # StaticSemanticRouter instance to shadow-evaluate alongside the live
    # context_aware router -- separate so its internal per-task decision
    # cache never shares state with the one actually being swept below.
    shadow_static_router = build_static_semantic_router()

    driver = BenchmarkDriver(
        providers=providers,
        mock_tooling=mock_tooling,
        metric_collector=collector,
        context_window_limit=REAL_CONTEXT_WINDOW_LIMIT,
        shadow_configs={
            "context_aware": ShadowConfig(
                static_router=shadow_static_router,
                local_model_id=LOCAL_MODEL_ID,
            ),
        },
    )

    routers: list[BaseRouter] = [
        build_static_semantic_router(),
        build_context_aware_router(),
        build_commercial_cloud_router(),
    ]

    print(f"subset={args.subset} n_repeats={args.n_repeats} output_dir={output_dir}")
    print(f"Running {len(tasks)} tasks x {len(routers)} routers x {args.n_repeats} repeats "
          f"= {len(tasks) * len(routers) * args.n_repeats} runs...")
    for task in tasks:
        print(f"  task: {task.id}")

    results = driver.run_matrix(tasks, routers, n_repeats=args.n_repeats)

    print(f"\nCompleted {len(results)} runs.\n")
    for router in routers:
        try:
            summary = collector.write_kpi_summary(router.name)
        except ValueError:
            print(f"{router.name}: no runs recorded")
            continue
        print(
            f"{router.name}: "
            f"task_success_rate={summary.task_success_rate:.2f} "
            f"wall_avoidance_rate={summary.wall_avoidance_rate:.2f} "
            f"silent_failure_recovery_rate={summary.silent_failure_recovery_rate:.2f} "
            f"routing_overhead_p50_ms={summary.routing_overhead_p50_ms:.1f} "
            f"sample_size={summary.sample_size}"
        )

    if driver.failures:
        print(f"\n{len(driver.failures)} run(s) hit expected runtime errors:")
        for failure in driver.failures:
            print(f"  {failure.run_id}: {failure.error_type}: {failure.message}")

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
