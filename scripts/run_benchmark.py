"""Reference benchmark run against real local Ollama models.

Implements the configuration documented in routing_benchmark_spec.md
section 8.1: no Anthropic API key is provisioned (avoiding billing beyond
the operator's existing Pro/Max subscription), so every "CLOUD" slot --
the CommercialCloudRouter's selection, the ContextAwareRouter's escalation
target, and the StaticSemanticRouter's cloud-leaning exemplars -- is
filled by a second, larger local Ollama model instead of a real cloud API
call. This validates the full routing/escalation pipeline end to end at
zero marginal cost; it does not measure a real local-vs-cloud tradeoff.

Usage:
    python scripts/run_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

from routing_benchmark.dataset import synthesize_dataset
from routing_benchmark.driver import BenchmarkDriver
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

# Bounds real generation time/length; routers don't set model_params today,
# so these constructor-level defaults are the only generation knobs in play.
# num_predict trimmed down from an initial 300 after the first real run hit
# CPU-only generation times that exceeded a 60s request timeout on 12/18
# calls (load average 8-10 on a CPU-only host, no GPU offload).
DEFAULT_GENERATION_OPTIONS = {"num_predict": 150, "temperature": 0.3}

# Default OllamaProvider.timeout_s (60s) was too short for CPU-only
# inference on this host -- bumped generously rather than risk a second
# round of timeouts eating most of the runs again.
REQUEST_TIMEOUT_S = 240.0

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


def select_subset(tasks: list[TaskCase]) -> list[TaskCase]:
    """A small, fast-to-run slice of the full matrix.

    The full matrix (192 tasks x 3 routers) against real local model
    inference would take a long time; this picks one baseline task per
    domain plus a couple of tasks specifically chosen to exercise the
    completion-wall and silent-tool-failure mechanics this suite exists to
    measure.
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
        default_options={"num_predict": 16, "temperature": 0.0},
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
        judge_timeout_ms=60_000.0,
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


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "results" / "run_local_standin"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = select_subset(synthesize_dataset(seed=0))

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
    driver = BenchmarkDriver(providers=providers, mock_tooling=mock_tooling, metric_collector=collector)

    routers: list[BaseRouter] = [
        build_static_semantic_router(),
        build_context_aware_router(),
        build_commercial_cloud_router(),
    ]

    print(f"Running {len(tasks)} tasks x {len(routers)} routers = {len(tasks) * len(routers)} runs...")
    for task in tasks:
        print(f"  task: {task.id}")

    results = driver.run_matrix(tasks, routers, n_repeats=1)

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
