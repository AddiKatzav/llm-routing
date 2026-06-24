"""File-persisting BaseMetricCollector for the LLM Routing Benchmark Suite.

Implements the "Run Store" target from section 7 of
``routing_benchmark_spec.md``: per-turn records appended to a JSONL file
and per-run summaries appended to a CSV file, plus KPI aggregation
matching section 5.1's formulas and a ``kpi_summary.json`` writer matching
section 7.3 (including the per-context-depth breakdown).

Known gaps relative to the *literal* section 7 schema, each because the
underlying typed objects don't carry the field, and adding it would mean
changing contracts established earlier in the suite:

- Per-turn ``task_id``/``domain``/``intent_complexity``/``context_depth``/
  ``failure_profile``/``finish_reason``/``tool_name`` are absent from
  ``turns.jsonl`` here -- ``TurnMetric`` has no back-reference to the
  ``TaskCase`` or the raw ``CompletionResult`` it came from, only to the
  ``RoutingDecision`` and ``TokenUsage``.
- Per-run ``repeat_seed``/``started_at_utc``/``finished_at_utc`` are
  absent from ``runs.csv`` -- ``RunResult`` doesn't track wall-clock run
  boundaries, and ``repeat_seed`` is recoverable by parsing the
  ``:repeatN`` suffix ``BenchmarkDriver`` already encodes into every
  ``run_id`` rather than needing its own column.

What's written here is the largest subset of section 7 derivable from
``TurnMetric``/``RunResult``/``TaskCase`` as they're actually defined.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from routing_benchmark.metrics import BaseMetricCollector, KPISummary, RunResult, TurnMetric
from routing_benchmark.models import ContextDepthLevel

__all__ = ["JsonlCsvMetricCollector"]

_RUNS_CSV_FIELDS = [
    "run_id",
    "task_id",
    "domain",
    "intent_complexity",
    "context_depth",
    "failure_profile",
    "router_name",
    "success",
    "total_turns",
    "wall_events",
    "silent_failures_injected",
    "silent_failures_recovered",
    "total_cost_usd",
    "total_routing_overhead_ms",
    "total_inference_latency_ms",
    "recorded_at_utc",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * pct
    lower, upper = int(rank), min(int(rank) + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


class JsonlCsvMetricCollector(BaseMetricCollector):
    """Appends TurnMetrics/RunResults to disk while caching both in memory
    so compute_kpis() can aggregate without re-parsing what was just written.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        turns_filename: str = "turns.jsonl",
        runs_filename: str = "runs.csv",
        kpi_summary_filename: str = "kpi_summary.json",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.turns_path = self.output_dir / turns_filename
        self.runs_path = self.output_dir / runs_filename
        self.kpi_summary_path = self.output_dir / kpi_summary_filename

        self._turns: list[TurnMetric] = []
        self._runs: list[RunResult] = []
        self._runs_csv_header_written = self.runs_path.exists() and self.runs_path.stat().st_size > 0

    def record_turn(self, run_id: str, turn_metric: TurnMetric) -> None:
        self._turns.append(turn_metric)
        record = {
            "run_id": run_id,
            "turn_index": turn_metric.turn_index,
            "router_name": turn_metric.router_name,
            "routing_target": turn_metric.routing_decision.target.value,
            "routing_model_id": turn_metric.routing_decision.model_id,
            "routing_reason": turn_metric.routing_decision.reason,
            "routing_latency_ms": turn_metric.routing_latency_ms,
            "inference_latency_ms": turn_metric.inference_latency_ms,
            "prompt_tokens": turn_metric.token_usage.prompt_tokens,
            "completion_tokens": turn_metric.token_usage.completion_tokens,
            "cost_usd": turn_metric.token_usage.cost_usd,
            "wall_hit": turn_metric.wall_hit,
            "silent_failure_detected": turn_metric.silent_failure_detected,
            "timestamp_utc": _utc_now_iso(),
        }
        with self.turns_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_run(self, run_result: RunResult) -> None:
        self._runs.append(run_result)
        total_routing_overhead_ms = sum(m.routing_latency_ms for m in run_result.turn_metrics)
        total_inference_latency_ms = sum(m.inference_latency_ms for m in run_result.turn_metrics)
        context_depth = run_result.task.context_depth

        row = {
            "run_id": run_result.run_id,
            "task_id": run_result.task.id,
            "domain": run_result.task.domain,
            "intent_complexity": run_result.task.complexity.value,
            "context_depth": context_depth.value if context_depth is not None else "",
            "failure_profile": run_result.task.failure_profile.value,
            "router_name": run_result.router_name,
            "success": run_result.success,
            "total_turns": run_result.total_turns,
            "wall_events": run_result.wall_events,
            "silent_failures_injected": run_result.silent_failures_injected,
            "silent_failures_recovered": run_result.silent_failures_recovered,
            "total_cost_usd": run_result.total_cost_usd,
            "total_routing_overhead_ms": total_routing_overhead_ms,
            "total_inference_latency_ms": total_inference_latency_ms,
            "recorded_at_utc": _utc_now_iso(),
        }

        with self.runs_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_RUNS_CSV_FIELDS)
            if not self._runs_csv_header_written:
                writer.writeheader()
                self._runs_csv_header_written = True
            writer.writerow(row)

    def compute_kpis(
        self,
        router_name: str,
        all_cloud_baseline_cost_usd: float | None = None,
    ) -> KPISummary:
        """Aggregate KPIs for one router from this session's recorded runs.

        Args:
            router_name: Restrict aggregation to runs using this router.
            all_cloud_baseline_cost_usd: Total cost of replaying the same
                tasks forcing target=CLOUD every turn, per spec section
                5.1's Cost Efficiency formula. This collector has no way
                to produce that baseline itself (it would require a
                separate all-cloud run of the same task set); pass it in
                if you have one, otherwise cost_efficiency is reported as
                NaN rather than silently defaulting to 0.0.
        """
        runs = [r for r in self._runs if r.router_name == router_name]
        if not runs:
            raise ValueError(f"no runs recorded for router {router_name!r}")

        turns = [t for t in self._turns if t.router_name == router_name]
        routing_latencies = sorted(t.routing_latency_ms for t in turns)

        wall_avoidance_rate = 1 - (sum(1 for r in runs if r.wall_events > 0) / len(runs))
        task_success_rate = sum(1 for r in runs if r.success) / len(runs)

        injected = sum(r.silent_failures_injected for r in runs)
        recovered = sum(r.silent_failures_recovered for r in runs)
        silent_failure_recovery_rate = (recovered / injected) if injected else 1.0

        total_cost = sum(r.total_cost_usd for r in runs)
        successful = sum(1 for r in runs if r.success)
        effective_cost_per_success = (total_cost / successful) if successful else float("inf")

        if all_cloud_baseline_cost_usd is not None and all_cloud_baseline_cost_usd > 0:
            cost_efficiency = 1 - (total_cost / all_cloud_baseline_cost_usd)
        else:
            cost_efficiency = float("nan")

        return KPISummary(
            router_name=router_name,
            wall_avoidance_rate=wall_avoidance_rate,
            routing_overhead_p50_ms=_percentile(routing_latencies, 0.5),
            routing_overhead_p95_ms=_percentile(routing_latencies, 0.95),
            task_success_rate=task_success_rate,
            silent_failure_recovery_rate=silent_failure_recovery_rate,
            cost_efficiency=cost_efficiency,
            effective_cost_per_success_usd=effective_cost_per_success,
            sample_size=len(runs),
        )

    def write_kpi_summary(
        self,
        router_name: str,
        all_cloud_baseline_cost_usd: float | None = None,
    ) -> KPISummary:
        """Compute KPIs for one router and persist them to kpi_summary.json.

        Matches spec section 7.3, including ``breakdown_by_context_depth``
        -- which ``KPISummary`` itself does not carry, since that field is
        part of the *file* schema, not the typed result object returned by
        ``compute_kpis()``. The file holds one entry per router (merged
        across calls), keyed by router_name, so multiple routers in the
        same benchmark session share one summary file.
        """
        summary = self.compute_kpis(router_name, all_cloud_baseline_cost_usd)
        runs = [r for r in self._runs if r.router_name == router_name]

        breakdown: dict[str, dict[str, float]] = {}
        for depth in ContextDepthLevel:
            depth_runs = [r for r in runs if r.task.context_depth == depth]
            if not depth_runs:
                continue
            breakdown[depth.value] = {
                "task_success_rate": sum(1 for r in depth_runs if r.success) / len(depth_runs),
                "wall_avoidance_rate": 1 - (sum(1 for r in depth_runs if r.wall_events > 0) / len(depth_runs)),
            }

        existing: dict[str, dict] = {}
        if self.kpi_summary_path.exists():
            existing = json.loads(self.kpi_summary_path.read_text(encoding="utf-8"))

        existing[router_name] = {
            "router_name": summary.router_name,
            "sample_size": summary.sample_size,
            "wall_avoidance_rate": summary.wall_avoidance_rate,
            "routing_overhead_p50_ms": summary.routing_overhead_p50_ms,
            "routing_overhead_p95_ms": summary.routing_overhead_p95_ms,
            "task_success_rate": summary.task_success_rate,
            "silent_failure_recovery_rate": summary.silent_failure_recovery_rate,
            "cost_efficiency": summary.cost_efficiency,
            "effective_cost_per_success_usd": summary.effective_cost_per_success_usd,
            "breakdown_by_context_depth": breakdown,
        }
        self.kpi_summary_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return summary
