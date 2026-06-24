import csv
import json
import math

import pytest

from routing_benchmark.metrics import RunResult, TurnMetric
from routing_benchmark.models import (
    ContextDepthLevel,
    IntentComplexity,
    ModelTarget,
    RoutingDecision,
    TaskCase,
    TokenUsage,
    ToolFailureProfile,
)
from routing_benchmark.persistence import JsonlCsvMetricCollector


def make_task(context_depth=ContextDepthLevel.SHALLOW, task_id="task-1") -> TaskCase:
    return TaskCase(
        id=task_id,
        domain="data_lookup",
        complexity=IntentComplexity.MODERATE,
        initial_prompt="What is the revenue?",
        synthetic_history=[],
        failure_profile=ToolFailureProfile.NONE,
        max_turns=5,
        expected_tool_calls=1,
        context_depth=context_depth,
    )


def make_turn_metric(run_id="run-1", turn_index=0, router_name="static_semantic", cost=0.01, routing_latency_ms=2.0, inference_latency_ms=20.0, wall_hit=False, silent_failure=False) -> TurnMetric:
    return TurnMetric(
        run_id=run_id,
        turn_index=turn_index,
        router_name=router_name,
        routing_decision=RoutingDecision(target=ModelTarget.LOCAL, model_id="llama3.1:8b", reason="static"),
        routing_latency_ms=routing_latency_ms,
        inference_latency_ms=inference_latency_ms,
        wall_hit=wall_hit,
        silent_failure_detected=silent_failure,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=cost),
    )


def make_run_result(task, run_id="run-1", router_name="static_semantic", success=True, wall_events=0, turn_metrics=None) -> RunResult:
    turn_metrics = turn_metrics if turn_metrics is not None else [make_turn_metric(run_id=run_id, router_name=router_name)]
    return RunResult(
        run_id=run_id,
        task=task,
        router_name=router_name,
        success=success,
        total_turns=len(turn_metrics),
        wall_events=wall_events,
        silent_failures_injected=0,
        silent_failures_recovered=0,
        total_cost_usd=sum(m.token_usage.cost_usd for m in turn_metrics),
        turn_metrics=turn_metrics,
    )


def test_constructor_creates_output_directory(tmp_path):
    output_dir = tmp_path / "nested" / "results"
    JsonlCsvMetricCollector(output_dir=output_dir)
    assert output_dir.exists()


def test_record_turn_appends_jsonl_line(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    metric = make_turn_metric(run_id="run-1", turn_index=0)

    collector.record_turn("run-1", metric)

    lines = collector.turns_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "run-1"
    assert record["routing_target"] == "local"
    assert record["routing_model_id"] == "llama3.1:8b"
    assert record["prompt_tokens"] == 10
    assert record["cost_usd"] == 0.01
    assert "timestamp_utc" in record


def test_record_turn_appends_multiple_lines_in_order(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    collector.record_turn("run-1", make_turn_metric(run_id="run-1", turn_index=0))
    collector.record_turn("run-1", make_turn_metric(run_id="run-1", turn_index=1))

    lines = collector.turns_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["turn_index"] == 0
    assert json.loads(lines[1])["turn_index"] == 1


def test_record_run_writes_csv_header_once(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()

    collector.record_run(make_run_result(task, run_id="run-1"))
    collector.record_run(make_run_result(task, run_id="run-2"))

    with collector.runs_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["run_id"] == "run-2"
    assert rows[0]["context_depth"] == "shallow"
    assert rows[0]["domain"] == "data_lookup"
    assert rows[0]["success"] == "True"


def test_record_run_computes_total_latencies_from_turn_metrics(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()
    turn_metrics = [
        make_turn_metric(turn_index=0, routing_latency_ms=2.0, inference_latency_ms=20.0),
        make_turn_metric(turn_index=1, routing_latency_ms=3.0, inference_latency_ms=30.0),
    ]
    collector.record_run(make_run_result(task, turn_metrics=turn_metrics))

    with collector.runs_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert float(row["total_routing_overhead_ms"]) == pytest.approx(5.0)
    assert float(row["total_inference_latency_ms"]) == pytest.approx(50.0)


def test_collector_resumes_append_mode_across_instances(tmp_path):
    task = make_task()
    first = JsonlCsvMetricCollector(output_dir=tmp_path)
    first.record_run(make_run_result(task, run_id="run-1"))

    second = JsonlCsvMetricCollector(output_dir=tmp_path)
    second.record_run(make_run_result(task, run_id="run-2"))

    with second.runs_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Header written only once across both instances/processes.
    assert len(rows) == 2
    assert [r["run_id"] for r in rows] == ["run-1", "run-2"]


def test_compute_kpis_raises_for_unknown_router(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    with pytest.raises(ValueError):
        collector.compute_kpis("never_recorded")


def test_compute_kpis_basic_formulas(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()

    collector.record_run(make_run_result(task, run_id="run-1", success=True, wall_events=0))
    collector.record_run(make_run_result(task, run_id="run-2", success=False, wall_events=1))

    summary = collector.compute_kpis("static_semantic")

    assert summary.sample_size == 2
    assert summary.wall_avoidance_rate == pytest.approx(0.5)
    assert summary.task_success_rate == pytest.approx(0.5)
    assert summary.silent_failure_recovery_rate == 1.0  # no silent failures injected
    assert math.isnan(summary.cost_efficiency)  # no baseline supplied


def test_compute_kpis_cost_efficiency_with_baseline(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()
    collector.record_run(make_run_result(task, run_id="run-1"))  # total_cost_usd = 0.01

    summary = collector.compute_kpis("static_semantic", all_cloud_baseline_cost_usd=0.10)

    assert summary.cost_efficiency == pytest.approx(0.9)


def test_write_kpi_summary_persists_breakdown_by_context_depth(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    shallow_task = make_task(context_depth=ContextDepthLevel.SHALLOW, task_id="task-shallow")
    near_wall_task = make_task(context_depth=ContextDepthLevel.NEAR_WALL, task_id="task-near-wall")

    collector.record_run(make_run_result(shallow_task, run_id="run-1", success=True, wall_events=0))
    collector.record_run(make_run_result(near_wall_task, run_id="run-2", success=False, wall_events=2))

    collector.write_kpi_summary("static_semantic")

    persisted = json.loads(collector.kpi_summary_path.read_text(encoding="utf-8"))
    entry = persisted["static_semantic"]

    assert entry["sample_size"] == 2
    assert entry["breakdown_by_context_depth"]["shallow"]["task_success_rate"] == 1.0
    assert entry["breakdown_by_context_depth"]["near_wall"]["task_success_rate"] == 0.0
    assert entry["breakdown_by_context_depth"]["near_wall"]["wall_avoidance_rate"] == 0.0


def test_write_kpi_summary_merges_multiple_routers_in_one_file(tmp_path):
    collector = JsonlCsvMetricCollector(output_dir=tmp_path)
    task = make_task()

    collector.record_run(make_run_result(task, run_id="run-1", router_name="static_semantic"))
    collector.write_kpi_summary("static_semantic")

    collector.record_run(make_run_result(task, run_id="run-2", router_name="context_aware"))
    collector.write_kpi_summary("context_aware")

    persisted = json.loads(collector.kpi_summary_path.read_text(encoding="utf-8"))
    assert set(persisted.keys()) == {"static_semantic", "context_aware"}
