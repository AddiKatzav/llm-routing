"""
Compute article-level metrics from benchmark results.

Three analytical threads:
  Thread 1 — Threshold-Latency-WAR Tradeoff (threshold sweep data)
  Thread 2 — Pre-emption Ablation (run_full_v2, near/over wall depths)
  Thread 3 — Escalation Quality (run_overnight_shadow, shadow fields)

Usage:
    python scripts/article_analysis.py
    # Prints summary tables; saves plots/article_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent

# Four threshold configurations for Thread 1.
# 0.50/0.65/0.80 use the "threshold_tuning" subset (2 domains, mid+near_wall only).
# 0.85 uses run_full_v2 (all depths, different domain scope — noted in results).
THRESHOLD_CONFIGS: dict[str, dict] = {
    "0.50": {
        "path": ROOT_DIR / "results/run_tuned_050",
        "dedup": True,   # crashed overnight run appended duplicates; keep last (post-fix)
        "domains": ["file_edit_simulation", "data_lookup"],
    },
    "0.65": {
        "path": ROOT_DIR / "results/run_tuned_065",
        "dedup": False,
        "domains": ["file_edit_simulation", "data_lookup"],
    },
    "0.80": {
        "path": ROOT_DIR / "results/run_tuned_080",
        "dedup": False,
        "domains": ["file_edit_simulation", "data_lookup"],
    },
    "0.85": {
        "path": ROOT_DIR / "results/results/run_full_v2",
        "dedup": False,
        "domains": None,  # data_lookup + multi_step_calculation (different from 0.50/0.65/0.80)
    },
}

SHADOW_RUN_PATH = ROOT_DIR / "results/run_overnight_shadow"
FULL_V2_PATH    = ROOT_DIR / "results/results/run_full_v2"
PLOTS_DIR       = ROOT_DIR / "plots"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_runs(
    path: Path,
    *,
    dedup: bool = False,
    domain_filter: list[str] | None = None,
) -> pd.DataFrame:
    runs = pd.read_csv(path / "runs.csv")
    runs["no_wall"] = runs["wall_events"] == 0
    if dedup:
        runs = runs.drop_duplicates(subset=["run_id"], keep="last")
    if domain_filter is not None:
        runs = runs[runs["domain"].isin(domain_filter)]
    return runs.reset_index(drop=True)


def load_turns(
    path: Path,
    *,
    valid_run_ids: set | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    with open(path / "turns.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    turns = pd.DataFrame(records) if records else pd.DataFrame()
    if valid_run_ids is not None and len(turns) > 0:
        turns = turns[turns["run_id"].isin(valid_run_ids)]
    return turns.reset_index(drop=True)


def classify_routing_path(reason: str | None) -> str:
    if not reason:
        return "other"
    if reason.startswith("context_proximity_to_wall"):
        return "rule"
    if reason.startswith("judge"):
        return "judge"
    if reason.startswith("failure_escalation"):
        return "failure_escalation"
    if reason.startswith("semantic"):
        return "semantic"
    if reason.startswith("cloud_router"):
        return "cloud_router"
    return "other"


def classify_ca_run(total_turns: int, run_id: str, ca_cloud_run_ids: set) -> str:
    if total_turns == 0:
        return "pre_empted"
    if run_id in ca_cloud_run_ids:
        return "mid_task_escalated"
    return "stayed_local"


# ---------------------------------------------------------------------------
# Thread 1: Threshold-Latency-WAR Tradeoff
# ---------------------------------------------------------------------------

def compute_thread1_threshold_sweep() -> dict:
    """
    For each threshold configuration, compute:
    - p50/p95 routing latency (context_aware turns only)
    - rule path % vs judge path % (from routing_reason)
    - WAR gap at near_wall and mid depths (CA - static)
    """
    results: dict[str, dict] = {}

    def war_pct(df: pd.DataFrame, depth: str) -> float:
        sub = df[df["context_depth"] == depth]["no_wall"]
        return float(sub.mean() * 100) if len(sub) > 0 else float("nan")

    for thresh, cfg in THRESHOLD_CONFIGS.items():
        runs = load_runs(cfg["path"], dedup=cfg["dedup"], domain_filter=cfg["domains"])
        valid_ids = set(runs["run_id"])
        turns = load_turns(cfg["path"], valid_run_ids=valid_ids)

        ca_runs = runs[runs["router_name"] == "context_aware"]
        st_runs = runs[runs["router_name"] == "static_semantic"]

        if len(turns) > 0:
            ca_turns = turns[turns["router_name"] == "context_aware"].copy()
            ca_turns["routing_path"] = ca_turns["routing_reason"].apply(classify_routing_path)
        else:
            ca_turns = pd.DataFrame()

        def safe_quantile(series: pd.Series, q: float) -> float:
            return float(series.quantile(q)) if len(series) > 0 else float("nan")

        p95 = safe_quantile(ca_turns["routing_latency_ms"], 0.95) if len(ca_turns) > 0 else float("nan")
        p50 = safe_quantile(ca_turns["routing_latency_ms"], 0.50) if len(ca_turns) > 0 else float("nan")
        rule_pct  = float((ca_turns["routing_path"] == "rule").mean()  * 100) if len(ca_turns) > 0 else float("nan")
        judge_pct = float((ca_turns["routing_path"] == "judge").mean() * 100) if len(ca_turns) > 0 else float("nan")

        war_ca_nw = war_pct(ca_runs, "near_wall")
        war_st_nw = war_pct(st_runs, "near_wall")
        war_ca_mid = war_pct(ca_runs, "mid")
        war_st_mid = war_pct(st_runs, "mid")

        results[thresh] = {
            "n_ca_runs":  int(len(ca_runs)),
            "n_st_runs":  int(len(st_runs)),
            "n_ca_turns": int(len(ca_turns)),
            "p95_latency_ms":  p95,
            "p50_latency_ms":  p50,
            "rule_pct":  rule_pct,
            "judge_pct": judge_pct,
            "war_ca_near_wall":    war_ca_nw,
            "war_st_near_wall":    war_st_nw,
            "war_gap_near_wall_pp": war_ca_nw - war_st_nw,
            "war_ca_mid":    war_ca_mid,
            "war_st_mid":    war_st_mid,
            "war_gap_mid_pp": war_ca_mid - war_st_mid,
            "domain_scope": cfg["domains"] if cfg["domains"] is not None else "all",
        }

    return results


# ---------------------------------------------------------------------------
# Thread 2: Pre-emption Ablation
# ---------------------------------------------------------------------------

def compute_thread2_preemption_ablation() -> dict:
    """
    From run_full_v2, classify every context_aware run at near_wall and over_wall
    into: pre_empted (0 turns) / mid_task_escalated / stayed_local.
    Compute WAR per class and the "adjusted gap" excluding pre-empted runs.
    """
    runs  = load_runs(FULL_V2_PATH)
    turns = load_turns(FULL_V2_PATH, valid_run_ids=set(runs["run_id"]))

    # Which CA runs ever sent a turn to cloud?
    ca_cloud_run_ids: set = set(
        turns[
            (turns["router_name"] == "context_aware") &
            (turns["routing_target"] == "cloud")
        ]["run_id"]
    )

    ca_runs = runs[runs["router_name"] == "context_aware"].copy()
    st_runs = runs[runs["router_name"] == "static_semantic"]

    ca_runs["run_class"] = ca_runs.apply(
        lambda r: classify_ca_run(int(r["total_turns"]), r["run_id"], ca_cloud_run_ids),
        axis=1,
    )

    results: dict[str, dict] = {}
    for depth in ("near_wall", "over_wall"):
        ca_d = ca_runs[ca_runs["context_depth"] == depth]
        st_d = st_runs[st_runs["context_depth"] == depth]

        class_counts = ca_d["run_class"].value_counts().to_dict()

        war_by_class: dict[str, float] = {}
        for cls in ("pre_empted", "mid_task_escalated", "stayed_local"):
            sub = ca_d[ca_d["run_class"] == cls]["no_wall"]
            war_by_class[cls] = float(sub.mean() * 100) if len(sub) > 0 else float("nan")

        ca_war = float(ca_d["no_wall"].mean() * 100) if len(ca_d) > 0 else float("nan")
        st_war = float(st_d["no_wall"].mean() * 100) if len(st_d) > 0 else float("nan")

        non_pre = ca_d[ca_d["run_class"] != "pre_empted"]["no_wall"]
        adj_war = float(non_pre.mean() * 100) if len(non_pre) > 0 else float("nan")

        results[depth] = {
            "n_total":              int(len(ca_d)),
            "n_pre_empted":         int(class_counts.get("pre_empted", 0)),
            "n_mid_task_escalated": int(class_counts.get("mid_task_escalated", 0)),
            "n_stayed_local":       int(class_counts.get("stayed_local", 0)),
            "war_by_class":         war_by_class,
            "ca_war_pct":   ca_war,
            "st_war_pct":   st_war,
            "war_gap_pp":   ca_war - st_war,
            "adj_war_pct":  adj_war,
            "adj_gap_pp":   (adj_war - st_war) if not (pd.isna(adj_war) or pd.isna(st_war)) else float("nan"),
        }
    return results


# ---------------------------------------------------------------------------
# Thread 3: Escalation Quality
# ---------------------------------------------------------------------------

def compute_thread3_escalation_quality() -> dict:
    """
    From run_overnight_shadow (the only dataset with shadow fields populated):
    - Escalation Precision: fraction of cloud escalations where local would have wall-hit
    - Escalation Recall: fraction of wall-risk turns that were caught
    - DDR: Decision Divergence Rate vs. shadow static router
    - Lead Time Headroom: 1 - context_occupancy_ratio at escalation moment
    """
    runs  = load_runs(SHADOW_RUN_PATH)
    turns = load_turns(SHADOW_RUN_PATH, valid_run_ids=set(runs["run_id"]))

    ca_turns = turns[turns["router_name"] == "context_aware"].copy()
    ca_turns["routing_path"] = ca_turns["routing_reason"].apply(classify_routing_path)

    ca_cloud = ca_turns[ca_turns["routing_target"] == "cloud"].copy()
    n_cloud = int(len(ca_cloud))

    # Precision: of escalated turns, how many were actually necessary?
    cloud_shadow_hits = ca_cloud["shadow_local_wall_hit"].dropna()
    precision = float(cloud_shadow_hits.mean()) if len(cloud_shadow_hits) > 0 else float("nan")

    # Recall: of all would-be wall-hit turns, how many were escalated to cloud?
    wall_risk = ca_turns[ca_turns["shadow_local_wall_hit"] == True]
    n_wall_risk = int(len(wall_risk))
    recall = float((wall_risk["routing_target"] == "cloud").mean()) if n_wall_risk > 0 else float("nan")

    # DDR: CA differs from shadow static decision
    with_shadow = ca_turns[ca_turns["shadow_static_decision_target"].notna()]
    ddr = float(
        (with_shadow["routing_target"] != with_shadow["shadow_static_decision_target"]).mean()
    ) if len(with_shadow) > 0 else float("nan")

    # Lead time: positive = escalated before wall; negative = after wall
    occ_at_esc = ca_cloud["context_occupancy_ratio"].dropna()
    lead_time   = 1.0 - occ_at_esc

    # Per routing-path breakdown for cloud escalations
    path_breakdown: dict[str, dict] = {}
    for path_type in ("rule", "failure_escalation", "judge"):
        sub = ca_cloud[ca_cloud["routing_path"] == path_type]
        if len(sub) > 0:
            sub_hits = sub["shadow_local_wall_hit"].dropna()
            path_breakdown[path_type] = {
                "n":               int(len(sub)),
                "mean_occupancy":  float(sub["context_occupancy_ratio"].mean()),
                "pct_before_wall": float((sub["context_occupancy_ratio"] < 1.0).mean() * 100),
                "precision":       float(sub_hits.mean()) if len(sub_hits) > 0 else float("nan"),
            }
        else:
            path_breakdown[path_type] = {"n": 0}

    return {
        "n_ca_cloud_escalations":  n_cloud,
        "n_wall_risk_turns":       n_wall_risk,
        "escalation_precision":    precision,
        "escalation_recall":       recall,
        "ddr":                     ddr,
        "lead_time_mean":          float(lead_time.mean())   if len(lead_time) > 0 else float("nan"),
        "lead_time_median":        float(lead_time.median()) if len(lead_time) > 0 else float("nan"),
        "pct_positive_headroom":   float((lead_time >= 0).mean() * 100) if len(lead_time) > 0 else float("nan"),
        "occupancy_at_escalation": occ_at_esc.tolist(),   # raw list for histogram in plots
        "lead_time_values":        lead_time.tolist(),    # raw list for histogram in plots
        "path_breakdown":          path_breakdown,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _f(v: float, spec: str = ".1f") -> str:
    return "N/A" if pd.isna(v) else f"{v:{spec}}"


def print_thread1_table(t1: dict) -> None:
    print("\n" + "=" * 95)
    print("THREAD 1: Threshold-Latency-WAR Tradeoff")
    print("=" * 95)
    print(f"{'Thresh':>7}  {'CA runs':>8}  {'p50 ms':>9}  {'p95 ms':>11}  {'Rule%':>6}  {'Judge%':>7}  {'WAR gap nw':>11}  Notes")
    print("-" * 95)
    for thresh in ("0.50", "0.65", "0.80", "0.85"):
        r    = t1[thresh]
        flag = "*" if thresh == "0.85" else " "
        scope = "2 domains" if isinstance(r["domain_scope"], list) else "all"
        print(
            f"{thresh+flag:>7}  {r['n_ca_runs']:>8}  {_f(r['p50_latency_ms'],'.2f'):>9}  "
            f"{_f(r['p95_latency_ms'],'.0f'):>11}  {_f(r['rule_pct']):>6}  "
            f"{_f(r['judge_pct']):>7}  {_f(r['war_gap_near_wall_pp'],'+.1f'):>10}pp  {scope}"
        )
    print("  * 0.85 uses different domain scope (data_lookup + multi_step_calculation)")


def print_thread2_table(t2: dict) -> None:
    print("\n" + "=" * 95)
    print("THREAD 2: Pre-emption Ablation (run_full_v2)")
    print("=" * 95)
    print(f"{'Depth':>12}  {'N':>4}  {'Pre-emp':>7}  {'Mid-task':>8}  {'Local':>5}  "
          f"{'CA WAR':>8}  {'St WAR':>7}  {'Gap':>7}  {'Adj gap':>8}")
    print("-" * 95)
    for depth in ("near_wall", "over_wall"):
        r = t2[depth]
        print(
            f"{depth:>12}  {r['n_total']:>4}  {r['n_pre_empted']:>7}  "
            f"{r['n_mid_task_escalated']:>8}  {r['n_stayed_local']:>5}  "
            f"{_f(r['ca_war_pct']):>7}%  {_f(r['st_war_pct']):>6}%  "
            f"{_f(r['war_gap_pp'],'+.1f'):>6}pp  {_f(r['adj_gap_pp'],'+.1f'):>7}pp"
        )


def print_thread3_table(t3: dict) -> None:
    print("\n" + "=" * 95)
    print("THREAD 3: Escalation Quality (run_overnight_shadow)")
    print("=" * 95)

    def _pct_frac(v: float) -> str:
        return "N/A" if pd.isna(v) else f"{v * 100:.1f}%"

    print(f"  Cloud escalations:          {t3['n_ca_cloud_escalations']}")
    print(f"  Wall-risk turns (shadow):   {t3['n_wall_risk_turns']}")
    print(f"  Escalation Precision:       {_pct_frac(t3['escalation_precision'])}")
    print(f"  Escalation Recall:          {_pct_frac(t3['escalation_recall'])}")
    print(f"  Decision Divergence Rate:   {_pct_frac(t3['ddr'])}")
    print(f"  Lead time headroom mean:    {_f(t3['lead_time_mean'], '.3f')}")
    print(f"  Lead time headroom median:  {_f(t3['lead_time_median'], '.3f')}")
    print(f"  Escalations before wall:    {_f(t3['pct_positive_headroom'])}%")
    if t3["path_breakdown"]:
        print("  Per routing-path breakdown (cloud escalations):")
        for ptype, stats in t3["path_breakdown"].items():
            if stats.get("n", 0) > 0:
                prec = stats.get("precision", float("nan"))
                prec_str = f"{prec * 100:.1f}%" if not pd.isna(prec) else "N/A"
                print(
                    f"    {ptype:22s}: n={stats['n']:>3}, "
                    f"mean_occ={stats.get('mean_occupancy', 0):.2f}x, "
                    f"before_wall={stats.get('pct_before_wall', 0):.1f}%, "
                    f"prec={prec_str}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing Thread 1: Threshold-Latency-WAR Tradeoff...")
    t1 = compute_thread1_threshold_sweep()
    print_thread1_table(t1)

    print("\nComputing Thread 2: Pre-emption Ablation...")
    t2 = compute_thread2_preemption_ablation()
    print_thread2_table(t2)

    print("\nComputing Thread 3: Escalation Quality...")
    t3 = compute_thread3_escalation_quality()
    print_thread3_table(t3)

    # Save JSON (omit raw value lists to keep file small)
    metrics_for_json = {
        "threshold_sweep":     t1,
        "preemption_ablation": t2,
        "escalation_quality":  {
            k: v for k, v in t3.items()
            if k not in ("occupancy_at_escalation", "lead_time_values")
        },
    }
    out_path = PLOTS_DIR / "article_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            metrics_for_json, f, indent=2,
            default=lambda x: None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x,
        )
    print(f"\nSaved metrics → {out_path}")


if __name__ == "__main__":
    main()
