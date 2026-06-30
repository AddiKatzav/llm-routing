"""Spec §5.2 compliance report for a completed benchmark run.

Reads a results directory (kpi_summary.json + runs.csv + turns.jsonl) and
prints a table comparing each spec acceptance criterion against the actual
values from the run. Designed to be run after any benchmark sweep to quickly
surface pass/fail status without manually cross-referencing two documents.

Usage:
    python scripts/analyze_results.py results/run_overnight_shadow
    python scripts/analyze_results.py results/run_overnight_shadow --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_kpi(results_dir: Path) -> dict:
    path = results_dir / "kpi_summary.json"
    if not path.exists():
        sys.exit(f"kpi_summary.json not found in {results_dir}. Run the benchmark first.")
    with path.open() as f:
        raw = json.load(f)
    # json.load turns NaN-as-literal into None for non-compliant JSON writers;
    # handle both the numeric NaN (written by our persistence layer) and None.
    return raw


def load_runs(results_dir: Path) -> list[dict]:
    path = results_dir / "runs.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_turns(results_dir: Path) -> list[dict]:
    path = results_dir / "turns.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: float | None, pct: bool = False, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "NaN"
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def _pass_fail(condition: bool | None) -> str:
    if condition is None:
        return "N/A"
    return "PASS" if condition else "FAIL"


def _router(kpi: dict, name: str) -> dict | None:
    return kpi.get(name)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_spec_52(kpi: dict) -> None:
    """Spec §5.2 — Acceptance thresholds."""
    print("=" * 72)
    print("SPEC §5.2 ACCEPTANCE CRITERIA")
    print("=" * 72)

    ca = _router(kpi, "context_aware")
    st = _router(kpi, "static_semantic")
    cc = _router(kpi, "commercial_cloud")

    if ca is None:
        print("  context_aware router not found in kpi_summary.json")
        return

    # --- Criterion 1: WAR +15pp at near_wall and over_wall ---
    print("\n1. Context-Aware WAR ≥ Static WAR + 15pp (near_wall + over_wall)")
    print(f"   {'Depth':<12} {'context_aware WAR':>18} {'static WAR':>12} {'gap':>8} {'status':>8}")
    print(f"   {'-'*12} {'-'*18} {'-'*12} {'-'*8} {'-'*8}")
    all_pass = True
    for depth in ("near_wall", "over_wall"):
        ca_war = (ca.get("breakdown_by_context_depth") or {}).get(depth, {}).get("wall_avoidance_rate")
        st_war = (st.get("breakdown_by_context_depth") or {}).get(depth, {}).get("wall_avoidance_rate") if st else None
        if ca_war is not None and st_war is not None:
            gap = ca_war - st_war
            ok = gap >= 0.15
            if not ok:
                all_pass = False
            print(f"   {depth:<12} {_fmt(ca_war, pct=True):>18} {_fmt(st_war, pct=True):>12} {_fmt(gap, pct=True):>8} {_pass_fail(ok):>8}")
        else:
            print(f"   {depth:<12} {'(no data)':>18}")
            all_pass = False
    print(f"   Overall: {_pass_fail(all_pass)}")

    # --- Criterion 2: routing overhead p95 ≤ 150ms ---
    p95 = ca.get("routing_overhead_p95_ms")
    ok2 = p95 is not None and p95 <= 150.0
    print(f"\n2. Context-Aware routing overhead p95 ≤ 150ms")
    print(f"   Actual p95: {_fmt(p95, decimals=1)}ms  →  {_pass_fail(ok2)}")
    if not ok2 and p95 is not None:
        print(f"   Note: {p95:.0f}ms exceeds 150ms; spec §5.2 classifies this as "
              f"'disqualified as impractical'. This is inherent to a real LLM judge "
              f"call (vs. a dedicated lightweight classifier) on CPU-only hardware.")

    # --- Criterion 3: CE ≥ 40% ---
    ce = ca.get("cost_efficiency")
    ce_nan = ce is None or (isinstance(ce, float) and math.isnan(ce))
    ok3 = not ce_nan and ce >= 0.40
    print(f"\n3. Context-Aware CE ≥ 40% vs all-cloud baseline")
    print(f"   Actual CE: {_fmt(ce, pct=True)}  →  {_pass_fail(None if ce_nan else ok3)}")
    if ce_nan:
        print("   Note: CE is NaN because all runs used local Ollama ($0 cost). "
              "A real API key is required to measure this criterion.")

    # --- Commercial cloud as ceiling reference ---
    print(f"\n4. Commercial Cloud Router TSR (upper-bound reference, no pass/fail threshold)")
    if cc:
        print(f"   commercial_cloud TSR: {_fmt(cc.get('task_success_rate'), pct=True)}")
        print(f"   context_aware TSR:    {_fmt(ca.get('task_success_rate'), pct=True)}")
        print(f"   static_semantic TSR:  {_fmt(st.get('task_success_rate'), pct=True) if st else 'N/A'}")
        ca_tsr = ca.get("task_success_rate") or 0
        cc_tsr = cc.get("task_success_rate") or 0
        print(f"   Gap to ceiling: {_fmt(cc_tsr - ca_tsr, pct=True)}")


def section_per_depth(kpi: dict) -> None:
    """Per-context-depth breakdown for all three routers."""
    print("\n" + "=" * 72)
    print("PER-DEPTH BREAKDOWN (TSR / WAR)")
    print("=" * 72)
    depths = ("shallow", "mid", "near_wall", "over_wall")
    routers = ("static_semantic", "context_aware", "commercial_cloud")
    short = {"static_semantic": "static", "context_aware": "ctx_aware", "commercial_cloud": "cloud"}

    print(f"\n  {'Depth':<12}", end="")
    for r in routers:
        print(f"  {short[r]:^22}", end="")
    print()
    print(f"  {'':<12}", end="")
    for _ in routers:
        print(f"  {'TSR':>10}  {'WAR':>8}  ", end="")
    print()
    print("  " + "-" * 71)

    for depth in depths:
        print(f"  {depth:<12}", end="")
        for r in routers:
            rd = kpi.get(r) or {}
            bd = (rd.get("breakdown_by_context_depth") or {}).get(depth, {})
            tsr = bd.get("task_success_rate")
            war = bd.get("wall_avoidance_rate")
            print(f"  {_fmt(tsr, pct=True):>10}  {_fmt(war, pct=True):>8}  ", end="")
        print()


def section_53_comparative(kpi: dict) -> None:
    """Spec §5.3 — Static vs. dynamic comparative metrics."""
    print("\n" + "=" * 72)
    print("SPEC §5.3 STATIC vs. DYNAMIC COMPARATIVE METRICS (context_aware only)")
    print("=" * 72)

    ca = _router(kpi, "context_aware")
    if not ca:
        return
    comp = ca.get("comparative_static_vs_dynamic")
    if not comp:
        print("  No comparative data found (shadow_config may not have been active).")
        return

    ddr = comp.get("decision_divergence_rate")
    ep = comp.get("escalation_precision")
    er = comp.get("escalation_recall")
    elt = comp.get("escalation_lead_time_headroom_mean")

    print(f"\n  Decision Divergence Rate (DDR):    {_fmt(ddr, pct=True)}")
    print(f"    (how often context_aware disagrees with static router)")
    print(f"\n  Escalation Precision:              {_fmt(ep, pct=True)}")
    print(f"    (of CLOUD escalations, % where LOCAL would have wall-hit)")
    print(f"\n  Escalation Recall:                 {_fmt(er, pct=True)}")
    print(f"    (of LOCAL wall-hit turns, % that were correctly escalated)")
    print(f"\n  Escalation Lead Time (headroom):   {_fmt(elt, decimals=3)}")
    print(f"    (1.0 - occupancy at escalation; positive = escalated before wall)")
    if elt is not None and elt < 0:
        print(f"    Note: negative headroom means escalation happened after the wall")
        print(f"    (typical for over_wall tasks where context starts at 1.1x limit)")

    bd = comp.get("breakdown_by_context_depth") or {}
    if bd:
        print(f"\n  Per-depth breakdown:")
        print(f"  {'Depth':<12} {'DDR':>8} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
        for depth in ("shallow", "mid", "near_wall", "over_wall"):
            d = bd.get(depth)
            if d:
                print(f"  {depth:<12} {_fmt(d.get('decision_divergence_rate'), pct=True):>8} "
                      f"{_fmt(d.get('escalation_precision'), pct=True):>10} "
                      f"{_fmt(d.get('escalation_recall'), pct=True):>8}")
            else:
                print(f"  {depth:<12} {'(no data)':>8}")


def section_routing_overhead(kpi: dict) -> None:
    """Routing overhead comparison across all routers."""
    print("\n" + "=" * 72)
    print("ROUTING OVERHEAD (LATENCY)")
    print("=" * 72)
    print(f"\n  {'Router':<20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'vs 150ms p95':>14}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*14}")
    for name in ("static_semantic", "context_aware", "commercial_cloud"):
        r = kpi.get(name) or {}
        p50 = r.get("routing_overhead_p50_ms")
        p95 = r.get("routing_overhead_p95_ms")
        vs = ""
        if name == "context_aware" and p95 is not None:
            if p95 <= 150:
                vs = "PASS"
            else:
                vs = f"FAIL ({p95/150:.0f}x over)"
        print(f"  {name:<20} {_fmt(p50, decimals=3):>10} {_fmt(p95, decimals=1):>10} {vs:>14}")


def section_run_summary(results_dir: Path, kpi: dict, runs: list[dict]) -> None:
    """High-level run summary."""
    print("\n" + "=" * 72)
    print("RUN SUMMARY")
    print("=" * 72)
    if runs:
        domains = set(r.get("domain", "") for r in runs)
        depths = set(r.get("context_depth", "") for r in runs)
        profiles = set(r.get("failure_profile", "") for r in runs)
        seeds = set(r.get("repeat_seed", "") for r in runs)
        routers_seen = set(r.get("router_name", "") for r in runs)
        total = len(runs)
        successes = sum(1 for r in runs if r.get("success", "").lower() == "true")
        print(f"  Results dir:     {results_dir}")
        print(f"  Total runs:      {total}  ({successes} succeeded, {_fmt(successes/total if total else 0, pct=True)} overall TSR)")
        print(f"  Routers:         {', '.join(sorted(routers_seen))}")
        print(f"  Domains:         {', '.join(sorted(domains))}")
        print(f"  Context depths:  {', '.join(sorted(depths))}")
        print(f"  Failure profiles:{', '.join(sorted(profiles))}")
        print(f"  Repeat seeds:    {len(seeds)} ({', '.join(sorted(seeds))})")

        spec_matrix = 3 * 4 * 4 * 4 * 3  # routers x domains x depths x profiles x intents
        spec_full = spec_matrix * 5
        print(f"\n  Spec full matrix: {spec_matrix} configs x 5 repeats = {spec_full} runs")
        print(f"  Coverage:        {total}/{spec_full} runs ({_fmt(total/spec_full, pct=True)} of full matrix)")
    else:
        print(f"  Results dir: {results_dir}")
        print("  (runs.csv not found or empty)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Path to a benchmark results directory")
    parser.add_argument("--verbose", action="store_true", help="Include per-depth comparative breakdown")
    args = parser.parse_args()

    kpi = load_kpi(args.results_dir)
    runs = load_runs(args.results_dir)

    section_run_summary(args.results_dir, kpi, runs)
    section_spec_52(kpi)
    section_per_depth(kpi)
    section_routing_overhead(kpi)
    section_53_comparative(kpi)
    print()


if __name__ == "__main__":
    main()
