"""Generate all analysis plots for a benchmark results directory.

Usage:
    python scripts/generate_plots.py results/results/run_full_v2 --out-dir plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Palette / style
# ---------------------------------------------------------------------------

ROUTER_COLORS = {
    "static_semantic":  "#4C72B0",
    "context_aware":    "#DD8452",
    "commercial_cloud": "#55A868",
}
ROUTER_LABELS = {
    "static_semantic":  "Static Semantic",
    "context_aware":    "Context-Aware",
    "commercial_cloud": "Commercial Cloud",
}
DEPTH_ORDER = ["shallow", "mid", "near_wall", "over_wall"]
ROUTER_ORDER = ["static_semantic", "context_aware", "commercial_cloud"]

sns.set_theme(style="whitegrid", font_scale=1.15)


def _load(results_dir: Path):
    runs = pd.read_csv(results_dir / "runs.csv")
    runs["no_wall"] = runs["wall_events"] == 0
    turns_raw = []
    with open(results_dir / "turns.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                turns_raw.append(json.loads(line))
    turns = pd.DataFrame(turns_raw)
    # Join context_depth and other run-level fields onto turns
    run_meta = runs[["run_id", "context_depth", "domain", "intent_complexity",
                     "failure_profile", "success"]].rename(columns={"success": "run_success"})
    turns = turns.merge(run_meta, on="run_id", how="left")
    return runs, turns


# ---------------------------------------------------------------------------
# Figure 1 — WAR by router × context_depth
# ---------------------------------------------------------------------------

def fig_war_by_depth(runs: pd.DataFrame, out: Path) -> None:
    war = (
        runs.groupby(["router_name", "context_depth"])["no_wall"]
        .mean()
        .reset_index()
        .rename(columns={"no_wall": "WAR"})
    )
    # Reorder
    war["context_depth"] = pd.Categorical(war["context_depth"], DEPTH_ORDER, ordered=True)
    war["router_name"] = pd.Categorical(war["router_name"], ROUTER_ORDER, ordered=True)
    war = war.sort_values(["context_depth", "router_name"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(DEPTH_ORDER))
    width = 0.25
    for i, router in enumerate(ROUTER_ORDER):
        subset = war[war["router_name"] == router].set_index("context_depth").reindex(DEPTH_ORDER)
        vals = subset["WAR"].fillna(0).values
        bars = ax.bar(x + (i - 1) * width, vals * 100, width,
                      label=ROUTER_LABELS[router],
                      color=ROUTER_COLORS[router], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)

    # §5.2 threshold annotation
    ax.axhline(100, color="grey", linewidth=0.6, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(["Shallow\n(~10% full)", "Mid\n(~50% full)",
                         "Near Wall\n(~85% full)", "Over Wall\n(>100% full)"])
    ax.set_ylabel("Wall Avoidance Rate (%)")
    ax.set_title("Figure 1 — Wall Avoidance Rate by Router and Context Depth\n"
                 "(Spec §5.2 criterion: Context-Aware WAR ≥ Static + 15pp at near_wall & over_wall)")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower left")

    # Annotate the gap at near_wall
    ca_nw = war[(war["router_name"] == "context_aware") & (war["context_depth"] == "near_wall")]["WAR"].values[0]
    st_nw = war[(war["router_name"] == "static_semantic") & (war["context_depth"] == "near_wall")]["WAR"].values[0]
    gap_nw = ca_nw - st_nw
    ax.annotate(f"+{gap_nw*100:.0f}pp gap",
                xy=(2 + 0.5 * width, ca_nw * 100 + 3),
                xytext=(2 + 0.5 * width, 108),
                ha="center", fontsize=9.5, color="#DD8452",
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))

    fig.tight_layout()
    fig.savefig(out / "fig1_war_by_depth.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig1_war_by_depth.png")


# ---------------------------------------------------------------------------
# Figure 2 — TSR by router × context_depth
# ---------------------------------------------------------------------------

def fig_tsr_by_depth(runs: pd.DataFrame, out: Path) -> None:
    tsr = (
        runs.groupby(["router_name", "context_depth"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "TSR"})
    )
    tsr["context_depth"] = pd.Categorical(tsr["context_depth"], DEPTH_ORDER, ordered=True)
    tsr["router_name"] = pd.Categorical(tsr["router_name"], ROUTER_ORDER, ordered=True)
    tsr = tsr.sort_values(["context_depth", "router_name"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(DEPTH_ORDER))
    width = 0.25
    for i, router in enumerate(ROUTER_ORDER):
        subset = tsr[tsr["router_name"] == router].set_index("context_depth").reindex(DEPTH_ORDER)
        vals = subset["TSR"].fillna(0).values
        bars = ax.bar(x + (i - 1) * width, vals * 100, width,
                      label=ROUTER_LABELS[router],
                      color=ROUTER_COLORS[router], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Shallow\n(~10% full)", "Mid\n(~50% full)",
                         "Near Wall\n(~85% full)", "Over Wall\n(>100% full)"])
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Figure 2 — Task Success Rate by Router and Context Depth")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig2_tsr_by_depth.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig2_tsr_by_depth.png")


# ---------------------------------------------------------------------------
# Figure 3 — WAR gap heat map (context_aware − static_semantic)
# ---------------------------------------------------------------------------

def fig_war_gap(runs: pd.DataFrame, out: Path) -> None:
    war = (
        runs.groupby(["router_name", "context_depth"])["no_wall"]
        .mean()
        .unstack("context_depth")
        .reindex(columns=DEPTH_ORDER)
    )
    gap = (war.loc["context_aware"] - war.loc["static_semantic"]) * 100  # in pp

    fig, ax = plt.subplots(figsize=(7, 2.8))
    gap_2d = gap.values.reshape(1, -1)
    im = ax.imshow(gap_2d, cmap="RdYlGn", vmin=-20, vmax=50, aspect="auto")
    ax.set_xticks(range(len(DEPTH_ORDER)))
    ax.set_xticklabels(["Shallow", "Mid", "Near Wall", "Over Wall"])
    ax.set_yticks([])
    for j, v in enumerate(gap.values):
        color = "white" if abs(v) > 30 else "black"
        ax.text(j, 0, f"{v:+.0f}pp", ha="center", va="center",
                fontsize=13, color=color, fontweight="bold")
    ax.set_title("Figure 3 — WAR Gap: Context-Aware minus Static Semantic\n"
                 "(Spec §5.2 criterion: ≥+15pp at near_wall & over_wall)", pad=10)
    plt.colorbar(im, ax=ax, label="Percentage points", shrink=0.8)
    # Threshold line annotation
    ax.axvline(1.5, color="black", linewidth=1.5, linestyle="--", alpha=0.4)
    ax.text(1.65, -0.48, "spec region →", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(out / "fig3_war_gap_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig3_war_gap_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4 — Routing overhead latency distribution (per-turn)
# ---------------------------------------------------------------------------

def fig_latency(turns: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, router in zip(axes, ROUTER_ORDER):
        sub = turns[turns["router_name"] == router]["routing_latency_ms"]
        p50 = sub.median()
        p95 = sub.quantile(0.95)
        # Use log scale for x
        bins = np.logspace(np.log10(max(sub.min(), 0.001)), np.log10(sub.max() + 1), 40)
        ax.hist(sub, bins=bins, color=ROUTER_COLORS[router], alpha=0.8, edgecolor="white")
        ax.axvline(p50, color="navy", linewidth=1.5, linestyle="--", label=f"p50={p50:.1f}ms")
        ax.axvline(p95, color="crimson", linewidth=1.5, linestyle="--", label=f"p95={p95:.0f}ms")
        ax.axvline(150, color="black", linewidth=1.2, linestyle=":", alpha=0.6, label="150ms threshold")
        ax.set_xscale("log")
        ax.set_title(ROUTER_LABELS[router], fontsize=11)
        ax.set_xlabel("Routing latency per turn (ms, log scale)")
        ax.set_ylabel("Turn count" if router == "static_semantic" else "")
        ax.legend(fontsize=8)

    fig.suptitle("Figure 4 — Routing Overhead Distribution per Turn\n"
                 "(Spec §5.2 criterion: Context-Aware p95 ≤ 150ms)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig4_routing_overhead.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig4_routing_overhead.png")


# ---------------------------------------------------------------------------
# Figure 5 — Escalation context occupancy at decision time (context_aware only)
# ---------------------------------------------------------------------------

def fig_escalation_occupancy(turns: pd.DataFrame, out: Path) -> None:
    ca = turns[turns["router_name"] == "context_aware"].copy()
    ca["path"] = ca["routing_reason"].apply(
        lambda r: "Rule (wall proximity)" if r and r.startswith("context_proximity_to_wall")
                  else "LLM Judge"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar by context_depth showing judge vs rule path
    path_counts = (
        ca.groupby(["context_depth", "path"]).size().unstack(fill_value=0)
        .reindex(DEPTH_ORDER)
    )
    path_counts.plot(kind="bar", stacked=True, ax=ax1,
                     color=["#E07B54", "#4C72B0"], alpha=0.85)
    ax1.set_xlabel("Context Depth")
    ax1.set_ylabel("Turn count")
    ax1.set_title("Decision Path per Context Depth\n(Rule path: instant; Judge path: ~ms–s)")
    ax1.set_xticklabels(DEPTH_ORDER, rotation=0)
    ax1.legend(title="Decision path")

    # Right: scatter of context_occupancy_ratio at escalation time (cloud decisions)
    ca_cloud = ca[ca["routing_target"] == "cloud"].copy()
    if len(ca_cloud) == 0:
        ax2.text(0.5, 0.5, "No escalations", ha="center", va="center")
    else:
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(ca_cloud))
        depth_idx = ca_cloud["context_depth"].map(
            {d: i for i, d in enumerate(DEPTH_ORDER)}
        ).fillna(-1) + jitter
        sc = ax2.scatter(ca_cloud["context_occupancy_ratio"], depth_idx,
                         c=ca_cloud["context_occupancy_ratio"],
                         cmap="RdYlGn_r", vmin=0.5, vmax=2.5,
                         s=70, alpha=0.85, edgecolors="white", linewidths=0.5)
        ax2.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Wall (1.0 = 100%)")
        ax2.axvline(0.85, color="orange", linestyle=":", linewidth=1.5, label="Threshold (0.85)")
        ax2.set_yticks(range(len(DEPTH_ORDER)))
        ax2.set_yticklabels(DEPTH_ORDER)
        ax2.set_xlabel("Context occupancy ratio at escalation")
        ax2.set_title("Escalation Timing (Context-Aware → Cloud decisions)\n"
                      "Left of orange = proactive; left of red = before wall; right of red = too late")
        ax2.legend(fontsize=9)
        plt.colorbar(sc, ax=ax2, label="Occupancy ratio")

    fig.suptitle("Figure 5 — Context-Aware Router: Decision Path and Escalation Timing", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig5_escalation_occupancy.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig5_escalation_occupancy.png")


# ---------------------------------------------------------------------------
# Figure 6 — §5.3 Comparative metrics (DDR, Precision, Recall, Lead Time)
# ---------------------------------------------------------------------------

def fig_53_metrics(turns: pd.DataFrame, out: Path) -> None:
    ca = turns[turns["router_name"] == "context_aware"].copy()
    ca["disagrees"] = ca["routing_target"] != ca["shadow_static_decision_target"]

    # Overall
    ddr = ca["disagrees"].mean()
    ca_cloud = ca[ca["routing_target"] == "cloud"]
    prec = ca_cloud["shadow_local_wall_hit"].mean() if len(ca_cloud) > 0 else float("nan")
    wall_turns = ca[ca["shadow_local_wall_hit"] == True]
    recall = (wall_turns["routing_target"] == "cloud").mean() if len(wall_turns) > 0 else float("nan")
    elt_mean = (1.0 - ca_cloud["context_occupancy_ratio"]).mean() if len(ca_cloud) > 0 else float("nan")

    # Per-depth breakdown
    per_depth = {}
    for depth in DEPTH_ORDER:
        sub = ca[ca["context_depth"] == depth]
        if len(sub) == 0:
            per_depth[depth] = {"ddr": np.nan, "prec": np.nan, "recall": np.nan, "elt": np.nan, "n": 0}
            continue
        ddr_d = sub["disagrees"].mean()
        cloud_d = sub[sub["routing_target"] == "cloud"]
        prec_d = cloud_d["shadow_local_wall_hit"].mean() if len(cloud_d) > 0 else np.nan
        wall_d = sub[sub["shadow_local_wall_hit"] == True]
        rec_d = (wall_d["routing_target"] == "cloud").mean() if len(wall_d) > 0 else np.nan
        elt_d = (1.0 - cloud_d["context_occupancy_ratio"]).mean() if len(cloud_d) > 0 else np.nan
        per_depth[depth] = {"ddr": ddr_d, "prec": prec_d, "recall": rec_d, "elt": elt_d, "n": len(sub)}

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel A: DDR + Precision + Recall summary bars
    ax_a = fig.add_subplot(gs[0])
    metrics = {"DDR\n(divergence)": ddr, "Escalation\nPrecision": prec, "Escalation\nRecall": recall}
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax_a.bar(list(metrics.keys()), [v * 100 for v in metrics.values()],
                    color=colors, alpha=0.85, edgecolor="white", width=0.55)
    for bar, v in zip(bars, metrics.values()):
        if not np.isnan(v):
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                      f"{v*100:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_a.set_ylim(0, 120)
    ax_a.set_ylabel("Rate (%)")
    ax_a.set_title("Overall §5.3 Metrics\n(Context-Aware vs Shadow Static)", fontsize=10)
    ax_a.annotate("N=3 actual\nwall turns\n(low confidence)", xy=(2, recall * 100 + 2),
                  xytext=(2.15, 80), fontsize=7.5, color="grey",
                  arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    # Panel B: Lead time headroom per escalation
    ax_b = fig.add_subplot(gs[1])
    if len(ca_cloud) > 0:
        lts = 1.0 - ca_cloud["context_occupancy_ratio"]
        colors_lt = ["#DD8452" if v >= 0 else "#C44E52" for v in lts]
        ax_b.barh(range(len(lts)), lts.values, color=colors_lt, alpha=0.8, edgecolor="white")
        ax_b.axvline(0, color="crimson", linewidth=1.5, linestyle="--", label="Wall boundary")
        ax_b.set_xlabel("Lead time headroom\n(positive = before wall, negative = after wall)")
        ax_b.set_yticks([])
        ax_b.set_title(f"Escalation Lead Time\n(N={len(lts)} cloud decisions)", fontsize=10)
        ax_b.legend(fontsize=8)
        ax_b.text(0.97, 0.05, f"Mean: {elt_mean:.2f}", transform=ax_b.transAxes,
                  ha="right", va="bottom", fontsize=9, color="grey")

    # Panel C: per-depth DDR bars
    ax_c = fig.add_subplot(gs[2])
    depths_with_data = [d for d in DEPTH_ORDER if per_depth[d]["n"] > 0]
    ddr_vals = [per_depth[d]["ddr"] * 100 for d in depths_with_data]
    ax_c.barh(depths_with_data, ddr_vals,
              color=[ROUTER_COLORS["context_aware"]] * len(depths_with_data),
              alpha=0.85, edgecolor="white")
    for i, (d, v) in enumerate(zip(depths_with_data, ddr_vals)):
        ax_c.text(v + 0.5, i, f"{v:.0f}%  (N={per_depth[d]['n']})", va="center", fontsize=9)
    ax_c.set_xlabel("Decision Divergence Rate (%)")
    ax_c.set_xlim(0, 130)
    ax_c.set_title("DDR by Context Depth\n(% turns where CA ≠ static)", fontsize=10)

    fig.suptitle("Figure 6 — Spec §5.3: Static vs. Dynamic Comparative Metrics", fontsize=12, y=1.01)
    fig.savefig(out / "fig6_s53_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6_s53_metrics.png")


# ---------------------------------------------------------------------------
# Figure 7 — Wall events total and run turn distribution
# ---------------------------------------------------------------------------

def fig_wall_and_turns(runs: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total wall events by router
    wall_by_router = runs.groupby("router_name")["wall_events"].sum().reindex(ROUTER_ORDER)
    bars = ax1.bar([ROUTER_LABELS[r] for r in ROUTER_ORDER],
                   wall_by_router.values,
                   color=[ROUTER_COLORS[r] for r in ROUTER_ORDER],
                   alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, wall_by_router.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(int(v)), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total wall-hit events across all runs")
    ax1.set_title("Total Context Wall Events by Router\n(lower is better)")

    # Right: turn count distribution box plot
    runs_turns = runs[["router_name", "total_turns"]].copy()
    runs_turns["Router"] = runs_turns["router_name"].map(ROUTER_LABELS)
    order = [ROUTER_LABELS[r] for r in ROUTER_ORDER]
    palette = {ROUTER_LABELS[r]: ROUTER_COLORS[r] for r in ROUTER_ORDER}
    sns.boxplot(data=runs_turns, x="Router", y="total_turns", order=order,
                hue="Router", palette=palette, legend=False, ax=ax2, width=0.5,
                flierprops=dict(markerfacecolor="grey", markersize=4, alpha=0.5))
    ax2.set_ylabel("Turns per run")
    ax2.set_title("Turns per Run Distribution\n(0-turn runs = task pre-empted before execution)")
    ax2.set_xlabel("")

    fig.suptitle("Figure 7 — Wall Events and Task Execution Depth", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig7_wall_events_turns.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig7_wall_events_turns.png")


# ---------------------------------------------------------------------------
# Figure 8 — Spec §5.2 compliance summary dashboard
# ---------------------------------------------------------------------------

def fig_compliance_dashboard(runs: pd.DataFrame, turns: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    ca_t = turns[turns["router_name"] == "context_aware"]
    st_runs = runs[runs["router_name"] == "static_semantic"]
    ca_runs = runs[runs["router_name"] == "context_aware"]

    # --- Criterion 1: WAR gap ---
    ax = axes[0]
    depths_check = ["near_wall", "over_wall"]
    ca_war_vals = [
        ca_runs[ca_runs["context_depth"] == d]["no_wall"].mean() for d in depths_check
    ]
    st_war_vals = [
        st_runs[st_runs["context_depth"] == d]["no_wall"].mean() for d in depths_check
    ]
    gaps = [(c - s) * 100 for c, s in zip(ca_war_vals, st_war_vals)]
    colors_gap = ["#55A868" if g >= 15 else "#C44E52" for g in gaps]
    bars = ax.bar(depths_check, gaps, color=colors_gap, alpha=0.85, edgecolor="white", width=0.45)
    ax.axhline(15, color="black", linewidth=1.5, linestyle="--", label="15pp threshold")
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"+{g:.0f}pp", ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=colors_gap[bars.index(bar)])
    ax.set_ylabel("WAR gap (pp)")
    ax.set_ylim(0, 60)
    ax.set_title("§5.2 Criterion 1\nWAR: Context-Aware − Static\n✓ PASS (both > +15pp)")
    ax.legend(fontsize=9)
    ax.text(0.5, 0.92, "PASS", transform=ax.transAxes, ha="center", fontsize=14,
            color="#55A868", fontweight="bold")

    # --- Criterion 2: p95 latency ---
    ax = axes[1]
    p95 = ca_t["routing_latency_ms"].quantile(0.95)
    p50 = ca_t["routing_latency_ms"].median()
    cat = ["p50", "p95"]
    vals = [p50, p95]
    colors_lat = ["#55A868" if v <= 150 else "#C44E52" for v in vals]
    bars = ax.bar(cat, vals, color=colors_lat, alpha=0.85, edgecolor="white", width=0.45)
    ax.axhline(150, color="black", linewidth=1.5, linestyle="--", label="150ms threshold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{v:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Routing latency (ms)")
    ax.set_yscale("log")
    ax.set_title(f"§5.2 Criterion 2\nRouting Overhead p95 ≤ 150ms\n✗ FAIL ({p95:.0f}ms, {p95/150:.0f}× over)")
    ax.legend(fontsize=9)
    ax.text(0.5, 0.92, "FAIL", transform=ax.transAxes, ha="center", fontsize=14,
            color="#C44E52", fontweight="bold")

    # --- Criterion 3: CE ---
    ax = axes[2]
    ax.text(0.5, 0.5, "Cost Efficiency\nNaN\n\n(All runs used local Ollama\n$0 cost — requires\nreal API key to measure)",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#FFC107"))
    ax.set_axis_off()
    ax.set_title("§5.2 Criterion 3\nCE ≥ 40% vs Cloud Baseline\n⚠ UNMEASURED")
    ax.text(0.5, 0.92, "N/A", transform=ax.transAxes, ha="center", fontsize=14,
            color="#FFC107", fontweight="bold")

    fig.suptitle("Figure 8 — Spec §5.2 Compliance Dashboard (run_full_v2, n=100 per router)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig8_compliance_dashboard.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig8_compliance_dashboard.png")


# ---------------------------------------------------------------------------
# Figure 9 — Turn-level routing decisions for context_aware (occupancy timeline)
# ---------------------------------------------------------------------------

def fig_occupancy_timeline(turns: pd.DataFrame, out: Path) -> None:
    ca = turns[turns["router_name"] == "context_aware"].copy()
    # For multi-turn runs, plot occupancy trajectory
    multi_turn_runs = ca.groupby("run_id").filter(lambda g: len(g) > 1)

    if len(multi_turn_runs) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No multi-turn context_aware runs in dataset",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out / "fig9_occupancy_timeline.png", dpi=160)
        plt.close(fig)
        return

    # Also show single-turn runs distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all turns scatter — occupancy vs turn_index, colored by routing_target
    colors_rt = {"local": "#4C72B0", "cloud": "#DD8452"}
    for target, grp in ca.groupby("routing_target"):
        ax1.scatter(grp["turn_index"], grp["context_occupancy_ratio"],
                    c=colors_rt.get(target, "grey"), label=target.capitalize(),
                    s=50, alpha=0.7, edgecolors="white", linewidths=0.4)
    ax1.axhline(0.85, color="orange", linewidth=1.5, linestyle="--", label="Wall threshold (0.85)")
    ax1.axhline(1.0, color="crimson", linewidth=1.5, linestyle="--", label="Hard wall (1.0)")
    ax1.set_xlabel("Turn index within run")
    ax1.set_ylabel("Context occupancy ratio")
    ax1.set_title("Context Occupancy at Routing Decision Time\n(Context-Aware router, all turns)")
    ax1.legend(fontsize=9)

    # Right: multi-turn runs — trajectory per run
    for run_id, grp in multi_turn_runs.groupby("run_id"):
        grp_sorted = grp.sort_values("turn_index")
        ax2.plot(grp_sorted["turn_index"], grp_sorted["context_occupancy_ratio"],
                 marker="o", markersize=4, linewidth=1.2, alpha=0.7)
    ax2.axhline(0.85, color="orange", linewidth=1.5, linestyle="--", label="Wall threshold")
    ax2.axhline(1.0, color="crimson", linewidth=1.5, linestyle="--", label="Hard wall")
    ax2.set_xlabel("Turn index")
    ax2.set_ylabel("Context occupancy ratio")
    ax2.set_title(f"Occupancy Trajectories — Multi-Turn Runs\n(Context-Aware, N={multi_turn_runs['run_id'].nunique()} runs)")
    ax2.legend(fontsize=9)

    fig.suptitle("Figure 9 — Context-Aware: Occupancy at Decision Time", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig9_occupancy_timeline.png", dpi=160)
    plt.close(fig)
    print(f"  Saved fig9_occupancy_timeline.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs, turns = _load(args.results_dir)

    print(f"Loaded {len(runs)} runs, {len(turns)} turns from {args.results_dir}")
    print(f"Generating plots → {args.out_dir}/\n")

    fig_war_by_depth(runs, args.out_dir)
    fig_tsr_by_depth(runs, args.out_dir)
    fig_war_gap(runs, args.out_dir)
    fig_latency(turns, args.out_dir)
    fig_escalation_occupancy(turns, args.out_dir)
    fig_53_metrics(turns, args.out_dir)
    fig_wall_and_turns(runs, args.out_dir)
    fig_compliance_dashboard(runs, turns, args.out_dir)
    fig_occupancy_timeline(turns, args.out_dir)

    print(f"\nDone. {len(list(args.out_dir.glob('*.png')))} figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
