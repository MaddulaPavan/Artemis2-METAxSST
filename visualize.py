"""
visualize.py — Visualization module for PreferenceAggregationEnv benchmark.

Generates 3 diagnostic plots:

  1. group_accuracy_by_scenario.png
     Grouped bar chart: per-group accuracy for greedy vs oracle across scenarios.
     Visually shows which group is harmed in each scenario.

  2. fairness_gap_comparison.png
     Horizontal bar chart: fairness gap for all policy x scenario combinations.
     The primary benchmark result plot.

  3. bias_strength_sweep.png
     Line chart: fairness gap and worst-group accuracy as bias_strength varies.
     Shows the critical tipping point where majority identity flips.

All plots follow a clean, minimal style — publication-appropriate.
"""

import os
from typing import Dict, List, Any

# Graceful import — plotting is optional
try:
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend (no display required)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


OUTPUT_DIR = "output"

# Clean, colorblind-accessible color palette
COLORS = {
    "group_0":  "#4C72B0",  # Steel blue  (Concise)
    "group_1":  "#55A868",  # Forest green (Detailed)
    "group_2":  "#C44E52",  # Crimson      (Technical)
    "greedy":   "#DD8452",  # Warm orange
    "robust":   "#8172B2",  # Purple
    "oracle":   "#64B5CD",  # Sky blue
    "random":   "#AAAAAA",  # Gray
}

GROUP_LABELS = ["G0: Concise", "G1: Detailed", "G2: Technical"]


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _require_mpl(fn_name: str) -> bool:
    if not _HAS_MPL:
        print(f"[visualize] matplotlib not available — skipping {fn_name}. "
              f"Install with: pip install matplotlib")
        return False
    return True


# ---------------------------------------------------------------------------
# Plot 1: Group accuracy by scenario (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_group_accuracy_by_scenario(
    all_results: Dict[str, Any],
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Grouped bar chart comparing per-group accuracy of greedy vs oracle
    across all benchmark scenarios.

    Each scenario shows 3 groups × 2 policies = 6 bars.
    Visually exposes which group is harmed and by how much.
    """
    if not _require_mpl("plot_group_accuracy_by_scenario"):
        return ""

    scenario_keys = list(all_results.keys())
    n_scenarios   = len(scenario_keys)
    n_groups      = 3

    fig, axes = plt.subplots(
        1, n_scenarios,
        figsize=(4 * n_scenarios, 5),
        sharey=True,
    )
    if n_scenarios == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Group Accuracy: Greedy Aggregated vs Oracle\n"
        "(Higher = Better | Fairness requires equal bars across groups)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    group_colors = [COLORS["group_0"], COLORS["group_1"], COLORS["group_2"]]
    bar_width    = 0.35
    x            = range(n_groups)

    for ax_idx, key in enumerate(scenario_keys):
        ax       = axes[ax_idx]
        exp_res  = all_results[key]
        scenario = exp_res["scenario"]
        res      = exp_res["results"]

        greedy_accs = res["standard"].get("greedy_aggregated", {}).get("accuracy_per_group", [0]*n_groups)
        oracle_accs = res["preference_aware"].get("oracle", {}).get("accuracy_per_group", [0]*n_groups)

        x_greedy = [xi - bar_width / 2 for xi in x]
        x_oracle = [xi + bar_width / 2 for xi in x]

        # Draw greedy bars with hatching
        for gi in range(n_groups):
            ax.bar(x_greedy[gi], greedy_accs[gi], bar_width,
                   color=group_colors[gi], alpha=0.6, hatch="//",
                   edgecolor="white", linewidth=0.5)

        # Draw oracle bars (solid)
        for gi in range(n_groups):
            ax.bar(x_oracle[gi], oracle_accs[gi], bar_width,
                   color=group_colors[gi], alpha=1.0,
                   edgecolor="white", linewidth=0.5)

        # Annotations
        dist = [round(w, 2) for w in scenario["group_distribution"]]
        title = f"Scenario {scenario['id']}: {scenario['name']}\ndist={dist}"
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(list(x))
        ax.set_xticklabels(["Concise", "Detailed", "Technical"], fontsize=8, rotation=15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Preference Group", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy (True Reward > 0)", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.6, hatch="//", label="Greedy Aggregated (standard)"),
        mpatches.Patch(facecolor="gray", alpha=1.0, label="Oracle (preference_aware)"),
        mpatches.Patch(facecolor=COLORS["group_0"], label="G0: Concise"),
        mpatches.Patch(facecolor=COLORS["group_1"], label="G1: Detailed"),
        mpatches.Patch(facecolor=COLORS["group_2"], label="G2: Technical"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    plt.tight_layout()

    path = ""
    if save:
        out = _ensure_output_dir()
        path = os.path.join(out, "group_accuracy_by_scenario.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Plot 2: Fairness gap comparison (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_fairness_gap_comparison(
    all_results: Dict[str, Any],
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Horizontal bar chart of fairness gap for each policy × scenario.

    The primary benchmark result visualization.
    A fairness gap of 0% = perfect fairness; 100% = catastrophic failure.
    """
    if not _require_mpl("plot_fairness_gap_comparison"):
        return ""

    # Collect data
    labels       = []
    greedy_gaps  = []
    robust_gaps  = []
    oracle_gaps  = []

    for key, exp_res in all_results.items():
        scenario = exp_res["scenario"]
        res      = exp_res["results"]
        label    = f"Sc.{scenario['id']}: {scenario['name']}"

        labels.append(label)
        greedy_gaps.append(
            res["standard"].get("greedy_aggregated", {}).get("fairness_gap", 0.0)
        )
        robust_gaps.append(
            res["standard"].get("group_robust", {}).get("fairness_gap", 0.0)
        )
        oracle_gaps.append(
            res["preference_aware"].get("oracle", {}).get("fairness_gap", 0.0)
        )

    n   = len(labels)
    y   = range(n)
    h   = 0.25
    fig, ax = plt.subplots(figsize=(9, 3 + n * 0.5))

    ax.barh([yi + h for yi in y], greedy_gaps, h, label="Greedy Aggregated",
            color=COLORS["greedy"], alpha=0.85)
    ax.barh([yi for yi in y],     robust_gaps, h, label="Group Robust",
            color=COLORS["robust"], alpha=0.85)
    ax.barh([yi - h for yi in y], oracle_gaps, h, label="Oracle (Fair)",
            color=COLORS["oracle"], alpha=0.85)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Fairness Gap  (max group acc − min group acc)", fontsize=10)
    ax.set_title(
        "Fairness Gap Across Scenarios and Policies\n"
        "Lower is Better  |  0% = Fair  |  100% = Catastrophic Minority Failure",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.axvline(x=0.0, color="black", linewidth=0.8)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="50% threshold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels on bars
    for xi, (g, r, o) in enumerate(zip(greedy_gaps, robust_gaps, oracle_gaps)):
        ax.text(g + 0.01, xi + h, f"{g:.0%}", va="center", fontsize=7, color="black")
        ax.text(r + 0.01, xi,     f"{r:.0%}", va="center", fontsize=7, color="black")
        ax.text(o + 0.01, xi - h, f"{o:.0%}", va="center", fontsize=7, color="black")

    plt.tight_layout()

    path = ""
    if save:
        out = _ensure_output_dir()
        path = os.path.join(out, "fairness_gap_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Plot 3: Bias strength sweep (line chart)
# ---------------------------------------------------------------------------

def plot_bias_strength_sweep(
    sweep_results: List[Dict[str, Any]],
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Line chart showing fairness gap and worst-group accuracy as
    bias_strength varies from 0.0 to 1.0.

    Key insight: there is a critical tipping point (~0.63 for [0.6, 0.3, 0.1])
    where the majority identity flips, but the fairness gap remains MAXIMAL
    on both sides. This proves aggregation bias is structural, not parametric.
    """
    if not _require_mpl("plot_bias_strength_sweep"):
        return ""

    bias_vals     = [r["bias_strength"]           for r in sweep_results]
    greedy_gaps   = [r["greedy_fairness_gap"]      for r in sweep_results]
    robust_gaps   = [r["robust_fairness_gap"]      for r in sweep_results]
    greedy_worst  = [r["greedy_worst_group_acc"]   for r in sweep_results]
    robust_worst  = [r["robust_worst_group_acc"]   for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # --- Panel 1: Fairness Gap ---
    ax1.plot(bias_vals, greedy_gaps, "o-", color=COLORS["greedy"],
             linewidth=2.0, markersize=6, label="Greedy Aggregated")
    ax1.plot(bias_vals, robust_gaps, "s--", color=COLORS["robust"],
             linewidth=2.0, markersize=6, label="Group Robust")

    ax1.set_xlabel("bias_strength", fontsize=10)
    ax1.set_ylabel("Fairness Gap", fontsize=10)
    ax1.set_title(
        "Fairness Gap vs. Bias Strength\n"
        "(Group distribution: [0.60, 0.30, 0.10])",
        fontsize=10, fontweight="bold",
    )
    ax1.set_ylim(-0.05, 1.15)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax1.axhline(y=0.0, color="green", linestyle=":", linewidth=1.0, alpha=0.6, label="Fair (0%)")
    ax1.axhline(y=1.0, color="red",   linestyle=":", linewidth=1.0, alpha=0.6, label="Max bias (100%)")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate tipping point (around bias_strength ≈ 0.63)
    ax1.axvline(x=0.63, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(0.65, 0.55, "Tipping\npoint\n(~0.63)", fontsize=8, color="gray")

    # --- Panel 2: Worst-Group Accuracy ---
    ax2.plot(bias_vals, greedy_worst, "o-", color=COLORS["greedy"],
             linewidth=2.0, markersize=6, label="Greedy Aggregated")
    ax2.plot(bias_vals, robust_worst, "s--", color=COLORS["robust"],
             linewidth=2.0, markersize=6, label="Group Robust")

    ax2.set_xlabel("bias_strength", fontsize=10)
    ax2.set_ylabel("Worst-Group Accuracy", fontsize=10)
    ax2.set_title(
        "Worst-Group Accuracy vs. Bias Strength\n"
        "(Higher = Better | Rawlsian fairness criterion)",
        fontsize=10, fontweight="bold",
    )
    ax2.set_ylim(-0.05, 1.10)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3, linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Effect of Bias Strength on Aggregation Fairness\n"
        "Key insight: Fairness gap remains near-maximal for ALL bias_strength values",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    path = ""
    if save:
        out = _ensure_output_dir()
        path = os.path.join(out, "bias_strength_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def save_all_plots(
    all_results: Dict[str, Any],
    sweep_results: List[Dict[str, Any]],
    show: bool = False,
) -> List[str]:
    """
    Generate and save all three diagnostic plots.

    Returns list of file paths to generated images.
    """
    paths = []

    p1 = plot_group_accuracy_by_scenario(all_results, save=True, show=show)
    if p1:
        paths.append(p1)

    p2 = plot_fairness_gap_comparison(all_results, save=True, show=show)
    if p2:
        paths.append(p2)

    p3 = plot_bias_strength_sweep(sweep_results, save=True, show=show)
    if p3:
        paths.append(p3)

    return paths
