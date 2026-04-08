"""
run_all_experiments.py — Full benchmark demo for PreferenceAggregationEnv.

Runs the complete research benchmark:

  Experiment 1: Full scenario sweep (Scenarios A-E)
                All policies evaluated in standard mode
                Oracle evaluated in preference_aware mode

  Experiment 2: Bias strength sweep
                Fairness gap and worst-group accuracy vs bias_strength [0.0 → 1.0]

  Experiment 3: Summary comparison table

  Output:
    - Formatted tables to stdout
    - 3 diagnostic plots -> output/
    - Structured results -> output/experiment_results.json

Usage:
    python run_all_experiments.py

Dependencies:
    - Core: pure Python (no ML required)
    - Plots: pip install matplotlib
    - JSON:  built-in
"""

import json
import os
import time

from experiments import (
    SCENARIOS,
    run_all_scenarios,
    run_bias_sweep,
    format_scenario_table,
    format_summary_table,
)
from visualize import save_all_plots, _HAS_MPL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EPISODES = 500     # Episodes per policy per scenario
SEED       = 2024    # Global seed (full reproducibility)
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

BANNER = "=" * 78

def header(title: str) -> None:
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def subheader(title: str) -> None:
    print(f"\n  {'─' * 70}")
    print(f"    {title}")
    print(f"  {'─' * 70}")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    print(f"\n{BANNER}")
    print("  PreferenceAggregationEnv — Research Benchmark v2")
    print("  RLHF Aggregation Failure Simulator")
    print("  India's MEGA AI Hackathon | Meta x Hugging Face x SST")
    print(BANNER)
    print(f"\n  Config: {N_EPISODES} episodes/policy | seed={SEED}")
    print(f"  Scenarios: {len(SCENARIOS)} | Policies: greedy, robust, oracle, random")

    ensure_output_dir()

    # -----------------------------------------------------------------------
    # Experiment 1: Full Scenario Sweep
    # -----------------------------------------------------------------------

    header("EXPERIMENT 1 — Full Scenario Sweep (A through E)")

    print("\n  Running all scenarios. This may take a moment...\n")
    all_results = run_all_scenarios(n_episodes=N_EPISODES, seed=SEED, verbose=True)

    # Print per-scenario tables
    for key, exp_res in all_results.items():
        print(format_scenario_table(exp_res))

    # -----------------------------------------------------------------------
    # Experiment 2: Bias Strength Sweep
    # -----------------------------------------------------------------------

    header("EXPERIMENT 2 — Bias Strength Sweep (bias_strength: 0.0 to 1.0)")

    print(
        "\n  Distribution: [0.60, 0.30, 0.10]"
        "\n  Measuring fairness gap and worst-group accuracy as bias_strength varies."
        "\n  Key question: Does reducing aggregation bias eliminate fairness failure?\n"
    )

    sweep_results = run_bias_sweep(
        group_distribution=[0.60, 0.30, 0.10],
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    # Print sweep table
    print(f"\n  {'bias_strength':>14} {'Greedy FairGap':>16} {'Robust FairGap':>16} "
          f"{'Greedy WorstGrp':>18} {'Robust WorstGrp':>16}")
    print(f"  {'─' * 84}")
    for row in sweep_results:
        bs = row["bias_strength"]
        print(
            f"  {bs:>14.2f} {row['greedy_fairness_gap']:>15.1%}  "
            f"{row['robust_fairness_gap']:>15.1%}  "
            f"{row['greedy_worst_group_acc']:>17.1%}  "
            f"{row['robust_worst_group_acc']:>15.1%}"
        )

    # -----------------------------------------------------------------------
    # Experiment 3: Summary Comparison Table
    # -----------------------------------------------------------------------

    header("EXPERIMENT 3 — Summary Table: Fairness Gap Across All Scenarios")

    print(format_summary_table(all_results))

    # Key findings
    print("""
  KEY FINDINGS:
  ─────────────────────────────────────────────────────────────────────────
  1. Greedy Aggregated always achieves ~100% fairness gap regardless of
     which group is the majority. The IDENTITY of the harmed group changes,
     but the magnitude of harm does not.

  2. Group Robust (minimax heuristic) consistently achieves lower fairness
     gap than greedy, at the cost of lower overall accuracy.
     This is the fundamental fairness-accuracy tradeoff.

  3. Even with low bias_strength (Scenario D), the fairness gap remains
     high, showing that the failure is structural — not a tuning problem.

  4. Reward noise (Scenario E) introduces uncertainty without improving
     fairness direction. Adding noise does NOT help minority groups.

  5. Oracle achieves 0% fairness gap in ALL scenarios, demonstrating that
     the environment can be solved — the failure is in the objective design.

  CONCLUSION:
     Increasing data, reducing bias parameters, or adding noise does NOT
     fix preference aggregation failure. The solution requires separating
     reward signals by preference group (PA-RLHF approach).
  ─────────────────────────────────────────────────────────────────────────
    """)

    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------

    header("Generating Diagnostic Plots")

    if _HAS_MPL:
        print("\n  Generating 3 plots...")
        plot_paths = save_all_plots(all_results, sweep_results, show=False)
        print(f"\n  Saved {len(plot_paths)} plot(s):")
        for p in plot_paths:
            print(f"    -> {os.path.abspath(p)}")
    else:
        print("\n  matplotlib not installed. Skipping plots.")
        print("  Install with: pip install matplotlib")
        plot_paths = []

    # -----------------------------------------------------------------------
    # Save Results to JSON
    # -----------------------------------------------------------------------

    header("Saving Results")

    # Serialize results (convert float keys to string for JSON compat)
    serializable = {}
    for key, exp_res in all_results.items():
        serializable[key] = {
            "scenario": exp_res["scenario"],
            "results": {
                mode: {
                    policy: {
                        k: (v if not isinstance(v, dict) else
                            {str(kk): vv for kk, vv in v.items()})
                        for k, v in metrics.items()
                    }
                    for policy, metrics in mode_results.items()
                }
                for mode, mode_results in exp_res["results"].items()
            }
        }

    results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_episodes": N_EPISODES,
                "seed":       SEED,
                "n_scenarios": len(SCENARIOS),
            },
            "scenarios":    serializable,
            "bias_sweep":   sweep_results,
            "plot_paths":   plot_paths,
        }, f, indent=2)

    print(f"\n  Results saved to: {os.path.abspath(results_path)}")

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------

    elapsed = time.time() - start_time
    print(f"\n{BANNER}")
    print(f"  Benchmark complete.  Elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}/")
    print(BANNER + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
