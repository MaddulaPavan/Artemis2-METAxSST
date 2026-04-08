"""
experiments.py — Experiment engine for PreferenceAggregationEnv benchmark.

Defines:
  - SCENARIOS    : curated set of benchmark configurations
  - run_experiment()   : single-scenario evaluator
  - run_all_scenarios(): full benchmark sweep

Scenario taxonomy:
  A — Majority Dominance  : [0.60, 0.30, 0.10] — classic RLHF setup
  B — Balanced Groups     : [0.33, 0.33, 0.34] — equal representation
  C — Minority Flip       : [0.20, 0.40, 0.40] — former majority is now minority
  D — Reduced Bias        : [0.60, 0.30, 0.10] + bias_strength=0.3
  E — Noisy Rewards       : [0.60, 0.30, 0.10] + noise_level=0.3

Scenarios A/B/C vary the population imbalance — showing the aggregated reward
always produces maximal fairness gap regardless of which group is the majority.

Scenarios D/E show that reducing bias_strength and adding noise don't
eliminate the fundamental problem — they only shift WHO is harmed.
"""

from typing import Dict, List, Any

from env import PreferenceAggregationEnv
from metrics import run_evaluation, format_metrics_row, format_table_header
from policies import (
    STANDARD_POLICIES,
    ORACLE_POLICY,
)

# ---------------------------------------------------------------------------
# Benchmark scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "majority_dominance": {
        "id":                "A",
        "name":              "Majority Dominance",
        "description":      "Group 0 holds 60% population weight — classic RLHF setup",
        "group_distribution": [0.60, 0.30, 0.10],
        "bias_strength":    1.0,
        "noise_level":      0.0,
    },
    "balanced_groups": {
        "id":                "B",
        "name":              "Balanced Groups",
        "description":      "Equal representation: no single group has population advantage",
        "group_distribution": [0.333, 0.333, 0.334],
        "bias_strength":    1.0,
        "noise_level":      0.0,
    },
    "minority_flip": {
        "id":                "C",
        "name":              "Minority Flip",
        "description":      "Former minority groups (1+2) hold 80% combined weight",
        "group_distribution": [0.20, 0.40, 0.40],
        "bias_strength":    1.0,
        "noise_level":      0.0,
    },
    "reduced_bias": {
        "id":                "D",
        "name":              "Reduced Bias (bias=0.3)",
        "description":      "Majority distribution but bias_strength=0.3 (moderate interpolation)",
        "group_distribution": [0.60, 0.30, 0.10],
        "bias_strength":    0.30,
        "noise_level":      0.0,
    },
    "noisy_rewards": {
        "id":                "E",
        "name":              "Noisy Rewards (sigma=0.3)",
        "description":      "Majority distribution with Gaussian reward corruption (sigma=0.3)",
        "group_distribution": [0.60, 0.30, 0.10],
        "bias_strength":    1.0,
        "noise_level":      0.30,
    },
}

# Bias sweep values for Experiment 3
BIAS_SWEEP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ---------------------------------------------------------------------------
# Single scenario evaluator
# ---------------------------------------------------------------------------

def run_experiment(
    scenario_key: str,
    n_episodes: int = 400,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a full policy comparison for a single scenario.

    For each policy × mode combination, evaluates the policy and records metrics.
    Returns a structured result dict suitable for aggregation and plotting.

    Parameters
    ----------
    scenario_key : str   Key in SCENARIOS dict
    n_episodes   : int   Episodes per policy evaluation
    seed         : int   Random seed
    verbose      : bool  Print progress during evaluation

    Returns
    -------
    {
        "scenario": dict           — scenario metadata
        "results": {
            "standard": {
                "random": metrics_dict,
                "greedy_aggregated": metrics_dict,
                "group_robust": metrics_dict,
            },
            "preference_aware": {
                "oracle": metrics_dict,
            }
        }
    }
    """
    scenario = SCENARIOS[scenario_key]
    env_kwargs = {
        "group_distribution": scenario["group_distribution"],
        "bias_strength":      scenario["bias_strength"],
        "noise_level":        scenario["noise_level"],
        "seed":               seed,
    }

    run_results = {"standard": {}, "preference_aware": {}}

    # --- Evaluate standard policies in STANDARD mode ---
    for policy_name, policy_fn in STANDARD_POLICIES.items():
        if verbose:
            print(f"    Evaluating {policy_name} [standard]...")
        env = PreferenceAggregationEnv(mode="standard", **env_kwargs)
        run_results["standard"][policy_name] = run_evaluation(
            env, policy_fn, n_episodes=n_episodes, seed=seed
        )

    # --- Evaluate oracle in PREFERENCE_AWARE mode ---
    if verbose:
        print(f"    Evaluating oracle [preference_aware]...")
    env = PreferenceAggregationEnv(mode="preference_aware", **env_kwargs)
    run_results["preference_aware"]["oracle"] = run_evaluation(
        env, ORACLE_POLICY, n_episodes=n_episodes, seed=seed, use_oracle=True
    )

    return {
        "scenario": scenario,
        "results":  run_results,
    }


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------

def run_all_scenarios(
    n_episodes: int = 400,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full benchmark across all scenarios.

    Parameters
    ----------
    n_episodes : int   Episodes per policy per scenario
    seed       : int   Global seed
    verbose    : bool  Print progress

    Returns
    -------
    dict {scenario_key: experiment_result}
    """
    all_results = {}

    for key, scenario in SCENARIOS.items():
        if verbose:
            print(f"\n  Running Scenario {scenario['id']}: {scenario['name']}...")
        all_results[key] = run_experiment(
            key, n_episodes=n_episodes, seed=seed, verbose=verbose
        )

    return all_results


# ---------------------------------------------------------------------------
# Bias strength sweep experiment
# ---------------------------------------------------------------------------

def run_bias_sweep(
    group_distribution: List[float] = None,
    n_episodes: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sweep bias_strength from 0.0 to 1.0 and record fairness metrics.

    Reveals the critical tipping point where majority dominance flips:
    at low bias_strength, the aggregated reward is nearly uniform, so
    the dominant group changes. At high bias_strength, the original
    majority dominates.

    Returns
    -------
    List of dicts: [{bias_strength, fairness_gap (standard), fairness_gap (robust)}]
    """
    from policies import greedy_aggregated_policy, group_robust_policy

    if group_distribution is None:
        group_distribution = [0.60, 0.30, 0.10]

    sweep_results = []

    for bs in BIAS_SWEEP_VALUES:
        env_std = PreferenceAggregationEnv(
            mode="standard",
            group_distribution=group_distribution,
            bias_strength=bs,
            seed=seed,
        )
        env_pa = PreferenceAggregationEnv(
            mode="preference_aware",
            group_distribution=group_distribution,
            bias_strength=bs,
            seed=seed,
        )

        met_greedy = run_evaluation(env_std, greedy_aggregated_policy,
                                    n_episodes=n_episodes, seed=seed)
        met_robust = run_evaluation(env_pa, group_robust_policy,
                                    n_episodes=n_episodes, seed=seed)

        sweep_results.append({
            "bias_strength":          bs,
            "greedy_fairness_gap":    met_greedy["fairness_gap"],
            "robust_fairness_gap":    met_robust["fairness_gap"],
            "greedy_worst_group_acc": met_greedy["worst_group_accuracy"],
            "robust_worst_group_acc": met_robust["worst_group_accuracy"],
            "greedy_accuracy_per_group": met_greedy["accuracy_per_group"],
            "robust_accuracy_per_group": met_robust["accuracy_per_group"],
        })

    return sweep_results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_scenario_table(experiment_result: Dict[str, Any]) -> str:
    """
    Format a single scenario experiment as a comparison table string.
    """
    scenario = experiment_result["scenario"]
    results  = experiment_result["results"]

    dist = [round(w, 2) for w in scenario["group_distribution"]]
    lines = [
        f"\n  Scenario {scenario['id']}: {scenario['name']}",
        f"  Distribution: {dist}  |  bias_strength: {scenario['bias_strength']}  |  noise: {scenario['noise_level']}",
        f"  {scenario['description']}",
        "",
        format_table_header(),
    ]

    # Standard mode policies
    for policy_name, metrics in results["standard"].items():
        label = f"{policy_name} [standard]"
        lines.append(format_metrics_row(label, metrics))

    # Preference-aware oracle
    for policy_name, metrics in results["preference_aware"].items():
        label = f"{policy_name} [pref_aware]"
        lines.append(format_metrics_row(label, metrics))

    return "\n".join(lines)


def format_summary_table(all_results: Dict[str, Any]) -> str:
    """
    Format a one-line-per-scenario summary table of fairness gaps.

    Shows how the greedy vs oracle gap evolves across scenarios.
    """
    lines = [
        "\n  SUMMARY: Fairness Gap by Scenario",
        "  " + "-" * 80,
        f"  {'Scenario':<30} {'Distribution':<20} {'Greedy Gap':>12} {'Robust Gap':>12} {'Oracle Gap':>11}",
        "  " + "-" * 80,
    ]

    for key, exp_res in all_results.items():
        scenario = exp_res["scenario"]
        res      = exp_res["results"]

        dist = [round(w, 2) for w in scenario["group_distribution"]]
        g_greedy = res["standard"].get("greedy_aggregated", {}).get("fairness_gap", float("nan"))
        g_robust = res["standard"].get("group_robust", {}).get("fairness_gap", float("nan"))
        g_oracle = res["preference_aware"].get("oracle", {}).get("fairness_gap", float("nan"))

        name = f"{scenario['id']}. {scenario['name']}"
        lines.append(
            f"  {name:<30} {str(dist):<20} {g_greedy:>11.1%}  {g_robust:>11.1%}  {g_oracle:>10.1%}"
        )

    lines.append("  " + "-" * 80)
    return "\n".join(lines)
