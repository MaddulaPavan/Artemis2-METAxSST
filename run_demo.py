"""
run_demo.py — End-to-end demonstration of PreferenceAggregationEnv.

This script:
  1. Initializes the environment in both modes
  2. Runs three policies: random, greedy-aggregated, oracle
  3. Prints per-group accuracy and fairness gap
  4. Shows a reward breakdown example for one episode
  5. Summarizes the core insight

Usage:
    python run_demo.py

No external dependencies required (pure Python + standard library).
Optional: pip install gymnasium  (for full Gym compatibility)
"""

import random

from env import PreferenceAggregationEnv
from evaluate import (
    evaluate,
    evaluate_oracle,
    random_policy,
    greedy_aggregated_policy,
    format_results,
)
from reward import reward_breakdown, GROUP_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EPISODES = 600     # Total evaluation episodes per experiment
SEED = 2024          # Global seed for reproducibility

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

DIVIDER = "=" * 72
SUBDIV  = "-" * 72

def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def subsection(title: str) -> None:
    print(f"\n  {SUBDIV}")
    print(f"    {title}")
    print(f"  {SUBDIV}")


# ---------------------------------------------------------------------------
# Experiment 1: Single-episode walkthrough
# ---------------------------------------------------------------------------

def demo_single_episode() -> None:
    section("STEP 1 — Single Episode Walkthrough")

    env = PreferenceAggregationEnv(mode="standard", seed=SEED)
    obs, info = env.reset()

    print(f"\n  Observation:")
    print(f"    Prompt     : {obs['prompt']}")
    print(f"    Response A : {obs['response_A'][:80]}...")
    print(f"    Response B : {obs['response_B'][:80]}...")
    print(f"\n  Hidden Group : {info['group_name']} (Group {info['group']})")

    print(f"\n  ── Reward Breakdown (action=0, choosing Response A) ──")
    breakdown = reward_breakdown(0, obs)
    for g_name, data in breakdown.items():
        if g_name == "aggregated":
            print(f"\n    Aggregated reward = {data:+.2f}")
        else:
            print(
                f"    {g_name:10s} | true_reward={data['reward']:+.1f} | "
                f"weight={data['weight']:.0%} | "
                f"contribution={data['weighted_contribution']:+.3f}"
            )

    print(f"\n  ── Reward Breakdown (action=1, choosing Response B) ──")
    breakdown_b = reward_breakdown(1, obs)
    for g_name, data in breakdown_b.items():
        if g_name == "aggregated":
            print(f"\n    Aggregated reward = {data:+.2f}")
        else:
            print(
                f"    {g_name:10s} | true_reward={data['reward']:+.1f} | "
                f"weight={data['weight']:.0%} | "
                f"contribution={data['weighted_contribution']:+.3f}"
            )

    print(
        f"\n  📌 Key Insight: Even when B is correct for Detailed/Technical groups,\n"
        f"     the aggregated reward favors A because Group 0 (Concise, 60%) dominates."
    )


# ---------------------------------------------------------------------------
# Experiment 2: Standard RLHF — Majority-Following Policy
# ---------------------------------------------------------------------------

def demo_standard_rlhf() -> None:
    section("STEP 2 — Standard RLHF: Greedy Aggregated Policy")

    print(
        "\n  Policy: greedy_aggregated_policy\n"
        "  → Picks whichever response maximizes the weighted aggregated reward.\n"
        "  → Simulates the behavior of a model trained on pooled human feedback.\n"
        "  → Majority group (Concise, 60%) dominates the learning signal."
    )

    env = PreferenceAggregationEnv(mode="standard", seed=SEED)
    results = evaluate(
        env,
        policy=greedy_aggregated_policy,
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    print(f"\n{format_results('Standard RLHF — Greedy Aggregated Policy', results)}")

    print(
        f"\n  ⚠️  Result: The policy perfectly satisfies the majority (Concise) group,\n"
        f"     but completely fails minority groups — Fairness Gap = {results['fairness_gap']:.0%}."
    )


# ---------------------------------------------------------------------------
# Experiment 3: Preference-Aware — Oracle Policy
# ---------------------------------------------------------------------------

def demo_preference_aware() -> None:
    section("STEP 3 — Preference-Aware Mode: Oracle Policy")

    print(
        "\n  Policy: oracle_policy\n"
        "  → Has access to the hidden preference group.\n"
        "  → Selects the response matching each group's true preference.\n"
        "  → Represents the ideal outcome of preference-separated training."
    )

    env = PreferenceAggregationEnv(mode="preference_aware", seed=SEED)
    results = evaluate_oracle(
        env,
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    print(f"\n{format_results('Preference-Aware Mode — Oracle Policy', results)}")

    print(
        f"\n  ✅ Result: All groups receive their preferred response.\n"
        f"     Fairness Gap = {results['fairness_gap']:.0%} — Near-zero bias."
    )


# ---------------------------------------------------------------------------
# Experiment 4: Random Policy Baseline (both modes)
# ---------------------------------------------------------------------------

def demo_random_baseline() -> None:
    section("STEP 4 — Baseline: Random Policy")

    print("\n  Policy: random_policy\n  → Randomly selects response A or B.")

    random.seed(SEED)

    env_std = PreferenceAggregationEnv(mode="standard", seed=SEED)
    results_std = evaluate(
        env_std,
        policy=random_policy,
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    env_pa = PreferenceAggregationEnv(mode="preference_aware", seed=SEED)
    results_pa = evaluate(
        env_pa,
        policy=random_policy,
        n_episodes=N_EPISODES,
        seed=SEED,
    )

    print(f"\n{format_results('Random Policy [standard mode]', results_std)}")
    print(f"\n{format_results('Random Policy [preference_aware mode]', results_pa)}")

    print(
        "\n  📊 Random baseline shows ~50% accuracy across all groups — the floor."
    )


# ---------------------------------------------------------------------------
# Experiment 5: Head-to-head summary table
# ---------------------------------------------------------------------------

def demo_summary_table() -> None:
    section("STEP 5 — Summary: Fairness Gap Comparison")

    # Run all policies
    env_std = PreferenceAggregationEnv(mode="standard", seed=SEED)
    env_pa  = PreferenceAggregationEnv(mode="preference_aware", seed=SEED)

    res_random_std = evaluate(env_std, random_policy,            n_episodes=N_EPISODES, seed=SEED)
    res_greedy     = evaluate(env_std, greedy_aggregated_policy,  n_episodes=N_EPISODES, seed=SEED)
    res_oracle     = evaluate_oracle(env_pa,                      n_episodes=N_EPISODES, seed=SEED)

    configs = [
        ("Random (standard)",       res_random_std),
        ("Greedy Aggregated (RLHF)", res_greedy),
        ("Oracle (preference-aware)", res_oracle),
    ]

    header = (
        f"\n  {'Policy':<30} {'Concise':>9} {'Detailed':>9} {'Technical':>10} {'Fairness Gap':>13}"
    )
    print(header)
    print(f"  {'─' * 73}")

    for label, res in configs:
        accs = res["accuracy_per_group"]
        gap = res["fairness_gap"]
        flag = (
            "✅ FAIR"       if gap < 0.10 else
            "⚠️  MOD BIAS"   if gap < 0.30 else
            "🚨 SEVERE BIAS"
        )
        print(
            f"  {label:<30} {accs[0]:>8.1%} {accs[1]:>9.1%} {accs[2]:>10.1%} "
            f"  {gap:>7.1%}  {flag}"
        )


# ---------------------------------------------------------------------------
# Core Insight
# ---------------------------------------------------------------------------

def print_core_insight() -> None:
    section("CORE INSIGHT")
    print("""
  The PreferenceAggregationEnv demonstrates a fundamental RLHF failure:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  When a single reward model is trained on pooled, heterogeneous     │
  │  annotator data, majority preferences systematically dominate.      │
  │                                                                     │
  │  Minority groups receive a reward signal polluted by majority       │
  │  preferences → the final policy under-serves them regardless of     │
  │  how long training runs.                                            │
  │                                                                     │
  │  This is NOT a training data size problem.                          │
  │  It is a structural consequence of aggregation itself.              │
  │                                                                     │
  │  Solution: Separate reward models per preference group, or use      │
  │  preference-aware training that conditions on group identity.       │
  └─────────────────────────────────────────────────────────────────────┘

  This environment provides a reproducible, minimal testbed for studying
  and benchmarking fairness interventions in RL alignment pipelines.
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  PreferenceAggregationEnv — RLHF Diagnostic Simulator")
    print("  India's MEGA AI Hackathon | Meta × Hugging Face × SST")
    print(f"{'=' * 72}")
    print(f"  Seed: {SEED} | Episodes per experiment: {N_EPISODES}")

    demo_single_episode()
    demo_standard_rlhf()
    demo_preference_aware()
    demo_random_baseline()
    demo_summary_table()
    print_core_insight()

    print(f"\n{'=' * 72}")
    print("  Run complete. All experiments finished successfully.")
    print(f"{'=' * 72}\n")
