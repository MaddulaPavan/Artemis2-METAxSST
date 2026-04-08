"""
evaluate.py — Evaluation framework for PreferenceAggregationEnv.

Metrics:
  - overall_accuracy    : fraction of steps where true reward > 0
  - accuracy_per_group  : per-group breakdown of above
  - fairness_gap        : max(group_acc) - min(group_acc)

Supports:
  - Any policy callable: obs → action
  - Oracle policy (receives hidden group via info)
"""

from typing import Callable, Dict, Any, Optional
import random

from reward import GROUP_NAMES


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    env,
    policy: Callable[[Dict[str, str]], int],
    n_episodes: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a policy over multiple episodes.

    The policy receives only the observation dict (no access to hidden group).
    True reward is computed from step info for accurate scoring.

    Args:
        env:        PreferenceAggregationEnv instance (any mode)
        policy:     Callable [[obs_dict] -> action (0 or 1)]
        n_episodes: Total number of evaluation episodes
        seed:       Random seed for reproducibility

    Returns:
        dict with:
            overall_accuracy    (float)       : fraction correct overall
            accuracy_per_group  (list[float]) : [acc_g0, acc_g1, acc_g2]
            fairness_gap        (float)       : max - min group accuracy
            group_totals        (dict)        : episode count per group
            group_correct       (dict)        : correct count per group
            n_episodes          (int)         : total episodes run
    """
    n_groups = 3
    group_correct = {g: 0 for g in range(n_groups)}
    group_total = {g: 0 for g in range(n_groups)}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        action = policy(obs)
        _, _, _, _, step_info = env.step(action)

        group = step_info["group"]
        correct = step_info["true_reward"] > 0

        group_total[group] += 1
        if correct:
            group_correct[group] += 1

    # Per-group accuracy
    accuracy_per_group = [
        (group_correct[g] / group_total[g]) if group_total[g] > 0 else 0.0
        for g in range(n_groups)
    ]

    overall_accuracy = sum(group_correct.values()) / n_episodes
    fairness_gap = max(accuracy_per_group) - min(accuracy_per_group)

    return {
        "overall_accuracy": overall_accuracy,
        "accuracy_per_group": accuracy_per_group,
        "fairness_gap": fairness_gap,
        "group_totals": group_total,
        "group_correct": group_correct,
        "n_episodes": n_episodes,
    }


def evaluate_oracle(
    env,
    n_episodes: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate an oracle policy that has perfect knowledge of the hidden group.

    This is the theoretical upper bound — a policy that always selects
    the response preferred by each episode's hidden group.

    Used to demonstrate what fairness looks like when preference aggregation
    bias is fully eliminated.

    Args:
        env:        PreferenceAggregationEnv instance
        n_episodes: Total number of evaluation episodes
        seed:       Random seed for reproducibility

    Returns:
        Same structure as evaluate()
    """
    from reward import group_reward

    n_groups = 3
    group_correct = {g: 0 for g in range(n_groups)}
    group_total = {g: 0 for g in range(n_groups)}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)

        # Oracle: peek at the hidden group from reset info
        hidden_group = info["group"]

        # Compute true reward for both actions and pick the better one
        r_a = group_reward(hidden_group, 0, obs)
        r_b = group_reward(hidden_group, 1, obs)
        action = 0 if r_a >= r_b else 1

        _, _, _, _, step_info = env.step(action)

        group = step_info["group"]
        correct = step_info["true_reward"] > 0

        group_total[group] += 1
        if correct:
            group_correct[group] += 1

    accuracy_per_group = [
        (group_correct[g] / group_total[g]) if group_total[g] > 0 else 0.0
        for g in range(n_groups)
    ]

    overall_accuracy = sum(group_correct.values()) / n_episodes
    fairness_gap = max(accuracy_per_group) - min(accuracy_per_group)

    return {
        "overall_accuracy": overall_accuracy,
        "accuracy_per_group": accuracy_per_group,
        "fairness_gap": fairness_gap,
        "group_totals": group_total,
        "group_correct": group_correct,
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Standard policies
# ---------------------------------------------------------------------------

def random_policy(obs: Dict[str, str]) -> int:
    """Baseline: choose uniformly at random."""
    return random.randint(0, 1)


def greedy_aggregated_policy(obs: Dict[str, str]) -> int:
    """
    Simulates a standard RLHF-trained agent.

    Selects the action that maximizes the weighted aggregated reward.
    Because Group 0 (concise) holds 60% population weight, this policy
    systematically learns the concise group's preference, ignoring others.
    """
    from reward import aggregated_reward
    r_a = aggregated_reward(0, obs)
    r_b = aggregated_reward(1, obs)
    return 0 if r_a >= r_b else 1


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_results(
    label: str,
    results: Dict[str, Any],
    indent: int = 2,
) -> str:
    """
    Format evaluation results as a human-readable string.

    Args:
        label:   Name of the policy/mode being reported
        results: Output of evaluate() or evaluate_oracle()
        indent:  Leading spaces

    Returns:
        Formatted multi-line string
    """
    pad = " " * indent
    lines = [
        f"{pad}┌─ {label}",
        f"{pad}│  Overall Accuracy    : {results['overall_accuracy']:.1%}",
    ]

    for g, acc in enumerate(results["accuracy_per_group"]):
        n = results["group_totals"][g]
        c = results["group_correct"][g]
        bar = _bar(acc)
        lines.append(
            f"{pad}│  Group {g} ({GROUP_NAMES[g]:10s}): {acc:.1%}  {bar}  ({c}/{n})"
        )

    fairness_gap = results["fairness_gap"]
    verdict = _fairness_verdict(fairness_gap)
    lines.append(f"{pad}│  Fairness Gap        : {fairness_gap:.1%}  ← {verdict}")
    lines.append(f"{pad}└{'─' * 55}")
    return "\n".join(lines)


def _bar(value: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar."""
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fairness_verdict(gap: float) -> str:
    if gap < 0.10:
        return "✅ FAIR"
    elif gap < 0.30:
        return "⚠️  MODERATE BIAS"
    elif gap < 0.60:
        return "🔴 HIGH BIAS"
    else:
        return "🚨 SEVERE BIAS (aggregation failure)"
