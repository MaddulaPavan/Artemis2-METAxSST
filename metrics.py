"""
metrics.py — Enhanced evaluation metrics for PreferenceAggregationEnv.

Metrics computed:
  - overall_accuracy      : fraction of steps where true_reward > 0
  - accuracy_per_group    : per-group breakdown
  - fairness_gap          : max(group_acc) - min(group_acc)         [primary]
  - variance_across_groups: population variance of group accuracies  [spread]
  - worst_group_accuracy  : min(group_acc)                          [tail risk]
  - best_group_accuracy   : max(group_acc)

Fairness framework:
  - fairness_gap       captures worst-case disparity (Rawlsian / minimax)
  - variance measures  symmetric spread across all groups
  - worst_group        identifies which group is being failed

These three metrics together provide a complete picture of fairness failure.
"""

import math
from typing import Dict, List, Any

from reward import GROUP_NAMES


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    group_correct: Dict[int, int],
    group_total: Dict[int, int],
    n_groups: int = 3,
) -> Dict[str, Any]:
    """
    Compute all fairness and accuracy metrics from raw episode tallies.

    Parameters
    ----------
    group_correct : dict  {group_id: number of correct episodes}
    group_total   : dict  {group_id: total episodes for that group}
    n_groups      : int   number of preference groups

    Returns
    -------
    dict with keys:
        overall_accuracy       (float)
        accuracy_per_group     (list[float])
        fairness_gap           (float)  ← primary fairness metric
        variance_across_groups (float)  ← spread metric
        worst_group_accuracy   (float)  ← tail risk
        best_group_accuracy    (float)
        worst_group_id         (int)
        best_group_id          (int)
        group_totals           (dict)
        group_correct          (dict)
        total_episodes         (int)
    """
    # Per-group accuracy
    accuracy_per_group = []
    for g in range(n_groups):
        if group_total.get(g, 0) > 0:
            accuracy_per_group.append(group_correct[g] / group_total[g])
        else:
            accuracy_per_group.append(0.0)

    total_episodes = sum(group_total.values())
    overall_accuracy = (
        sum(group_correct.values()) / total_episodes
        if total_episodes > 0 else 0.0
    )

    fairness_gap = max(accuracy_per_group) - min(accuracy_per_group)

    # Population variance across groups (measure of spread)
    mean_acc = sum(accuracy_per_group) / n_groups
    variance_across_groups = sum(
        (a - mean_acc) ** 2 for a in accuracy_per_group
    ) / n_groups

    worst_group_accuracy = min(accuracy_per_group)
    best_group_accuracy  = max(accuracy_per_group)
    worst_group_id = accuracy_per_group.index(worst_group_accuracy)
    best_group_id  = accuracy_per_group.index(best_group_accuracy)

    return {
        "overall_accuracy":       overall_accuracy,
        "accuracy_per_group":     accuracy_per_group,
        "fairness_gap":           fairness_gap,
        "variance_across_groups": variance_across_groups,
        "worst_group_accuracy":   worst_group_accuracy,
        "best_group_accuracy":    best_group_accuracy,
        "worst_group_id":         worst_group_id,
        "best_group_id":          best_group_id,
        "group_totals":           dict(group_total),
        "group_correct":          dict(group_correct),
        "total_episodes":         total_episodes,
    }


# ---------------------------------------------------------------------------
# Single-run evaluator
# ---------------------------------------------------------------------------

def run_evaluation(
    env,
    policy_fn,
    n_episodes: int = 300,
    seed: int = 42,
    use_oracle: bool = False,
) -> Dict[str, Any]:
    """
    Run a policy in the environment and compute metrics.

    Parameters
    ----------
    env         : PreferenceAggregationEnv instance
    policy_fn   : callable (obs -> action) OR oracle callable (obs, group -> action)
    n_episodes  : int
    seed        : int
    use_oracle  : bool — if True, policy_fn receives (obs, group) instead of (obs,)

    Returns
    -------
    dict from compute_metrics()
    """
    n_groups = env.n_groups
    group_correct = {g: 0 for g in range(n_groups)}
    group_total   = {g: 0 for g in range(n_groups)}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)

        if use_oracle:
            action = policy_fn(obs, info["group"])
        else:
            action = policy_fn(obs)

        _, _, _, _, step_info = env.step(action)

        g = step_info["group"]
        group_total[g] += 1
        if step_info["true_reward"] > 0:
            group_correct[g] += 1

    return compute_metrics(group_correct, group_total, n_groups)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fairness_verdict(gap: float) -> str:
    """Return a human-readable fairness label for a given gap value."""
    if gap < 0.05:
        return "FAIR"
    elif gap < 0.25:
        return "MODERATE BIAS"
    elif gap < 0.60:
        return "HIGH BIAS"
    else:
        return "SEVERE BIAS"


def fairness_icon(gap: float) -> str:
    if gap < 0.05:
        return "FAIR"
    elif gap < 0.25:
        return "MOD"
    elif gap < 0.60:
        return "HIGH"
    else:
        return "CRIT"


def ascii_bar(value: float, width: int = 16) -> str:
    """ASCII progress bar for terminal output."""
    filled = int(round(value * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def format_metrics_row(
    policy_label: str,
    metrics: Dict[str, Any],
    col_widths: tuple = (26, 10, 8, 8, 8, 12, 8),
) -> str:
    """
    Format a single result row for a comparison table.

    Columns: Policy | Overall | G0 | G1 | G2 | FairnessGap | Verdict
    """
    accs = metrics["accuracy_per_group"]
    gap  = metrics["fairness_gap"]
    verdict = fairness_icon(gap)

    return (
        f"  {policy_label:<{col_widths[0]}} "
        f"{metrics['overall_accuracy']:>{col_widths[1]}.1%} "
        f"{accs[0]:>{col_widths[2]}.1%} "
        f"{accs[1]:>{col_widths[3]}.1%} "
        f"{accs[2]:>{col_widths[4]}.1%} "
        f"{gap:>{col_widths[5]}.1%}  "
        f"{verdict}"
    )


def format_table_header(col_widths: tuple = (26, 10, 8, 8, 8, 12, 8)) -> str:
    """Return a formatted table header string."""
    labels = ["Policy", "Overall", "G0(Con)", "G1(Det)", "G2(Tec)", "FairGap", "Verdict"]
    header = "  " + "  ".join(
        f"{lbl:<{w}}" if i == 0 else f"{lbl:>{w}}"
        for i, (lbl, w) in enumerate(zip(labels, col_widths))
    )
    sep = "  " + "-" * (sum(col_widths) + 2 * len(col_widths))
    return f"{header}\n{sep}"
