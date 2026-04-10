"""
env/reward.py — Reward computation for PreferenceAggregationEnv.

Groups:
  0 — Concise   : prefers shorter response
  1 — Detailed  : prefers longer response
  2 — Technical : prefers higher technical vocabulary density

Rewards are in [0.0, 1.0] so that aggregated rewards are also in [0.0, 1.0].
This ensures: score = sum(rewards) / max_steps is naturally in [0.0, 1.0].
"""

from typing import List

TECHNICAL_VOCAB = {
    "algorithm", "parameter", "function", "optimization", "gradient",
    "neural", "vector", "matrix", "computational", "architecture",
    "inference", "training", "model", "hyperparameter", "objective",
    "loss", "backpropagation", "regularization", "stochastic", "convergence",
    "variance", "embedding", "latent", "posterior", "distribution",
    "probabilistic", "deterministic", "differentiable", "softmax",
    "parameterized", "activation", "encoder", "decoder", "policy",
    "reward", "markov", "entropy", "divergence", "expectation",
    "approximation", "discrete", "continuous", "normalization",
    "tokenization", "vocabulary", "sampling", "fine-tuning",
}

GROUP_NAMES = {0: "Concise", 1: "Detailed", 2: "Technical"}


def _length(text: str) -> int:
    return len(text.strip())


def _technical_score(text: str) -> float:
    words = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split()]
    if not words:
        return 0.0
    return sum(1 for w in words if w in TECHNICAL_VOCAB) / len(words)


def technical_score(text: str) -> float:
    """Fraction of tokens in TECHNICAL_VOCAB (public; used by baselines / diagnostics)."""
    return _technical_score(text)


def group_reward(group: int, action: int, response_a: str, response_b: str) -> float:
    """
    Compute true reward for a given preference group.

    Returns 1.0 if action matches group preference, 0.0 otherwise.
    Using {0, 1} (not {-1, +1}) so aggregated rewards stay in [0, 1].
    """
    if group == 0:  # Concise: prefers shorter
        preferred = 0 if _length(response_a) <= _length(response_b) else 1
    elif group == 1:  # Detailed: prefers longer
        preferred = 0 if _length(response_a) >= _length(response_b) else 1
    elif group == 2:  # Technical: prefers higher technical density
        score_a = _technical_score(response_a)
        score_b = _technical_score(response_b)
        preferred = 0 if score_a >= score_b else 1
    else:
        raise ValueError(f"Unknown group: {group}")

    return 1.0 if action == preferred else 0.0


def compute_all_group_rewards(action: int, response_a: str, response_b: str) -> List[float]:
    """Return [reward_g0, reward_g1, reward_g2] for diagnostic breakdown."""
    return [group_reward(g, action, response_a, response_b) for g in range(3)]


def sharpen_aggregated_reward(x: float, gamma: float = 1.55) -> float:
    """
    Map [0,1] → [0,1] with stronger separation toward 0 and 1 (clearer policy signal).

    gamma=1 is identity. Does not change per-group satisfaction; graders use group_rewards only.
    """
    if gamma <= 1.0:
        return max(0.0, min(1.0, x))
    y = 0.5 + gamma * (x - 0.5)
    return max(0.0, min(1.0, y))


def compute_aggregated_reward(
    action: int,
    response_a: str,
    response_b: str,
    weights: List[float],
) -> float:
    """
    Weighted sum of group rewards — the standard RLHF aggregated reward.

    Returns float in [0.0, 1.0] since each group_reward ∈ {0, 1}
    and weights sum to 1.0.
    """
    total = 0.0
    for g, w in enumerate(weights):
        total += w * group_reward(g, action, response_a, response_b)
    return total


def counterfactual_reward_margin(
    response_a: str,
    response_b: str,
    weights: List[float],
) -> float:
    """|R(choose A) - R(choose B)| under the same population weights (diagnostic)."""
    r0 = compute_aggregated_reward(0, response_a, response_b, weights)
    r1 = compute_aggregated_reward(1, response_a, response_b, weights)
    return abs(r0 - r1)


def compute_fairness_gap(group_rewards_history: List[List[float]]) -> float:
    """
    Compute fairness gap from episode history of per-group rewards.

    fairness_gap = max(group_accuracy) - min(group_accuracy)
    """
    if not group_rewards_history:
        return 0.0
    n = len(group_rewards_history)
    n_groups = len(group_rewards_history[0])
    group_accs = [
        sum(step[g] for step in group_rewards_history) / n
        for g in range(n_groups)
    ]
    return max(group_accs) - min(group_accs)
