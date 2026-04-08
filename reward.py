"""
reward.py — Reward logic for PreferenceAggregationEnv.

v2 changes:
  - aggregated_reward() now accepts an optional `weights` parameter.
    When supplied (by env.py), it uses env-computed effective_weights
    (bias_strength-interpolated), enabling the parameterized bias sweep.
  - compute_effective_weights() helper exposed for standalone use.

Defines:
  - Per-group true reward functions (concise, detailed, technical)
  - Aggregated reward (standard RLHF — majority-weighted)
  - Preference-aware reward (fair — uses hidden group only)

Group definitions:
  0 → Concise   (prefers shorter responses)
  1 → Detailed  (prefers longer responses)
  2 → Technical (prefers higher technical vocabulary density)

Population weights (simulating majority→minority imbalance):
  Group 0: 60%  ← majority
  Group 1: 30%
  Group 2: 10%  ← minority
"""

# ---------------------------------------------------------------------------
# Technical vocabulary used to score response technicality
# ---------------------------------------------------------------------------
TECHNICAL_VOCABULARY = {
    "algorithm", "parameter", "function", "implementation", "optimization",
    "gradient", "neural", "vector", "matrix", "computational", "architecture",
    "inference", "training", "model", "dataset", "hyperparameter", "objective",
    "loss", "backpropagation", "regularization", "stochastic", "convergence",
    "variance", "embedding", "latent", "posterior", "likelihood", "distribution",
    "probabilistic", "deterministic", "iterative", "recursive", "polynomial",
    "parameterized", "non-linear", "activation", "encoder", "decoder", "policy",
    "reward", "markov", "transition", "entropy", "divergence", "expectation",
    "estimator", "approximation", "discrete", "continuous", "differentiable",
    "softmax", "projection", "dimensionality", "manifold", "representation",
    "factorize", "normalize", "decompose", "constraint", "regularize",
    "tokenization", "vocabulary", "subword", "sampling", "temperature",
    "fine-tuning", "pretrained", "downstream", "generalization", "empirical",
}

# Population weights simulating annotator imbalance
GROUP_WEIGHTS = {
    0: 0.60,   # Concise   → majority
    1: 0.30,   # Detailed
    2: 0.10,   # Technical → minority
}

GROUP_NAMES = {0: "Concise", 1: "Detailed", 2: "Technical"}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _length_score(text: str) -> int:
    """Return character length of text (stripped)."""
    return len(text.strip())


def _technical_score(text: str) -> float:
    """
    Return the fraction of words that belong to the technical vocabulary.
    Higher = more technical.
    """
    words = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split()]
    if not words:
        return 0.0
    tech_count = sum(1 for w in words if w in TECHNICAL_VOCABULARY)
    return tech_count / len(words)


# ---------------------------------------------------------------------------
# Per-group true reward
# ---------------------------------------------------------------------------

def group_reward(group: int, action: int, state: dict) -> float:
    """
    Compute the TRUE reward for a given preference group.

    Returns:
        +1.0  if the chosen response matches the group's preference
        -1.0  otherwise

    Args:
        group:  0 (concise), 1 (detailed), 2 (technical)
        action: 0 (choose response_A) or 1 (choose response_B)
        state:  observation dict with keys 'response_A', 'response_B'
    """
    resp_A = state["response_A"]
    resp_B = state["response_B"]

    if group == 0:
        # Concise: prefers the shorter response
        preferred = 0 if _length_score(resp_A) <= _length_score(resp_B) else 1

    elif group == 1:
        # Detailed: prefers the longer response
        preferred = 0 if _length_score(resp_A) >= _length_score(resp_B) else 1

    elif group == 2:
        # Technical: prefers the response with higher technical vocabulary density
        score_A = _technical_score(resp_A)
        score_B = _technical_score(resp_B)
        preferred = 0 if score_A >= score_B else 1

    else:
        raise ValueError(f"Unknown group: {group}")

    return 1.0 if action == preferred else -1.0


# ---------------------------------------------------------------------------
# Aggregated reward (Standard RLHF — biased)
# ---------------------------------------------------------------------------

def compute_effective_weights(
    group_distribution: list,
    bias_strength: float = 1.0,
) -> list:
    """
    Compute effective reward weights by interpolating between
    uniform weights and group_distribution using bias_strength.

    bias_strength = 0.0  →  uniform weights (no aggregation bias)
    bias_strength = 1.0  →  full group_distribution (maximum bias)

    Args:
        group_distribution: List of population weights (will be normalized)
        bias_strength:      Float in [0.0, 1.0]

    Returns:
        List of effective weights (normalized, sums to 1)
    """
    n = len(group_distribution)
    total = sum(group_distribution)
    dist = [w / total for w in group_distribution]
    uniform = [1.0 / n] * n
    effective = [
        (1.0 - bias_strength) * uniform[g] + bias_strength * dist[g]
        for g in range(n)
    ]
    eff_total = sum(effective)
    return [w / eff_total for w in effective]


def aggregated_reward(
    action: int,
    state: dict,
    weights: list = None,
) -> float:
    """
    Compute the aggregated reward across ALL preference groups.

    Simulates standard RLHF where a single reward model is trained
    on pooled, heterogeneous annotator data.

    When called from the env, `weights` = env.effective_weights, which
    encodes both the population distribution and the bias_strength
    interpolation. This allows fine-grained control over aggregation bias.

    Args:
        action:  0 or 1
        state:   observation dict
        weights: Optional list of per-group weights (default: GROUP_WEIGHTS)

    Returns:
        Weighted sum in range [-1.0, +1.0]
    """
    if weights is None:
        weights = list(GROUP_WEIGHTS.values())

    total = 0.0
    for g, w in enumerate(weights):
        total += w * group_reward(g, action, state)
    return total


# ---------------------------------------------------------------------------
# Preference-aware reward (Fair mode)
# ---------------------------------------------------------------------------

def preference_aware_reward(group: int, action: int, state: dict) -> float:
    """
    Compute the reward using ONLY the hidden preference group.

    No cross-group interference. Each agent is evaluated against
    its own group's true preferences.

    Returns:
        Identical to group_reward(group, action, state)
    """
    return group_reward(group, action, state)


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def reward_breakdown(
    action: int,
    state: dict,
    weights: list = None,
) -> dict:
    """
    Return per-group rewards and the aggregated reward for diagnostics.

    Accepts optional `weights` to reflect env-specific effective_weights.
    """
    if weights is None:
        weights = list(GROUP_WEIGHTS.values())

    breakdown = {}
    for g in range(len(GROUP_WEIGHTS)):
        r = group_reward(g, action, state)
        w = weights[g]
        breakdown[GROUP_NAMES[g]] = {
            "reward": r,
            "weight": round(w, 4),
            "weighted_contribution": round(w * r, 4),
        }
    breakdown["aggregated"] = aggregated_reward(action, state, weights=weights)
    return breakdown
