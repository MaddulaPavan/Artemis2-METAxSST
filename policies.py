"""
policies.py — Policy definitions for PreferenceAggregationEnv.

All policies share the same base signature:
    policy(obs: dict) -> int  (0 or 1)

Oracle policy has extended signature:
    oracle_policy(obs: dict, group: int) -> int

Policies implemented:
  1. random_policy           — uniform random baseline
  2. greedy_aggregated_policy — simulates standard RLHF-trained agent
  3. group_robust_policy      — minimax fairness heuristic (original)
  4. oracle_policy            — perfect per-group knowledge (upper bound)

Design note on group_robust_policy
------------------------------------
Without knowing the hidden group, a deterministic policy must commit to one
action — which always perfectly satisfies some groups at the expense of others.

The group-robust approach uses equal-weight voting across all groups:
  - Compute each group's preferred action independently
  - Assign equal vote weight (ignoring population imbalance)
  - Select action stochastically proportional to vote counts

This is a minimax-fairness heuristic implementing the Rawlsian maximin
criterion: maximize the minimum expected accuracy across groups.

It achieves a fairness gap strictly less than the greedy policy
while requiring no group identity information.
"""

import random
from typing import Dict

from reward import group_reward, aggregated_reward

# Number of preference groups
N_GROUPS = 3


# ---------------------------------------------------------------------------
# 1. Random Policy (Baseline)
# ---------------------------------------------------------------------------

def random_policy(obs: Dict[str, str]) -> int:
    """
    Uniform random baseline.

    Expected per-group accuracy: ~50% (independent of group).
    Fairness gap: ~0% (fair but useless).
    """
    return random.randint(0, 1)


# ---------------------------------------------------------------------------
# 2. Greedy Aggregated Policy (Simulates Standard RLHF)
# ---------------------------------------------------------------------------

def greedy_aggregated_policy(obs: Dict[str, str]) -> int:
    """
    Simulates the behavior of a standard RLHF-trained agent.

    Selects the action that maximizes the aggregated reward signal
    (using the default GROUP_WEIGHTS from reward.py).

    This is what an ideal optimizer trained on the pooled reward model
    would converge to. Because Group 0 (Concise) holds 60% of the weight,
    this policy learns the majority group's preference, systematically
    ignoring minority groups.

    Expected behavior with standard [0.60, 0.30, 0.10] distribution:
      - Group 0 accuracy: ~100% (always satisfies majority)
      - Group 1 accuracy: ~0%   (always wrong)
      - Group 2 accuracy: ~0%   (always wrong)
      - Fairness gap: ~100%

    Note: With different group_distributions the majority may shift,
    but the fairness gap remains ~100% for any non-uniform distribution.
    """
    r_a = aggregated_reward(0, obs)
    r_b = aggregated_reward(1, obs)
    return 0 if r_a >= r_b else 1


# ---------------------------------------------------------------------------
# 3. Group-Robust Policy (Original — Minimax Fairness Heuristic)
# ---------------------------------------------------------------------------

def group_robust_policy(obs: Dict[str, str]) -> int:
    """
    Minimax fairness heuristic — equalizes expected accuracy across groups.

    Implements a probabilistic equal-vote strategy:
      (1) Compute the preferred action for each group independently
      (2) Assign equal vote weight to all groups (no population bias)
      (3) Sample an action proportional to the group vote counts

    This decouples the decision from population imbalance and directs
    the policy toward the Rawlsian social optimum: maximizing the
    minimum expected accuracy across all groups.

    With 3 groups where 1 prefers A and 2 prefer B:
      - P(pick A) = 1/3,  P(pick B) = 2/3
      - Group 0 accuracy: ~33%,  Groups 1&2: ~67%
      - Fairness gap: ~33%

    Key insight: This achieves strictly better fairness than the greedy
    policy (100% gap) without any knowledge of the hidden preference group.
    It represents the best achievable fairness via heuristic in this setting.
    """
    # Compute each group's preferred action (equal weight, ignoring population)
    votes_for_A = sum(
        1 for g in range(N_GROUPS)
        if group_reward(g, 0, obs) >= group_reward(g, 1, obs)
    )
    votes_for_B = N_GROUPS - votes_for_A

    # Stochastic selection proportional to vote ratio
    prob_A = votes_for_A / N_GROUPS
    return 0 if random.random() < prob_A else 1


# ---------------------------------------------------------------------------
# 4. Oracle Policy (Upper Bound — Requires Hidden Group)
# ---------------------------------------------------------------------------

def oracle_policy(obs: Dict[str, str], group: int) -> int:
    """
    Ideal fair policy with perfect knowledge of the hidden preference group.

    Represents the theoretical optimum: a policy that knows each user's
    preference group and selects accordingly.

    This is unachievable in practice (groups are hidden), but serves as:
      - The upper bound oracle for fairness benchmarking
      - The target that preference-aware training systems approximate
      - Proof that the environment CAN achieve 0% fairness gap

    Expected behavior:
      - All groups: ~100% accuracy
      - Fairness gap: ~0%
    """
    r_a = group_reward(group, 0, obs)
    r_b = group_reward(group, 1, obs)
    return 0 if r_a >= r_b else 1


# ---------------------------------------------------------------------------
# Policy registry (for use in experiment runner)
# ---------------------------------------------------------------------------

# Standard policies (obs -> action)
STANDARD_POLICIES = {
    "random":           random_policy,
    "greedy_aggregated": greedy_aggregated_policy,
    "group_robust":     group_robust_policy,
}

# Oracle requires hidden group — handled separately in experiment runner
ORACLE_POLICY = oracle_policy
