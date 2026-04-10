"""
env/tasks.py — Task definitions and deterministic graders for PreferenceAggregationEnv.

Three tasks with increasing difficulty:

  Task 1 — majority_dominance (Easy)
    Group distribution: [0.70, 0.20, 0.10]
    Majority (Concise) holds 70% weight. Aggregated reward clearly favors response_a.
    Grader: fraction of steps satisfying the majority (Concise) group.

  Task 2 — mixed_preferences (Medium)
    Group distribution: [0.40, 0.35, 0.25]
    No supermajority. Groups 1+2 combined (0.60) slightly outweigh group 0.
    Aggregated reward signal is weaker and can favor either response.
    Grader: mean accuracy across ALL groups (balanced evaluation).

  Task 3 — fairness_collapse (Hard)
    Group distribution: [0.333, 0.333, 0.334]
    Perfectly balanced. Any deterministic policy collapses fairness gap to 100%.
    Only a randomized or preference-aware agent can achieve low fairness gap.
    Grader: 1.0 - fairness_gap. Deterministic agents score near 0.0.
"""

from typing import List, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Embedded dataset (prompt-response pairs; expanded with tie-break cases)
# response_a = concise/simple, response_b = detailed/technical
# Some rows use similar lengths so Concise vs Detailed ties; Technical density
# then decides preference — reduces naive "always pick the shorter" heuristics.
# ---------------------------------------------------------------------------

DATASET: List[Dict[str, str]] = [
    {
        "prompt": "What is a neural network?",
        "response_a": "A neural network learns from data using layered computations.",
        "response_b": (
            "A neural network is a parameterized computational graph of hierarchically stacked layers, "
            "each performing weighted linear transformations followed by non-linear activation functions. "
            "Training uses backpropagation and stochastic gradient descent to minimize a differentiable "
            "loss function across millions of parameters."
        ),
    },
    {
        "prompt": "Explain reinforcement learning.",
        "response_a": "An agent learns to take actions that maximize cumulative reward over time.",
        "response_b": (
            "Reinforcement learning formalizes sequential decision-making as a Markov Decision Process "
            "(S, A, P, R, gamma). The agent policy pi: S->A interacts with the environment by observing "
            "states, selecting actions, and receiving scalar rewards. The objective maximizes expected "
            "discounted return G_t = sum(gamma^k * r_{t+k}), estimated via temporal difference learning "
            "or direct policy gradient optimization."
        ),
    },
    {
        "prompt": "What is gradient descent?",
        "response_a": "Gradient descent updates model parameters in the direction that reduces the loss.",
        "response_b": (
            "Gradient descent is an iterative first-order optimization algorithm that minimizes a "
            "differentiable objective L(theta) by updating: theta_{t+1} = theta_t - eta * grad(L). "
            "Stochastic variants approximate the true gradient using mini-batches. Adaptive methods "
            "like Adam and RMSProp address non-stationary gradient magnitudes for improved convergence."
        ),
    },
    {
        "prompt": "What is RLHF?",
        "response_a": "RLHF trains language models using human feedback as a reward signal.",
        "response_b": (
            "Reinforcement Learning from Human Feedback (RLHF) is a three-stage alignment paradigm: "
            "supervised fine-tuning on demonstrations, reward model training via Bradley-Terry pairwise "
            "preference modeling, and policy optimization using PPO with a KL-divergence penalty against "
            "the reference policy to prevent reward hacking and distributional collapse."
        ),
    },
    {
        "prompt": "What is attention in transformers?",
        "response_a": "Attention lets each token focus on relevant parts of the input when generating output.",
        "response_b": (
            "Transformer attention computes Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, where Q, K, V "
            "are linear projections of input embeddings scaled by sqrt(d_k) to prevent softmax saturation. "
            "Multi-head attention runs h parallel attention functions and concatenates outputs, enabling "
            "joint attendance across multiple representation subspaces."
        ),
    },
    {
        "prompt": "How does a language model generate text?",
        "response_a": "A language model predicts the next word given previous context, one token at a time.",
        "response_b": (
            "Autoregressive language models factorize joint probability as P(x) = prod P(x_t | x_{<t}). "
            "At each decoding step, the model computes a logit distribution over vocabulary via softmax, "
            "then samples using greedy decoding, top-k sampling, nucleus (top-p) sampling, or "
            "temperature scaling. KV-caching amortizes quadratic attention cost across decoding steps."
        ),
    },
    {
        "prompt": "What is overfitting?",
        "response_a": "Overfitting is when a model memorizes training data and fails on new examples.",
        "response_b": (
            "Overfitting occurs when model capacity exceeds what the training distribution requires, "
            "resulting in low empirical risk but high generalization error. Mitigations include L1/L2 "
            "regularization, dropout, early stopping on held-out validation loss, data augmentation, "
            "and complexity constraints via pruning or architectural bounds."
        ),
    },
    {
        "prompt": "What is a reward model in RL?",
        "response_a": "A reward model scores agent actions to guide learning.",
        "response_b": (
            "A reward model r_phi: S x A -> R is a parametric function trained via maximum likelihood "
            "on pairwise preference data under the Bradley-Terry model: P(y1 > y2) = sigma(r(y1) - r(y2)). "
            "Its out-of-distribution generalization critically determines policy optimization stability "
            "and susceptibility to reward hacking."
        ),
    },
    {
        "prompt": "What is transfer learning?",
        "response_a": "Transfer learning reuses knowledge from one task to improve performance on another.",
        "response_b": (
            "Transfer learning leverages representations learned on large-scale source data to initialize "
            "target task optimization. Techniques include full fine-tuning, linear probing, and "
            "parameter-efficient methods (LoRA, prefix tuning, adapters). Transfer effectiveness depends "
            "on source-target domain gap and label structure alignment."
        ),
    },
    {
        "prompt": "What is a Markov Decision Process?",
        "response_a": "An MDP models sequential decisions where outcomes depend on the current state and action.",
        "response_b": (
            "An MDP is defined by (S, A, P, R, gamma, rho_0): state space S, action space A, "
            "stochastic transition kernel P: S x A -> Delta(S), reward function R: S x A -> R, "
            "discount gamma in [0,1), and initial state distribution. The Markov property asserts "
            "P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_0..s_t, a_0..a_t), enabling Bellman decomposition."
        ),
    },
    {
        "prompt": "What is entropy in information theory?",
        "response_a": "Entropy measures the uncertainty or unpredictability of a random variable.",
        "response_b": (
            "Shannon entropy H(X) = -sum p(x) log p(x) quantifies expected information content. "
            "Cross-entropy H(p,q) = -sum p(x) log q(x) measures encoding cost under mismatched model q. "
            "KL divergence D_KL(p||q) = H(p,q) - H(p) quantifies distributional discrepancy and underpins "
            "variational inference and KL-penalized policy optimization in RLHF."
        ),
    },
    {
        "prompt": "What is model alignment?",
        "response_a": "Alignment ensures AI systems behave in accordance with human values and intentions.",
        "response_b": (
            "Alignment is the research program ensuring AI behavior conforms to intended objectives across "
            "all deployment conditions. Failure modes include reward hacking, distributional shift, "
            "deceptive alignment, and preference aggregation failure under heterogeneous annotator "
            "populations. Approaches include RLHF, Constitutional AI, debate, and interpretability-based "
            "monitoring."
        ),
    },
    {
        "prompt": "What is a policy gradient method?",
        "response_a": "Policy gradient methods optimize an agent's strategy by following the reward gradient.",
        "response_b": (
            "Policy gradient methods estimate and ascend the gradient of expected return: "
            "grad J(theta) = E_pi[grad log pi_theta(a|s) * Q^pi(s,a)]. REINFORCE uses Monte Carlo "
            "return estimates. Actor-critic methods replace Q^pi with advantage A^pi = Q^pi - V^pi "
            "to reduce variance. PPO clips the surrogate objective ratio to stabilize updates."
        ),
    },
    {
        "prompt": "What is tokenization in NLP?",
        "response_a": "Tokenization splits text into smaller units like words or subwords for model input.",
        "response_b": (
            "Tokenization maps raw strings to integer indices from a fixed vocabulary. Modern LLMs use "
            "subword algorithms: BPE iteratively merges frequent character n-grams; WordPiece maximizes "
            "language model likelihood. Vocabulary size and merge strategy directly affect sequence length, "
            "OOV rate, and morphological representation of complex languages and code."
        ),
    },
    {
        "prompt": "What is a vector database?",
        "response_a": "A vector database stores embeddings and retrieves them by similarity.",
        "response_b": (
            "Vector databases support approximate nearest-neighbor (ANN) retrieval over high-dimensional "
            "embedding vectors using HNSW graphs or IVF with product quantization to reduce retrieval "
            "complexity from O(n*d) to O(log n). They are integral to retrieval-augmented generation "
            "pipelines for semantic search and external memory injection during LLM inference."
        ),
    },
    # Tie-break rows: similar lengths; technical vocabulary density breaks ties.
    {
        "prompt": "What is batch normalization?",
        "response_a": "Batch normalization stabilizes training by normalizing layer inputs across a batch.",
        "response_b": (
            "Batch normalization standardizes activations using batch mean and variance, then applies "
            "learnable scale and shift; it reduces internal covariate shift and allows higher learning rates."
        ),
    },
    {
        "prompt": "What is dropout?",
        "response_a": "Dropout randomly disables neurons during training to reduce overfitting.",
        "response_b": (
            "Dropout samples a sub-network each step by zeroing activations with probability p; at inference, "
            "weights are scaled by (1-p) to match expected activations under the training distribution."
        ),
    },
    {
        "prompt": "What is a learning rate schedule?",
        "response_a": "A schedule changes the learning rate over training time to improve convergence.",
        "response_b": (
            "Learning rate schedules include step decay, cosine annealing, and warmup; they balance fast early "
            "progress with stable late-stage optimization near minima."
        ),
    },
    {
        "prompt": "What is data augmentation?",
        "response_a": "Data augmentation creates modified training examples to improve generalization.",
        "response_b": (
            "Augmentation applies label-preserving transforms (crop, rotate, paraphrase) to expand effective "
            "dataset diversity and reduce overfitting under limited supervision."
        ),
    },
]


# ---------------------------------------------------------------------------
# Grader functions (deterministic — only use episode_history)
# ---------------------------------------------------------------------------

def grade_majority_dominance(episode_history: List[Dict[str, Any]]) -> float:
    """
    Task 1 — Easy grader.

    Measures: fraction of steps where the Concise group (index 0) was satisfied.
    The majority group (70% weight) should dominate; this score quantifies
    how well the agent learned to serve the majority.

    Score range: [0.0, 1.0]
    """
    if not episode_history:
        return 0.0
    satisfied = sum(
        1 for step in episode_history
        if step["group_rewards"][0] >= 1.0  # group 0 (Concise) satisfied
    )
    return satisfied / len(episode_history)


def grade_mixed_preferences(episode_history: List[Dict[str, Any]]) -> float:
    """
    Task 2 — Medium grader.

    Measures: mean accuracy averaged across ALL three preference groups.
    Rewards agents that partially satisfy multiple groups, not just the
    dominant one.

    Score range: [0.0, 1.0]
    """
    if not episode_history:
        return 0.0
    n_groups = 3
    group_totals = [0.0] * n_groups
    for step in episode_history:
        for g in range(n_groups):
            group_totals[g] += step["group_rewards"][g]
    group_accs = [t / len(episode_history) for t in group_totals]
    return sum(group_accs) / n_groups


def grade_fairness_collapse(episode_history: List[Dict[str, Any]]) -> float:
    """
    Task 3 — Hard grader.

    Measures: 1.0 - fairness_gap across preference groups.

    With perfectly balanced groups [0.333, 0.333, 0.334], any deterministic
    agent satisfies exactly one response preference at all times, causing
    fairness_gap = 1.0 -> score = 0.0.

    Only a randomized or group-aware policy can achieve low fairness gap.
    This directly exposes the structural failure of preference aggregation.

    Score range: [0.0, 1.0]
    Perfect agent: 1.0 | Any deterministic agent: ~0.0
    """
    if not episode_history:
        return 0.0
    n_groups = 3
    group_totals = [0.0] * n_groups
    for step in episode_history:
        for g in range(n_groups):
            group_totals[g] += step["group_rewards"][g]
    group_accs = [t / len(episode_history) for t in group_totals]
    fairness_gap = max(group_accs) - min(group_accs)
    return max(0.0, 1.0 - fairness_gap)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "majority_dominance": {
        "id":                "majority_dominance",
        "name":              "Task 1: Majority Dominance",
        "difficulty":        "easy",
        "description": (
            "Group 0 (Concise) holds 70% population weight. The aggregated RLHF reward "
            "clearly favors response_a at every step. An optimal agent learns the majority "
            "preference and achieves ~0.70 score. Minority groups are systematically ignored."
        ),
        "group_distribution": [0.70, 0.20, 0.10],
        "max_steps":         10,
        "context": (
            "Select the response that best serves a general user audience. "
            "Clarity and brevity are valued."
        ),
        "grader":            grade_majority_dominance,
        "expected_baseline": 0.70,
    },
    "mixed_preferences": {
        "id":                "mixed_preferences",
        "name":              "Task 2: Mixed Preferences",
        "difficulty":        "medium",
        "description": (
            "Group distribution [0.40, 0.35, 0.25]. Groups 1+2 combined (0.60) slightly "
            "outweigh Group 0 (0.40), shifting the aggregated reward signal toward response_b. "
            "The agent must navigate conflicting preference signals. "
            "Grader rewards balanced satisfaction across all groups."
        ),
        "group_distribution": [0.40, 0.35, 0.25],
        "max_steps":         10,
        "context": (
            "Select the response that best serves users with varying expertise. "
            "Consider both depth and accessibility."
        ),
        "grader":            grade_mixed_preferences,
        "expected_baseline": 0.50,
    },
    "fairness_collapse": {
        "id":                "fairness_collapse",
        "name":              "Task 3: Fairness Collapse",
        "difficulty":        "hard",
        "description": (
            "Perfectly balanced groups [0.333, 0.333, 0.334]. Groups 1+2 combined prefer "
            "response_b; Group 0 prefers response_a. Any deterministic policy satisfies "
            "at most 2 out of 3 groups -> fairness_gap = 1.0 -> grader score = 0.0. "
            "This exposes the structural impossibility of fair aggregation without group identity."
        ),
        "group_distribution": [0.333, 0.333, 0.334],
        "max_steps":         10,
        "context": (
            "Select the response that best serves a diverse user base with equal representation "
            "of concise, detailed, and technical preference profiles."
        ),
        "grader":            grade_fairness_collapse,
        "expected_baseline": 0.10,
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]
