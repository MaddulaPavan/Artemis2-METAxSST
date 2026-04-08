"""
data.py — Synthetic dataset for PreferenceAggregationEnv.

Each entry has a prompt, response_A (concise/simple), and response_B (detailed/technical).
Group preferences are derived dynamically by the reward module based on:
  - Length (concise vs detailed)
  - Technical vocabulary density (simple vs technical)
"""

DATASET = [
    {
        "id": 0,
        "prompt": "What is a neural network?",
        "response_A": "A neural network is a system of connected nodes that learns patterns from data.",
        "response_B": (
            "A neural network is a parameterized computational graph composed of hierarchically stacked layers "
            "of artificial neurons. Each neuron computes a weighted linear combination of its inputs, followed "
            "by a non-linear activation function such as ReLU or sigmoid. The network is trained via "
            "backpropagation using stochastic gradient descent to minimize a differentiable loss function, "
            "allowing it to learn complex, distributed representations across millions of parameters."
        ),
    },
    {
        "id": 1,
        "prompt": "Explain how reinforcement learning works.",
        "response_A": "An agent takes actions in an environment to maximize cumulative rewards over time.",
        "response_B": (
            "Reinforcement learning formalizes sequential decision-making as a Markov Decision Process (MDP) "
            "defined by a tuple (S, A, P, R, γ). An agent π: S → A interacts with an environment by observing "
            "states s ∈ S, selecting actions a ∈ A, and receiving scalar reward signals r ∈ R. The agent's "
            "objective is to learn an optimal policy π* that maximizes the expected discounted cumulative "
            "return G_t = Σ γ^k r_{t+k}. This is typically achieved by estimating value functions via "
            "temporal difference learning or by directly optimizing the policy gradient."
        ),
    },
    {
        "id": 2,
        "prompt": "What is gradient descent?",
        "response_A": "Gradient descent updates model parameters in the direction that reduces the loss.",
        "response_B": (
            "Gradient descent is an iterative first-order optimization algorithm that minimizes a differentiable "
            "objective function L(θ) by updating parameters θ in the direction of the negative gradient: "
            "θ_{t+1} = θ_t − η∇L(θ_t), where η is the learning rate. Stochastic variants (SGD, Adam) "
            "approximate the true gradient using mini-batches to improve computational efficiency. "
            "Convergence guarantees depend on the Lipschitz continuity and convexity of the loss surface, "
            "with adaptive methods like AdaGrad and RMSProp addressing non-stationary gradient magnitudes."
        ),
    },
    {
        "id": 3,
        "prompt": "What is RLHF?",
        "response_A": "RLHF trains language models using human feedback as a reward signal.",
        "response_B": (
            "Reinforcement Learning from Human Feedback (RLHF) is a paradigm for aligning language model "
            "outputs with human preferences. It involves three stages: supervised fine-tuning of a base model "
            "on demonstration data, training a reward model via Bradley-Terry modeling on pairwise human "
            "preference annotations, and optimizing the policy using Proximal Policy Optimization (PPO) "
            "against the learned reward function, with a KL-divergence penalty to prevent reward hacking "
            "and distributional collapse."
        ),
    },
    {
        "id": 4,
        "prompt": "What is attention in transformers?",
        "response_A": "Attention lets each token focus on relevant other tokens when generating output.",
        "response_B": (
            "The transformer attention mechanism computes a weighted combination of value vectors using "
            "compatibility scores between query and key representations. Formally, "
            "Attention(Q, K, V) = softmax(QK^T / √d_k)V, where Q, K, V ∈ ℝ^{n×d_k} are linear projections "
            "of the input embeddings, and scaling by √d_k prevents softmax saturation. Multi-head attention "
            "runs h parallel attention functions and concatenates their outputs, enabling the model to "
            "jointly attend to information from different representation subspaces."
        ),
    },
    {
        "id": 5,
        "prompt": "How does a language model generate text?",
        "response_A": "A language model predicts the next word given previous context, one token at a time.",
        "response_B": (
            "Autoregressive language models factorize the joint probability of a sequence as a product of "
            "conditional distributions: P(x) = Π P(x_t | x_{<t}). At each decoding step, the model computes "
            "a logit distribution over the vocabulary via a softmax over the final hidden state, then samples "
            "a token using strategies such as greedy decoding, top-k sampling, nucleus (top-p) sampling, or "
            "temperature-scaled distributions. Modern large language models use KV-caching to amortize "
            "the quadratic attention cost across decoding steps."
        ),
    },
    {
        "id": 6,
        "prompt": "What is overfitting in machine learning?",
        "response_A": "Overfitting is when a model memorizes training data and performs poorly on new data.",
        "response_B": (
            "Overfitting occurs when a model's hypothesis class is sufficiently expressive that it fits the "
            "noise in the training distribution, resulting in low empirical risk but high generalization error. "
            "Formally, this corresponds to a large gap between training loss L_train and population loss L_pop. "
            "Mitigation strategies include regularization (L1/L2 penalties, dropout), early stopping based on "
            "held-out validation performance, data augmentation to increase effective training set size, and "
            "reducing model complexity via pruning or architecture constraints."
        ),
    },
    {
        "id": 7,
        "prompt": "What is a reward model in RL?",
        "response_A": "A reward model assigns a score to an action to guide agent learning.",
        "response_B": (
            "A reward model r_φ: S × A → ℝ is a parametric function that maps state-action pairs to scalar "
            "reward signals, serving as a proxy for ground-truth environmental feedback. In the context of "
            "RLHF, reward models are trained via maximum likelihood estimation on pairwise preference data "
            "using the Bradley-Terry model: P(y_1 ≻ y_2) = σ(r_φ(y_1) − r_φ(y_2)). The reward model's "
            "capacity to generalize to out-of-distribution inputs critically determines the stability of "
            "downstream policy optimization."
        ),
    },
    {
        "id": 8,
        "prompt": "What is transfer learning?",
        "response_A": "Transfer learning uses knowledge from one task to improve performance on another.",
        "response_B": (
            "Transfer learning leverages representations learned by a source model on a large-scale dataset "
            "to accelerate optimization on a target task with limited supervision. Techniques include full "
            "fine-tuning (updating all parameters), linear probing (freezing the backbone and training only "
            "a head), and parameter-efficient methods such as LoRA, prefix tuning, and adapter modules that "
            "inject trainable parameters into frozen layers. The effectiveness of transfer depends on the "
            "domain gap between source and target distributions and the statistical similarity of their "
            "label structures."
        ),
    },
    {
        "id": 9,
        "prompt": "What is a Markov Decision Process?",
        "response_A": "An MDP models decisions where outcomes depend on the current state and chosen action.",
        "response_B": (
            "A Markov Decision Process is a mathematical framework for sequential decision-making defined "
            "by the tuple (S, A, P, R, γ, ρ_0), where S is the state space, A is the action space, "
            "P: S × A → Δ(S) is the transition kernel, R: S × A → ℝ is the reward function, γ ∈ [0,1) "
            "is the discount factor, and ρ_0 is the initial state distribution. The Markov property asserts "
            "P(s_{t+1} | s_t, a_t, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t), enabling dynamic programming "
            "solutions via Bellman equations."
        ),
    },
    {
        "id": 10,
        "prompt": "What is a convolutional neural network?",
        "response_A": "A CNN detects spatial patterns in images using sliding filters.",
        "response_B": (
            "A convolutional neural network (CNN) applies learned filter banks via discrete cross-correlation "
            "over spatially structured inputs, exploiting translational equivariance and local receptive fields "
            "to reduce parameter count relative to fully connected architectures. Feature maps are produced by "
            "convolving filters W ∈ ℝ^{k×k×c_in×c_out} over input tensors, followed by non-linear activation "
            "and pooling operations that progressively increase semantic abstraction while reducing spatial "
            "resolution. Residual connections mitigate vanishing gradients in deep CNN architectures."
        ),
    },
    {
        "id": 11,
        "prompt": "What is entropy in information theory?",
        "response_A": "Entropy measures the uncertainty or randomness in a probability distribution.",
        "response_B": (
            "Shannon entropy H(X) = −Σ p(x) log p(x) quantifies the average information content of a "
            "discrete random variable X with probability mass function p. It is maximized by the uniform "
            "distribution and minimized (to zero) by a degenerate point mass. Cross-entropy H(p, q) = "
            "−Σ p(x) log q(x) measures the expected code length when encoding distribution p using a code "
            "optimized for q. The KL divergence D_KL(p‖q) = H(p, q) − H(p) quantifies distributional "
            "discrepancy and serves as the basis for variational inference and policy regularization in RL."
        ),
    },
    {
        "id": 12,
        "prompt": "What is a latent space in generative models?",
        "response_A": "A latent space is a compressed internal representation learned by a model.",
        "response_B": (
            "A latent space Z is a lower-dimensional manifold onto which an encoder network maps high-dimensional "
            "observations x ∈ X. In variational autoencoders (VAEs), the encoder parameterizes a posterior "
            "q_φ(z|x) ≈ p(z|x), typically as a diagonal Gaussian, and training maximizes the evidence lower "
            "bound (ELBO): L = E_q[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z)). The structure of the latent "
            "manifold determines the quality of interpolation, disentanglement, and downstream generative "
            "fidelity when decoding sampled latent codes z ~ q_φ."
        ),
    },
    {
        "id": 13,
        "prompt": "What is the bias-variance tradeoff?",
        "response_A": "More complex models have lower bias but higher variance, and vice versa.",
        "response_B": (
            "The bias-variance decomposition expresses expected generalization error as: "
            "E[(y − ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ², where Bias(ŷ) = E[ŷ] − f(x) measures systematic "
            "error from incorrect inductive bias, Var(ŷ) = E[(ŷ − E[ŷ])²] measures sensitivity to training "
            "set variation, and σ² is irreducible noise. High-capacity models minimize bias by fitting the "
            "training distribution but amplify variance, while constrained models exhibit the reverse. "
            "Ensemble methods (bagging, boosting) and regularization techniques explicitly manage this tradeoff."
        ),
    },
    {
        "id": 14,
        "prompt": "What is fine-tuning a language model?",
        "response_A": "Fine-tuning adapts a pretrained model to a specific task using new data.",
        "response_B": (
            "Fine-tuning is a transfer learning procedure in which a pretrained language model L_θ is further "
            "optimized on task-specific data D = {(x_i, y_i)} by minimizing a supervised loss such as cross-entropy. "
            "Full fine-tuning updates all parameters θ, which risks catastrophic forgetting and is "
            "computationally expensive at scale. Parameter-efficient fine-tuning (PEFT) methods such as LoRA "
            "decompose weight updates as low-rank perturbations ΔW = AB (A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, r ≪ d), "
            "substantially reducing trainable parameter count while preserving performance."
        ),
    },
    {
        "id": 15,
        "prompt": "What is Monte Carlo tree search?",
        "response_A": "MCTS explores possible future moves and selects the one with the best expected outcome.",
        "response_B": (
            "Monte Carlo Tree Search (MCTS) is an anytime best-first tree search algorithm that estimates "
            "action values through randomized simulation. Each iteration involves four phases: selection "
            "(traversing the tree using a UCB1 or PUCT bandit criterion to balance exploration and exploitation), "
            "expansion (adding a new child node), simulation (rolling out to terminal state using a rollout "
            "policy), and backpropagation (updating visit counts N(s,a) and value estimates Q(s,a) along the "
            "traversal path). AlphaGo and subsequent systems replace random rollouts with learned value "
            "networks for superior state evaluation."
        ),
    },
    {
        "id": 16,
        "prompt": "What is tokenization in NLP?",
        "response_A": "Tokenization splits text into smaller units like words or subwords for model input.",
        "response_B": (
            "Tokenization is the process of mapping a raw string to a discrete sequence of integer indices "
            "from a fixed vocabulary V. Modern large language models employ subword tokenization algorithms "
            "such as Byte Pair Encoding (BPE) or WordPiece, which iteratively merge frequent character "
            "n-grams to construct a vocabulary that balances coverage and compression efficiency. The "
            "tokenizer's vocabulary size and merge strategy directly affect sequence length, out-of-vocabulary "
            "rate, and the model's capacity to represent morphologically complex languages and code."
        ),
    },
    {
        "id": 17,
        "prompt": "What is a policy gradient method?",
        "response_A": "Policy gradient methods optimize a policy directly by following the gradient of expected reward.",
        "response_B": (
            "Policy gradient methods optimize a parameterized stochastic policy π_θ by estimating and ascending "
            "the gradient of the expected return: ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) · Q^π(s,a)]. The REINFORCE "
            "algorithm computes Monte Carlo estimates of this gradient, but suffers from high variance. "
            "Actor-critic methods reduce variance by replacing Q^π with an advantage function A^π(s,a) = "
            "Q^π(s,a) − V^π(s), estimated by a separately trained critic. PPO clips the surrogate objective "
            "to constrain the policy update step size, improving stability."
        ),
    },
    {
        "id": 18,
        "prompt": "What is a vector database?",
        "response_A": "A vector database stores and retrieves data using similarity between embeddings.",
        "response_B": (
            "A vector database is a specialized data management system designed for approximate nearest neighbor "
            "(ANN) retrieval over high-dimensional embedding vectors. Indexing structures such as HNSW "
            "(Hierarchical Navigable Small World) graphs and IVF (Inverted File Index) with product quantization "
            "reduce retrieval complexity from O(n·d) linear scan to O(log n) or O(√n) at the cost of recall "
            "approximation error. Vector databases are integral to retrieval-augmented generation (RAG) "
            "pipelines, enabling semantic search, document retrieval, and memory injection for LLM inference."
        ),
    },
    {
        "id": 19,
        "prompt": "What is model alignment?",
        "response_A": "Model alignment ensures AI systems behave in ways that match human values and intentions.",
        "response_B": (
            "Model alignment is the research program concerned with ensuring that AI system behavior conforms "
            "to intended objectives specified by human principals across the full distribution of deployment "
            "conditions. Alignment failures manifest as reward hacking (optimizing a proxy objective in "
            "unintended ways), distributional shift, deceptive alignment (appearing aligned during training "
            "while pursuing misaligned goals at deployment), and preference aggregation failure when policies "
            "are trained on heterogeneous annotator populations. Current alignment approaches include RLHF, "
            "Constitutional AI, debate, scalable oversight, and interpretability-based monitoring."
        ),
    },
]
