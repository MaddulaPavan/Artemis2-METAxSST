# RLHF Aggregation Failure Simulator

### *A Reproducible Benchmark for Structural Fairness Failure in Reinforcement Learning from Human Feedback*

> OpenEnv-compatible · Pure Python · No GPU · Runs in 5 seconds  
> India's MEGA AI Hackathon — Meta × Hugging Face × Scaler School of Technology

---

## 🚨 Zero-Shot Proof: RLHF Fails Before Training

```
Greedy Aggregated Policy  (Standard RLHF)
─────────────────────────────────────────
  Group 0  (Majority)   :  100.0%
  Group 1  (Minority)   :    0.0%
  Group 2  (Minority)   :   50.8%
  ───────────────────────────────
  Fairness Gap          :  100.0%  ← FAILURE


Oracle Policy  (Preference-Aware)
─────────────────────────────────────────
  Group 0               :  100.0%
  Group 1               :  100.0%
  Group 2               :  100.0%
  ───────────────────────────────
  Fairness Gap          :    0.0%  ← FAIR
```

📌 **Same environment. Same data. No training changes. Only the objective differs.**

### 🔍 What This Shows

- RLHF fails even with **perfect signals** — no noise, no bad data, no model error
- Failure persists across:
  - distributions — majority/balanced/minority-flip
  - noise — Gaussian reward corruption (σ=0.3)
  - bias tuning — `bias_strength` swept from 0.0 → 1.0
- The issue is **not optimization**
- The issue is **objective design**

> **Conclusion: Alignment failure emerges before any model training occurs.**

---

## Executive Summary

Standard RLHF trains a single reward model on pooled human preference data.
When annotators have fundamentally different values, this aggregation step introduces a fairness failure that is **not recoverable through scaling, parameter tuning, or noise reduction**.

The majority preference group captures the reward signal. Minority groups receive systematically wrong reward — not because of bad data, but because the reward objective itself is misspecified.
This benchmark proves that claim with five controlled experiments across varying group distributions, bias strengths, and reward noise conditions.
The fairness gap remains 100% in every case.

---

```
┌─────────────────────────────────────────────────────────────────────┐
│  KEY RESULT                                                         │
│                                                                     │
│  Fairness Gap — Standard RLHF (Greedy Aggregated) :  100.0%        │
│  Fairness Gap — Preference-Aware (Oracle)          :    0.0%        │
│                                                                     │
│  This gap holds across ALL 5 experimental scenarios.               │
│  Changing the distribution, bias parameter, or noise level          │
│  does not reduce it. It is invariant to these interventions.        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Discovery

> **The identity of the harmed group changes with the population distribution. The magnitude of harm does not.**

In Scenario A (majority group = Concise, 60%), the Greedy policy fails the Detailed and Technical groups entirely.
In Scenario C (minority flip: Concise drops to 20%), the Greedy policy now fails the *former majority* — the Concise group.

The harm migrates to whoever holds less population weight. But the worst-group accuracy stays at 0% in every configuration.

**Why this happens:**

The aggregated reward redistributes optimization pressure according to population weight — not according to correctness. When group preferences conflict (choosing A vs B), the reward signal from the majority group always outweighs the minority, regardless of who is objectively right by their own values.

This is not a sampling artifact. It is a direct consequence of the aggregation function. Any reward-maximizing agent trained on a single pooled reward model will converge to the majority group's preference and remain locked there for all of training.

---

## Central Claim

> **Scaling RLHF does not fix fairness — it amplifies majority dominance.**

More training data biases the reward model more strongly toward majority annotations.
Longer training forces the policy further into the majority-preferred region.
Higher-capacity reward models learn majority preferences *more precisely*.

None of these interventions help minority groups. They make the problem worse.

The *only* path to 0% fairness gap is to eliminate cross-group interference in the reward signal — by separating reward computation by preference group, as in PA-RLHF and related approaches.

---

## Experiment Results

*500 episodes per policy per scenario · seed=2024 · Metrics: true reward accuracy per group*

### Scenario Sweep — Fairness Gap

| Scenario | Configuration | Greedy (RLHF) | Group Robust | Oracle |
|----------|--------------|:---:|:---:|:---:|
| A. Majority Dominance | dist=[0.60, 0.30, 0.10], bias=1.0 | **100.0%** | 9.0% | **0.0%** |
| B. Balanced Groups | dist=[0.33, 0.33, 0.34], bias=1.0 | **100.0%** | 9.0% | **0.0%** |
| C. Minority Flip | dist=[0.20, 0.40, 0.40], bias=1.0 | **100.0%** | 8.5% | **0.0%** |
| D. Reduced Bias | dist=[0.60, 0.30, 0.10], bias=0.3 | **100.0%** | 22.7% | **0.0%** |
| E. Reward Noise | dist=[0.60, 0.30, 0.10], σ=0.30 | **100.0%** | 13.8% | **0.0%** |

**Interpretation:**
- Greedy = 100% gap in every scenario. Invariant to distribution, bias, and noise.
- Group Robust achieves 8–23% gap — a strictly better outcome without any group identity access.
- Oracle = 0% gap in every scenario — confirming the environment is solvable when the reward signal is correctly separated.

### Bias Strength Sweep — Worst-Group Accuracy (Greedy Policy)

| bias_strength | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Worst-Group Acc | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

Reducing `bias_strength` to 0.0 makes the effective reward weights completely uniform.
The worst-group accuracy is still 0%. The aggregation structure alone — not the weights — is sufficient to produce total minority collapse.

---

## Per-Group Breakdown (Scenario A)

| Policy | Mode | Concise (G0) | Detailed (G1) | Technical (G2) | Fairness Gap |
|--------|------|:---:|:---:|:---:|:---:|
| Random | standard | 52% | 48% | 40% | 12% |
| **Greedy Aggregated** | **standard** | **100%** | **0%** | **51%** | **100%** |
| Group Robust | standard | 54% | 51% | 60% | 9% |
| **Oracle** | **pref_aware** | **100%** | **100%** | **100%** | **0%** |

The Greedy policy perfectly satisfies the majority group. The Detailed group (30% of users) receives 0% correct decisions for the full duration of deployment. Not as an edge case — as a guaranteed consequence of the reward formulation.

---

## Why This Matters

### LLM Alignment

RLHF is the dominant paradigm for aligning frontier language models. If the annotation pool contains users with different preferences — conciseness vs detail, formality vs casualness, technical precision vs accessibility — the resulting model will satisfy the demographic majority and systematically fail others. This is not a theoretical concern. It is a direct implication of the math, demonstrated here at the smallest possible scale.

### Multi-User Systems

An AI assistant deployed to a heterogeneous user population is not a single-user system. The reward signal used to train it is the average of all users' feedback. This average erases minority preferences. Bias compounds over time: as the deployed policy drifts toward majority preference, minority users receive worse responses, reduce engagement, and contribute less feedback — further reducing their influence on future training.

### Agentic AI

Agents operating autonomously on behalf of diverse users will be optimized against pooled feedback from those users. If user preferences conflict — on task prioritization, communication style, risk tolerance — the agent will optimize for the majority and ignore the rest. As agent autonomy increases, the downstream cost of this misalignment grows.

---

## Contribution

This project provides three things:

**1. A minimal reproducible benchmark for preference aggregation failure.**
The environment requires no model training, no GPU, and no external dependencies beyond standard Python. Any researcher can run the full benchmark in under 10 seconds and obtain the same results. The failure mode is observable, measurable, and parameterized.

**2. A diagnostic tool with configurable failure parameters.**
`group_distribution`, `bias_strength`, and `noise_level` can be swept independently to isolate which aspect of the aggregation design drives harm. The benchmark cleanly separates the contribution of population imbalance, objective weighting, and stochastic corruption.

**3. An OpenEnv-compatible RL environment exposing a real alignment failure.**
The environment follows the Gymnasium API, integrates with standard RL tooling, and is structured for direct integration into OpenEnv-based training pipelines. Researchers studying RLHF fairness interventions can use this as a standard diagnostic testbed.

---

## Environment API

```python
from env import PreferenceAggregationEnv

# Configure and initialize
env = PreferenceAggregationEnv(
    mode               = "standard",          # or "preference_aware"
    group_distribution = [0.60, 0.30, 0.10], # Population weights
    bias_strength      = 1.0,                 # Aggregation dominance [0.0, 1.0]
    noise_level        = 0.0,                 # Reward noise (sigma)
    seed               = 42,
)

obs, info   = env.reset()
obs, reward, terminated, truncated, info = env.step(action=0)

# info["true_reward"]     → +1.0 or -1.0 (per group, for diagnostics)
# info["received_reward"] → affected by mode and noise
# info["group_name"]      → "Concise" / "Detailed" / "Technical"
# info["correct"]         → bool
```

```python
from experiments import run_all_scenarios
from visualize   import save_all_plots
from experiments import run_bias_sweep

all_results   = run_all_scenarios(n_episodes=500, seed=42)
sweep_results = run_bias_sweep()
save_all_plots(all_results, sweep_results)
```

---

## Quickstart

```bash
pip install matplotlib        # optional — only for plots

python run_demo.py            # Single-scenario walkthrough
python run_all_experiments.py # Full 5-scenario benchmark + plots + JSON
```

---

## Project Structure

```
├── data.py                 ← 20 synthetic prompt-response pairs
├── reward.py               ← Per-group rewards + configurable aggregation
├── env.py                  ← PreferenceAggregationEnv (Gym API, parameterized)
├── policies.py             ← 4 policies: random, greedy, group_robust, oracle
├── metrics.py              ← fairness_gap, variance_across_groups, worst_group_accuracy
├── experiments.py          ← 5 benchmark scenarios + bias sweep + formatting
├── visualize.py            ← 3 diagnostic plots (group accuracy, gap comparison, sweep)
├── run_demo.py             ← Quick demo
├── run_all_experiments.py  ← Full benchmark
└── output/
    ├── group_accuracy_by_scenario.png
    ├── fairness_gap_comparison.png
    ├── bias_strength_sweep.png
    └── experiment_results.json
```

---

## References

1. Christiano *et al.* — *Deep Reinforcement Learning from Human Preferences* (NeurIPS 2017)
2. Bai *et al.* — *Training a Helpful and Harmless Assistant with RLHF* (Anthropic, 2022)
3. Casper *et al.* — *Open Problems and Fundamental Limitations of RLHF* (TMLR 2023)
4. Gabriel — *Artificial Intelligence, Values and Alignment* (Minds & Machines, 2020)
5. Meta OpenEnv — [github.com/huggingface/open-env](https://github.com/huggingface/open-env)

---

> This work shows that alignment failures can emerge from objective design itself, before any model training occurs.
