---
title: RLHF Aggregation Failure Simulator
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
---

# PreferenceAggregationEnv

**An OpenEnv environment that exposes structural fairness failure in RLHF reward aggregation.**

> Built for the **Meta OpenEnv Hackathon** — Meta x Hugging Face x Scaler School of Technology

| | |
|---|---|
| **Team** | Artemis 2 |
| **Lead** | M P V S Gopinadh (`pavanmaddula44@gmail.com`) |
| **Member** | Kappara Lakshmi Sindhu (`klsindhu68@gmail.com`) |
| **Space** | [pavanmaddula-artemis2-sp.hf.space](https://pavanmaddula-artemis2-sp.hf.space) |
| **SDK** | Docker · Python 3.11 · FastAPI · Port 7860 |
| **Runtime** | ~2 min · CPU only · No GPU required |

---

## What This Environment Does

Standard RLHF aggregates preferences from diverse annotators into a single reward model. When those annotators disagree (e.g., some prefer concise answers, others want detail), the majority group dominates and minority preferences are silenced.

**This environment simulates that exact process.** An agent evaluates AI responses under conflicting user preferences — the same task RLHF annotators perform in production. The reward the agent receives is the population-weighted average, just like real RLHF. The result: a fairness gap of 100% that no deterministic policy can close.

This is not a game or a toy. It models a real task (preference annotation) and exposes a real limitation of a system used to align every major LLM.

---

## Key Result

```
Task 1 (Easy)   — Grader: 1.00   Agent follows majority, scores well
Task 2 (Medium) — Grader: 0.37   Conflicting signals, partial success
Task 3 (Hard)   — Grader: 0.00   Structural impossibility, all agents fail
```

Any deterministic policy scores **0.0** on the hard task. This is not a capability gap — it's a design limitation of aggregated reward objectives.

---

## Tasks

| ID | Difficulty | Group Distribution | Objective | Grader |
|----|-----------|-------------------|-----------|--------|
| `majority_dominance` | Easy | [0.70, 0.20, 0.10] | Learn the majority preference | Fraction of steps satisfying majority group |
| `mixed_preferences` | Medium | [0.40, 0.35, 0.25] | Navigate conflicting reward signals | Mean accuracy across all 3 groups |
| `fairness_collapse` | Hard | [0.333, 0.333, 0.334] | Achieve fairness across balanced groups | `1.0 - fairness_gap` |

- All graders are **deterministic** — same actions produce same score
- All scores are in **[0.0, 1.0]**
- Scores **vary with agent actions** — graders never return a fixed value
- The hard task **genuinely challenges frontier models** — structural impossibility

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | Task prompt shown to the agent |
| `response_a` | `str` | Candidate response A (concise style) |
| `response_b` | `str` | Candidate response B (detailed style) |
| `previous_reward` | `float` | Reward from previous step |
| `step_count` | `int` | Current step index |
| `task_id` | `str` | Active task identifier |
| `context` | `str` | Task-level instruction |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `select_response` | `int` | `0` = choose response A, `1` = choose response B |

## Reward Function

```
reward = sum( group_weight[g] * group_reward(g, action) )  for g in {0, 1, 2}
```

- **Range:** [0.0, 1.0] per step
- **Dense:** computed every step, not just end-of-episode
- **Partial:** varies continuously with actions (not binary)
- **Meaningful:** choosing the majority-preferred response gives higher aggregated reward, but hurts minority groups — exactly the RLHF tradeoff

The `info` dict returns `per_group_rewards`, `fairness_gap`, `true_reward`, and `hidden_group` for diagnostics.

---

## API

All models are Pydantic-typed. Full OpenEnv spec: `step(action)`, `reset()`, `state()`.

### Python (direct import)

```python
from env.environment import PreferenceAggregationEnv
from env.models import Action

env = PreferenceAggregationEnv(task_id="majority_dominance", seed=42)
result = env.reset()

for step in range(10):
    obs = result.observation
    result = env.step(Action(select_response=0))
    print(f"Step {step+1}: reward={result.reward:.2f}, done={result.done}")
    if result.done:
        print(f"Grader score: {result.info['grader_score']}")
```

### HTTP (deployed Space)

```bash
# Health check
curl https://pavanmaddula-artemis2-sp.hf.space/health

# Start episode
curl -X POST "https://pavanmaddula-artemis2-sp.hf.space/reset?task_id=majority_dominance&seed=42"

# Take action
curl -X POST https://pavanmaddula-artemis2-sp.hf.space/step \
  -H "Content-Type: application/json" -d '{"select_response": 0}'

# Get state
curl https://pavanmaddula-artemis2-sp.hf.space/state

# List tasks
curl https://pavanmaddula-artemis2-sp.hf.space/tasks
```

---

## Setup

### Install

```bash
pip install -r requirements.txt
```

### Run locally

```bash
python -m uvicorn env.environment:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t pref-agg-env .
docker run -p 7860:7860 pref-agg-env
```

### Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | (required for inference) |

---

## Baseline Scores

Scores from running `inference.py` with a greedy LLM agent (always picks higher-reward response):

| Task | Score | Grader | Interpretation |
|------|-------|--------|----------------|
| `majority_dominance` | 0.70 | 1.00 | Agent learns majority easily |
| `mixed_preferences` | 0.50 | 0.37 | Partial tradeoff, no group fully satisfied |
| `fairness_collapse` | 0.33 | 0.00 | Structural failure — 100% fairness gap |

---

## Project Structure

```
├── inference.py           Baseline inference script (root, required)
├── openenv.yaml           Task registry + environment metadata
├── Dockerfile             Docker config (port 7860, health check)
├── requirements.txt       Dependencies
├── README.md              This file
└── env/
    ├── __init__.py         Package init
    ├── environment.py      PreferenceAggregationEnv + FastAPI server
    ├── models.py           Pydantic: Observation, Action, Reward, StepResult
    ├── reward.py           Per-group + aggregated reward functions
    └── tasks.py            3 tasks, 15-prompt dataset, deterministic graders
```

---

## Why This Matters

- **Real-world task:** This models exactly what RLHF annotators do — evaluate and select AI responses
- **Real-world failure:** Preference aggregation is how every major LLM is aligned today; this failure mode is structural
- **Novel domain:** No existing OpenEnv environment exposes RLHF fairness collapse as a learnable benchmark
- **Useful:** Researchers studying reward model fairness can use this as a diagnostic testbed

---

## Hackathon Submission

This environment is **Team Artemis 2's** Round 1 submission for:

> **India's Biggest Meta OpenEnv Hackathon**  
> Built on Meta's OpenEnv — the foundation for next-gen RL environments used by leading AI labs.

| | |
|---|---|
| **Event** | Meta OpenEnv Hackathon — Round 1 |
| **Organizers** | Meta, Hugging Face, Scaler School of Technology |
| **Round** | Round 1 — Build a Mini RL Environment (25 Mar - 8 Apr 2026) |
| **Team** | **Artemis 2** |
| **Lead** | M P V S Gopinadh (`pavanmaddula44@gmail.com`) |
| **Member** | Kappara Lakshmi Sindhu (`klsindhu68@gmail.com`) |
| **Space** | [huggingface.co/spaces/PavanMaddula/Artemis2-SP](https://huggingface.co/spaces/PavanMaddula/Artemis2-SP) |
| **GitHub** | [github.com/MaddulaPavan/Artemis2-METAxSST](https://github.com/MaddulaPavan/Artemis2-METAxSST) |
