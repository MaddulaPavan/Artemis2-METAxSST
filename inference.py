"""
inference.py — Baseline inference script for PreferenceAggregationEnv.

Runs an LLM agent against all 3 tasks and emits structured logs.

REQUIRED ENVIRONMENT VARIABLES:
  API_BASE_URL   LLM API endpoint (OpenAI-compatible)
  MODEL_NAME     Model identifier
  HF_TOKEN       API key / HuggingFace token

USAGE:
  export API_BASE_URL="https://api-inference.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py

LOG FORMAT (strict — deviation causes evaluation failure):
  [START] {"task": ..., "env": ..., "model": ...}
  [STEP]  {"step": ..., "action": ..., "reward": ..., "done": ..., "error": ...}
  [END]   {"success": ..., "steps": ..., "score": ..., "rewards": [...]}
"""

import os
import sys
import json
import asyncio
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (EXACT names required by spec)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-70B-Instruct")
API_KEY:      str = os.getenv("HF_TOKEN",      "")   # HF_TOKEN maps to API_KEY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK        = "preference-aggregation-env"
MAX_STEPS        = 10
MAX_TOTAL_REWARD = float(MAX_STEPS)   # max reward per step = 1.0
SUCCESS_THRESHOLD = 0.50
TEMPERATURE      = 0.0
MAX_TOKENS       = 8

SYSTEM_PROMPT = (
    "You are evaluating AI-generated responses. "
    "Given a prompt and two responses (A and B), select the one that better serves the user. "
    "Reply with ONLY the letter A or B — nothing else."
)

TASK_IDS = ["majority_dominance", "mixed_preferences", "fairness_collapse"]

# ---------------------------------------------------------------------------
# Structured logging (EXACT format — do not modify)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    payload = {
        "step":   step,
        "action": action,
        "reward": round(reward, 4),
        "done":   done,
        "error":  error,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    payload = {
        "success": success,
        "steps":   steps,
        "score":   round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
    }
    print(f"[END] {json.dumps(payload)}", flush=True)

# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def get_model_action(
    client:     OpenAI,
    obs_prompt: str,
    response_a: str,
    response_b: str,
    context:    str,
    step:       int,
    last_reward: float,
    history:    List[str],
) -> str:
    """
    Call the LLM and parse its response as 'A' or 'B'.
    Falls back to 'A' on any error.
    """
    user_prompt = (
        f"Context: {context}\n\n"
        f"Prompt: {obs_prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}\n\n"
        f"Step {step} | Previous reward: {last_reward:.2f}\n"
        "Which response is better? Reply with only A or B."
    )
    try:
        completion = client.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip().upper()
        # Parse: accept "A", "B", or first character
        if text and text[0] in ("A", "B"):
            return text[0]
        return "A"  # safe default
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return "A"

# ---------------------------------------------------------------------------
# Single task runner (async — matches OpenEnv sample pattern)
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_id: str) -> dict:
    """
    Run one full episode for a given task.

    Returns dict with: score, success, steps, rewards, grader_score.
    """
    # Import here to avoid circular issues
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env.environment import PreferenceAggregationEnv
    from env.models import Action

    env    = PreferenceAggregationEnv(task_id=task_id, seed=42)
    result = env.reset()  # OpenENV.reset()

    rewards:      List[float]    = []
    history:      List[str]      = []
    steps_taken:  int            = 0
    last_reward:  float          = 0.0
    grader_score: Optional[float] = None
    score:        float          = 0.0
    success:      bool           = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            obs   = result.observation
            error = None

            # Get LLM decision
            choice_letter = get_model_action(
                client      = client,
                obs_prompt  = obs.prompt,
                response_a  = obs.response_a,
                response_b  = obs.response_b,
                context     = obs.context,
                step        = step,
                last_reward = last_reward,
                history     = history,
            )

            action_int = 0 if choice_letter == "A" else 1

            # Step environment
            try:
                result = env.step(Action(select_response=action_int))
            except Exception as e:
                error = str(e)
                reward = 0.0
                done   = True
            else:
                reward = result.reward
                done   = result.done
                if done and "grader_score" in result.info:
                    grader_score = result.info["grader_score"]

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            history.append(
                f"Step {step}: chose {choice_letter} -> reward {reward:.2f}"
            )

            log_step(
                step   = step,
                action = choice_letter,
                reward = reward,
                done   = done,
                error  = error,
            )

            if done:
                break

    except Exception as outer_exc:
        print(f"[DEBUG] Episode error: {outer_exc}", flush=True)

    # Compute final score
    score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score   = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
    success = score >= SUCCESS_THRESHOLD

    log_end(
        success = success,
        steps   = steps_taken,
        score   = score,
        rewards = rewards,
    )

    return {
        "task_id":      task_id,
        "score":        score,
        "success":      success,
        "steps":        steps_taken,
        "rewards":      rewards,
        "grader_score": grader_score,
    }


# ---------------------------------------------------------------------------
# Main (async — matches sample inference.py pattern exactly)
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("[DEBUG] HF_TOKEN not set. Set HF_TOKEN to your API key.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY if API_KEY else "dummy")

    all_scores = []
    for task_id in TASK_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)

        task_result = await run_task(client, task_id)
        all_scores.append(task_result["score"])

        print(
            f"[DEBUG] Task complete | score={task_result['score']:.4f} "
            f"| grader={task_result['grader_score']}",
            flush=True,
        )

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[DEBUG] === OVERALL SCORE: {overall:.4f} ===", flush=True)
    print(
        f"[DEBUG] Per-task: "
        + " | ".join(f"{tid}={s:.3f}" for tid, s in zip(TASK_IDS, all_scores)),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
