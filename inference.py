"""
inference.py — Baseline inference script for PreferenceAggregationEnv.

Runs an LLM agent against all 3 tasks and emits structured logs.

REQUIRED ENVIRONMENT VARIABLES:
  API_BASE_URL     LLM API endpoint (OpenAI-compatible)
  MODEL_NAME       Model identifier
  HF_TOKEN         Preferred API key (Hugging Face / router)
  OPENAI_API_KEY   Alternative API key (OpenAI-compatible; accepted if HF_TOKEN unset)

OPTIONAL:
  INFERENCE_MODE   "llm" (default) or "heuristic" — deterministic baseline that picks
                   the response with higher population-weighted aggregate reward
                   (strong, reproducible; uses task weights only, no hidden group).

STDOUT FORMAT (mandatory — any deviation fails evaluation):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (EXACT names required by spec)
# ---------------------------------------------------------------------------

API_KEY: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME:   str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
INFERENCE_MODE: str = os.getenv("INFERENCE_MODE", "llm").strip().lower()
HEURISTIC_MODEL_LABEL = "heuristic-population-aggregate"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK        = "preference-aggregation-env"
TASK_IDS         = ["majority_dominance", "mixed_preferences", "fairness_collapse"]
MAX_STEPS        = 10
MAX_TOTAL_REWARD = float(MAX_STEPS)   # max reward per step = 1.0
SUCCESS_THRESHOLD = 0.50
TEMPERATURE      = 0.7
MAX_TOKENS       = 8

SYSTEM_PROMPT = (
    "You are collecting RLHF-style preferences between two model responses. "
    "Read the Context line first — it states who the answer should serve and what to optimize "
    "(brevity vs depth vs technical density). "
    "Choose the single response (A or B) that best matches that Context for the user question. "
    "Reply with ONLY the letter A or B — nothing else."
)


def model_label_for_logs() -> str:
    return HEURISTIC_MODEL_LABEL if INFERENCE_MODE == "heuristic" else MODEL_NAME

# ---------------------------------------------------------------------------
# Structured logging (EXACT key=value format from sample spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def get_model_action(
    client:      OpenAI,
    obs_prompt:  str,
    response_a:  str,
    response_b:  str,
    context:     str,
    step:        int,
    last_reward: float,
    history:     List[str],
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
        if text and text[0] in ("A", "B"):
            return text[0]
        return "A"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "A"


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_id: str) -> dict:
    """
    Run one full episode for a given task.
    Returns dict with: score, success, steps, rewards, grader_score.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env.environment import PreferenceAggregationEnv
    from env.models import Action
    from env.tasks import TASKS
    from env.reward import compute_aggregated_reward

    env    = PreferenceAggregationEnv(task_id=task_id, seed=42)
    result = env.reset()  # OpenENV.reset()

    rewards:      List[float]     = []
    history:      List[str]       = []
    steps_taken:  int             = 0
    last_reward:  float           = 0.0
    grader_score: Optional[float] = None
    score:        float           = 0.0
    success:      bool            = False

    log_start(task=task_id, env=BENCHMARK, model=model_label_for_logs())

    try:
        for step in range(1, MAX_STEPS + 1):
            obs   = result.observation
            error = None

            if INFERENCE_MODE == "heuristic":
                weights = TASKS[task_id]["group_distribution"]
                r0 = compute_aggregated_reward(
                    0, obs.response_a, obs.response_b, weights
                )
                r1 = compute_aggregated_reward(
                    1, obs.response_a, obs.response_b, weights
                )
                choice_letter = "A" if r0 >= r1 else "B"
            else:
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
                error  = str(e)
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
# Main (async — matches sample inference.py pattern)
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY and INFERENCE_MODE != "heuristic":
        print(
            "[DEBUG] HF_TOKEN / OPENAI_API_KEY not set. Set one for LLM mode.",
            flush=True,
        )

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
