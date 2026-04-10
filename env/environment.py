"""
env/environment.py — PreferenceAggregationEnv core class + FastAPI server.

The environment class works standalone (direct import) AND as an HTTP server
(Docker / HuggingFace Space deployment).

HTTP Endpoints:
  POST /reset?task_id=...&seed=...   → ResetResult
  POST /step                         → StepResult  (body: Action)
  GET  /state                        → EnvironmentState
  GET  /tasks                        → list of task metadata
  GET  /health                       → {"status": "ok"}
"""

import random
import os
from typing import Optional, Dict, Any, List

from env.models import (
    Observation, Action, Reward, StepResult, ResetResult, EnvironmentState
)
from env.tasks import TASKS, get_task, DATASET
from env.reward import (
    group_reward,
    compute_all_group_rewards,
    compute_aggregated_reward,
    compute_fairness_gap,
    counterfactual_reward_margin,
    sharpen_aggregated_reward,
    GROUP_NAMES,
)

def _reward_sharpen_gamma() -> float:
    raw = os.getenv("REWARD_SHARPEN_GAMMA", "1.55").strip()
    try:
        g = float(raw)
    except ValueError:
        g = 1.55
    return max(1.0, min(g, 2.5))


def _reward_sharpen_gamma_for_task(task_id: str) -> float:
    """Hard task uses stronger sharpening so A vs B policy signal stays crisp under balanced weights."""
    if task_id != "fairness_collapse":
        return _reward_sharpen_gamma()
    raw = os.getenv("REWARD_SHARPEN_GAMMA_HARD", "1.90").strip()
    try:
        g = float(raw)
    except ValueError:
        g = 1.90
    return max(1.0, min(g, 2.5))

# ---------------------------------------------------------------------------
# Core environment class
# ---------------------------------------------------------------------------

class PreferenceAggregationEnv:
    """
    Preference Aggregation Failure Environment — OpenEnv v2.

    Simulates AI response selection under heterogeneous user preferences.
    Demonstrates that RLHF reward aggregation produces structural fairness
    failure independent of distribution, noise, or bias parameters.

    Parameters
    ----------
    task_id : str
        One of: "majority_dominance", "mixed_preferences", "fairness_collapse"
    seed    : int
        Random seed for full reproducibility.
    """

    def __init__(self, task_id: str = "majority_dominance", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(TASKS.keys())}")

        self.task_id = task_id
        self.task    = TASKS[task_id]
        self.seed    = seed
        self._rng    = random.Random(seed)

        # Episode state
        self._obs:             Optional[Observation] = None
        self._hidden_group:    Optional[int]         = None
        self._step_count:      int                   = 0
        self._episode_done:    bool                  = False
        self._episode_rewards: List[float]           = []
        self._episode_history: List[Dict[str, Any]]  = []
        self._grader_score:    Optional[float]       = None

    # ------------------------------------------------------------------
    # OpenEnv Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, task_id: Optional[str] = None) -> ResetResult:
        """
        Begin a new episode.

        Samples a random prompt-response pair and a hidden preference group
        (weighted by task's group_distribution).

        Returns ResetResult with initial Observation and episode metadata.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        if task_id is not None and task_id != self.task_id:
            self.task_id = task_id
            self.task    = TASKS[task_id]

        sample = self._rng.choice(DATASET)
        groups = list(range(3))
        self._hidden_group = self._rng.choices(
            groups,
            weights=self.task["group_distribution"],
            k=1
        )[0]

        self._obs = Observation(
            prompt      = sample["prompt"],
            response_a  = sample["response_a"],
            response_b  = sample["response_b"],
            previous_reward = 0.0,
            step_count  = 0,
            task_id     = self.task_id,
            context     = self.task["context"],
        )
        self._step_count      = 0
        self._episode_done    = False
        self._episode_rewards = []
        self._episode_history = []
        self._grader_score    = None

        return ResetResult(
            observation = self._obs,
            done        = False,
            info        = {
                "task_id":           self.task_id,
                "task_name":         self.task["name"],
                "difficulty":        self.task["difficulty"],
                "max_steps":         self.task["max_steps"],
                "group_distribution": self.task["group_distribution"],
            },
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute one decision step.

        The agent selects response_a (0) or response_b (1).
        Reward = aggregated weighted sum across all preference groups.
        Hidden group determines true reward for diagnostic purposes.

        Returns StepResult with next Observation, reward, done flag, and
        full diagnostic info (grader_score on final step).
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if not isinstance(action, Action):
            action = Action(select_response=int(action))

        act  = action.select_response
        obs  = self._obs
        dist = self.task["group_distribution"]

        # Compute per-group and aggregated rewards
        per_group = compute_all_group_rewards(act, obs.response_a, obs.response_b)
        agg       = compute_aggregated_reward(act, obs.response_a, obs.response_b, dist)
        true_r    = group_reward(self._hidden_group, act, obs.response_a, obs.response_b)

        gamma = _reward_sharpen_gamma_for_task(self.task_id)
        step_reward = sharpen_aggregated_reward(agg, gamma)

        # Running fairness gap
        self._episode_history.append({
            "step":         self._step_count + 1,
            "action":       act,
            "reward":       step_reward,
            "true_reward":  true_r,
            "group":        self._hidden_group,
            "group_rewards": per_group,
        })
        self._episode_rewards.append(step_reward)
        self._step_count += 1

        gap   = compute_fairness_gap([h["group_rewards"] for h in self._episode_history])
        done  = self._step_count >= self.task["max_steps"]
        self._episode_done = done

        # Run grader on final step
        if done:
            self._grader_score = self.task["grader"](self._episode_history)

        # Advance to next prompt (re-sample for next step)
        if not done:
            sample = self._rng.choice(DATASET)
            self._obs = Observation(
                prompt          = sample["prompt"],
                response_a      = sample["response_a"],
                response_b      = sample["response_b"],
                previous_reward = step_reward,
                step_count      = self._step_count,
                task_id         = self.task_id,
                context         = self.task["context"],
            )

        margin = counterfactual_reward_margin(obs.response_a, obs.response_b, dist)
        breakdown = {
            f"group_{g}_weighted": round(dist[g] * per_group[g], 6)
            for g in range(len(dist))
        }
        info: Dict[str, Any] = {
            "true_reward":   true_r,
            "aggregated_reward": agg,
            "policy_reward": step_reward,
            "reward_sharpen_gamma": gamma,
            "reward_sharpen_mode": "hard" if self.task_id == "fairness_collapse" else "default",
            "per_group_rewards": per_group,
            "weight_breakdown": breakdown,
            "action_margin": round(margin, 6),
            "hidden_group":  self._hidden_group,
            "group_name":    GROUP_NAMES[self._hidden_group],
            "fairness_gap":  round(gap, 4),
            "step_count":    self._step_count,
        }
        if done and self._grader_score is not None:
            info["grader_score"] = self._grader_score

        return StepResult(
            observation = self._obs,
            reward      = step_reward,
            done        = done,
            info        = info,
        )

    def state(self) -> Dict[str, Any]:
        """
        Return full internal environment state.

        Includes hidden group identity (for diagnostics/research).
        Not available to the agent during normal operation.
        """
        gap = compute_fairness_gap(
            [h["group_rewards"] for h in self._episode_history]
        ) if self._episode_history else 0.0

        dist = self.task["group_distribution"]
        return {
            "task_id":            self.task_id,
            "hidden_group":       self._hidden_group,
            "group_name":         GROUP_NAMES.get(self._hidden_group, "unknown"),
            "step_count":         self._step_count,
            "episode_done":       self._episode_done,
            "episode_rewards":    self._episode_rewards,
            "group_distribution": dist,
            "effective_weights":  list(dist),
            "fairness_gap":       round(gap, 4),
            "grader_score":       self._grader_score,
        }

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Return metadata for all available tasks."""
        return [
            {
                "id":          t["id"],
                "name":        t["name"],
                "difficulty":  t["difficulty"],
                "description": t["description"],
                "max_steps":   t["max_steps"],
                "expected_baseline": t.get("expected_baseline"),
            }
            for t in TASKS.values()
        ]


# ---------------------------------------------------------------------------
# FastAPI server (for Docker / HuggingFace Space deployment)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as _BM
    import uvicorn

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title       = "PreferenceAggregationEnv",
        description = "OpenEnv environment: RLHF preference aggregation failure simulator",
        version     = "2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # Global environment instance (single-agent evaluation model)
    _env: Optional[PreferenceAggregationEnv] = None

    @app.get("/")
    async def root():
        return {
            "name": "PreferenceAggregationEnv",
            "description": "OpenEnv benchmark: RLHF preference aggregation failure simulator",
            "version": "2.0.0",
            "endpoints": {
                "POST /reset": "Begin a new episode",
                "POST /step": "Take an action",
                "GET /state": "Get current state",
                "GET /tasks": "List available tasks",
                "GET /health": "Health check",
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "preference-aggregation-env", "version": "2.0.0"}

    @app.post("/reset")
    async def http_reset(task_id: str = "majority_dominance", seed: int = 42):
        global _env
        _env = PreferenceAggregationEnv(task_id=task_id, seed=seed)
        result = _env.reset()
        return result.dict()

    @app.post("/step")
    async def http_step(action: Action):
        global _env
        if _env is None:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        result = _env.step(action)
        return result.dict()

    @app.get("/state")
    async def http_state():
        global _env
        if _env is None:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        return _env.state()

    @app.get("/tasks")
    async def http_tasks():
        temp = PreferenceAggregationEnv()
        return temp.get_tasks()

    def serve(host: str = "0.0.0.0", port: int = 7860):
        """Start the FastAPI server."""
        uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    if _FASTAPI_AVAILABLE:
        serve(port=port)
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
