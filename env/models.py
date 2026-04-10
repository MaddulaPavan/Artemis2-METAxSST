"""
env/models.py — Typed Pydantic models for PreferenceAggregationEnv.

All models follow the OpenEnv spec: Observation, Action, Reward.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class Observation(BaseModel):
    """Full observation returned to the agent each step."""
    prompt: str = Field(..., description="The task prompt shown to the agent")
    response_a: str = Field(..., description="Candidate response A (typically concise)")
    response_b: str = Field(..., description="Candidate response B (typically detailed)")
    previous_reward: float = Field(default=0.0, description="Reward received on previous step")
    step_count: int = Field(default=0, description="Current step index (0-based)")
    task_id: str = Field(..., description="Active task identifier")
    context: str = Field(default="", description="Task-level instruction context for the agent")


class Action(BaseModel):
    """Agent action: select one of two candidate responses."""
    select_response: int = Field(
        ..., ge=0, le=1,
        description="0 = choose response_a, 1 = choose response_b"
    )


class Reward(BaseModel):
    """Structured reward with full diagnostic breakdown."""
    value: float = Field(..., description="Scalar reward in [0.0, 1.0]")
    true_reward: float = Field(..., description="True reward for hidden preference group")
    aggregated_reward: float = Field(..., description="Weighted reward across all groups")
    fairness_gap: float = Field(..., description="max(group_acc) - min(group_acc) this episode")
    group_rewards: List[float] = Field(
        default_factory=list,
        description="Per-group reward for this step [g0, g1, g2]"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-group weighted contribution to aggregated reward"
    )


class StepResult(BaseModel):
    """Full result returned by env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Result returned by env.reset()."""
    observation: Observation
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    """Full internal state returned by env.state()."""
    task_id: str
    hidden_group: Optional[int]
    group_name: Optional[str]
    step_count: int
    episode_done: bool
    episode_rewards: List[float]
    group_distribution: List[float]
    effective_weights: List[float]
    fairness_gap: float = 0.0
    grader_score: Optional[float] = None
