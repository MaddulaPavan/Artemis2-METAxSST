"""Determinism, bounds, and episode contract tests for PreferenceAggregationEnv."""

import pytest

from env.environment import PreferenceAggregationEnv
from env.models import Action


TASK_IDS = ("majority_dominance", "mixed_preferences", "fairness_collapse")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reset_is_deterministic_for_same_seed(task_id: str) -> None:
    a = PreferenceAggregationEnv(task_id=task_id, seed=123)
    b = PreferenceAggregationEnv(task_id=task_id, seed=123)
    ra, rb = a.reset(), b.reset()
    assert ra.observation.prompt == rb.observation.prompt
    assert ra.observation.response_a == rb.observation.response_a
    assert ra.observation.response_b == rb.observation.response_b


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_episode_length_and_grader_bounds(task_id: str) -> None:
    env = PreferenceAggregationEnv(task_id=task_id, seed=7)
    env.reset()
    last = None
    for _ in range(20):
        last = env.step(Action(select_response=0))
        if last.done:
            break
    assert last is not None
    assert last.done
    assert "grader_score" in last.info
    g = last.info["grader_score"]
    assert isinstance(g, float)
    assert 0.0 <= g <= 1.0


def test_state_includes_effective_weights() -> None:
    env = PreferenceAggregationEnv(task_id="fairness_collapse", seed=1)
    env.reset()
    env.step(Action(select_response=1))
    s = env.state()
    assert "effective_weights" in s
    assert s["effective_weights"] == [0.333, 0.333, 0.334]
    assert "fairness_gap" in s


def test_aggregated_reward_in_unit_interval() -> None:
    env = PreferenceAggregationEnv(task_id="mixed_preferences", seed=0)
    env.reset()
    for _ in range(10):
        r = env.step(Action(select_response=1))
        assert 0.0 <= r.reward <= 1.0
        if r.done:
            break
