"""
env.py — PreferenceAggregationEnv v2

A parameterized, OpenEnv-compatible RL diagnostic environment for studying
preference aggregation failure in RLHF systems.

New in v2:
  - group_distribution : configurable population weights for each preference group
  - bias_strength       : interpolates reward weighting between uniform and group_distribution
  - noise_level         : Gaussian reward noise (simulates annotation noise)
  - episode_length      : reserved parameter for future multi-step extensions
  - effective_weights   : computed blend of uniform and group_distribution weights

Design rationale:
  bias_strength = 0.0  →  uniform effective weights (no aggregation bias)
  bias_strength = 1.0  →  full group_distribution weights (baseline RLHF bias)

  This axis allows systematic study of how much the aggregation design
  contributes to fairness failure, independent of the data distribution.
"""

import random
import math
from typing import Optional, List, Tuple, Dict, Any

from data import DATASET
from reward import group_reward, aggregated_reward, GROUP_NAMES

# ---------------------------------------------------------------------------
# Minimal Gym-compatible base (no external dependency required)
# ---------------------------------------------------------------------------

class _EnvBase:
    """Drop-in Gymnasium-compatible base class (no installation required)."""

    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


try:
    import gymnasium as gym
    _BASE_CLASS = gym.Env
    _HAS_GYM = True
except ImportError:
    _BASE_CLASS = _EnvBase
    _HAS_GYM = False


# ---------------------------------------------------------------------------
# Core Environment
# ---------------------------------------------------------------------------

class PreferenceAggregationEnv(_BASE_CLASS):
    """
    Preference Aggregation Failure Environment — RLHF Diagnostic Simulator (v2).

    Exposes a configurable testbed for studying how reward aggregation
    design choices produce systematic fairness failures across preference groups.

    Parameters
    ----------
    mode : str
        "standard"         — aggregated RLHF reward (biased)
        "preference_aware" — per-group true reward (fair)
    group_distribution : list[float]
        Population weight of each preference group. Controls:
          (1) the group sampling probability in reset()
          (2) the base for effective_weights computation in step()
        Default: [0.60, 0.30, 0.10] — classic majority/minority split
    bias_strength : float in [0.0, 1.0]
        Interpolation coefficient for reward weighting:
          0.0 → uniform weights (aggregation is unbiased by population)
          1.0 → full group_distribution weights (maximum aggregation bias)
        Controls how strongly the population imbalance pollutes the reward.
    noise_level : float >= 0.0
        Std dev of zero-mean Gaussian noise added to the reward signal.
        Simulates annotation noise in the reward model.
    episode_length : int
        Reserved for multi-step episode support. Currently fixed at 1.
    seed : int, optional
        Random seed for full reproducibility.

    Attributes
    ----------
    effective_weights : list[float]
        The actual weights used in the aggregated reward computation.
        Interpolated between uniform and group_distribution via bias_strength.
    """

    metadata = {"render_modes": ["human"]}
    VALID_MODES = ("standard", "preference_aware")
    N_GROUPS = 3

    def __init__(
        self,
        mode: str = "standard",
        group_distribution: Optional[List[float]] = None,
        bias_strength: float = 1.0,
        noise_level: float = 0.0,
        episode_length: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # ---------- Validate parameters ----------
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        if not (0.0 <= bias_strength <= 1.0):
            raise ValueError(f"bias_strength must be in [0.0, 1.0], got {bias_strength}")
        if noise_level < 0.0:
            raise ValueError(f"noise_level must be >= 0.0, got {noise_level}")

        # ---------- Store configuration ----------
        self.mode = mode
        self.bias_strength = bias_strength
        self.noise_level = noise_level
        self.episode_length = episode_length
        self.dataset = DATASET
        self.n_groups = self.N_GROUPS

        # ---------- Normalize and store group distribution ----------
        if group_distribution is None:
            group_distribution = [0.60, 0.30, 0.10]
        if len(group_distribution) != self.N_GROUPS:
            raise ValueError(
                f"group_distribution must have {self.N_GROUPS} elements, "
                f"got {len(group_distribution)}"
            )
        total = sum(group_distribution)
        self.group_distribution = [w / total for w in group_distribution]

        # ---------- Compute effective weights ----------
        # Interpolate between uniform and group_distribution using bias_strength.
        # At bias_strength=0: uniform weights → no population bias in reward.
        # At bias_strength=1: full group_distribution → maximum bias.
        uniform = [1.0 / self.N_GROUPS] * self.N_GROUPS
        self.effective_weights = [
            (1.0 - bias_strength) * uniform[g] + bias_strength * self.group_distribution[g]
            for g in range(self.N_GROUPS)
        ]

        # ---------- Seeded RNG ----------
        self._rng = random.Random(seed)
        self._seed = seed

        # ---------- Gymnasium action/observation spaces (if available) ----------
        if _HAS_GYM:
            from gymnasium import spaces
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Dict({
                "prompt":     spaces.Text(max_length=512),
                "response_A": spaces.Text(max_length=2048),
                "response_B": spaces.Text(max_length=2048),
            })

        # ---------- Episode state ----------
        self._current_obs: Optional[Dict[str, str]] = None
        self._current_group: Optional[int] = None
        self._episode_done: bool = False
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Begin a new episode.

        Samples:
          - A prompt-response pair uniformly from the dataset
          - A hidden preference group, weighted by group_distribution

        Returns
        -------
        obs  : dict  {"prompt", "response_A", "response_B"}
        info : dict  {"group", "group_name", "effective_weights"}
        """
        if seed is not None:
            self._rng = random.Random(seed + self._episode_count)

        # Sample dataset entry
        sample = self._rng.choice(self.dataset)

        # Sample preference group (weighted by population distribution)
        groups = list(range(self.N_GROUPS))
        self._current_group = self._rng.choices(
            groups, weights=self.group_distribution, k=1
        )[0]

        self._current_obs = {
            "prompt":     sample["prompt"],
            "response_A": sample["response_A"],
            "response_B": sample["response_B"],
        }
        self._episode_done = False
        self._episode_count += 1

        info = {
            "group":            self._current_group,
            "group_name":       GROUP_NAMES[self._current_group],
            "effective_weights": self.effective_weights,
        }
        return self._current_obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, str], float, bool, bool, Dict[str, Any]]:
        """
        Execute a decision (bandit-style: one step per episode).

        Parameters
        ----------
        action : int   0 = choose response_A,  1 = choose response_B

        Returns
        -------
        obs        : Same observation (episode ends after 1 step)
        reward     : float — based on current mode + noise
        terminated : bool  — always True (single-step episode)
        truncated  : bool  — always False
        info       : dict  — diagnostics including true_reward, group
        """
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}. Must be 0 or 1.")

        obs   = self._current_obs
        group = self._current_group

        # Compute reward based on mode
        if self.mode == "standard":
            reward = aggregated_reward(action, obs, weights=self.effective_weights)
        else:
            reward = group_reward(group, action, obs)

        # Always compute true reward for diagnostics
        true_reward = group_reward(group, action, obs)

        # Optionally corrupt reward with noise
        if self.noise_level > 0.0:
            noise = self._rng.gauss(0.0, self.noise_level)
            reward = max(-2.0, min(2.0, reward + noise))  # soft clip

        self._episode_done = True

        info = {
            "group":           group,
            "group_name":      GROUP_NAMES[group],
            "true_reward":     true_reward,
            "received_reward": reward,
            "mode":            self.mode,
            "action_chosen":   "A" if action == 0 else "B",
            "correct":         true_reward > 0,
        }

        return obs, reward, True, False, info

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Print current episode state."""
        if self._current_obs is None:
            print("[PreferenceAggregationEnv] No active episode. Call reset() first.")
            return

        obs   = self._current_obs
        group = self._current_group
        sep   = "=" * 70

        print(f"\n{sep}")
        print(f"  MODE: {self.mode.upper().replace('_', ' ')}")
        print(f"  bias_strength={self.bias_strength:.2f}  "
              f"noise_level={self.noise_level:.2f}  "
              f"dist={[round(w, 2) for w in self.group_distribution]}")
        print(f"  Hidden Group: {GROUP_NAMES[group]} (Group {group})")
        print(f"{sep}")
        print(f"  PROMPT:     {obs['prompt']}")
        print(f"  RESPONSE A: {obs['response_A'][:90]}...")
        print(f"  RESPONSE B: {obs['response_B'][:90]}...")
        print(f"{sep}\n")

    def get_config(self) -> Dict[str, Any]:
        """Return full environment configuration as a serializable dict."""
        return {
            "mode":               self.mode,
            "group_distribution": self.group_distribution,
            "bias_strength":      self.bias_strength,
            "noise_level":        self.noise_level,
            "episode_length":     self.episode_length,
            "effective_weights":  self.effective_weights,
            "n_groups":           self.n_groups,
            "n_prompts":          len(self.dataset),
        }

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        dist = [round(w, 2) for w in self.group_distribution]
        return (
            f"PreferenceAggregationEnv("
            f"mode='{self.mode}', "
            f"dist={dist}, "
            f"bias={self.bias_strength:.2f}, "
            f"noise={self.noise_level:.2f})"
        )
