"""Simple 2-state navigation environment for Quantum RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EnvironmentConfig:
    """Configuration for the navigation environment."""

    max_episode_steps: int = 20
    correct_reward: float = 1.0
    incorrect_penalty: float = -0.1


class SimpleNavigationEnv:
    """
    Minimal navigation MDP.

    State space:
    - 0: Searching
    - 1: Target Found

    Action space:
    - 0: Move Left
    - 1: Move Right

    Dynamics:
    - From state 0, action 1 transitions to state 1.
    - Any other action keeps the agent in state 0.
    - Episode terminates on reaching state 1 or max steps.
    """

    ACTION_MEANINGS = {0: "Move Left", 1: "Move Right"}
    STATE_MEANINGS = {0: "Searching", 1: "Target Found"}
    OPTIMAL_ACTION = {0: 1}

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self._state: int = 0
        self._step_count: int = 0
        self._done: bool = False

    @property
    def action_space(self) -> int:
        """Return action-space cardinality."""
        return 2

    @property
    def observation_space(self) -> int:
        """Return observation-space cardinality."""
        return 2

    def reset(self) -> int:
        """Reset to the Searching state."""
        self._state = 0
        self._step_count = 0
        self._done = False
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Apply action and return transition tuple."""
        if self._done:
            return self._state, 0.0, True, {"warning": "Episode already terminated."}
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}. Valid actions are 0 and 1.")

        self._step_count += 1
        optimal_action = self.OPTIMAL_ACTION[self._state]
        action_correct = action == optimal_action
        reward = self.config.correct_reward if action_correct else self.config.incorrect_penalty

        if self._state == 0 and action == 1:
            self._state = 1

        reached_target = self._state == 1
        timed_out = self._step_count >= self.config.max_episode_steps
        self._done = reached_target or timed_out

        info = {
            "step": self._step_count,
            "state_name": self.STATE_MEANINGS[self._state],
            "action_name": self.ACTION_MEANINGS[action],
            "optimal_action": optimal_action,
            "action_correct": action_correct,
            "reached_target": reached_target,
            "timed_out": timed_out,
        }
        return self._state, reward, self._done, info

    def render(self) -> str:
        """Text render for quick debugging."""
        return f"State={self._state} ({self.STATE_MEANINGS[self._state]}), step={self._step_count}"


class SimpleNavigationGymEnv:
    """Thin Gymnasium-style wrapper."""

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.env = SimpleNavigationEnv(config=config)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        del seed, options
        return self.env.reset(), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        next_state, reward, done, info = self.env.step(action)
        terminated = done and info["reached_target"]
        truncated = done and info["timed_out"]
        return next_state, reward, terminated, truncated, info

    def render(self) -> None:
        print(self.env.render())

    def close(self) -> None:
        return None

