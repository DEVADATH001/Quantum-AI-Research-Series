"""Sequential navigation environments for quantum RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class EnvironmentConfig:
    """Configuration for the key-and-door navigation task."""

    n_positions: int = 4
    start_positions: tuple[int, ...] = (1, 2)
    key_position: int = 0
    goal_position: int = 3
    max_episode_steps: int = 8
    step_penalty: float = -0.02
    wall_penalty: float = -0.05
    locked_goal_penalty: float = -0.25
    key_reward: float = 0.15
    goal_reward: float = 1.0
    progress_reward_scale: float = 0.05
    slip_probability: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        if self.n_positions < 3:
            raise ValueError("n_positions must be at least 3.")
        if not self.start_positions:
            raise ValueError("start_positions must contain at least one position.")
        if any(pos < 0 or pos >= self.n_positions for pos in self.start_positions):
            raise ValueError("All start positions must lie inside the corridor.")
        if not (0 <= self.key_position < self.n_positions):
            raise ValueError("key_position must lie inside the corridor.")
        if not (0 <= self.goal_position < self.n_positions):
            raise ValueError("goal_position must lie inside the corridor.")
        if self.key_position == self.goal_position:
            raise ValueError("key_position and goal_position must differ.")
        if not (0.0 <= self.slip_probability < 1.0):
            raise ValueError("slip_probability must be in [0, 1).")


class KeyDoorNavigationEnv:
    """
    Sequential corridor task with delayed reward and phase-dependent control.

    The agent starts without a key. The goal tile is locked until the key tile
    has been visited, so the optimal policy must first move toward the key and
    then reverse direction toward the goal.
    """

    ACTION_MEANINGS = {0: "Move Left", 1: "Move Right"}

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self._position = self.config.start_positions[0]
        self._has_key = False
        self._step_count = 0
        self._done = False

    @property
    def action_space(self) -> int:
        return 2

    @property
    def observation_space(self) -> int:
        return self.config.n_positions * 2

    @property
    def state(self) -> int:
        return self.encode_state(position=self._position, has_key=self._has_key)

    def encode_state(self, position: int, has_key: bool) -> int:
        return int(position + (self.config.n_positions if has_key else 0))

    def decode_state(self, state: int) -> tuple[int, bool]:
        if state < 0 or state >= self.observation_space:
            raise ValueError(f"State {state} is outside [0, {self.observation_space}).")
        has_key = state >= self.config.n_positions
        position = state % self.config.n_positions
        return position, has_key

    def state_label(self, state: int) -> str:
        position, has_key = self.decode_state(state)
        return f"pos={position},key={int(has_key)}"

    def optimal_action(self, state: int) -> int:
        position, has_key = self.decode_state(state)
        target_position = self.config.goal_position if has_key else self.config.key_position
        return 1 if position < target_position else 0

    def reset(self) -> int:
        self._position = int(self.rng.choice(self.config.start_positions))
        self._has_key = False
        self._step_count = 0
        self._done = False
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        if self._done:
            return self.state, 0.0, True, {"warning": "Episode already terminated."}
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}. Valid actions are 0 and 1.")

        self._step_count += 1
        slipped = bool(self.rng.random() < self.config.slip_probability)
        executed_action = 1 - action if slipped else action
        delta = -1 if executed_action == 0 else 1

        reward = self.config.step_penalty
        target_position = self.config.goal_position if self._has_key else self.config.key_position
        prev_distance = abs(self._position - target_position)
        proposed_position = int(np.clip(self._position + delta, 0, self.config.n_positions - 1))
        hit_wall = proposed_position == self._position and (
            (delta < 0 and self._position == 0)
            or (delta > 0 and self._position == self.config.n_positions - 1)
        )

        if hit_wall:
            reward += self.config.wall_penalty
        elif proposed_position == self.config.goal_position and not self._has_key:
            proposed_position = self._position
            reward += self.config.locked_goal_penalty
        else:
            self._position = proposed_position

        target_position = self.config.goal_position if self._has_key else self.config.key_position
        new_distance = abs(self._position - target_position)
        reward += self.config.progress_reward_scale * float(prev_distance - new_distance)

        key_collected = False
        if self._position == self.config.key_position and not self._has_key:
            self._has_key = True
            key_collected = True
            reward += self.config.key_reward

        reached_goal = self._position == self.config.goal_position and self._has_key
        if reached_goal:
            reward += self.config.goal_reward

        timed_out = self._step_count >= self.config.max_episode_steps
        self._done = reached_goal or timed_out

        info = {
            "step": self._step_count,
            "state_name": self.state_label(self.state),
            "action_name": self.ACTION_MEANINGS[action],
            "executed_action_name": self.ACTION_MEANINGS[executed_action],
            "slipped": slipped,
            "hit_wall": hit_wall,
            "key_collected": key_collected,
            "has_key": self._has_key,
            "reached_goal": reached_goal,
            "timed_out": timed_out,
            "optimal_action": self.optimal_action(self.state),
        }
        return self.state, float(reward), self._done, info

    def render(self) -> str:
        corridor = []
        for pos in range(self.config.n_positions):
            token = "."
            if pos == self.config.key_position:
                token = "K"
            if pos == self.config.goal_position:
                token = "G"
            if pos == self._position:
                token = "A"
            corridor.append(token)
        return (
            f"{''.join(corridor)} | step={self._step_count} | "
            f"has_key={int(self._has_key)}"
        )

    def all_state_labels(self) -> list[str]:
        return [self.state_label(state) for state in range(self.observation_space)]


class SimpleNavigationEnv(KeyDoorNavigationEnv):
    """Backward-compatible alias for older imports."""


class KeyDoorNavigationGymEnv:
    """Thin Gymnasium-style wrapper around the sequential task."""

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.env = KeyDoorNavigationEnv(config=config)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        del options
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset(), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        next_state, reward, done, info = self.env.step(action)
        terminated = done and info["reached_goal"]
        truncated = done and info["timed_out"]
        return next_state, reward, terminated, truncated, info

    def render(self) -> None:
        print(self.env.render())

    def close(self) -> None:
        return None
