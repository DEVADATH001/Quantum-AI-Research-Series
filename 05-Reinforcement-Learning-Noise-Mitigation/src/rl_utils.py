"""Shared reinforcement-learning utilities."""

from __future__ import annotations

import numpy as np


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Compute reward-to-go returns for a trajectory."""

    returns = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def baseline_adjusted_returns(
    baseline_buffer: np.ndarray,
    returns: np.ndarray,
    decay: float | None,
) -> np.ndarray:
    """Return advantages using an action-independent timestep baseline."""

    if decay is None:
        return returns.copy()
    return returns - baseline_buffer[: returns.size]


def update_timestep_baseline(
    baseline_buffer: np.ndarray,
    returns: np.ndarray,
    decay: float | None,
) -> None:
    """Update a timestep-indexed exponential moving-average baseline in place."""

    if decay is None:
        return
    clipped_decay = float(np.clip(decay, 0.0, 0.999999))
    baseline = baseline_buffer[: returns.size]
    baseline *= clipped_decay
    baseline += (1.0 - clipped_decay) * returns
    baseline_buffer[: returns.size] = baseline


def generalized_advantage_estimation(
    rewards: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and bootstrap returns."""

    rewards_arr = np.asarray(rewards, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if rewards_arr.size != values_arr.size:
        raise ValueError("rewards and values must have matching lengths.")
    if rewards_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    advantages = np.zeros_like(rewards_arr)
    running_advantage = 0.0
    next_value = float(bootstrap_value)
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)

    for idx in range(rewards_arr.size - 1, -1, -1):
        delta = rewards_arr[idx] + gamma * next_value - values_arr[idx]
        running_advantage = delta + gamma * gae_lambda * running_advantage
        advantages[idx] = running_advantage
        next_value = values_arr[idx]

    returns = advantages + values_arr
    return advantages, returns
