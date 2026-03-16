"""Author: DEVADATH H K

Project: Quantum RL Noise Mitigation

Evaluation and plotting utilities for QRL stability analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agent.quantum_policy import QuantumPolicyNetwork
from src.runtime_executor import QuantumRuntimeExecutor

logger = logging.getLogger(__name__)

def moving_average(values: list[float], window: int) -> np.ndarray:
    """Compute moving average with edge padding."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    kernel = np.ones(window) / float(window)
    return np.convolve(arr, kernel, mode="same")

def find_convergence_episode(
    rewards: list[float],
    threshold: float,
    window: int,
) -> int | None:
    """Return first episode index (1-based) where moving average crosses threshold."""
    ma = moving_average(rewards, window=window)
    crossed = np.where(ma >= threshold)[0]
    if crossed.size == 0:
        return None
    return int(crossed[0] + 1)

def convergence_runtime_seconds(
    episode_runtime_sec: list[float],
    convergence_episode: int | None,
) -> float:
    """Return cumulative runtime until convergence or full runtime if not converged."""
    runtimes = np.asarray(episode_runtime_sec, dtype=float)
    if runtimes.size == 0:
        return 0.0
    if convergence_episode is None:
        return float(runtimes.sum())
    cutoff = max(1, convergence_episode)
    return float(runtimes[:cutoff].sum())

def plot_learning_curves(
    histories: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_avg_window: int = 10,
) -> None:
    """Plot reward-vs-episode curves for ideal/noisy/mitigated modes."""
    plt.figure(figsize=(10, 6), dpi=150)
    for mode, record in histories.items():
        rewards = record["episode_rewards"]
        episodes = np.arange(1, len(rewards) + 1)
        ma = moving_average(rewards, moving_avg_window)
        plt.plot(episodes, ma, linewidth=2.0, label=f"{mode} (MA)")
        plt.plot(episodes, rewards, linewidth=0.8, alpha=0.25)
    plt.title("Quantum RL Learning Curves")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved learning curves to %s", output_path)

def plot_convergence_comparison(
    mode_summary: dict[str, dict[str, float | int | None]],
    output_path: str | Path,
) -> None:
    """Plot convergence runtime comparison across execution modes."""
    modes = list(mode_summary.keys())
    runtimes = [float(mode_summary[m]["convergence_runtime_sec"]) for m in modes]

    plt.figure(figsize=(8, 5), dpi=150)
    bars = plt.bar(modes, runtimes, color=["#2ca02c", "#d62728", "#1f77b4"][: len(modes)])
    plt.title("Convergence Time by Execution Mode")
    plt.ylabel("Time to Convergence (seconds)")
    plt.grid(axis="y", alpha=0.3)
    for bar, runtime in zip(bars, runtimes):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{runtime:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved convergence comparison to %s", output_path)

def plot_final_policy(
    policy: QuantumPolicyNetwork,
    executor: QuantumRuntimeExecutor,
    parameters: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot final action probabilities for each state."""
    states = [0, 1]
    labels = ["Move Left", "Move Right"]
    probs_by_state = [
        policy.action_probabilities(state=s, parameters=parameters, executor=executor) for s in states
    ]

    x = np.arange(len(states))
    width = 0.35

    plt.figure(figsize=(8, 5), dpi=150)
    left_probs = [p[0] for p in probs_by_state]
    right_probs = [p[1] for p in probs_by_state]
    plt.bar(x - width / 2.0, left_probs, width, label=labels[0], color="#ff7f0e")
    plt.bar(x + width / 2.0, right_probs, width, label=labels[1], color="#2ca02c")
    plt.xticks(x, ["Searching", "Target Found"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Action Probability")
    plt.title("Final Learned Policy")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved final policy plot to %s", output_path)

