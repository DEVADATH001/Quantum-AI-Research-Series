"""Evaluation and plotting utilities for the upgraded QRL benchmark."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agent.quantum_policy import QuantumPolicyNetwork
from environments.simple_nav_env import EnvironmentConfig, KeyDoorNavigationEnv
from src.baselines import MLPSoftmaxPolicy, TabularSoftmaxPolicy
from src.research_stats import summarize_scalar_distribution
from src.runtime_executor import QuantumRuntimeExecutor

logger = logging.getLogger(__name__)


def _palette(n: int) -> list[str]:
    base = ["#2f6f8f", "#bf5b17", "#3a9d5d", "#7f3c8d", "#c98a1d", "#5b8c5a"]
    if n <= len(base):
        return base[:n]
    repeats = int(np.ceil(n / len(base)))
    return (base * repeats)[:n]


def moving_average(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def normalized_auc(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    return float(np.trapezoid(arr, dx=1.0) / (arr.size - 1))


def find_convergence_episode(
    successes: list[bool] | list[float],
    threshold: float,
    window: int,
) -> int | None:
    success_ma = moving_average(np.asarray(successes, dtype=float), window=window)
    crossed = np.where(success_ma >= threshold)[0]
    if crossed.size == 0:
        return None
    return int(crossed[0] + 1)


def convergence_runtime_seconds(
    episode_runtime_sec: list[float],
    convergence_episode: int | None,
) -> float:
    runtimes = np.asarray(episode_runtime_sec, dtype=float)
    if runtimes.size == 0:
        return 0.0
    if convergence_episode is None:
        return float(runtimes.sum())
    return float(runtimes[: max(1, convergence_episode)].sum())


def evaluate_quantum_policy(
    policy: QuantumPolicyNetwork,
    executor: QuantumRuntimeExecutor,
    parameters: np.ndarray,
    env_config: EnvironmentConfig,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    env_payload = asdict(env_config)
    env_payload["seed"] = seed
    env = KeyDoorNavigationEnv(config=EnvironmentConfig(**env_payload))
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        for _ in range(env.config.max_episode_steps):
            probs = policy.action_probabilities(state=state, parameters=parameters, executor=executor)
            action = int(rng.choice(policy.n_actions, p=probs))
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if done:
                success = bool(info.get("reached_goal", False))
                break
        rewards.append(total_reward)
        successes.append(success)
        lengths.append(steps)

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "avg_length": float(np.mean(lengths)),
    }


def evaluate_tabular_policy(
    parameters: np.ndarray,
    env_config: EnvironmentConfig,
    n_episodes: int,
    seed: int,
    temperature: float = 1.0,
) -> dict[str, float]:
    env_payload = asdict(env_config)
    env_payload["seed"] = seed
    env = KeyDoorNavigationEnv(config=EnvironmentConfig(**env_payload))
    policy = TabularSoftmaxPolicy(
        n_states=env.observation_space,
        n_actions=env.action_space,
        temperature=temperature,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    rewards: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        for _ in range(env.config.max_episode_steps):
            probs = policy.action_probabilities(state, parameters)
            action = int(rng.choice(policy.n_actions, p=probs))
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if done:
                success = bool(info.get("reached_goal", False))
                break
        rewards.append(total_reward)
        successes.append(success)
        lengths.append(steps)

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "avg_length": float(np.mean(lengths)),
    }


def evaluate_mlp_policy(
    parameters: np.ndarray,
    env_config: EnvironmentConfig,
    n_episodes: int,
    seed: int,
    hidden_dim: int = 16,
    temperature: float = 1.0,
) -> dict[str, float]:
    env_payload = asdict(env_config)
    env_payload["seed"] = seed
    env = KeyDoorNavigationEnv(config=EnvironmentConfig(**env_payload))
    policy = MLPSoftmaxPolicy(
        n_states=env.observation_space,
        n_actions=env.action_space,
        hidden_dim=hidden_dim,
        temperature=temperature,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    rewards: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        for _ in range(env.config.max_episode_steps):
            probs = policy.action_probabilities(state, parameters)
            action = int(rng.choice(policy.n_actions, p=probs))
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if done:
                success = bool(info.get("reached_goal", False))
                break
        rewards.append(total_reward)
        successes.append(success)
        lengths.append(steps)

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "avg_length": float(np.mean(lengths)),
    }


def summarize_history(
    history: dict[str, Any],
    convergence_threshold: float,
    moving_avg_window: int,
) -> dict[str, float | int | None]:
    convergence_episode = find_convergence_episode(
        history["episode_success"],
        threshold=convergence_threshold,
        window=moving_avg_window,
    )
    return {
        "convergence_episode": convergence_episode,
        "convergence_runtime_sec": convergence_runtime_seconds(
            history["episode_runtime_sec"],
            convergence_episode,
        ),
        "final_avg_reward": float(np.mean(history["episode_rewards"][-moving_avg_window:])),
        "final_success_rate": float(np.mean(history["episode_success"][-moving_avg_window:])),
        "reward_auc": normalized_auc(history["episode_rewards"]),
        "success_auc": normalized_auc(np.asarray(history["episode_success"], dtype=float)),
        "grad_norm_final": float(np.mean(history.get("grad_norm_history", [0.0])[-moving_avg_window:])),
        "total_runtime_sec": float(history["total_runtime_sec"]),
    }


def aggregate_histories(
    records: list[dict[str, Any]],
    convergence_threshold: float,
    moving_avg_window: int,
) -> dict[str, Any]:
    if not records:
        return {}

    reward_curves = np.asarray([record["episode_rewards"] for record in records], dtype=float)
    success_curves = np.asarray([record["episode_success"] for record in records], dtype=float)
    length_curves = np.asarray([record["episode_lengths"] for record in records], dtype=float)
    grad_curves = np.asarray([record.get("grad_norm_history", []) for record in records], dtype=float)

    reward_ma = np.asarray([moving_average(curve, moving_avg_window) for curve in reward_curves], dtype=float)
    success_ma = np.asarray([moving_average(curve, moving_avg_window) for curve in success_curves], dtype=float)

    per_seed_metrics = [
        summarize_history(record, convergence_threshold, moving_avg_window) for record in records
    ]
    convergence_values = [m["convergence_episode"] for m in per_seed_metrics if m["convergence_episode"] is not None]
    eval_rewards = [record["evaluation"]["avg_reward"] for record in records if "evaluation" in record]
    eval_success = [record["evaluation"]["success_rate"] for record in records if "evaluation" in record]
    distribution_fields = {
        "final_avg_reward": [m["final_avg_reward"] for m in per_seed_metrics],
        "final_success_rate": [m["final_success_rate"] for m in per_seed_metrics],
        "reward_auc": [m["reward_auc"] for m in per_seed_metrics],
        "success_auc": [m["success_auc"] for m in per_seed_metrics],
        "grad_norm_final": [m["grad_norm_final"] for m in per_seed_metrics],
        "convergence_episode": convergence_values,
        "convergence_runtime": [m["convergence_runtime_sec"] for m in per_seed_metrics],
        "total_runtime_sec": [m["total_runtime_sec"] for m in per_seed_metrics],
        "eval_reward": eval_rewards,
        "eval_success": eval_success,
    }

    aggregate = {
        "seeds": [record["seed"] for record in records],
        "seed_count": int(len(records)),
        "reward_curve_mean": reward_curves.mean(axis=0).tolist(),
        "reward_curve_std": reward_curves.std(axis=0).tolist(),
        "success_curve_mean": success_curves.mean(axis=0).tolist(),
        "success_curve_std": success_curves.std(axis=0).tolist(),
        "length_curve_mean": length_curves.mean(axis=0).tolist(),
        "length_curve_std": length_curves.std(axis=0).tolist(),
        "reward_ma_mean": reward_ma.mean(axis=0).tolist(),
        "reward_ma_std": reward_ma.std(axis=0).tolist(),
        "success_ma_mean": success_ma.mean(axis=0).tolist(),
        "success_ma_std": success_ma.std(axis=0).tolist(),
        "grad_norm_curve_mean": grad_curves.mean(axis=0).tolist()
        if grad_curves.ndim == 2 and grad_curves.shape[1] > 0
        else [],
        "grad_norm_curve_std": grad_curves.std(axis=0).tolist()
        if grad_curves.ndim == 2 and grad_curves.shape[1] > 0
        else [],
        "final_avg_reward_mean": float(np.mean([m["final_avg_reward"] for m in per_seed_metrics])),
        "final_avg_reward_std": float(np.std([m["final_avg_reward"] for m in per_seed_metrics])),
        "final_success_rate_mean": float(np.mean([m["final_success_rate"] for m in per_seed_metrics])),
        "final_success_rate_std": float(np.std([m["final_success_rate"] for m in per_seed_metrics])),
        "reward_auc_mean": float(np.mean([m["reward_auc"] for m in per_seed_metrics])),
        "reward_auc_std": float(np.std([m["reward_auc"] for m in per_seed_metrics])),
        "success_auc_mean": float(np.mean([m["success_auc"] for m in per_seed_metrics])),
        "success_auc_std": float(np.std([m["success_auc"] for m in per_seed_metrics])),
        "grad_norm_final_mean": float(np.mean([m["grad_norm_final"] for m in per_seed_metrics])),
        "grad_norm_final_std": float(np.std([m["grad_norm_final"] for m in per_seed_metrics])),
        "convergence_rate": float(len(convergence_values) / len(records)),
        "convergence_episode_mean": float(np.mean(convergence_values)) if convergence_values else None,
        "convergence_episode_std": float(np.std(convergence_values)) if convergence_values else None,
        "convergence_runtime_mean": float(np.mean([m["convergence_runtime_sec"] for m in per_seed_metrics])),
        "convergence_runtime_std": float(np.std([m["convergence_runtime_sec"] for m in per_seed_metrics])),
        "total_runtime_mean": float(np.mean([m["total_runtime_sec"] for m in per_seed_metrics])),
        "total_runtime_std": float(np.std([m["total_runtime_sec"] for m in per_seed_metrics])),
        "eval_reward_mean": float(np.mean(eval_rewards)) if eval_rewards else None,
        "eval_reward_std": float(np.std(eval_rewards)) if eval_rewards else None,
        "eval_success_mean": float(np.mean(eval_success)) if eval_success else None,
        "eval_success_std": float(np.std(eval_success)) if eval_success else None,
        "per_seed_metrics": per_seed_metrics,
    }
    for prefix, values in distribution_fields.items():
        stats = summarize_scalar_distribution(values)
        aggregate[f"{prefix}_count"] = stats["count"]
        aggregate[f"{prefix}_sem"] = stats["sem"]
        aggregate[f"{prefix}_ci95"] = stats["ci95"]
    return aggregate


def plot_learning_curves(
    aggregate_histories_by_label: dict[str, dict[str, Any]],
    output_path: str | Path,
    moving_avg_window: int = 10,
    random_baseline_reward: float | None = None,
) -> None:
    plt.figure(figsize=(10, 6), dpi=150)
    for label, summary in aggregate_histories_by_label.items():
        reward_ma_mean = np.asarray(summary.get("reward_ma_mean", []), dtype=float)
        reward_ma_std = np.asarray(summary.get("reward_ma_std", []), dtype=float)
        if reward_ma_mean.size == 0:
            continue
        episodes = np.arange(1, reward_ma_mean.size + 1)
        plt.plot(episodes, reward_ma_mean, linewidth=2.0, label=label)
        plt.fill_between(
            episodes,
            reward_ma_mean - reward_ma_std,
            reward_ma_mean + reward_ma_std,
            alpha=0.15,
        )

    if random_baseline_reward is not None:
        plt.axhline(
            random_baseline_reward,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="Random baseline",
        )

    plt.title(f"Learning Curves (moving average window = {moving_avg_window})")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.legend()
    plt.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved learning curves to %s", output_path)


def plot_convergence_comparison(
    aggregate_histories_by_label: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    labels = list(aggregate_histories_by_label.keys())
    means = [
        np.nan if aggregate_histories_by_label[label].get("convergence_episode_mean") is None
        else float(aggregate_histories_by_label[label]["convergence_episode_mean"])
        for label in labels
    ]
    stds = [
        0.0 if aggregate_histories_by_label[label].get("convergence_episode_std") is None
        else float(aggregate_histories_by_label[label]["convergence_episode_std"])
        for label in labels
    ]

    plt.figure(figsize=(9, 5), dpi=150)
    x = np.arange(len(labels))
    bars = plt.bar(x, means, yerr=stds, capsize=4, color=_palette(len(labels)))
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Convergence Episode")
    plt.title("Convergence Comparison")
    plt.grid(axis="y", alpha=0.3)
    for bar, mean_value in zip(bars, means):
        if np.isnan(mean_value):
            plt.text(bar.get_x() + bar.get_width() / 2.0, 0.1, "NR", ha="center", va="bottom", fontsize=9)
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{mean_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved convergence comparison to %s", output_path)


def plot_baseline_comparison(
    aggregate_histories_by_label: dict[str, dict[str, Any]],
    output_path: str | Path,
    metric_key: str = "eval_success_mean",
    ylabel: str = "Evaluation Success Rate",
) -> None:
    labels = list(aggregate_histories_by_label.keys())
    means = [float(aggregate_histories_by_label[label].get(metric_key) or 0.0) for label in labels]
    stds = [
        float(aggregate_histories_by_label[label].get(metric_key.replace("_mean", "_std")) or 0.0)
        for label in labels
    ]

    plt.figure(figsize=(9, 5), dpi=150)
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4, color=_palette(len(labels)))
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title("Method Comparison")
    plt.grid(axis="y", alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved baseline comparison to %s", output_path)


def extract_quantum_policy_matrix(
    policy: QuantumPolicyNetwork,
    executor: QuantumRuntimeExecutor,
    parameters: np.ndarray,
    n_states: int,
) -> np.ndarray:
    return np.asarray(
        [
            policy.action_probabilities(state=state, parameters=parameters, executor=executor)
            for state in range(n_states)
        ],
        dtype=float,
    )


def extract_tabular_policy_matrix(parameters: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    matrix = []
    for row in np.asarray(parameters, dtype=float):
        matrix.append(np.exp((row - np.max(row)) / max(temperature, 1e-8)))
    probs = np.asarray(matrix, dtype=float)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def extract_mlp_policy_matrix(
    parameters: np.ndarray,
    n_states: int,
    n_actions: int,
    hidden_dim: int = 16,
    temperature: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    policy = MLPSoftmaxPolicy(
        n_states=n_states,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
        temperature=temperature,
        seed=seed,
    )
    return np.asarray(
        [policy.action_probabilities(state, parameters) for state in range(n_states)],
        dtype=float,
    )


def plot_policy_heatmaps(
    policy_matrices: dict[str, np.ndarray],
    state_labels: list[str],
    action_labels: list[str],
    output_path: str | Path,
) -> None:
    n_plots = len(policy_matrices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), dpi=150, squeeze=False)

    for ax, (title, matrix) in zip(axes[0], policy_matrices.items()):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Action")
        ax.set_ylabel("State")
        ax.set_xticks(np.arange(len(action_labels)))
        ax.set_xticklabels(action_labels)
        ax.set_yticks(np.arange(len(state_labels)))
        ax.set_yticklabels(state_labels)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved policy heatmaps to %s", output_path)
