"""Classical baselines for the sequential navigation benchmark."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from environments.simple_nav_env import EnvironmentConfig, KeyDoorNavigationEnv
from src.optim import AdamOptimizer
from src.rl_utils import (
    baseline_adjusted_returns,
    discounted_returns,
    generalized_advantage_estimation,
    update_timestep_baseline,
)
from utils.qiskit_helpers import softmax

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TabularBaselineConfig:
    """Training hyperparameters for the tabular REINFORCE baseline."""

    n_episodes: int = 80
    gamma: float = 0.99
    learning_rate: float = 0.05
    max_episode_steps: int = 8
    baseline_decay: float | None = 0.9
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0
    temperature: float = 1.0


@dataclass(slots=True)
class MLPBaselineConfig:
    """Training hyperparameters for the MLP REINFORCE baseline."""

    n_episodes: int = 80
    gamma: float = 0.99
    learning_rate: float = 0.03
    max_episode_steps: int = 8
    baseline_decay: float | None = 0.9
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0
    hidden_dim: int = 16
    init_scale: float = 0.1
    temperature: float = 1.0


@dataclass(slots=True)
class MLPActorCriticConfig:
    """Training hyperparameters for the MLP actor-critic baseline."""

    n_episodes: int = 80
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_learning_rate: float = 0.03
    critic_learning_rate: float = 0.02
    max_episode_steps: int = 8
    baseline_decay: float | None = None
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0
    actor_hidden_dim: int = 16
    critic_hidden_dim: int = 16
    init_scale: float = 0.1
    temperature: float = 1.0
    value_loss_coeff: float = 0.5


@dataclass(slots=True)
class BaselineTrainingHistory:
    """Training outputs for a classical baseline."""

    name: str
    episode_rewards: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_runtime_sec: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    value_loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    total_runtime_sec: float = 0.0
    final_parameters: list[Any] = field(default_factory=list)


class TabularSoftmaxPolicy:
    """Tabular softmax policy for discrete-state environments."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def initial_parameters(self, scale: float = 0.05) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=scale, size=(self.n_states, self.n_actions))

    def action_probabilities(self, state: int, parameters: np.ndarray) -> np.ndarray:
        return softmax(parameters[state], temperature=self.temperature)

    def sample_action(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(self.n_actions, p=probs))


class MLPSoftmaxPolicy:
    """Small one-hidden-layer softmax policy for discrete-state environments."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_dim: int = 16,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1.")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.parameter_count = (
            self.n_states * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.n_actions
            + self.n_actions
        )

    def initial_parameters(self, scale: float = 0.1) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=scale, size=self.parameter_count)

    def _unpack(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        flat = np.asarray(parameters, dtype=float).reshape(-1)
        if flat.size != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} parameters for the MLP policy, received {flat.size}."
            )

        idx = 0
        w1_size = self.n_states * self.hidden_dim
        w1 = flat[idx : idx + w1_size].reshape(self.n_states, self.hidden_dim)
        idx += w1_size
        b1 = flat[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w2_size = self.hidden_dim * self.n_actions
        w2 = flat[idx : idx + w2_size].reshape(self.hidden_dim, self.n_actions)
        idx += w2_size
        b2 = flat[idx : idx + self.n_actions]
        return w1, b1, w2, b2

    def _one_hot(self, state: int) -> np.ndarray:
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State {state} is outside [0, {self.n_states}).")
        features = np.zeros(self.n_states, dtype=float)
        features[state] = 1.0
        return features

    def forward(self, state: int, parameters: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        w1, b1, w2, b2 = self._unpack(parameters)
        x = self._one_hot(state)
        hidden_pre = x @ w1 + b1
        hidden = np.tanh(hidden_pre)
        logits = hidden @ w2 + b2
        probs = softmax(logits, temperature=self.temperature)
        return probs, {
            "x": x,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
            "logits": logits,
            "probs": probs,
            "w2": w2,
        }

    def action_probabilities(self, state: int, parameters: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(state, parameters)
        return probs

    def sample_action(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(self.n_actions, p=probs))

    def gradient_step(
        self,
        cache: dict[str, np.ndarray],
        action: int,
        weight: float,
        entropy_coeff: float,
    ) -> np.ndarray:
        x = cache["x"]
        hidden_pre = cache["hidden_pre"]
        hidden = cache["hidden"]
        probs = cache["probs"]
        w2 = cache["w2"]

        temperature = max(self.temperature, 1e-8)
        one_hot_action = np.zeros(self.n_actions, dtype=float)
        one_hot_action[action] = 1.0
        avg_log_p = float(np.sum(probs * np.log(probs + 1e-9)))
        d_logits = (float(weight) * (probs - one_hot_action)) / temperature
        d_logits += (
            float(entropy_coeff) * probs * (np.log(probs + 1e-9) - avg_log_p)
        ) / temperature

        grad_w2 = np.outer(hidden, d_logits)
        grad_b2 = d_logits
        d_hidden = w2 @ d_logits
        d_hidden_pre = d_hidden * (1.0 - np.tanh(hidden_pre) ** 2)
        grad_w1 = np.outer(x, d_hidden_pre)
        grad_b1 = d_hidden_pre

        return np.concatenate(
            [
                grad_w1.reshape(-1),
                grad_b1.reshape(-1),
                grad_w2.reshape(-1),
                grad_b2.reshape(-1),
            ]
        )


class MLPValueFunction:
    """Small one-hidden-layer value network on one-hot state features."""

    def __init__(
        self,
        n_states: int,
        hidden_dim: int = 16,
        seed: int = 42,
    ) -> None:
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1.")
        self.n_states = int(n_states)
        self.hidden_dim = int(hidden_dim)
        self.rng = np.random.default_rng(seed)
        self.parameter_count = self.n_states * self.hidden_dim + self.hidden_dim + self.hidden_dim + 1

    def initial_parameters(self, scale: float = 0.1) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=scale, size=self.parameter_count)

    def _unpack(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        flat = np.asarray(parameters, dtype=float).reshape(-1)
        if flat.size != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} parameters for the value network, received {flat.size}."
            )
        idx = 0
        w1_size = self.n_states * self.hidden_dim
        w1 = flat[idx : idx + w1_size].reshape(self.n_states, self.hidden_dim)
        idx += w1_size
        b1 = flat[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w2 = flat[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        b2 = float(flat[idx])
        return w1, b1, w2, b2

    def _one_hot(self, state: int) -> np.ndarray:
        if state < 0 or state >= self.n_states:
            raise ValueError(f"State {state} is outside [0, {self.n_states}).")
        x = np.zeros(self.n_states, dtype=float)
        x[state] = 1.0
        return x

    def forward(self, state: int, parameters: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        w1, b1, w2, b2 = self._unpack(parameters)
        x = self._one_hot(state)
        hidden_pre = x @ w1 + b1
        hidden = np.tanh(hidden_pre)
        value = float(hidden @ w2 + b2)
        return value, {
            "x": x,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
            "w2": w2,
        }

    def value(self, state: int, parameters: np.ndarray) -> float:
        value, _ = self.forward(state, parameters)
        return value

    def gradient_step(
        self,
        cache: dict[str, np.ndarray],
        target: float,
        predicted_value: float,
    ) -> tuple[np.ndarray, float]:
        error = float(predicted_value - target)
        hidden = cache["hidden"]
        hidden_pre = cache["hidden_pre"]
        x = cache["x"]
        w2 = cache["w2"]

        d_value = error
        grad_w2 = hidden * d_value
        grad_b2 = np.array([d_value], dtype=float)
        d_hidden = w2 * d_value
        d_hidden_pre = d_hidden * (1.0 - np.tanh(hidden_pre) ** 2)
        grad_w1 = np.outer(x, d_hidden_pre)
        grad_b1 = d_hidden_pre
        grads = np.concatenate(
            [
                grad_w1.reshape(-1),
                grad_b1.reshape(-1),
                grad_w2.reshape(-1),
                grad_b2,
            ]
        )
        return grads, 0.5 * (error**2)

def _rollout_policy(
    env: KeyDoorNavigationEnv,
    policy: Any,
    parameters: np.ndarray,
    max_episode_steps: int,
) -> tuple[list[int], list[int], list[float], bool]:
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    success = False

    state = env.reset()
    for _ in range(max_episode_steps):
        probs = policy.action_probabilities(state, parameters)
        action = policy.sample_action(probs)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
        state = next_state
        if done:
            success = bool(info.get("reached_goal", False))
            break
    return states, actions, rewards, success

def train_tabular_reinforce(
    env_config: EnvironmentConfig,
    config: TabularBaselineConfig | None = None,
) -> BaselineTrainingHistory:
    """Train a tabular softmax REINFORCE baseline."""

    cfg = config or TabularBaselineConfig()
    env = KeyDoorNavigationEnv(config=env_config)
    policy = TabularSoftmaxPolicy(
        n_states=env.observation_space,
        n_actions=env.action_space,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )
    params = policy.initial_parameters()
    optimizer = AdamOptimizer(lr=cfg.learning_rate, grad_clip=cfg.grad_clip)
    history = BaselineTrainingHistory(name="tabular_reinforce")
    timestep_baseline = np.zeros(cfg.max_episode_steps, dtype=float)
    train_start = time.perf_counter()

    for episode in range(1, cfg.n_episodes + 1):
        episode_start = time.perf_counter()
        states, actions, rewards, success = _rollout_policy(
            env=env,
            policy=policy,
            parameters=params,
            max_episode_steps=cfg.max_episode_steps,
        )
        returns = discounted_returns(rewards, gamma=cfg.gamma)
        advantages = baseline_adjusted_returns(
            baseline_buffer=timestep_baseline,
            returns=returns,
            decay=cfg.baseline_decay,
        )

        grads = np.zeros_like(params)
        loss = 0.0
        for state, action, weight in zip(states, actions, advantages):
            probs = policy.action_probabilities(state, params)
            d_log_pi = -probs
            d_log_pi[action] += 1.0
            d_log_pi /= max(policy.temperature, 1e-8)

            avg_log_p = np.sum(probs * np.log(probs + 1e-9))
            d_entropy = -(probs * (np.log(probs + 1e-9) - avg_log_p)) / max(policy.temperature, 1e-8)
            grads[state] += -float(weight) * d_log_pi - cfg.entropy_coeff * d_entropy

            loss += -float(weight) * np.log(np.clip(probs[action], 1e-9, 1.0))
            loss -= cfg.entropy_coeff * float(-np.sum(probs * np.log(probs + 1e-9)))

        update_timestep_baseline(
            baseline_buffer=timestep_baseline,
            returns=returns,
            decay=cfg.baseline_decay,
        )

        grad_norm = float(np.linalg.norm(grads))
        params, _ = optimizer.step(params=params, grads=grads)
        episode_runtime = time.perf_counter() - episode_start

        history.episode_rewards.append(float(np.sum(rewards)))
        history.episode_success.append(success)
        history.episode_lengths.append(len(rewards))
        history.episode_runtime_sec.append(episode_runtime)
        history.loss_history.append(float(loss))
        history.grad_norm_history.append(grad_norm)

        if episode % cfg.log_every == 0 or episode == 1:
            logger.info(
                "baseline=tabular episode=%d avg_reward=%.3f success=%.1f%%",
                episode,
                float(np.mean(history.episode_rewards[-cfg.log_every :])),
                float(np.mean(history.episode_success[-cfg.log_every :])) * 100.0,
            )

    history.total_runtime_sec = time.perf_counter() - train_start
    history.final_parameters = params.tolist()
    return history


def train_mlp_reinforce(
    env_config: EnvironmentConfig,
    config: MLPBaselineConfig | None = None,
) -> BaselineTrainingHistory:
    """Train a small classical MLP policy with REINFORCE."""

    cfg = config or MLPBaselineConfig()
    env = KeyDoorNavigationEnv(config=env_config)
    policy = MLPSoftmaxPolicy(
        n_states=env.observation_space,
        n_actions=env.action_space,
        hidden_dim=cfg.hidden_dim,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )
    params = policy.initial_parameters(scale=cfg.init_scale)
    optimizer = AdamOptimizer(lr=cfg.learning_rate, grad_clip=cfg.grad_clip)
    history = BaselineTrainingHistory(name="mlp_reinforce")
    timestep_baseline = np.zeros(cfg.max_episode_steps, dtype=float)
    train_start = time.perf_counter()

    for episode in range(1, cfg.n_episodes + 1):
        episode_start = time.perf_counter()
        states, actions, rewards, success = _rollout_policy(
            env=env,
            policy=policy,
            parameters=params,
            max_episode_steps=cfg.max_episode_steps,
        )
        returns = discounted_returns(rewards, gamma=cfg.gamma)
        advantages = baseline_adjusted_returns(
            baseline_buffer=timestep_baseline,
            returns=returns,
            decay=cfg.baseline_decay,
        )

        grads = np.zeros_like(params)
        loss = 0.0
        for state, action, weight in zip(states, actions, advantages):
            probs, cache = policy.forward(state, params)
            grads += policy.gradient_step(
                cache=cache,
                action=action,
                weight=float(weight),
                entropy_coeff=cfg.entropy_coeff,
            )
            loss += -float(weight) * np.log(np.clip(probs[action], 1e-9, 1.0))
            loss -= cfg.entropy_coeff * float(-np.sum(probs * np.log(probs + 1e-9)))

        update_timestep_baseline(
            baseline_buffer=timestep_baseline,
            returns=returns,
            decay=cfg.baseline_decay,
        )

        grad_norm = float(np.linalg.norm(grads))
        params, _ = optimizer.step(params=params, grads=grads)
        episode_runtime = time.perf_counter() - episode_start

        history.episode_rewards.append(float(np.sum(rewards)))
        history.episode_success.append(success)
        history.episode_lengths.append(len(rewards))
        history.episode_runtime_sec.append(episode_runtime)
        history.loss_history.append(float(loss))
        history.grad_norm_history.append(grad_norm)

        if episode % cfg.log_every == 0 or episode == 1:
            logger.info(
                "baseline=mlp episode=%d avg_reward=%.3f success=%.1f%%",
                episode,
                float(np.mean(history.episode_rewards[-cfg.log_every :])),
                float(np.mean(history.episode_success[-cfg.log_every :])) * 100.0,
            )

    history.total_runtime_sec = time.perf_counter() - train_start
    history.final_parameters = params.tolist()
    return history


def train_mlp_actor_critic(
    env_config: EnvironmentConfig,
    config: MLPActorCriticConfig | None = None,
) -> BaselineTrainingHistory:
    """Train a small classical actor-critic with GAE."""

    cfg = config or MLPActorCriticConfig()
    env = KeyDoorNavigationEnv(config=env_config)
    actor = MLPSoftmaxPolicy(
        n_states=env.observation_space,
        n_actions=env.action_space,
        hidden_dim=cfg.actor_hidden_dim,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )
    critic = MLPValueFunction(
        n_states=env.observation_space,
        hidden_dim=cfg.critic_hidden_dim,
        seed=cfg.seed + 1,
    )
    actor_params = actor.initial_parameters(scale=cfg.init_scale)
    critic_params = critic.initial_parameters(scale=cfg.init_scale)
    actor_optimizer = AdamOptimizer(lr=cfg.actor_learning_rate, grad_clip=cfg.grad_clip)
    critic_optimizer = AdamOptimizer(lr=cfg.critic_learning_rate, grad_clip=cfg.grad_clip)
    history = BaselineTrainingHistory(name="mlp_actor_critic")
    train_start = time.perf_counter()

    for episode in range(1, cfg.n_episodes + 1):
        episode_start = time.perf_counter()
        states, actions, rewards, success = _rollout_policy(
            env=env,
            policy=actor,
            parameters=actor_params,
            max_episode_steps=cfg.max_episode_steps,
        )
        values = np.asarray([critic.value(state, critic_params) for state in states], dtype=float)
        advantages, returns = generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            bootstrap_value=0.0,
        )

        actor_grads = np.zeros_like(actor_params)
        critic_grads = np.zeros_like(critic_params)
        policy_loss = 0.0
        value_loss = 0.0

        for state, action, advantage, target_return in zip(states, actions, advantages, returns):
            probs, actor_cache = actor.forward(state, actor_params)
            actor_grads += actor.gradient_step(
                cache=actor_cache,
                action=action,
                weight=float(advantage),
                entropy_coeff=cfg.entropy_coeff,
            )
            policy_loss += -float(advantage) * np.log(np.clip(probs[action], 1e-9, 1.0))
            policy_loss -= cfg.entropy_coeff * float(-np.sum(probs * np.log(probs + 1e-9)))

            predicted_value, critic_cache = critic.forward(state, critic_params)
            critic_step_grads, sample_value_loss = critic.gradient_step(
                cache=critic_cache,
                target=float(target_return),
                predicted_value=predicted_value,
            )
            critic_grads += critic_step_grads
            value_loss += float(sample_value_loss)

        actor_grad_norm = float(np.linalg.norm(actor_grads))
        actor_params, _ = actor_optimizer.step(params=actor_params, grads=actor_grads)
        critic_params, _ = critic_optimizer.step(
            params=critic_params,
            grads=cfg.value_loss_coeff * critic_grads,
        )

        episode_runtime = time.perf_counter() - episode_start
        history.episode_rewards.append(float(np.sum(rewards)))
        history.episode_success.append(success)
        history.episode_lengths.append(len(rewards))
        history.episode_runtime_sec.append(episode_runtime)
        history.loss_history.append(float(policy_loss))
        history.value_loss_history.append(float(value_loss))
        history.grad_norm_history.append(actor_grad_norm)

        if episode % cfg.log_every == 0 or episode == 1:
            logger.info(
                "baseline=mlp_actor_critic episode=%d avg_reward=%.3f success=%.1f%%",
                episode,
                float(np.mean(history.episode_rewards[-cfg.log_every :])),
                float(np.mean(history.episode_success[-cfg.log_every :])) * 100.0,
            )

    history.total_runtime_sec = time.perf_counter() - train_start
    history.final_parameters = actor_params.tolist()
    return history


def evaluate_random_policy(
    env_config: EnvironmentConfig,
    n_episodes: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Monte-Carlo evaluation for a uniform random policy."""

    env_payload = asdict(env_config)
    env_payload["seed"] = seed
    env = KeyDoorNavigationEnv(config=EnvironmentConfig(**env_payload))
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    successes: list[bool] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        env.reset()
        total_reward = 0.0
        steps = 0
        success = False
        for _ in range(env_config.max_episode_steps):
            action = int(rng.integers(env.action_space))
            _, reward, done, info = env.step(action)
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
