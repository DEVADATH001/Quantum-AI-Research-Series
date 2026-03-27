"""Quantum actor-critic learner with a measurement-defined actor and classical critic."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from agent.gradient_estimator import ParameterShiftGradientEstimator
from agent.quantum_policy import QuantumPolicyNetwork
from environments.simple_nav_env import EnvironmentConfig, KeyDoorNavigationEnv
from src.baselines import MLPValueFunction
from src.optim import AdamOptimizer, OptimizerStepStats
from src.rl_utils import generalized_advantage_estimation
from src.runtime_executor import QuantumRuntimeExecutor
from utils.qiskit_helpers import project_probabilities

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QuantumActorCriticConfig:
    """Training hyperparameters for the quantum actor-critic learner."""

    n_episodes: int = 80
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_learning_rate: float = 0.02
    critic_learning_rate: float = 0.02
    max_episode_steps: int = 8
    episodes_per_update: int = 4
    probability_floor: float | None = None
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0
    selection_eval_episodes: int = 4
    track_best_parameters: bool = True
    critic_hidden_dim: int = 16
    critic_init_scale: float = 0.1
    value_loss_coeff: float = 0.5
    lr_plateau_patience: int | None = 3
    lr_decay: float = 0.5
    min_learning_rate: float = 0.002


@dataclass(slots=True)
class ActorCriticTrainingHistory:
    """Training outputs for the quantum actor-critic learner."""

    mode: str
    episode_rewards: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_runtime_sec: list[float] = field(default_factory=list)
    parameter_history: list[list[float]] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    value_loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    applied_grad_norm_history: list[float] = field(default_factory=list)
    update_norm_history: list[float] = field(default_factory=list)
    learning_rate_history: list[float] = field(default_factory=list)
    validation_reward_history: list[float] = field(default_factory=list)
    validation_success_history: list[float] = field(default_factory=list)
    optimizer_clipped_history: list[bool] = field(default_factory=list)
    total_runtime_sec: float = 0.0
    final_parameters: list[float] = field(default_factory=list)
    last_parameters: list[float] = field(default_factory=list)
    best_parameters: list[float] = field(default_factory=list)
    best_episode: int | None = None
    num_skipped_updates: int = 0
    num_lr_decays: int = 0


class QuantumActorCriticLearner:
    """Train a measurement-defined quantum actor using a classical value critic."""

    def __init__(
        self,
        policy: QuantumPolicyNetwork,
        env: KeyDoorNavigationEnv,
        executor: QuantumRuntimeExecutor,
        gradient_estimator: ParameterShiftGradientEstimator,
        config: QuantumActorCriticConfig | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.executor = executor
        self.gradient_estimator = gradient_estimator
        self.config = config or QuantumActorCriticConfig()
        self.actor_optimizer = AdamOptimizer(
            lr=self.config.actor_learning_rate,
            grad_clip=self.config.grad_clip,
        )
        self.critic_optimizer = AdamOptimizer(
            lr=self.config.critic_learning_rate,
            grad_clip=self.config.grad_clip,
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.current_params = self.policy.initial_parameters()
        self.critic = MLPValueFunction(
            n_states=self.env.observation_space,
            hidden_dim=self.config.critic_hidden_dim,
            seed=self.config.seed + 1,
        )
        self.critic_params = self.critic.initial_parameters(scale=self.config.critic_init_scale)
        self._validation_counter = 0
        self._updates_since_improvement = 0

    def _effective_probability_floor(self) -> float:
        if self.config.probability_floor is not None:
            return max(0.0, float(self.config.probability_floor))
        shots = max(2, int(getattr(self.executor.config, "shots", 128)))
        return 0.5 / float(shots)

    def save_model(self, path: str | Path, parameters: np.ndarray) -> None:
        np.save(str(path), parameters)
        logger.info("Model saved to %s", path)

    def _stabilize_probabilities(self, probs: np.ndarray) -> np.ndarray:
        return project_probabilities(probs, floor=self._effective_probability_floor())

    def _rollout(
        self,
        parameters: np.ndarray | None = None,
    ) -> tuple[list[int], list[int], list[float], list[float], bool]:
        states: list[int] = []
        actions: list[int] = []
        rewards: list[float] = []
        values: list[float] = []
        success = False

        active_params = self.current_params if parameters is None else np.asarray(parameters, dtype=float)
        state = self.env.reset()
        for _ in range(self.config.max_episode_steps):
            probs = self._stabilize_probabilities(
                self.policy.action_probabilities(
                    state=state,
                    parameters=active_params,
                    executor=self.executor,
                )
            )
            action = int(self.rng.choice(self.policy.n_actions, p=probs))
            value = self.critic.value(state, self.critic_params)
            next_state, reward, done, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            values.append(float(value))
            state = next_state
            if done:
                success = bool(info.get("reached_goal", False))
                break
        return states, actions, rewards, values, success

    def _validation_metrics(self, parameters: np.ndarray, n_episodes: int) -> tuple[float, float]:
        if n_episodes <= 0:
            return 0.0, 0.0

        env_payload = asdict(self.env.config)
        env_payload["seed"] = self.config.seed + 100_000 + self._validation_counter
        validation_env = KeyDoorNavigationEnv(config=EnvironmentConfig(**env_payload))
        validation_rng = np.random.default_rng(self.config.seed + 200_000 + self._validation_counter)
        self._validation_counter += 1

        rewards: list[float] = []
        successes: list[bool] = []
        for _ in range(n_episodes):
            state = validation_env.reset()
            total_reward = 0.0
            success = False
            for _ in range(validation_env.config.max_episode_steps):
                probs = self._stabilize_probabilities(
                    self.policy.action_probabilities(
                        state=state,
                        parameters=parameters,
                        executor=self.executor,
                    )
                )
                action = int(validation_rng.choice(self.policy.n_actions, p=probs))
                state, reward, done, info = validation_env.step(action)
                total_reward += float(reward)
                if done:
                    success = bool(info.get("reached_goal", False))
                    break
            rewards.append(total_reward)
            successes.append(success)
        return float(np.mean(rewards)), float(np.mean(successes))

    def train(self, initial_parameters: np.ndarray | None = None) -> ActorCriticTrainingHistory:
        self.current_params = (
            np.asarray(initial_parameters, dtype=float)
            if initial_parameters is not None
            else self.policy.initial_parameters()
        )
        self._validation_counter = 0
        self._updates_since_improvement = 0

        history = ActorCriticTrainingHistory(mode=self.executor.mode)
        train_start = time.perf_counter()
        episodes_per_update = max(1, int(self.config.episodes_per_update))
        probability_floor = self._effective_probability_floor()

        batch_actor_grads: list[np.ndarray] = []
        batch_value_grads: list[np.ndarray] = []
        batch_policy_losses: list[float] = []
        batch_value_losses: list[float] = []
        batch_rewards: list[float] = []
        batch_successes: list[bool] = []
        batch_episode_indices: list[int] = []

        best_params = self.current_params.copy()
        best_episode: int | None = None
        best_score = (-np.inf, -np.inf)
        last_step_stats = OptimizerStepStats(0.0, 0.0, 0.0, False)
        last_validation_reward = 0.0
        last_validation_success = 0.0

        for episode in range(1, self.config.n_episodes + 1):
            episode_start = time.perf_counter()
            states, actions, rewards, values, success = self._rollout(parameters=self.current_params)
            advantages, returns = generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                bootstrap_value=0.0,
            )

            actor_grads, policy_loss = self.gradient_estimator.estimate_reinforce_gradient(
                policy=self.policy,
                executor=self.executor,
                states=states,
                actions=actions,
                weights=advantages,
                parameters=self.current_params,
                entropy_coeff=self.config.entropy_coeff,
                probability_floor=probability_floor,
            )

            critic_grads = np.zeros_like(self.critic_params)
            value_loss = 0.0
            for state, target_return in zip(states, returns):
                predicted_value, critic_cache = self.critic.forward(state, self.critic_params)
                step_grads, sample_value_loss = self.critic.gradient_step(
                    cache=critic_cache,
                    target=float(target_return),
                    predicted_value=predicted_value,
                )
                critic_grads += step_grads
                value_loss += float(sample_value_loss)

            total_reward = float(np.sum(rewards))
            history.episode_rewards.append(total_reward)
            history.episode_success.append(success)
            history.episode_lengths.append(len(rewards))
            history.loss_history.append(float(policy_loss))
            history.value_loss_history.append(float(value_loss))

            batch_actor_grads.append(actor_grads)
            batch_value_grads.append(critic_grads)
            batch_policy_losses.append(float(policy_loss))
            batch_value_losses.append(float(value_loss))
            batch_rewards.append(total_reward)
            batch_successes.append(success)
            batch_episode_indices.append(len(history.episode_rewards) - 1)

            if len(batch_actor_grads) >= episodes_per_update or episode == self.config.n_episodes:
                mean_actor_grad = np.mean(np.stack(batch_actor_grads, axis=0), axis=0)
                mean_value_grad = np.mean(np.stack(batch_value_grads, axis=0), axis=0)

                if not np.all(np.isfinite(mean_actor_grad)):
                    history.num_skipped_updates += 1
                    last_step_stats = OptimizerStepStats(0.0, 0.0, 0.0, False)
                else:
                    self.current_params, last_step_stats = self.actor_optimizer.step(
                        params=self.current_params,
                        grads=mean_actor_grad,
                    )
                    self.critic_params, _ = self.critic_optimizer.step(
                        params=self.critic_params,
                        grads=self.config.value_loss_coeff * mean_value_grad,
                    )

                if self.config.selection_eval_episodes > 0:
                    last_validation_reward, last_validation_success = self._validation_metrics(
                        parameters=self.current_params,
                        n_episodes=self.config.selection_eval_episodes,
                    )
                else:
                    last_validation_reward = float(np.mean(batch_rewards))
                    last_validation_success = float(np.mean(batch_successes))

                current_score = (last_validation_success, last_validation_reward)
                if current_score > best_score:
                    best_score = current_score
                    best_params = self.current_params.copy()
                    best_episode = episode
                    self._updates_since_improvement = 0
                else:
                    self._updates_since_improvement += 1

                if (
                    self.config.lr_plateau_patience is not None
                    and self._updates_since_improvement >= self.config.lr_plateau_patience
                ):
                    new_lr = max(self.config.min_learning_rate, self.actor_optimizer.lr * self.config.lr_decay)
                    if new_lr < self.actor_optimizer.lr - 1e-12:
                        self.actor_optimizer.set_learning_rate(new_lr)
                        history.num_lr_decays += 1
                        logger.info(
                            "mode=%s decayed actor learning rate to %.6f at episode=%d after a validation plateau.",
                            self.executor.mode,
                            self.actor_optimizer.lr,
                            episode,
                        )
                    self._updates_since_improvement = 0

                for _ in batch_episode_indices:
                    history.grad_norm_history.append(last_step_stats.raw_grad_norm)
                    history.applied_grad_norm_history.append(last_step_stats.applied_grad_norm)
                    history.update_norm_history.append(last_step_stats.update_norm)
                    history.learning_rate_history.append(self.actor_optimizer.lr)
                    history.validation_reward_history.append(last_validation_reward)
                    history.validation_success_history.append(last_validation_success)
                    history.optimizer_clipped_history.append(last_step_stats.clipped)
                    history.parameter_history.append(self.current_params.tolist())

                batch_actor_grads.clear()
                batch_value_grads.clear()
                batch_policy_losses.clear()
                batch_value_losses.clear()
                batch_rewards.clear()
                batch_successes.clear()
                batch_episode_indices.clear()

            episode_runtime = time.perf_counter() - episode_start
            history.episode_runtime_sec.append(episode_runtime)

            if episode % self.config.log_every == 0 or episode == 1:
                reward_window = history.episode_rewards[-self.config.log_every :]
                success_window = history.episode_success[-self.config.log_every :]
                logger.info(
                    "mode=%s actor_critic episode=%d reward=%.3f avg_reward=%.3f success=%.1f%% "
                    "policy_loss=%.4f value_loss=%.4f grad_norm=%.4f lr=%.5f val_success=%.1f%%",
                    self.executor.mode,
                    episode,
                    total_reward,
                    float(np.mean(reward_window)),
                    float(np.mean(success_window)) * 100.0,
                    float(history.loss_history[-1]),
                    float(history.value_loss_history[-1]),
                    last_step_stats.raw_grad_norm,
                    self.actor_optimizer.lr,
                    last_validation_success * 100.0,
                )

        history.total_runtime_sec = time.perf_counter() - train_start
        history.last_parameters = self.current_params.tolist()
        if self.config.track_best_parameters:
            self.current_params = best_params.copy()
            history.final_parameters = best_params.tolist()
            history.best_parameters = best_params.tolist()
        else:
            history.final_parameters = self.current_params.tolist()
            history.best_parameters = self.current_params.tolist()
        history.best_episode = best_episode
        return history
