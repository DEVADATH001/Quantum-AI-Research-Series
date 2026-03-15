"""REINFORCE learner for the quantum policy network."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from agent.gradient_estimator import ParameterShiftGradientEstimator
from agent.quantum_policy import QuantumPolicyNetwork
from environments.simple_nav_env import SimpleNavigationEnv
from src.runtime_executor import QuantumRuntimeExecutor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReinforceConfig:
    """Training hyperparameters for REINFORCE."""

    n_episodes: int = 200
    gamma: float = 0.99
    learning_rate: float = 0.05
    max_episode_steps: int = 20
    normalize_returns: bool = True
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0


@dataclass(slots=True)
class TrainingHistory:
    """Container for training outputs."""

    mode: str
    episode_rewards: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_runtime_sec: list[float] = field(default_factory=list)
    parameter_history: list[list[float]] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    total_runtime_sec: float = 0.0
    final_parameters: list[float] = field(default_factory=list)


class AdamOptimizer:
    """Lightweight Adam optimizer for NumPy parameter arrays with gradient clipping."""

    def __init__(
        self,
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one Adam step with optional gradient clipping."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Gradient Clipping to stabilize noisy quantum gradients
        if self.grad_clip is not None:
            norm = np.linalg.norm(grads)
            if norm > self.grad_clip:
                grads = grads * (self.grad_clip / (norm + 1e-6))
                
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads * grads)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ReinforceLearner:
    """Train a quantum policy with REINFORCE and parameter-shift gradients."""

    def __init__(
        self,
        policy: QuantumPolicyNetwork,
        env: SimpleNavigationEnv,
        executor: QuantumRuntimeExecutor,
        gradient_estimator: ParameterShiftGradientEstimator,
        config: ReinforceConfig | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.executor = executor
        self.gradient_estimator = gradient_estimator
        self.config = config or ReinforceConfig()
        self.optimizer = AdamOptimizer(
            lr=self.config.learning_rate,
            grad_clip=self.config.grad_clip
        )
        self.rng = np.random.default_rng(self.config.seed)

    def save_model(self, path: str | Path, parameters: np.ndarray) -> None:
        """Save policy parameters to disk."""
        np.save(str(path), parameters)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str | Path) -> np.ndarray:
        """Load policy parameters from disk."""
        return np.load(str(path))

    def _discounted_returns(self, rewards: list[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=float)
        running = 0.0
        for idx in reversed(range(len(rewards))):
            running = rewards[idx] + self.config.gamma * running
            returns[idx] = running
        if self.config.normalize_returns and returns.size > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _rollout(self) -> tuple[list[int], list[int], list[float], bool]:
        states: list[int] = []
        actions: list[int] = []
        rewards: list[float] = []
        success = False

        state = self.env.reset()
        for _ in range(self.config.max_episode_steps):
            probs = self.policy.action_probabilities(
                state=state,
                parameters=self.current_params,
                executor=self.executor,
            )
            action = self.policy.sample_action(probs)
            next_state, reward, done, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))

            state = next_state
            if done:
                success = info.get("reached_target", False)
                break
        return states, actions, rewards, success

    current_params: np.ndarray = field(init=False)

    def train(
        self,
        initial_parameters: np.ndarray | None = None,
    ) -> TrainingHistory:
        """Train policy and return full history."""
        self.current_params = (
            np.asarray(initial_parameters, dtype=float)
            if initial_parameters is not None
            else self.policy.initial_parameters()
        )
        history = TrainingHistory(mode=self.executor.mode)
        train_start = time.perf_counter()

        for episode in range(1, self.config.n_episodes + 1):
            episode_start = time.perf_counter()
            states, actions, rewards, success = self._rollout()
            returns = self._discounted_returns(rewards)
            grads, loss = self.gradient_estimator.estimate_reinforce_gradient(
                policy=self.policy,
                executor=self.executor,
                states=states,
                actions=actions,
                returns=returns,
                parameters=self.current_params,
                entropy_coeff=self.config.entropy_coeff,
            )
            self.current_params = self.optimizer.step(params=self.current_params, grads=grads)

            episode_runtime = time.perf_counter() - episode_start
            total_reward = float(np.sum(rewards))

            history.episode_rewards.append(total_reward)
            history.episode_success.append(success)
            history.episode_lengths.append(len(rewards))
            history.loss_history.append(loss)
            history.episode_runtime_sec.append(episode_runtime)
            history.parameter_history.append(self.current_params.tolist())

            if episode % self.config.log_every == 0 or episode == 1:
                success_rate = np.mean(history.episode_success[-self.config.log_every:])
                logger.info(
                    "mode=%s episode=%d reward=%.3f success=%.1f%% loss=%.4f",
                    self.executor.mode,
                    episode,
                    total_reward,
                    success_rate * 100,
                    loss,
                )

        history.total_runtime_sec = time.perf_counter() - train_start
        history.final_parameters = self.current_params.tolist()
        return history

