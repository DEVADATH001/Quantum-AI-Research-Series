"""Quantum method wrappers."""

from __future__ import annotations

from agent.actor_critic_learner import QuantumActorCriticConfig, QuantumActorCriticLearner
from agent.reinforce_learner import ReinforceConfig, ReinforceLearner


def train_quantum_reinforce_method(*, policy, env, executor, gradient_estimator, config: ReinforceConfig):
    learner = ReinforceLearner(
        policy=policy,
        env=env,
        executor=executor,
        gradient_estimator=gradient_estimator,
        config=config,
    )
    return learner.train()


def train_quantum_actor_critic_method(*, policy, env, executor, gradient_estimator, config: QuantumActorCriticConfig):
    learner = QuantumActorCriticLearner(
        policy=policy,
        env=env,
        executor=executor,
        gradient_estimator=gradient_estimator,
        config=config,
    )
    return learner.train()
