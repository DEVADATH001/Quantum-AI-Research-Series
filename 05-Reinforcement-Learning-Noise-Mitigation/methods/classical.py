"""Classical method wrappers."""

from __future__ import annotations

from src.baselines import MLPActorCriticConfig, train_mlp_actor_critic


def train_mlp_actor_critic_method(*, env_config, config: MLPActorCriticConfig):
    return train_mlp_actor_critic(env_config=env_config, config=config)
