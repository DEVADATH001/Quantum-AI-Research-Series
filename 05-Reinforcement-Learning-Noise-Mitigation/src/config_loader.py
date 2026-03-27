
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field

from core.seeds import MAIN_BENCHMARK_SEEDS

class ExperimentConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: list(MAIN_BENCHMARK_SEEDS))
    n_eval_episodes: int = 32
    run_tabular_baseline: bool = True
    run_mlp_baseline: bool = True
    run_mlp_actor_critic: bool = True
    run_random_baseline: bool = True
    run_quantum_actor_critic: bool = True

class ResultsConfig(BaseModel):
    output_dir: str = "results"
    log_level: str = "INFO"

class EnvironmentConfig(BaseModel):
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

class PolicyConfig(BaseModel):
    num_qubits: int = 3
    reuploads: int = 1
    ansatz: str = "RealAmplitudes"
    ansatz_reps: int = 1
    entanglement: str = "linear"
    state_encoding: str = "hybrid"
    seed: int = 42

class TrainingConfig(BaseModel):
    n_episodes: int = 80
    gamma: float = 0.99
    learning_rate: float = 0.03
    max_episode_steps: int = 8
    baseline_decay: float | None = 0.9
    episodes_per_update: int = 4
    probability_floor: float | None = None
    log_every: int = 10
    seed: int = 42
    entropy_coeff: float = 0.01
    grad_clip: float | None = 1.0
    shift: float = 3.141592653589793 / 2.0
    selection_eval_episodes: int = 4
    track_best_parameters: bool = True
    lr_plateau_patience: int | None = 3
    lr_decay: float = 0.5
    min_learning_rate: float = 0.002

class TabularBaselineConfig(BaseModel):
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


class MLPBaselineConfig(BaseModel):
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


class MLPActorCriticConfig(BaseModel):
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


class QuantumActorCriticConfig(BaseModel):
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
    shift: float = 3.141592653589793 / 2.0
    selection_eval_episodes: int = 4
    track_best_parameters: bool = True
    critic_hidden_dim: int = 16
    critic_init_scale: float = 0.1
    value_loss_coeff: float = 0.5
    lr_plateau_patience: int | None = 3
    lr_decay: float = 0.5
    min_learning_rate: float = 0.002

class ZNEConfig(BaseModel):
    scale_factors: list[float] = [1.0, 2.0, 3.0]
    extrapolation: str = "linear"
    degree: int = 1

class MitigationConfig(BaseModel):
    enabled: bool = True
    method: str = "both"
    readout_correction: bool = True
    zne: ZNEConfig = Field(default_factory=ZNEConfig)

class QuantumExecutionConfig(BaseModel):
    shots: int = 512
    backend_name: str = "ibm_osaka"
    optimization_level: int = 1
    resilience_level: int = 0
    compact_noise_model: bool = False

class EvaluationConfig(BaseModel):
    convergence_threshold: float = 0.9
    moving_avg_window: int = 8

class AppConfig(BaseModel):
    seed: int = 42
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    quantum_policy: PolicyConfig = Field(default_factory=PolicyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    baselines: TabularBaselineConfig = Field(default_factory=TabularBaselineConfig)
    mlp_baseline: MLPBaselineConfig = Field(default_factory=MLPBaselineConfig)
    mlp_actor_critic: MLPActorCriticConfig = Field(default_factory=MLPActorCriticConfig)
    quantum_actor_critic: QuantumActorCriticConfig = Field(default_factory=QuantumActorCriticConfig)
    mitigation: MitigationConfig = Field(default_factory=MitigationConfig)
    quantum_execution: QuantumExecutionConfig = Field(default_factory=QuantumExecutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(source: str | Path | Mapping[str, Any] | AppConfig) -> AppConfig:
    if isinstance(source, AppConfig):
        return source
    if isinstance(source, Mapping):
        return AppConfig(**dict(source))
    with Path(source).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AppConfig(**data)


def dump_config(config: AppConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
