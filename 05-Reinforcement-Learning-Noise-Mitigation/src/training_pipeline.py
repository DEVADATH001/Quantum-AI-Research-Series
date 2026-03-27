"""End-to-end training pipeline for the upgraded QRL benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.schemas import EvalResult, MethodSpec, RunResult, ScenarioSpec
from agent.actor_critic_learner import (
    ActorCriticTrainingHistory,
    QuantumActorCriticConfig,
    QuantumActorCriticLearner,
)
from agent.gradient_estimator import GradientEstimatorConfig, ParameterShiftGradientEstimator
from agent.quantum_policy import PolicyConfig as QuantumPolicyConfig, QuantumPolicyNetwork
from agent.reinforce_learner import ReinforceConfig, ReinforceLearner, TrainingHistory
from environments.simple_nav_env import EnvironmentConfig as EnvRuntimeConfig, KeyDoorNavigationEnv
from src.baselines import (
    BaselineTrainingHistory,
    MLPActorCriticConfig,
    MLPBaselineConfig as MLPBaselineTrainConfig,
    TabularBaselineConfig as BaselineTrainConfig,
    evaluate_random_policy,
    train_mlp_actor_critic,
    train_mlp_reinforce,
    train_tabular_reinforce,
)
from src.config_loader import AppConfig, dump_config, load_config
from src.evaluation import (
    aggregate_histories,
    evaluate_mlp_policy,
    evaluate_quantum_policy,
    evaluate_tabular_policy,
    extract_mlp_policy_matrix,
    extract_quantum_policy_matrix,
    extract_tabular_policy_matrix,
    plot_baseline_comparison,
    plot_convergence_comparison,
    plot_learning_curves,
    plot_policy_heatmaps,
    summarize_history,
)
from src.mitigation_engine import MitigationConfig, MitigationEngine
from src.project_paths import path_relative_to_project, resolve_project_path
from src.research_stats import paired_method_comparison
from src.result_store import ExperimentResultStore
from src.runtime_executor import QuantumRuntimeExecutor, RuntimeConfig
from src.run_metadata import build_run_manifest
from utils.qiskit_helpers import configure_logging, set_global_seed

logger = logging.getLogger(__name__)


def _quantum_history_to_dict(
    history: TrainingHistory | ActorCriticTrainingHistory,
    seed: int,
    evaluation: dict[str, float],
    convergence_threshold: float | None = None,
    moving_avg_window: int | None = None,
    algorithm: str = "quantum_reinforce",
) -> dict[str, Any]:
    record = {
        "seed": seed,
        "algorithm": algorithm,
        "mode": history.mode,
        "episode_rewards": history.episode_rewards,
        "episode_success": history.episode_success,
        "episode_lengths": history.episode_lengths,
        "episode_runtime_sec": history.episode_runtime_sec,
        "parameter_history": history.parameter_history,
        "loss_history": history.loss_history,
        "value_loss_history": getattr(history, "value_loss_history", []),
        "grad_norm_history": history.grad_norm_history,
        "applied_grad_norm_history": history.applied_grad_norm_history,
        "update_norm_history": history.update_norm_history,
        "learning_rate_history": history.learning_rate_history,
        "validation_reward_history": history.validation_reward_history,
        "validation_success_history": history.validation_success_history,
        "optimizer_clipped_history": history.optimizer_clipped_history,
        "total_runtime_sec": history.total_runtime_sec,
        "final_parameters": history.final_parameters,
        "last_parameters": history.last_parameters,
        "best_parameters": history.best_parameters,
        "best_episode": history.best_episode,
        "num_skipped_updates": history.num_skipped_updates,
        "num_lr_decays": history.num_lr_decays,
        "evaluation": evaluation,
    }
    if convergence_threshold is not None and moving_avg_window is not None:
        record["summary_metrics"] = summarize_history(
            record,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_avg_window,
        )
    return record


def _baseline_history_to_dict(
    history: BaselineTrainingHistory,
    seed: int,
    evaluation: dict[str, float],
    convergence_threshold: float | None = None,
    moving_avg_window: int | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    record = {
        "seed": seed,
        "algorithm": algorithm or history.name,
        "name": history.name,
        "episode_rewards": history.episode_rewards,
        "episode_success": history.episode_success,
        "episode_lengths": history.episode_lengths,
        "episode_runtime_sec": history.episode_runtime_sec,
        "loss_history": history.loss_history,
        "value_loss_history": history.value_loss_history,
        "grad_norm_history": history.grad_norm_history,
        "total_runtime_sec": history.total_runtime_sec,
        "final_parameters": history.final_parameters,
        "evaluation": evaluation,
    }
    if convergence_threshold is not None and moving_avg_window is not None:
        record["summary_metrics"] = summarize_history(
            record,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_avg_window,
        )
    return record


def _build_mitigation_engine(config: AppConfig, resilience_level: int) -> MitigationEngine:
    mitigation_cfg = config.mitigation
    mitigation_enabled = mitigation_cfg.enabled
    mitigation_method = mitigation_cfg.method.lower()
    enable_readout_correction = mitigation_enabled and mitigation_cfg.readout_correction
    enable_zne = mitigation_enabled and mitigation_method in {"both", "zne"}

    return MitigationEngine(
        MitigationConfig(
            resilience_level=resilience_level,
            enable_readout_correction=enable_readout_correction,
            enable_zne=enable_zne,
            scale_factors=tuple(mitigation_cfg.zne.scale_factors),
            extrapolation=mitigation_cfg.zne.extrapolation,
            polynomial_degree=mitigation_cfg.zne.degree,
        )
    )


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "matplotlib",
        "pandas",
        "PyYAML",
        "pydantic",
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "qiskit-machine-learning",
    ]
    versions: dict[str, str] = {}
    for package_name in packages:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _evaluation_metric(metric_key: str) -> Any:
    def _metric(record: dict[str, Any]) -> float | None:
        return record.get("evaluation", {}).get(metric_key)

    return _metric


def _summary_metric(metric_key: str) -> Any:
    def _metric(record: dict[str, Any]) -> float | None:
        if metric_key in record:
            return record.get(metric_key)
        return record.get("summary_metrics", {}).get(metric_key)

    return _metric


def _build_statistical_analysis(
    quantum_histories_by_mode: dict[str, list[dict[str, Any]]],
    classical_histories_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    comparisons: dict[str, dict[str, Any]] = {}
    metric_extractors = {
        "eval_success": _evaluation_metric("success_rate"),
        "eval_reward": _evaluation_metric("avg_reward"),
        "reward_auc": _summary_metric("reward_auc"),
        "final_success_rate": _summary_metric("final_success_rate"),
    }

    comparison_pairs: list[tuple[str, list[dict[str, Any]], str, list[dict[str, Any]]]] = [
        ("ideal", quantum_histories_by_mode["ideal"], "noisy", quantum_histories_by_mode["noisy"]),
        ("mitigated", quantum_histories_by_mode["mitigated"], "noisy", quantum_histories_by_mode["noisy"]),
    ]
    for baseline_name, baseline_histories in classical_histories_by_name.items():
        if not baseline_histories:
            continue
        comparison_pairs.extend(
            [
                ("ideal", quantum_histories_by_mode["ideal"], baseline_name, baseline_histories),
                ("mitigated", quantum_histories_by_mode["mitigated"], baseline_name, baseline_histories),
                ("noisy", quantum_histories_by_mode["noisy"], baseline_name, baseline_histories),
            ]
        )

    for left_name, left_records, right_name, right_records in comparison_pairs:
        pair_key = f"{left_name}_vs_{right_name}"
        comparisons[pair_key] = {
            metric_name: paired_method_comparison(
                left_records,
                right_records,
                metric_name=metric_name,
                metric_fn=metric_fn,
            )
            for metric_name, metric_fn in metric_extractors.items()
        }

    return comparisons


def _config_source_label(config_source: str | Path | Mapping[str, Any] | AppConfig) -> str:
    if isinstance(config_source, AppConfig):
        return "in_memory_app_config"
    if isinstance(config_source, Mapping):
        return "in_memory_mapping"
    return path_relative_to_project(resolve_project_path(config_source))


def _scenario_spec_from_config(config: AppConfig) -> ScenarioSpec:
    output_name = Path(config.results.output_dir).name or "scenario"
    return ScenarioSpec(
        name=output_name,
        description=f"Auto-generated scenario spec for {output_name}.",
        seeds=list(config.experiment.seeds),
        n_eval_episodes=int(config.experiment.n_eval_episodes),
        environment=config.environment.model_dump(mode="json"),
        quantum_policy=config.quantum_policy.model_dump(mode="json"),
    )


def _save_run_result(
    *,
    result_store: ExperimentResultStore,
    path: Path,
    scenario_spec: ScenarioSpec,
    method_spec: MethodSpec,
    seed: int,
    output_dir: Path,
    training_log_path: Path,
    weights_path: Path | None,
    evaluation: dict[str, float],
    summary_metrics: dict[str, Any] | None,
) -> None:
    run_result = RunResult(
        scenario_spec=scenario_spec,
        method_spec=method_spec,
        seed=seed,
        output_dir=str(output_dir.resolve()),
        training_log_path=str(training_log_path.resolve()),
        weights_path=str(weights_path.resolve()) if weights_path is not None else None,
        evaluation=EvalResult(**evaluation),
        summary_metrics=summary_metrics or {},
    )
    result_store.save_json(path, run_result.model_dump(mode="json"))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _quantum_shot_estimate_per_seed(
    *,
    config: AppConfig,
    parameter_count: int,
    aggregate_record: dict[str, Any],
    mode: str,
) -> dict[str, float]:
    length_curve = aggregate_record.get("length_curve_mean", [])
    avg_episode_length = (
        float(np.mean(np.asarray(length_curve, dtype=float)))
        if length_curve
        else float(config.environment.max_episode_steps)
    )
    parameter_bindings_per_timestep = int(2 + 2 * parameter_count)
    n_training_episodes = int(config.training.n_episodes)
    n_eval_episodes = int(config.experiment.n_eval_episodes)
    selection_eval_episodes = int(config.training.selection_eval_episodes)
    episodes_per_update = max(1, int(config.training.episodes_per_update))
    optimizer_updates = int(np.ceil(n_training_episodes / episodes_per_update))

    training_parameter_bindings = float(
        n_training_episodes * avg_episode_length * parameter_bindings_per_timestep
        + optimizer_updates * selection_eval_episodes * avg_episode_length
    )
    evaluation_parameter_bindings = float(n_eval_episodes * avg_episode_length)
    total_parameter_bindings = training_parameter_bindings + evaluation_parameter_bindings

    circuit_multiplier = 1
    mitigation_method = config.mitigation.method.lower()
    if (
        mode == "mitigated"
        and config.mitigation.enabled
        and mitigation_method in {"both", "zne"}
        and config.mitigation.zne.scale_factors
    ):
        circuit_multiplier = len(config.mitigation.zne.scale_factors)

    total_shots = float(total_parameter_bindings * circuit_multiplier * int(config.quantum_execution.shots))
    return {
        "avg_episode_length_estimate": avg_episode_length,
        "parameter_bindings_per_timestep": float(parameter_bindings_per_timestep),
        "training_parameter_bindings_per_seed": training_parameter_bindings,
        "evaluation_parameter_bindings_per_seed": evaluation_parameter_bindings,
        "total_parameter_bindings_per_seed": total_parameter_bindings,
        "estimated_circuit_executions_per_logical_query": float(circuit_multiplier),
        "estimated_total_shots_per_seed": total_shots,
    }


def _build_resource_efficiency_summary(
    *,
    config: AppConfig,
    reference_env: KeyDoorNavigationEnv,
    quantum_aggregates_by_algorithm: dict[str, dict[str, dict[str, Any]]],
    classical_aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_policy = QuantumPolicyNetwork(
        n_actions=reference_env.action_space,
        n_observations=reference_env.observation_space,
        config=QuantumPolicyConfig(**config.quantum_policy.model_dump()),
    )

    quantum_metrics: dict[str, dict[str, Any]] = {}
    for algorithm_name, quantum_aggregates in quantum_aggregates_by_algorithm.items():
        algorithm_metrics: dict[str, Any] = {}
        for mode, aggregate_record in quantum_aggregates.items():
            if not aggregate_record:
                continue
            shot_estimate = _quantum_shot_estimate_per_seed(
                config=config,
                parameter_count=reference_policy.parameter_count,
                aggregate_record=aggregate_record,
                mode=mode,
            )
            total_runtime_mean = aggregate_record.get("total_runtime_mean")
            eval_success_mean = aggregate_record.get("eval_success_mean")
            reward_auc_mean = aggregate_record.get("reward_auc_mean")
            algorithm_metrics[mode] = {
                **shot_estimate,
                "eval_success_per_runtime_sec": _safe_ratio(eval_success_mean, total_runtime_mean),
                "reward_auc_per_runtime_sec": _safe_ratio(reward_auc_mean, total_runtime_mean),
                "eval_success_per_million_shots": _safe_ratio(
                    eval_success_mean,
                    shot_estimate["estimated_total_shots_per_seed"] / 1_000_000.0,
                ),
            }
        if algorithm_metrics:
            quantum_metrics[algorithm_name] = algorithm_metrics

    classical_metrics: dict[str, dict[str, float | None]] = {}
    for name, aggregate_record in classical_aggregates.items():
        if not aggregate_record:
            continue
        total_runtime_mean = aggregate_record.get("total_runtime_mean")
        eval_success_mean = aggregate_record.get("eval_success_mean")
        reward_auc_mean = aggregate_record.get("reward_auc_mean")
        classical_metrics[name] = {
            "eval_success_per_runtime_sec": _safe_ratio(eval_success_mean, total_runtime_mean),
            "reward_auc_per_runtime_sec": _safe_ratio(reward_auc_mean, total_runtime_mean),
        }

    return {
        "quantum": quantum_metrics,
        "classical": classical_metrics,
    }


def run_training_pipeline(config_source: str | Path | Mapping[str, Any] | AppConfig) -> dict[str, Any]:
    config = load_config(config_source)
    configure_logging(config.results.log_level)
    config_source_label = _config_source_label(config_source)

    experiment_cfg = config.experiment
    seeds = experiment_cfg.seeds
    n_eval_episodes = experiment_cfg.n_eval_episodes
    run_tabular_baseline = experiment_cfg.run_tabular_baseline
    run_mlp_baseline = getattr(experiment_cfg, "run_mlp_baseline", True)
    run_mlp_actor_critic = getattr(experiment_cfg, "run_mlp_actor_critic", True)
    run_random_baseline = experiment_cfg.run_random_baseline
    run_quantum_actor_critic = getattr(experiment_cfg, "run_quantum_actor_critic", True)

    output_dir = resolve_project_path(config.results.output_dir)
    result_store = ExperimentResultStore(output_dir=output_dir)
    scenario_spec = _scenario_spec_from_config(config)
    result_store.save_json(
        output_dir / "run_manifest.json",
        build_run_manifest(
            config_source=config_source_label,
            resolved_config=dump_config(config),
            output_dir=output_dir,
            command=sys.argv,
        ),
    )

    quantum_exec_cfg = config.quantum_execution
    eval_cfg = config.evaluation
    convergence_threshold = eval_cfg.convergence_threshold
    moving_window = eval_cfg.moving_avg_window

    training_start = time.perf_counter()
    quantum_histories_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in ("ideal", "noisy", "mitigated")}
    quantum_actor_critic_histories_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in ("ideal", "noisy", "mitigated")
    }
    tabular_baseline_histories: list[dict[str, Any]] = []
    mlp_baseline_histories: list[dict[str, Any]] = []
    mlp_actor_critic_histories: list[dict[str, Any]] = []
    random_baseline_metrics: list[dict[str, float]] = []

    for seed in seeds:
        set_global_seed(seed)
        env_cfg = EnvRuntimeConfig(**config.environment.model_dump())
        env_cfg.seed = seed
        
        policy_cfg = QuantumPolicyConfig(**config.quantum_policy.model_dump())
        policy_cfg.seed = seed

        reinforce_cfg = ReinforceConfig(**config.training.model_dump(exclude={"shift"}))
        reinforce_cfg.seed = seed
        reinforce_cfg.max_episode_steps = env_cfg.max_episode_steps

        baseline_cfg = BaselineTrainConfig(**config.baselines.model_dump())
        baseline_cfg.seed = seed
        baseline_cfg.max_episode_steps = env_cfg.max_episode_steps
        mlp_baseline_cfg = MLPBaselineTrainConfig(**config.mlp_baseline.model_dump())
        mlp_baseline_cfg.seed = seed
        mlp_baseline_cfg.max_episode_steps = env_cfg.max_episode_steps
        mlp_actor_critic_cfg = MLPActorCriticConfig(**config.mlp_actor_critic.model_dump())
        mlp_actor_critic_cfg.seed = seed
        mlp_actor_critic_cfg.max_episode_steps = env_cfg.max_episode_steps
        quantum_actor_critic_cfg = QuantumActorCriticConfig(**config.quantum_actor_critic.model_dump(exclude={"shift"}))
        quantum_actor_critic_cfg.seed = seed
        quantum_actor_critic_cfg.max_episode_steps = env_cfg.max_episode_steps
        
        grad_shift = config.training.shift
        gradient_estimator = ParameterShiftGradientEstimator(GradientEstimatorConfig(shift=grad_shift))
        actor_critic_gradient_estimator = ParameterShiftGradientEstimator(
            GradientEstimatorConfig(shift=config.quantum_actor_critic.shift)
        )

        seed_output_dir = result_store.quantum_seed_dir(seed)
        env = KeyDoorNavigationEnv(config=env_cfg)
        baseline_policy = QuantumPolicyNetwork(
            n_actions=env.action_space,
            n_observations=env.observation_space,
            config=policy_cfg,
        )
        initial_parameters = baseline_policy.initial_parameters()

        for mode in ("ideal", "noisy", "mitigated"):
            logger.info("Starting quantum run seed=%d mode=%s", seed, mode)
            env = KeyDoorNavigationEnv(config=env_cfg)
            policy = QuantumPolicyNetwork(
                n_actions=env.action_space,
                n_observations=env.observation_space,
                config=policy_cfg,
            )
            resilience_level = quantum_exec_cfg.resilience_level
            if mode == "mitigated":
                resilience_level = max(2, resilience_level)
            elif mode == "noisy":
                resilience_level = 0

            mitigation_engine = _build_mitigation_engine(config, resilience_level=resilience_level)
            executor = QuantumRuntimeExecutor(
                config=RuntimeConfig(
                    mode=mode,
                    shots=quantum_exec_cfg.shots,
                    backend_name=quantum_exec_cfg.backend_name,
                    optimization_level=quantum_exec_cfg.optimization_level,
                    resilience_level=resilience_level,
                    seed=seed,
                    compact_noise_model=quantum_exec_cfg.compact_noise_model,
                ),
                mitigation_engine=mitigation_engine,
            )
            learner = ReinforceLearner(
                policy=policy,
                env=env,
                executor=executor,
                gradient_estimator=gradient_estimator,
                config=reinforce_cfg,
            )
            history = learner.train(initial_parameters=initial_parameters.copy())
            final_parameters = np.asarray(history.final_parameters, dtype=float)
            evaluation = evaluate_quantum_policy(
                policy=policy,
                executor=executor,
                parameters=final_parameters,
                env_config=env_cfg,
                n_episodes=n_eval_episodes,
                seed=seed + 10_000,
            )
            history_dict = _quantum_history_to_dict(
                history=history,
                seed=seed,
                evaluation=evaluation,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
            )
            quantum_histories_by_mode[mode].append(history_dict)

            history_log_path = seed_output_dir / f"{mode}_training_log.json"
            weights_path = seed_output_dir / f"{mode}_weights.npy"
            result_store.save_json(history_log_path, history_dict)
            result_store.save_episode_csv(
                seed_output_dir / f"{mode}_episode_metrics.csv",
                rewards=history.episode_rewards,
                successes=history.episode_success,
                runtimes=history.episode_runtime_sec,
                grad_norms=history.grad_norm_history,
            )
            learner.save_model(weights_path, final_parameters)
            _save_run_result(
                result_store=result_store,
                path=seed_output_dir / f"{mode}_run_result.json",
                scenario_spec=scenario_spec,
                method_spec=MethodSpec(
                    name="quantum_reinforce",
                    family="quantum",
                    training_algorithm="reinforce",
                    execution_mode=mode,
                ),
                seed=seed,
                output_dir=seed_output_dir,
                training_log_path=history_log_path,
                weights_path=weights_path,
                evaluation=evaluation,
                summary_metrics=history_dict.get("summary_metrics", {}),
            )

            if run_quantum_actor_critic:
                logger.info("Starting quantum actor-critic seed=%d mode=%s", seed, mode)
                ac_env = KeyDoorNavigationEnv(config=env_cfg)
                ac_policy = QuantumPolicyNetwork(
                    n_actions=ac_env.action_space,
                    n_observations=ac_env.observation_space,
                    config=policy_cfg,
                )
                ac_learner = QuantumActorCriticLearner(
                    policy=ac_policy,
                    env=ac_env,
                    executor=executor,
                    gradient_estimator=actor_critic_gradient_estimator,
                    config=quantum_actor_critic_cfg,
                )
                ac_history = ac_learner.train(initial_parameters=initial_parameters.copy())
                ac_final_parameters = np.asarray(ac_history.final_parameters, dtype=float)
                ac_evaluation = evaluate_quantum_policy(
                    policy=ac_policy,
                    executor=executor,
                    parameters=ac_final_parameters,
                    env_config=env_cfg,
                    n_episodes=n_eval_episodes,
                    seed=seed + 15_000,
                )
                ac_history_dict = _quantum_history_to_dict(
                    history=ac_history,
                    seed=seed,
                    evaluation=ac_evaluation,
                    convergence_threshold=convergence_threshold,
                    moving_avg_window=moving_window,
                    algorithm="quantum_actor_critic",
                )
                quantum_actor_critic_histories_by_mode[mode].append(ac_history_dict)
                ac_history_log_path = seed_output_dir / f"quantum_actor_critic_{mode}_training_log.json"
                ac_weights_path = seed_output_dir / f"quantum_actor_critic_{mode}_weights.npy"
                result_store.save_json(ac_history_log_path, ac_history_dict)
                result_store.save_episode_csv(
                    seed_output_dir / f"quantum_actor_critic_{mode}_episode_metrics.csv",
                    rewards=ac_history.episode_rewards,
                    successes=ac_history.episode_success,
                    runtimes=ac_history.episode_runtime_sec,
                    grad_norms=ac_history.grad_norm_history,
                )
                ac_learner.save_model(ac_weights_path, ac_final_parameters)
                _save_run_result(
                    result_store=result_store,
                    path=seed_output_dir / f"quantum_actor_critic_{mode}_run_result.json",
                    scenario_spec=scenario_spec,
                    method_spec=MethodSpec(
                        name="quantum_actor_critic",
                        family="quantum",
                        training_algorithm="actor_critic",
                        execution_mode=mode,
                    ),
                    seed=seed,
                    output_dir=seed_output_dir,
                    training_log_path=ac_history_log_path,
                    weights_path=ac_weights_path,
                    evaluation=ac_evaluation,
                    summary_metrics=ac_history_dict.get("summary_metrics", {}),
                )
            executor.close()

        if run_tabular_baseline:
            logger.info("Starting tabular baseline seed=%d", seed)
            baseline_history = train_tabular_reinforce(env_config=env_cfg, config=baseline_cfg)
            baseline_parameters = np.asarray(baseline_history.final_parameters, dtype=float)
            baseline_evaluation = evaluate_tabular_policy(
                parameters=baseline_parameters,
                env_config=env_cfg,
                n_episodes=n_eval_episodes,
                seed=seed + 20_000,
                temperature=baseline_cfg.temperature,
            )
            baseline_dict = _baseline_history_to_dict(
                history=baseline_history,
                seed=seed,
                evaluation=baseline_evaluation,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
            )
            tabular_baseline_histories.append(baseline_dict)
            seed_baseline_dir = result_store.baseline_seed_dir(seed)
            baseline_log_path = seed_baseline_dir / "tabular_reinforce_training_log.json"
            baseline_weights_path = seed_baseline_dir / "tabular_reinforce_weights.npy"
            result_store.save_json(baseline_log_path, baseline_dict)
            result_store.save_episode_csv(
                seed_baseline_dir / "tabular_reinforce_episode_metrics.csv",
                rewards=baseline_history.episode_rewards,
                successes=baseline_history.episode_success,
                runtimes=baseline_history.episode_runtime_sec,
                grad_norms=baseline_history.grad_norm_history,
            )
            result_store.save_numpy(baseline_weights_path, baseline_parameters)
            _save_run_result(
                result_store=result_store,
                path=seed_baseline_dir / "tabular_reinforce_run_result.json",
                scenario_spec=scenario_spec,
                method_spec=MethodSpec(
                    name="tabular_reinforce",
                    family="classical",
                    training_algorithm="reinforce",
                ),
                seed=seed,
                output_dir=seed_baseline_dir,
                training_log_path=baseline_log_path,
                weights_path=baseline_weights_path,
                evaluation=baseline_evaluation,
                summary_metrics=baseline_dict.get("summary_metrics", {}),
            )

        if run_mlp_baseline:
            logger.info("Starting MLP baseline seed=%d", seed)
            mlp_history = train_mlp_reinforce(env_config=env_cfg, config=mlp_baseline_cfg)
            mlp_parameters = np.asarray(mlp_history.final_parameters, dtype=float)
            mlp_evaluation = evaluate_mlp_policy(
                parameters=mlp_parameters,
                env_config=env_cfg,
                n_episodes=n_eval_episodes,
                seed=seed + 25_000,
                hidden_dim=mlp_baseline_cfg.hidden_dim,
                temperature=mlp_baseline_cfg.temperature,
            )
            mlp_dict = _baseline_history_to_dict(
                history=mlp_history,
                seed=seed,
                evaluation=mlp_evaluation,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
            )
            mlp_baseline_histories.append(mlp_dict)
            seed_baseline_dir = result_store.baseline_seed_dir(seed)
            mlp_log_path = seed_baseline_dir / "mlp_reinforce_training_log.json"
            mlp_weights_path = seed_baseline_dir / "mlp_reinforce_weights.npy"
            result_store.save_json(mlp_log_path, mlp_dict)
            result_store.save_episode_csv(
                seed_baseline_dir / "mlp_reinforce_episode_metrics.csv",
                rewards=mlp_history.episode_rewards,
                successes=mlp_history.episode_success,
                runtimes=mlp_history.episode_runtime_sec,
                grad_norms=mlp_history.grad_norm_history,
            )
            result_store.save_numpy(mlp_weights_path, mlp_parameters)
            _save_run_result(
                result_store=result_store,
                path=seed_baseline_dir / "mlp_reinforce_run_result.json",
                scenario_spec=scenario_spec,
                method_spec=MethodSpec(
                    name="mlp_reinforce",
                    family="classical",
                    training_algorithm="reinforce",
                ),
                seed=seed,
                output_dir=seed_baseline_dir,
                training_log_path=mlp_log_path,
                weights_path=mlp_weights_path,
                evaluation=mlp_evaluation,
                summary_metrics=mlp_dict.get("summary_metrics", {}),
            )

        if run_mlp_actor_critic:
            logger.info("Starting MLP actor-critic baseline seed=%d", seed)
            mlp_actor_critic_history = train_mlp_actor_critic(env_config=env_cfg, config=mlp_actor_critic_cfg)
            mlp_actor_critic_parameters = np.asarray(mlp_actor_critic_history.final_parameters, dtype=float)
            mlp_actor_critic_evaluation = evaluate_mlp_policy(
                parameters=mlp_actor_critic_parameters,
                env_config=env_cfg,
                n_episodes=n_eval_episodes,
                seed=seed + 27_500,
                hidden_dim=mlp_actor_critic_cfg.actor_hidden_dim,
                temperature=mlp_actor_critic_cfg.temperature,
            )
            mlp_actor_critic_dict = _baseline_history_to_dict(
                history=mlp_actor_critic_history,
                seed=seed,
                evaluation=mlp_actor_critic_evaluation,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
                algorithm="mlp_actor_critic",
            )
            mlp_actor_critic_histories.append(mlp_actor_critic_dict)
            seed_baseline_dir = result_store.baseline_seed_dir(seed)
            mlp_ac_log_path = seed_baseline_dir / "mlp_actor_critic_training_log.json"
            mlp_ac_weights_path = seed_baseline_dir / "mlp_actor_critic_weights.npy"
            result_store.save_json(mlp_ac_log_path, mlp_actor_critic_dict)
            result_store.save_episode_csv(
                seed_baseline_dir / "mlp_actor_critic_episode_metrics.csv",
                rewards=mlp_actor_critic_history.episode_rewards,
                successes=mlp_actor_critic_history.episode_success,
                runtimes=mlp_actor_critic_history.episode_runtime_sec,
                grad_norms=mlp_actor_critic_history.grad_norm_history,
            )
            result_store.save_numpy(mlp_ac_weights_path, mlp_actor_critic_parameters)
            _save_run_result(
                result_store=result_store,
                path=seed_baseline_dir / "mlp_actor_critic_run_result.json",
                scenario_spec=scenario_spec,
                method_spec=MethodSpec(
                    name="mlp_actor_critic",
                    family="classical",
                    training_algorithm="actor_critic",
                ),
                seed=seed,
                output_dir=seed_baseline_dir,
                training_log_path=mlp_ac_log_path,
                weights_path=mlp_ac_weights_path,
                evaluation=mlp_actor_critic_evaluation,
                summary_metrics=mlp_actor_critic_dict.get("summary_metrics", {}),
            )

        if run_random_baseline:
            random_baseline_metrics.append(
                evaluate_random_policy(
                    env_config=env_cfg,
                    n_episodes=max(200, n_eval_episodes),
                    seed=seed + 30_000,
                )
            )

    aggregate_quantum = {
        mode: aggregate_histories(
            records=records,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_window,
        )
        for mode, records in quantum_histories_by_mode.items()
    }
    aggregate_quantum_actor_critic = {
        mode: aggregate_histories(
            records=records,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_window,
        )
        for mode, records in quantum_actor_critic_histories_by_mode.items()
    }
    aggregate_tabular = (
        aggregate_histories(
            records=tabular_baseline_histories,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_window,
        )
        if tabular_baseline_histories
        else {}
    )
    aggregate_mlp = (
        aggregate_histories(
            records=mlp_baseline_histories,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_window,
        )
        if mlp_baseline_histories
        else {}
    )
    aggregate_mlp_actor_critic = (
        aggregate_histories(
            records=mlp_actor_critic_histories,
            convergence_threshold=convergence_threshold,
            moving_avg_window=moving_window,
        )
        if mlp_actor_critic_histories
        else {}
    )

    random_summary = None
    if random_baseline_metrics:
        random_summary = {
            "avg_reward_mean": float(np.mean([m["avg_reward"] for m in random_baseline_metrics])),
            "avg_reward_std": float(np.std([m["avg_reward"] for m in random_baseline_metrics])),
            "success_rate_mean": float(np.mean([m["success_rate"] for m in random_baseline_metrics])),
            "success_rate_std": float(np.std([m["success_rate"] for m in random_baseline_metrics])),
            "avg_length_mean": float(np.mean([m["avg_length"] for m in random_baseline_metrics])),
            "avg_length_std": float(np.std([m["avg_length"] for m in random_baseline_metrics])),
        }

    plot_records = {
        "Quantum REINFORCE ideal": aggregate_quantum["ideal"],
        "Quantum REINFORCE noisy": aggregate_quantum["noisy"],
        "Quantum REINFORCE mitigated": aggregate_quantum["mitigated"],
    }
    if any(aggregate_quantum_actor_critic.values()):
        plot_records["Quantum A2C ideal"] = aggregate_quantum_actor_critic["ideal"]
        plot_records["Quantum A2C noisy"] = aggregate_quantum_actor_critic["noisy"]
        plot_records["Quantum A2C mitigated"] = aggregate_quantum_actor_critic["mitigated"]
    if aggregate_tabular:
        plot_records["Tabular REINFORCE"] = aggregate_tabular
    if aggregate_mlp:
        plot_records["MLP REINFORCE"] = aggregate_mlp
    if aggregate_mlp_actor_critic:
        plot_records["MLP Actor-Critic"] = aggregate_mlp_actor_critic

    plot_learning_curves(
        aggregate_histories_by_label=plot_records,
        output_path=output_dir / "learning_curves.png",
        moving_avg_window=moving_window,
        random_baseline_reward=random_summary["avg_reward_mean"] if random_summary else None,
    )
    plot_convergence_comparison(
        aggregate_histories_by_label=plot_records,
        output_path=output_dir / "convergence_comparison.png",
    )
    plot_baseline_comparison(
        aggregate_histories_by_label=plot_records,
        output_path=output_dir / "baseline_comparison.png",
        metric_key="eval_success_mean",
        ylabel="Evaluation Success Rate",
    )

    reference_env = KeyDoorNavigationEnv(config=EnvRuntimeConfig(**config.environment.model_dump()))
    state_labels = reference_env.all_state_labels()
    action_labels = [reference_env.ACTION_MEANINGS[idx] for idx in range(reference_env.action_space)]

    policy_matrices: dict[str, np.ndarray] = {}
    for label, record_group in (
        ("Quantum REINFORCE mitigated", quantum_histories_by_mode["mitigated"]),
        ("Quantum A2C mitigated", quantum_actor_critic_histories_by_mode["mitigated"]),
    ):
        if not record_group:
            continue
        best_record = max(
            record_group,
            key=lambda record: (
                record["evaluation"]["success_rate"],
                record["evaluation"]["avg_reward"],
            ),
        )
        best_policy_cfg = QuantumPolicyConfig(**config.quantum_policy.model_dump())
        best_policy_cfg.seed = int(best_record["seed"])
        best_policy = QuantumPolicyNetwork(
            n_actions=reference_env.action_space,
            n_observations=reference_env.observation_space,
            config=best_policy_cfg,
        )
        best_executor = QuantumRuntimeExecutor(
            config=RuntimeConfig(
                mode="mitigated",
                shots=quantum_exec_cfg.shots,
                backend_name=quantum_exec_cfg.backend_name,
                optimization_level=quantum_exec_cfg.optimization_level,
                resilience_level=max(2, quantum_exec_cfg.resilience_level),
                seed=int(best_record["seed"]),
                compact_noise_model=quantum_exec_cfg.compact_noise_model,
            ),
            mitigation_engine=_build_mitigation_engine(
                config,
                resilience_level=max(2, quantum_exec_cfg.resilience_level),
            ),
        )
        policy_matrices[label] = extract_quantum_policy_matrix(
            policy=best_policy,
            executor=best_executor,
            parameters=np.asarray(best_record["final_parameters"], dtype=float),
            n_states=reference_env.observation_space,
        )
        best_executor.close()

    if tabular_baseline_histories:
        best_tabular_record = max(
            tabular_baseline_histories,
            key=lambda record: (
                record["evaluation"]["success_rate"],
                record["evaluation"]["avg_reward"],
            ),
        )
        
        best_baseline_cfg = BaselineTrainConfig(**config.baselines.model_dump())
        best_baseline_cfg.seed = int(best_tabular_record["seed"])
        
        policy_matrices["Tabular baseline"] = extract_tabular_policy_matrix(
            np.asarray(best_tabular_record["final_parameters"], dtype=float),
            temperature=best_baseline_cfg.temperature,
        )
    if mlp_baseline_histories:
        best_mlp_record = max(
            mlp_baseline_histories,
            key=lambda record: (
                record["evaluation"]["success_rate"],
                record["evaluation"]["avg_reward"],
            ),
        )
        best_mlp_cfg = MLPBaselineTrainConfig(**config.mlp_baseline.model_dump())
        best_mlp_cfg.seed = int(best_mlp_record["seed"])
        policy_matrices["MLP baseline"] = extract_mlp_policy_matrix(
            parameters=np.asarray(best_mlp_record["final_parameters"], dtype=float),
            n_states=reference_env.observation_space,
            n_actions=reference_env.action_space,
            hidden_dim=best_mlp_cfg.hidden_dim,
            temperature=best_mlp_cfg.temperature,
            seed=best_mlp_cfg.seed,
        )
    if mlp_actor_critic_histories:
        best_mlp_ac_record = max(
            mlp_actor_critic_histories,
            key=lambda record: (
                record["evaluation"]["success_rate"],
                record["evaluation"]["avg_reward"],
            ),
        )
        best_mlp_ac_cfg = MLPActorCriticConfig(**config.mlp_actor_critic.model_dump())
        best_mlp_ac_cfg.seed = int(best_mlp_ac_record["seed"])
        policy_matrices["MLP actor-critic"] = extract_mlp_policy_matrix(
            parameters=np.asarray(best_mlp_ac_record["final_parameters"], dtype=float),
            n_states=reference_env.observation_space,
            n_actions=reference_env.action_space,
            hidden_dim=best_mlp_ac_cfg.actor_hidden_dim,
            temperature=best_mlp_ac_cfg.temperature,
            seed=best_mlp_ac_cfg.seed,
        )

    plot_policy_heatmaps(
        policy_matrices=policy_matrices,
        state_labels=state_labels,
        action_labels=action_labels,
        output_path=output_dir / "final_policy_plot.png",
    )

    resource_efficiency = _build_resource_efficiency_summary(
        config=config,
        reference_env=reference_env,
        quantum_aggregates_by_algorithm={
            "quantum_reinforce": aggregate_quantum,
            "quantum_actor_critic": aggregate_quantum_actor_critic,
        },
        classical_aggregates={
            "tabular_reinforce": aggregate_tabular,
            "mlp_reinforce": aggregate_mlp,
            "mlp_actor_critic": aggregate_mlp_actor_critic,
        },
    )

    statistical_analysis = _build_statistical_analysis(
        quantum_histories_by_mode=quantum_histories_by_mode,
        classical_histories_by_name={
            "tabular": tabular_baseline_histories,
            "mlp": mlp_baseline_histories,
            "mlp_actor_critic": mlp_actor_critic_histories,
        },
    )
    statistical_analysis_quantum_actor_critic = _build_statistical_analysis(
        quantum_histories_by_mode=quantum_actor_critic_histories_by_mode,
        classical_histories_by_name={
            "tabular": tabular_baseline_histories,
            "mlp": mlp_baseline_histories,
            "mlp_actor_critic": mlp_actor_critic_histories,
        },
    )

    summary = {
        "config_source": config_source_label,
        "seeds": seeds,
        "n_eval_episodes": n_eval_episodes,
        "environment": config.environment.model_dump(),
        "quantum": aggregate_quantum,
        "quantum_actor_critic": aggregate_quantum_actor_critic,
        "tabular_baseline": aggregate_tabular,
        "mlp_baseline": aggregate_mlp,
        "mlp_actor_critic": aggregate_mlp_actor_critic,
        "random_baseline": random_summary,
        "methods": {
            "quantum_reinforce": aggregate_quantum,
            "quantum_actor_critic": aggregate_quantum_actor_critic,
            "tabular_reinforce": aggregate_tabular,
            "mlp_reinforce": aggregate_mlp,
            "mlp_actor_critic": aggregate_mlp_actor_critic,
            "random": random_summary,
        },
        "resource_efficiency": resource_efficiency,
        "mitigation_gain_over_noisy": {
            "eval_success_delta": (
                None
                if aggregate_quantum["mitigated"].get("eval_success_mean") is None
                or aggregate_quantum["noisy"].get("eval_success_mean") is None
                else float(
                    aggregate_quantum["mitigated"]["eval_success_mean"]
                    - aggregate_quantum["noisy"]["eval_success_mean"]
                )
            ),
            "reward_auc_delta": float(
                aggregate_quantum["mitigated"]["reward_auc_mean"]
                - aggregate_quantum["noisy"]["reward_auc_mean"]
            ),
        },
        "statistical_analysis": statistical_analysis,
        "statistical_analysis_quantum_actor_critic": statistical_analysis_quantum_actor_critic,
        "reproducibility": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "command": sys.argv,
            "run_manifest_path": str((output_dir / "run_manifest.json").resolve()),
            "package_versions": _package_versions(),
            "seed_count": len(seeds),
            "per_seed_logs_present": {
                "quantum": bool(quantum_histories_by_mode["ideal"]),
                "quantum_actor_critic": bool(quantum_actor_critic_histories_by_mode["ideal"]),
                "tabular_baseline": bool(tabular_baseline_histories),
                "mlp_baseline": bool(mlp_baseline_histories),
                "mlp_actor_critic": bool(mlp_actor_critic_histories),
                "random_baseline": bool(random_baseline_metrics),
            },
        },
        "total_runtime_sec": time.perf_counter() - training_start,
    }
    result_store.save_json(output_dir / "summary.json", summary)
    logger.info("Training pipeline complete in %.2f seconds", summary["total_runtime_sec"])
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upgraded Quantum RL Noise-Mitigation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to YAML config file.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_training_pipeline(args.config)


if __name__ == "__main__":
    main()
