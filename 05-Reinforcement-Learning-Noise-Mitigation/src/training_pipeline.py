"""End-to-end training pipeline for quantum RL noise-mitigation experiments."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.gradient_estimator import GradientEstimatorConfig, ParameterShiftGradientEstimator
from agent.quantum_policy import PolicyConfig, QuantumPolicyNetwork
from agent.reinforce_learner import ReinforceConfig, ReinforceLearner
from environments.simple_nav_env import EnvironmentConfig, SimpleNavigationEnv
from src.evaluation import (
    convergence_runtime_seconds,
    find_convergence_episode,
    plot_convergence_comparison,
    plot_final_policy,
    plot_learning_curves,
)
from src.mitigation_engine import MitigationConfig, MitigationEngine
from src.runtime_executor import QuantumRuntimeExecutor, RuntimeConfig
from utils.qiskit_helpers import configure_logging, ensure_dir, save_json, set_global_seed

logger = logging.getLogger(__name__)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _to_env_config(payload: dict[str, Any]) -> EnvironmentConfig:
    return EnvironmentConfig(
        max_episode_steps=int(payload.get("max_episode_steps", 20)),
        correct_reward=float(payload.get("correct_reward", 1.0)),
        incorrect_penalty=float(payload.get("incorrect_penalty", -0.1)),
    )


def _to_policy_config(payload: dict[str, Any], seed: int) -> PolicyConfig:
    return PolicyConfig(
        num_qubits=int(payload.get("num_qubits", 2)),
        reps=int(payload.get("reps", 2)),
        entanglement=str(payload.get("entanglement", "full")),
        ansatz=str(payload.get("ansatz", "RealAmplitudes")),
        temperature=float(payload.get("temperature", 1.0)),
        seed=seed,
    )


def _to_reinforce_config(payload: dict[str, Any], seed: int, max_steps: int) -> ReinforceConfig:
    return ReinforceConfig(
        n_episodes=int(payload.get("n_episodes", 200)),
        gamma=float(payload.get("gamma", 0.99)),
        learning_rate=float(payload.get("learning_rate", 0.05)),
        max_episode_steps=max_steps,
        normalize_returns=bool(payload.get("normalize_returns", True)),
        log_every=int(payload.get("log_every", 10)),
        seed=seed,
        entropy_coeff=float(payload.get("entropy_coeff", 0.01)),
        grad_clip=payload.get("grad_clip", 1.0),
    )


def _save_episode_csv(path: Path, rewards: list[float], runtimes: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "reward", "runtime_sec"])
        for idx, (reward, runtime_sec) in enumerate(zip(rewards, runtimes), start=1):
            writer.writerow([idx, reward, runtime_sec])


def run_training_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run full experiment for ideal, noisy, and mitigated modes."""
    config = _load_config(config_path)
    configure_logging(config.get("results", {}).get("log_level", "INFO"))

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    output_dir = ensure_dir(config.get("results", {}).get("output_dir", "results"))

    env_cfg = _to_env_config(config.get("environment", {}))
    policy_cfg = _to_policy_config(config.get("quantum_policy", {}), seed=seed)
    reinforce_cfg = _to_reinforce_config(
        config.get("training", {}),
        seed=seed,
        max_steps=env_cfg.max_episode_steps,
    )

    grad_shift = float(config.get("training", {}).get("shift", np.pi / 2.0))
    gradient_estimator = ParameterShiftGradientEstimator(
        GradientEstimatorConfig(shift=grad_shift)
    )

    quantum_exec_cfg = config.get("quantum_execution", {})
    eval_cfg = config.get("evaluation", {})
    convergence_threshold = float(eval_cfg.get("convergence_threshold", 0.8))
    moving_window = int(eval_cfg.get("moving_avg_window", 10))

    baseline_policy = QuantumPolicyNetwork(n_actions=2, config=policy_cfg)
    initial_parameters = baseline_policy.initial_parameters()

    histories: dict[str, dict[str, Any]] = {}
    mode_summary: dict[str, dict[str, float | int | None]] = {}

    training_start = time.perf_counter()
    for mode in ("ideal", "noisy", "mitigated"):
        logger.info("Starting training mode=%s", mode)
        env = SimpleNavigationEnv(config=env_cfg)
        policy = QuantumPolicyNetwork(n_actions=2, config=policy_cfg)

        resilience_level = int(quantum_exec_cfg.get("resilience_level", 0))
        if mode == "mitigated":
            resilience_level = max(2, resilience_level)

        mitigation_cfg = config.get("mitigation", {})
        mitigation_enabled = bool(mitigation_cfg.get("enabled", True))
        mitigation_method = str(mitigation_cfg.get("method", "both")).lower()
        enable_trex = mitigation_enabled and mitigation_method in {"both", "trex"}
        enable_zne = mitigation_enabled and mitigation_method in {"both", "zne"}

        mitigation_engine = MitigationEngine(
            MitigationConfig(
                resilience_level=resilience_level,
                enable_trex=enable_trex,
                enable_zne=enable_zne,
                scale_factors=tuple(
                    float(x)
                    for x in mitigation_cfg
                    .get("zne", {})
                    .get("scale_factors", [1.0, 1.5, 2.0, 2.5, 3.0])
                ),
                extrapolation=str(
                    mitigation_cfg.get("zne", {}).get("extrapolation", "polynomial")
                ),
                polynomial_degree=int(mitigation_cfg.get("zne", {}).get("degree", 2)),
            )
        )

        executor = QuantumRuntimeExecutor(
            config=RuntimeConfig(
                mode=mode,
                shots=int(quantum_exec_cfg.get("shots", 1024)),
                backend_name=str(quantum_exec_cfg.get("backend_name", "ibm_osaka")),
                optimization_level=int(quantum_exec_cfg.get("optimization_level", 1)),
                resilience_level=resilience_level,
                seed=seed,
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

        history_dict = {
            "mode": history.mode,
            "episode_rewards": history.episode_rewards,
            "episode_success": history.episode_success,
            "episode_lengths": history.episode_lengths,
            "episode_runtime_sec": history.episode_runtime_sec,
            "parameter_history": history.parameter_history,
            "loss_history": history.loss_history,
            "total_runtime_sec": history.total_runtime_sec,
            "final_parameters": history.final_parameters,
        }
        histories[mode] = history_dict

        convergence_episode = find_convergence_episode(
            history.episode_rewards,
            threshold=convergence_threshold,
            window=moving_window,
        )
        mode_summary[mode] = {
            "convergence_episode": convergence_episode,
            "convergence_runtime_sec": convergence_runtime_seconds(
                history.episode_runtime_sec,
                convergence_episode,
            ),
            "total_runtime_sec": history.total_runtime_sec,
            "final_avg_reward": float(np.mean(history.episode_rewards[-moving_window:])),
            "final_success_rate": float(np.mean(history.episode_success[-moving_window:])),
        }

        save_json(output_dir / f"{mode}_training_log.json", history_dict)
        _save_episode_csv(
            output_dir / f"{mode}_episode_metrics.csv",
            rewards=history.episode_rewards,
            runtimes=history.episode_runtime_sec,
        )
        # Save model parameters
        learner.save_model(output_dir / f"{mode}_weights.npy", np.array(history.final_parameters))
        
        # Close executor (closes session)
        executor.close()

    total_runtime = time.perf_counter() - training_start

    plot_learning_curves(
        histories=histories,
        output_path=output_dir / "learning_curves.png",
        moving_avg_window=moving_window,
    )
    plot_convergence_comparison(
        mode_summary=mode_summary,
        output_path=output_dir / "convergence_comparison.png",
    )

    # Use mitigated policy parameters for final policy visualization.
    final_policy = QuantumPolicyNetwork(n_actions=2, config=policy_cfg)
    final_executor = QuantumRuntimeExecutor(
        RuntimeConfig(
            mode="mitigated",
            shots=int(quantum_exec_cfg.get("shots", 1024)),
            backend_name=str(quantum_exec_cfg.get("backend_name", "ibm_osaka")),
            optimization_level=int(quantum_exec_cfg.get("optimization_level", 1)),
            resilience_level=max(2, int(quantum_exec_cfg.get("resilience_level", 0))),
            seed=seed,
        ),
        mitigation_engine=MitigationEngine(
            MitigationConfig(resilience_level=2),
        ),
    )
    plot_final_policy(
        policy=final_policy,
        executor=final_executor,
        parameters=np.asarray(histories["mitigated"]["final_parameters"], dtype=float),
        output_path=output_dir / "final_policy_plot.png",
    )

    summary = {
        "config_path": str(Path(config_path).resolve()),
        "seed": seed,
        "mode_summary": mode_summary,
        "total_runtime_sec": total_runtime,
    }
    save_json(output_dir / "summary.json", summary)
    logger.info("Training pipeline complete in %.2f seconds", total_runtime)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum RL Noise-Mitigation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to YAML config file.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_training_pipeline(args.config)
