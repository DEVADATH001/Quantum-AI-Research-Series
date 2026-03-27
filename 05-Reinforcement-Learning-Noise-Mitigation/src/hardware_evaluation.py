"""Evaluate a saved quantum checkpoint on a simulator, fake backend, or IBM hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent.quantum_policy import PolicyConfig as QuantumPolicyConfig, QuantumPolicyNetwork
from environments.simple_nav_env import EnvironmentConfig as EnvRuntimeConfig, KeyDoorNavigationEnv
from src.config_loader import AppConfig, load_config
from src.evaluation import evaluate_quantum_policy
from src.mitigation_engine import MitigationConfig, MitigationEngine
from src.project_paths import resolve_project_path
from src.runtime_executor import QuantumRuntimeExecutor, RuntimeConfig
from utils.qiskit_helpers import configure_logging, save_json, set_global_seed


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint evaluation for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to the project YAML config.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to a `.npy` checkpoint file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hardware",
        choices=["ideal", "noisy", "mitigated", "hardware"],
        help="Execution mode for checkpoint evaluation.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for the number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the execution seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(resolve_project_path(args.config))
    configure_logging(config.results.log_level)

    seed = int(args.seed if args.seed is not None else config.seed)
    set_global_seed(seed)

    env_cfg = EnvRuntimeConfig(**config.environment.model_dump())
    env_cfg.seed = seed
    env = KeyDoorNavigationEnv(config=env_cfg)

    policy_cfg = QuantumPolicyConfig(**config.quantum_policy.model_dump())
    policy_cfg.seed = seed
    policy = QuantumPolicyNetwork(
        n_actions=env.action_space,
        n_observations=env.observation_space,
        config=policy_cfg,
    )

    resilience_level = int(config.quantum_execution.resilience_level)
    if args.mode == "mitigated":
        resilience_level = max(2, resilience_level)
    elif args.mode == "noisy":
        resilience_level = 0

    executor = QuantumRuntimeExecutor(
        config=RuntimeConfig(
            mode=args.mode,
            shots=config.quantum_execution.shots,
            backend_name=config.quantum_execution.backend_name,
            optimization_level=config.quantum_execution.optimization_level,
            resilience_level=resilience_level,
            seed=seed,
            compact_noise_model=config.quantum_execution.compact_noise_model,
        ),
        mitigation_engine=_build_mitigation_engine(config, resilience_level=resilience_level),
    )

    try:
        weights = np.load(resolve_project_path(args.weights))
        metrics = evaluate_quantum_policy(
            policy=policy,
            executor=executor,
            parameters=np.asarray(weights, dtype=float),
            env_config=env_cfg,
            n_episodes=int(args.episodes if args.episodes is not None else config.experiment.n_eval_episodes),
            seed=seed + 40_000,
        )
    finally:
        executor.close()

    payload = {
        "config_path": str(resolve_project_path(args.config)),
        "weights_path": str(resolve_project_path(args.weights)),
        "mode": args.mode,
        "backend_name": config.quantum_execution.backend_name,
        "shots": int(config.quantum_execution.shots),
        "seed": seed,
        "evaluation": metrics,
        "notes": [
            "This script is intended for the realistic workflow: train in simulation, then evaluate a frozen checkpoint on hardware or a hardware-like backend.",
            "The local hardware path uses the project's runtime executor. If you select `hardware`, an authenticated IBM Runtime environment is required.",
        ],
    }

    output_path = resolve_project_path(args.output) if args.output else resolve_project_path(config.results.output_dir) / f"{args.mode}_checkpoint_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
