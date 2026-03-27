"""Hardware-feasibility audit for IBM-targeted QRL experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile

from agent.quantum_policy import PolicyConfig as QuantumPolicyConfig, QuantumPolicyNetwork
from environments.simple_nav_env import EnvironmentConfig as EnvRuntimeConfig, KeyDoorNavigationEnv
from src.config_loader import AppConfig, load_config
from src.mitigation_engine import fold_circuit_for_noise_scaling
from src.noise_models import resolve_fake_backend
from src.project_paths import resolve_project_path
from utils.qiskit_helpers import configure_logging, save_json


def _serialize_op_counts(op_counts: dict[str, Any]) -> dict[str, int]:
    return {str(name): int(count) for name, count in op_counts.items()}


def _two_qubit_gate_count(op_counts: dict[str, int]) -> int:
    return int(sum(count for name, count in op_counts.items() if name in {"cx", "cz", "ecr"}))


def _build_circuit_report(
    circuit,
    backend,
    optimization_level: int,
    seed: int,
) -> dict[str, Any]:
    logical_ops = _serialize_op_counts(dict(circuit.count_ops()))
    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )
    transpiled_ops = _serialize_op_counts(dict(transpiled.count_ops()))
    return {
        "logical_qubits": int(circuit.num_qubits),
        "logical_depth": int(circuit.depth()),
        "logical_size": int(circuit.size()),
        "logical_ops": logical_ops,
        "transpiled_depth": int(transpiled.depth()),
        "transpiled_size": int(transpiled.size()),
        "transpiled_ops": transpiled_ops,
        "transpiled_two_qubit_gates": _two_qubit_gate_count(transpiled_ops),
    }


def _load_empirical_lengths(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        return {}

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    quantum = payload.get("quantum", {})
    empirical: dict[str, float] = {}
    for mode in ("ideal", "noisy", "mitigated"):
        length_curve = quantum.get(mode, {}).get("length_curve_mean", [])
        if not length_curve:
            continue
        empirical[mode] = float(np.mean(np.asarray(length_curve, dtype=float)))
    return empirical


def _mode_scale_reports(
    config: AppConfig,
    base_circuit,
    backend,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    base_report = _build_circuit_report(
        circuit=base_circuit,
        backend=backend,
        optimization_level=config.quantum_execution.optimization_level,
        seed=config.seed,
    )

    zne_enabled = (
        config.mitigation.enabled
        and config.mitigation.method.lower() in {"both", "zne"}
        and bool(config.mitigation.zne.scale_factors)
    )

    folded_reports: list[dict[str, Any]] = []
    if zne_enabled:
        for requested_scale in config.mitigation.zne.scale_factors:
            folded_circuit, achieved_scale = fold_circuit_for_noise_scaling(
                circuit=base_circuit,
                scale_factor=float(requested_scale),
            )
            report = _build_circuit_report(
                circuit=folded_circuit,
                backend=backend,
                optimization_level=0,
                seed=config.seed,
            )
            report["requested_scale"] = float(requested_scale)
            report["achieved_scale"] = float(achieved_scale)
            folded_reports.append(report)

    return {"base": base_report}, folded_reports


def _workload_estimate(
    *,
    config: AppConfig,
    parameter_count: int,
    steps_per_episode: float,
    per_call_scale_reports: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    max_steps = float(steps_per_episode)
    shots = int(config.quantum_execution.shots)
    n_episodes = int(config.training.n_episodes)
    n_eval_episodes = int(config.experiment.n_eval_episodes)
    episodes_per_update = max(1, int(config.training.episodes_per_update))
    selection_eval_episodes = max(0, int(config.training.selection_eval_episodes))
    updates = int(math.ceil(n_episodes / episodes_per_update))

    parameter_bindings_per_timestep = int(2 + 2 * parameter_count)
    training_parameter_bindings = float(
        n_episodes * max_steps * parameter_bindings_per_timestep
        + updates * selection_eval_episodes * max_steps
    )
    evaluation_parameter_bindings = float(n_eval_episodes * max_steps)
    total_parameter_bindings = training_parameter_bindings + evaluation_parameter_bindings

    circuit_multiplier = max(1, len(per_call_scale_reports))
    total_circuit_executions = float(total_parameter_bindings * circuit_multiplier)
    total_shots = float(total_circuit_executions * shots)
    total_two_qubit_gates = float(
        total_parameter_bindings
        * shots
        * sum(report["transpiled_two_qubit_gates"] for report in per_call_scale_reports)
    )

    return {
        "label": label,
        "steps_per_episode_assumption": float(max_steps),
        "parameter_bindings_per_timestep": parameter_bindings_per_timestep,
        "optimizer_updates_per_seed": updates,
        "training_parameter_bindings_per_seed": training_parameter_bindings,
        "evaluation_parameter_bindings_per_seed": evaluation_parameter_bindings,
        "total_parameter_bindings_per_seed": total_parameter_bindings,
        "circuit_executions_per_logical_query": circuit_multiplier,
        "estimated_total_circuit_executions_per_seed": total_circuit_executions,
        "estimated_total_shots_per_seed": total_shots,
        "estimated_total_two_qubit_gate_executions_per_seed": total_two_qubit_gates,
    }


def build_hardware_feasibility_report(config: AppConfig) -> dict[str, Any]:
    env = KeyDoorNavigationEnv(config=EnvRuntimeConfig(**config.environment.model_dump()))
    policy = QuantumPolicyNetwork(
        n_actions=env.action_space,
        n_observations=env.observation_space,
        config=QuantumPolicyConfig(**config.quantum_policy.model_dump()),
    )

    backend = resolve_fake_backend(config.quantum_execution.backend_name)
    if backend is None:
        raise RuntimeError(
            f"Could not resolve fake backend for '{config.quantum_execution.backend_name}'. "
            "The hardware audit needs the fake backend metadata for transpilation."
        )

    circuit_reports, folded_reports = _mode_scale_reports(
        config=config,
        base_circuit=policy._parameterized_circuit,
        backend=backend,
    )
    base_report = circuit_reports["base"]

    output_dir = resolve_project_path(config.results.output_dir)
    empirical_lengths = _load_empirical_lengths(output_dir / "summary.json")

    zne_enabled = (
        config.mitigation.enabled
        and config.mitigation.method.lower() in {"both", "zne"}
        and bool(folded_reports)
    )

    unmitigated_scales = [base_report]
    mitigated_scales = folded_reports if zne_enabled else [base_report]

    worst_case_workloads = {
        "ideal_or_noisy": _workload_estimate(
            config=config,
            parameter_count=policy.parameter_count,
            steps_per_episode=float(config.environment.max_episode_steps),
            per_call_scale_reports=unmitigated_scales,
            label="worst_case",
        ),
        "mitigated": _workload_estimate(
            config=config,
            parameter_count=policy.parameter_count,
            steps_per_episode=float(config.environment.max_episode_steps),
            per_call_scale_reports=mitigated_scales,
            label="worst_case",
        ),
    }

    empirical_workloads: dict[str, dict[str, Any]] = {}
    for mode, steps_per_episode in empirical_lengths.items():
        scale_reports = mitigated_scales if mode == "mitigated" else unmitigated_scales
        empirical_workloads[mode] = _workload_estimate(
            config=config,
            parameter_count=policy.parameter_count,
            steps_per_episode=float(steps_per_episode),
            per_call_scale_reports=scale_reports,
            label="empirical",
        )

    single_circuit_feasible = (
        base_report["logical_qubits"] <= 5
        and base_report["transpiled_two_qubit_gates"] <= 12
        and base_report["transpiled_depth"] <= 128
    )

    reasons: list[str] = []
    if worst_case_workloads["ideal_or_noisy"]["estimated_total_shots_per_seed"] > 1.0e5:
        reasons.append(
            "Full parameter-shift training is too shot-heavy for routine IBM hardware use. "
            f"The current default config is about {worst_case_workloads['ideal_or_noisy']['estimated_total_shots_per_seed']:,.0f} "
            "shots per seed even before ZNE."
        )
    if zne_enabled:
        reasons.append(
            "In-loop ZNE is not a realistic default for hardware training. The current mitigated path "
            f"expands each logical policy query into {len(mitigated_scales)} circuit executions and "
            f"about {worst_case_workloads['mitigated']['estimated_total_shots_per_seed']:,.0f} shots per seed."
        )
    if config.training.selection_eval_episodes > 0:
        reasons.append(
            "Validation inside training adds repeated hardware evaluations after each optimizer update."
        )
    if config.quantum_execution.shots < 256:
        reasons.append(
            f"{config.quantum_execution.shots} shots per circuit is low for stable hardware policy-gradient estimation."
        )
    if config.quantum_execution.compact_noise_model:
        reasons.append(
            "The default noisy simulation uses a compact averaged noise model, so it does not fully capture "
            "layout-specific heterogeneity, crosstalk, or scheduling effects."
        )

    suggested_modifications = [
        "Keep the 3-qubit, 1-repetition policy. The per-circuit footprint is already small enough for IBM hardware.",
        "Do not train the policy end-to-end on hardware. Train in simulation or on the fake backend, then run only held-out evaluation episodes on hardware.",
        "For hardware rehearsal, set `compact_noise_model: false` so the fake backend preserves qubit-specific noise and coupling constraints.",
        "Disable in-loop ZNE on hardware. Use readout-focused mitigation only, or reserve ZNE for a tiny final calibration study.",
        "Use more shots for hardware evaluation, typically 512 to 2048, but cut evaluation episode count and remove in-training validation to keep queue cost bounded.",
        "Use the provided `config/hardware_realistic.yaml` as a starting point for simulator-train and hardware-eval workflows.",
    ]

    return {
        "backend_name": config.quantum_execution.backend_name,
        "single_circuit_ibm_hardware_feasible": single_circuit_feasible,
        "full_training_ibm_hardware_realistic": False,
        "why_full_training_is_not_realistic": reasons,
        "suggested_modifications": suggested_modifications,
        "policy": {
            "parameter_count": int(policy.parameter_count),
            "action_register_qubits": int(policy.action_register_size),
            "latent_qubits": int(len(policy.latent_qubits)),
            "state_count": int(env.observation_space),
            "action_count": int(env.action_space),
        },
        "base_circuit": base_report,
        "zne_scale_circuits": folded_reports,
        "worst_case_workloads": worst_case_workloads,
        "empirical_workloads_from_latest_summary": empirical_workloads,
        "noise_model_review": {
            "compact_noise_model": bool(config.quantum_execution.compact_noise_model),
            "readout_correction_enabled": bool(config.mitigation.readout_correction),
            "zne_enabled": bool(zne_enabled),
            "zne_scale_factors": [float(scale) for scale in config.mitigation.zne.scale_factors],
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IBM hardware-feasibility audit for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to the project YAML config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to <results.output_dir>/hardware_audit.json.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(resolve_project_path(args.config))
    configure_logging(config.results.log_level)
    report = build_hardware_feasibility_report(config)
    output_path = resolve_project_path(args.output) if args.output else resolve_project_path(config.results.output_dir) / "hardware_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
