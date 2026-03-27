"""Config-driven hyperparameter sweep for the QRL benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.project_paths import resolve_project_path
from src.training_pipeline import run_training_pipeline
from utils.qiskit_helpers import configure_logging, ensure_dir, save_json


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]

def _lookup_metric(payload: dict[str, Any], dotted_path: str) -> float:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, dict):
            raise KeyError(f"Cannot descend into non-dictionary value while resolving '{dotted_path}'.")
        value = value[key]
    return float(value)


def _build_trial_configs(base_config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    quantum_learning_rates = _parse_float_list(args.quantum_learning_rates)
    quantum_entropy = _parse_float_list(args.quantum_entropy)
    tabular_learning_rates = _parse_float_list(args.tabular_learning_rates)
    mlp_learning_rates = _parse_float_list(args.mlp_learning_rates)

    trials: list[dict[str, Any]] = []
    for idx, (q_lr, q_ent, tab_lr, mlp_lr) in enumerate(
        itertools.product(quantum_learning_rates, quantum_entropy, tabular_learning_rates, mlp_learning_rates),
        start=1,
    ):
        trial = deepcopy(base_config)
        trial["training"]["learning_rate"] = q_lr
        trial["training"]["entropy_coeff"] = q_ent
        trial["baselines"]["learning_rate"] = tab_lr
        if "mlp_baseline" in trial:
            trial["mlp_baseline"]["learning_rate"] = mlp_lr
        trial.setdefault("results", {})
        trial["results"]["output_dir"] = str(Path(args.output_root) / f"trial_{idx:03d}")
        trial["results"]["log_level"] = args.log_level
        trials.append(trial)
    return trials


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/smoke_test.yaml",
        help="Base YAML config for the sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results_sweep",
        help="Root directory where trial outputs and the leaderboard are written.",
    )
    parser.add_argument(
        "--quantum-learning-rates",
        type=str,
        default="0.02,0.03",
        help="Comma-separated learning rates for the quantum learner.",
    )
    parser.add_argument(
        "--quantum-entropy",
        type=str,
        default="0.01,0.02",
        help="Comma-separated entropy coefficients for the quantum learner.",
    )
    parser.add_argument(
        "--tabular-learning-rates",
        type=str,
        default="0.05",
        help="Comma-separated learning rates for the tabular baseline.",
    )
    parser.add_argument(
        "--mlp-learning-rates",
        type=str,
        default="0.03",
        help="Comma-separated learning rates for the MLP baseline.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="quantum.mitigated.eval_success_mean",
        help="Dotted metric path used to rank trials.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on the number of evaluated trials.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for trial runs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    base_path = resolve_project_path(args.config)
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    output_root = ensure_dir(resolve_project_path(args.output_root))
    args.output_root = str(output_root)

    trials = _build_trial_configs(base_config, args)
    if args.max_trials is not None:
        trials = trials[: max(0, int(args.max_trials))]

    leaderboard: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None

    for trial_idx, trial_cfg in enumerate(trials, start=1):
        trial_output_dir = ensure_dir(trial_cfg["results"]["output_dir"])
        summary = run_training_pipeline(trial_cfg)
        objective_value = _lookup_metric(summary, args.objective)
        trial_record = {
            "trial_index": trial_idx,
            "config_overrides": {
                "training.learning_rate": trial_cfg["training"]["learning_rate"],
                "training.entropy_coeff": trial_cfg["training"]["entropy_coeff"],
                "baselines.learning_rate": trial_cfg["baselines"]["learning_rate"],
                "mlp_baseline.learning_rate": trial_cfg.get("mlp_baseline", {}).get("learning_rate"),
            },
            "objective": args.objective,
            "objective_value": objective_value,
            "output_dir": str(trial_output_dir.resolve()),
        }
        leaderboard.append(trial_record)
        if best_trial is None or objective_value > float(best_trial["objective_value"]):
            best_trial = trial_record

    report = {
        "base_config_path": str(base_path.resolve()),
        "objective": args.objective,
        "trial_count": len(leaderboard),
        "leaderboard": sorted(leaderboard, key=lambda item: float(item["objective_value"]), reverse=True),
        "best_trial": best_trial,
    }
    save_json(output_root / "hyperparameter_sweep.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
