"""Run environment robustness sweeps for the QRL benchmark."""

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


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Environment robustness sweep for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Base experiment config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results_environment_sweep",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--slip-probabilities",
        type=str,
        default="0.0,0.05,0.1",
        help="Comma-separated slip probabilities.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=str,
        default="6,8",
        help="Comma-separated episode-length caps.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for the sweep.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    base_path = resolve_project_path(args.config)
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_root = ensure_dir(resolve_project_path(args.output_root))

    slip_values = _parse_float_list(args.slip_probabilities)
    horizon_values = _parse_int_list(args.max_episode_steps)
    leaderboard: list[dict[str, Any]] = []

    for condition_idx, (slip_prob, horizon) in enumerate(itertools.product(slip_values, horizon_values), start=1):
        trial_cfg = deepcopy(base_config)
        trial_cfg.setdefault("environment", {})
        trial_cfg["environment"]["slip_probability"] = float(slip_prob)
        trial_cfg["environment"]["max_episode_steps"] = int(horizon)
        trial_cfg.setdefault("training", {})
        trial_cfg["training"]["max_episode_steps"] = int(horizon)
        trial_cfg.setdefault("baselines", {})
        trial_cfg["baselines"]["max_episode_steps"] = int(horizon)
        if "mlp_baseline" in trial_cfg:
            trial_cfg["mlp_baseline"]["max_episode_steps"] = int(horizon)
        trial_cfg.setdefault("results", {})
        condition_label = f"slip_{slip_prob:.2f}_horizon_{horizon}"
        trial_cfg["results"]["output_dir"] = str(output_root / condition_label)
        trial_cfg["results"]["log_level"] = args.log_level

        summary = run_training_pipeline(trial_cfg)
        leaderboard.append(
            {
                "condition_index": condition_idx,
                "condition": condition_label,
                "slip_probability": float(slip_prob),
                "max_episode_steps": int(horizon),
                "output_dir": str((output_root / condition_label).resolve()),
                "quantum_mitigated_eval_success": summary["quantum"]["mitigated"].get("eval_success_mean"),
                "quantum_noisy_eval_success": summary["quantum"]["noisy"].get("eval_success_mean"),
                "tabular_eval_success": summary.get("tabular_baseline", {}).get("eval_success_mean"),
                "mlp_eval_success": summary.get("mlp_baseline", {}).get("eval_success_mean"),
            }
        )

    payload = {
        "base_config_path": str(base_path.resolve()),
        "conditions": leaderboard,
    }
    save_json(output_root / "environment_sweep_report.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
