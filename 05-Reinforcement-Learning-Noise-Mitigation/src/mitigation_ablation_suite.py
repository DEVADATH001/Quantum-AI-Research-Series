"""Run mitigation ablations for the QRL benchmark."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.project_paths import resolve_project_path
from src.training_pipeline import run_training_pipeline
from utils.qiskit_helpers import configure_logging, ensure_dir, save_json


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mitigation ablation suite for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Base experiment config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results_mitigation_ablation",
        help="Root directory for ablation outputs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for ablation runs.",
    )
    return parser


def _condition_overrides() -> dict[str, dict[str, Any]]:
    return {
        "none": {
            "mitigation": {
                "enabled": False,
                "readout_correction": False,
                "method": "both",
            }
        },
        "readout_only": {
            "mitigation": {
                "enabled": True,
                "readout_correction": True,
                "method": "readout",
            }
        },
        "zne_only": {
            "mitigation": {
                "enabled": True,
                "readout_correction": False,
                "method": "zne",
            }
        },
        "both": {
            "mitigation": {
                "enabled": True,
                "readout_correction": True,
                "method": "both",
            }
        },
    }


def _merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    base_path = resolve_project_path(args.config)
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_root = ensure_dir(resolve_project_path(args.output_root))

    report_rows: list[dict[str, Any]] = []
    for condition_name, overrides in _condition_overrides().items():
        trial_config = _merge_nested(base_config, overrides)
        trial_config.setdefault("results", {})
        trial_config["results"]["output_dir"] = str(output_root / condition_name)
        trial_config["results"]["log_level"] = args.log_level

        summary = run_training_pipeline(trial_config)
        report_rows.append(
            {
                "condition": condition_name,
                "output_dir": str((output_root / condition_name).resolve()),
                "mitigated_eval_success": summary["quantum"]["mitigated"].get("eval_success_mean"),
                "mitigated_eval_reward": summary["quantum"]["mitigated"].get("eval_reward_mean"),
                "mitigated_reward_auc": summary["quantum"]["mitigated"].get("reward_auc_mean"),
                "mitigation_gain_over_noisy": summary.get("mitigation_gain_over_noisy", {}),
            }
        )

    payload = {
        "base_config_path": str(base_path.resolve()),
        "conditions": report_rows,
    }
    save_json(output_root / "ablation_report.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
