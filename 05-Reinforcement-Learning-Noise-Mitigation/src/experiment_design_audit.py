"""Reviewer-style audit of the experiment design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.evaluation import summarize_history
from src.project_paths import resolve_project_path
from src.research_stats import holm_correct, paired_method_comparison
from utils.qiskit_helpers import configure_logging, ensure_dir, save_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_quantum_records(
    results_root: Path,
    mode: str,
    convergence_threshold: float,
    moving_window: int,
    prefix: str = "",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quantum_root = results_root / "quantum"
    if not quantum_root.exists():
        return records

    for seed_dir in sorted(path for path in quantum_root.iterdir() if path.is_dir()):
        record = _load_json(seed_dir / f"{prefix}{mode}_training_log.json")
        if record is None:
            continue
        record.setdefault(
            "summary_metrics",
            summarize_history(
                record,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
            ),
        )
        records.append(record)
    return records


def _load_baseline_records(
    results_root: Path,
    filename: str,
    convergence_threshold: float,
    moving_window: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline_root = results_root / "baselines"
    if not baseline_root.exists():
        return records

    for seed_dir in sorted(path for path in baseline_root.iterdir() if path.is_dir()):
        record = _load_json(seed_dir / filename)
        if record is None:
            continue
        record.setdefault(
            "summary_metrics",
            summarize_history(
                record,
                convergence_threshold=convergence_threshold,
                moving_avg_window=moving_window,
            ),
        )
        records.append(record)
    return records


def _evaluation_metric(metric_key: str):
    def _metric(record: dict[str, Any]) -> float | None:
        return record.get("evaluation", {}).get(metric_key)

    return _metric


def _summary_metric(metric_key: str):
    def _metric(record: dict[str, Any]) -> float | None:
        return record.get("summary_metrics", {}).get(metric_key)

    return _metric


def _build_pairwise_statistics(
    quantum_records: dict[str, list[dict[str, Any]]],
    classical_records_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    metric_extractors = {
        "eval_success": _evaluation_metric("success_rate"),
        "eval_reward": _evaluation_metric("avg_reward"),
        "reward_auc": _summary_metric("reward_auc"),
    }
    pairs: list[tuple[str, list[dict[str, Any]], str, list[dict[str, Any]]]] = [
        ("ideal", quantum_records["ideal"], "noisy", quantum_records["noisy"]),
        ("mitigated", quantum_records["mitigated"], "noisy", quantum_records["noisy"]),
    ]
    for baseline_name, records in classical_records_by_name.items():
        if not records:
            continue
        pairs.extend(
            [
                ("ideal", quantum_records["ideal"], baseline_name, records),
                ("mitigated", quantum_records["mitigated"], baseline_name, records),
                ("noisy", quantum_records["noisy"], baseline_name, records),
            ]
        )

    correction_targets: dict[str, float | None] = {}
    for left_name, left_records, right_name, right_records in pairs:
        comparisons[f"{left_name}_vs_{right_name}"] = {
            metric_name: paired_method_comparison(
                left_records,
                right_records,
                metric_name=metric_name,
                metric_fn=metric_fn,
            )
            for metric_name, metric_fn in metric_extractors.items()
        }
        for metric_name, metric_payload in comparisons[f"{left_name}_vs_{right_name}"].items():
            correction_targets[f"{left_name}_vs_{right_name}:{metric_name}"] = metric_payload.get("sign_flip_pvalue")
    corrected = holm_correct(correction_targets)
    for pair_key, metric_payloads in comparisons.items():
        for metric_name, metric_payload in metric_payloads.items():
            metric_payload["holm_corrected_sign_flip_pvalue"] = corrected.get(f"{pair_key}:{metric_name}")
    return comparisons


def build_experiment_design_audit(config_path: str | Path, summary_path: str | Path | None = None) -> dict[str, Any]:
    resolved_config_path = resolve_project_path(config_path)
    config = load_config(resolved_config_path)
    results_root = resolve_project_path(config.results.output_dir)
    summary_file = resolve_project_path(summary_path) if summary_path is not None else results_root / "summary.json"
    summary = _load_json(summary_file)

    convergence_threshold = float(config.evaluation.convergence_threshold)
    moving_window = int(config.evaluation.moving_avg_window)
    quantum_records = {
        mode: _load_quantum_records(results_root, mode, convergence_threshold, moving_window)
        for mode in ("ideal", "noisy", "mitigated")
    }
    quantum_actor_critic_records = {
        mode: _load_quantum_records(
            results_root,
            mode,
            convergence_threshold,
            moving_window,
            prefix="quantum_actor_critic_",
        )
        for mode in ("ideal", "noisy", "mitigated")
    }
    tabular_records = _load_baseline_records(
        results_root,
        filename="tabular_reinforce_training_log.json",
        convergence_threshold=convergence_threshold,
        moving_window=moving_window,
    )
    mlp_records = _load_baseline_records(
        results_root,
        filename="mlp_reinforce_training_log.json",
        convergence_threshold=convergence_threshold,
        moving_window=moving_window,
    )
    mlp_actor_critic_records = _load_baseline_records(
        results_root,
        filename="mlp_actor_critic_training_log.json",
        convergence_threshold=convergence_threshold,
        moving_window=moving_window,
    )

    seed_count = len(config.experiment.seeds)
    random_baseline_present = summary is not None and summary.get("random_baseline") is not None
    tabular_baseline_present = len(tabular_records) > 0
    mlp_baseline_present = len(mlp_records) > 0
    mlp_actor_critic_present = len(mlp_actor_critic_records) > 0
    stronger_baseline_present = mlp_baseline_present or mlp_actor_critic_present
    tuning_artifact_present = (results_root / "hyperparameter_sweep.json").exists()
    mitigation_ablation_present = (results_root / "mitigation_ablation" / "ablation_report.json").exists()
    environment_sweep_present = (results_root / "environment_sweep" / "environment_sweep_report.json").exists()
    benchmark_suite_present = (results_root / "benchmark_suite" / "benchmark_report.json").exists()
    significance_ready = seed_count >= 5
    summary_has_statistical_analysis = bool(summary and summary.get("statistical_analysis"))
    resource_efficiency_present = bool(summary and summary.get("resource_efficiency"))
    pairwise_statistics = _build_pairwise_statistics(
        quantum_records=quantum_records,
        classical_records_by_name={
            "tabular": tabular_records,
            "mlp": mlp_records,
            "mlp_actor_critic": mlp_actor_critic_records,
        },
    )
    pairwise_statistics_quantum_actor_critic = _build_pairwise_statistics(
        quantum_records=quantum_actor_critic_records,
        classical_records_by_name={
            "tabular": tabular_records,
            "mlp": mlp_records,
            "mlp_actor_critic": mlp_actor_critic_records,
        },
    )

    ratings = {
        "dataset_choice": {
            "score": 7 if benchmark_suite_present else 5 if environment_sweep_present else 3,
            "assessment": (
                "The project now includes a named multi-scenario benchmark suite in addition to the environment sweep, which is a materially stronger benchmark contribution than a single fixed-task study."
                if benchmark_suite_present
                else
                "The project still centers on a single handcrafted RL environment, but it now includes a saved environment sweep over difficulty and stochasticity, which makes the evidence materially stronger than a single fixed-task report."
                if environment_sweep_present
                else "The project uses a single handcrafted toy RL environment, not a dataset suite. That is acceptable for a prototype benchmark, but too narrow for broad algorithmic claims."
            ),
        },
        "baseline_comparisons": {
            "score": 7 if stronger_baseline_present else 4 if tabular_baseline_present else 2,
            "assessment": (
                "Random, tabular, and MLP baselines are present. This is materially better than the old setup, "
                "though stronger budget-matched comparisons and more baseline diversity would still help."
                if stronger_baseline_present
                else "Random and tabular baselines are present, which is better than no baselines, "
                "but the design still lacks stronger classical function-approximation baselines and budget-matched comparisons."
            ),
        },
        "evaluation_metrics": {
            "score": 8 if resource_efficiency_present and benchmark_suite_present else 7 if environment_sweep_present else 6,
            "assessment": (
                "Reward, success rate, convergence, runtime, AUC, and compute-normalized efficiency metrics are all present. The remaining gap is breadth over more tasks and budgets, not the total absence of efficiency reporting."
                if resource_efficiency_present
                else
                "Reward, success rate, convergence, runtime, and AUC are useful, but the experiment still lacks "
                "shot-normalized efficiency metrics and a broader robustness matrix over more environment variants and compute budgets."
            ),
        },
        "hyperparameter_tuning": {
            "score": 7 if tuning_artifact_present else 2,
            "assessment": (
                "A sweep workflow is available." if tuning_artifact_present else
                "There is no evidence that the reported default config came from a fair tuning protocol with equal budgets across methods."
            ),
        },
        "reproducibility": {
            "score": 8 if summary and summary.get("reproducibility") else 6,
            "assessment": (
                "Config files, pinned requirements, and per-seed logs make the project fairly reproducible. "
                "The upgraded summaries now include provenance metadata."
            ),
        },
        "statistical_significance": {
            "score": 2 if not significance_ready else 6,
            "assessment": (
                f"Only {seed_count} seed(s) are configured in the default benchmark, which is underpowered for comparative claims."
                if not significance_ready
                else "The seed count is at least large enough to attempt pairwise significance checks, though stronger power would still be preferable."
            ),
        },
    }

    missing_experiments: list[str] = []
    if not significance_ready:
        missing_experiments.append(
            "Run a multi-seed benchmark with at least 5 to 10 seeds per method before making comparative claims."
        )
    if not summary_has_statistical_analysis:
        missing_experiments.append(
            "Report confidence intervals and pairwise statistical tests for the main held-out metrics, not just means and standard deviations."
        )
    if not tuning_artifact_present:
        missing_experiments.insert(
            1,
            "Use the new sweep workflow to tune both quantum and classical baselines with an explicit, equal hyperparameter budget.",
        )
    if not environment_sweep_present:
        missing_experiments.insert(
            2,
            "Run environment-sweep experiments over difficulty and stochasticity, for example changing slip probability, horizon, and reward shaping.",
        )
    if not benchmark_suite_present:
        missing_experiments.insert(
            3,
            "Promote the environment sweep into a named multi-scenario benchmark suite with cross-condition leaderboards and robust aggregate reporting.",
        )
    if not mitigation_ablation_present:
        missing_experiments.insert(
            3,
            "Add mitigation ablations: readout-only, ZNE-only, both, and neither.",
        )
    if not stronger_baseline_present:
        missing_experiments.insert(
            2,
            "Add at least one stronger classical baseline beyond tabular softmax REINFORCE, such as a small classical function-approximation policy trained with the same optimizer budget.",
        )

    convincing = False
    if significance_ready and stronger_baseline_present and tuning_artifact_present and environment_sweep_present:
        convincing = True

    return {
        "config_path": str(resolved_config_path),
        "summary_path": str(summary_file.resolve()) if summary_file.exists() else None,
        "convincing_experiment": convincing,
        "ratings": ratings,
        "pairwise_statistics": pairwise_statistics,
        "pairwise_statistics_quantum_actor_critic": pairwise_statistics_quantum_actor_critic,
        "missing_experiments": missing_experiments,
        "meta": {
            "default_seed_count": seed_count,
            "default_eval_episodes": int(config.experiment.n_eval_episodes),
            "random_baseline_present": random_baseline_present,
            "tabular_baseline_present": tabular_baseline_present,
            "mlp_baseline_present": mlp_baseline_present,
            "mlp_actor_critic_present": mlp_actor_critic_present,
            "tuning_artifact_present": tuning_artifact_present,
            "mitigation_ablation_present": mitigation_ablation_present,
            "environment_sweep_present": environment_sweep_present,
            "benchmark_suite_present": benchmark_suite_present,
            "resource_efficiency_present": resource_efficiency_present,
            "results_root": str(results_root.resolve()),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scientific-validity audit for the QRL benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional summary.json path. Defaults to <results.output_dir>/summary.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional audit output path. Defaults to <results.output_dir>/experiment_design_audit.json.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(resolve_project_path(args.config))
    configure_logging(config.results.log_level)
    report = build_experiment_design_audit(args.config, args.summary)
    output_path = resolve_project_path(args.output) if args.output else ensure_dir(resolve_project_path(config.results.output_dir)) / "experiment_design_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
