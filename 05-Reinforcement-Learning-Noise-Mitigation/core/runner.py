"""Stable public runner APIs for training, evaluation, benchmarking, and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schemas import BenchmarkSuiteResult, EvalResult, MethodSpec, PaperReportBundle, RunResult, ScenarioSpec
from reporting.paper_report import build_paper_report
from src.benchmark_suite import run_benchmark_suite
from src.project_paths import resolve_project_path
from src.training_pipeline import run_training_pipeline


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def train_method(
    method_spec: MethodSpec,
    scenario_spec: ScenarioSpec,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one scenario configuration through the existing training pipeline."""

    config_payload = {
        "experiment": {
            "seeds": scenario_spec.seeds,
            "n_eval_episodes": scenario_spec.n_eval_episodes,
            "run_tabular_baseline": False,
            "run_mlp_baseline": False,
            "run_mlp_actor_critic": False,
            "run_random_baseline": False,
            "run_quantum_actor_critic": False,
        },
        "environment": scenario_spec.environment,
        "quantum_policy": scenario_spec.quantum_policy,
        "results": {
            "output_dir": str(resolve_project_path(output_root or f"results/{scenario_spec.name}/{method_spec.name}")),
        },
    }
    if method_spec.name == "random":
        config_payload["experiment"]["run_random_baseline"] = True
    elif method_spec.name == "tabular_reinforce":
        config_payload["experiment"]["run_tabular_baseline"] = True
        config_payload.setdefault("baselines", {})
    elif method_spec.name == "mlp_reinforce":
        config_payload["experiment"]["run_mlp_baseline"] = True
        config_payload.setdefault("mlp_baseline", {})
    elif method_spec.name == "mlp_actor_critic":
        config_payload["experiment"]["run_mlp_actor_critic"] = True
        config_payload.setdefault("mlp_actor_critic", {})
    elif method_spec.name == "quantum_reinforce":
        pass
    elif method_spec.name == "quantum_actor_critic":
        config_payload["experiment"]["run_quantum_actor_critic"] = True
    config_payload = _deep_merge(config_payload, method_spec.config)
    return run_training_pipeline(config_payload)


def evaluate_checkpoint(
    run_result: RunResult | dict[str, Any],
    scenario_spec: ScenarioSpec,
    execution_mode: str,
) -> EvalResult:
    """Return the saved evaluation block from an existing run-result-like payload."""

    payload = RunResult(**run_result) if isinstance(run_result, dict) else run_result
    evaluation = payload.evaluation
    if payload.method_spec.execution_mode is not None and payload.method_spec.execution_mode != execution_mode:
        evaluation = payload.evaluation
    return EvalResult(**evaluation.model_dump())


def run_benchmark_suite_api(
    suite_path: str | Path,
    *,
    output_root: str | Path | None = None,
    max_scenarios: int | None = None,
) -> BenchmarkSuiteResult:
    """Programmatic wrapper around the benchmark-suite runner."""

    payload = run_benchmark_suite(
        suite_path=resolve_project_path(suite_path),
        output_root=resolve_project_path(output_root) if output_root is not None else None,
        max_scenarios=max_scenarios,
    )
    return BenchmarkSuiteResult(**payload)


def build_paper_report_api(results_root: str | Path) -> PaperReportBundle:
    """Programmatic wrapper around the report builder."""

    payload = build_paper_report(resolve_project_path(results_root))
    return PaperReportBundle(**payload)
