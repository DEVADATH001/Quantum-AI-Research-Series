"""Run the multi-scenario benchmark suite for the research-grade QRL system."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from benchmarks.scenario_registry import make_research_benchmark_suite, make_smoke_benchmark_suite
from core.schemas import BenchmarkSuiteResult
from src.project_paths import path_relative_to_project, resolve_project_path
from src.training_pipeline import run_training_pipeline
from utils.qiskit_helpers import configure_logging, ensure_dir, save_json

METHOD_SPECS: tuple[tuple[str, str], ...] = (
    ("random", "Random baseline"),
    ("tabular_reinforce", "Tabular REINFORCE"),
    ("mlp_reinforce", "MLP REINFORCE"),
    ("mlp_actor_critic", "MLP Actor-Critic"),
    ("quantum_reinforce_ideal", "Quantum REINFORCE ideal"),
    ("quantum_reinforce_noisy", "Quantum REINFORCE noisy"),
    ("quantum_reinforce_mitigated", "Quantum REINFORCE mitigated"),
    ("quantum_actor_critic_ideal", "Quantum Actor-Critic ideal"),
    ("quantum_actor_critic_noisy", "Quantum Actor-Critic noisy"),
    ("quantum_actor_critic_mitigated", "Quantum Actor-Critic mitigated"),
)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _sync_episode_limits(config_payload: dict[str, Any]) -> None:
    max_episode_steps = config_payload.get("environment", {}).get("max_episode_steps")
    if max_episode_steps is None:
        return

    episode_keys = ("training", "baselines", "mlp_baseline", "mlp_actor_critic", "quantum_actor_critic")
    for key in episode_keys:
        config_payload.setdefault(key, {})["max_episode_steps"] = int(max_episode_steps)


def _load_suite_definition(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark suite YAML must load to a dictionary.")
    if "base_config" not in payload:
        raise ValueError("Benchmark suite YAML must define 'base_config'.")
    return payload


def _resolve_scenarios(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_source = payload.get("scenario_source")
    if scenario_source == "research_registry":
        return make_research_benchmark_suite(base_config=str(payload["base_config"]))["scenarios"]
    if scenario_source == "smoke_registry":
        return make_smoke_benchmark_suite(base_config=str(payload["base_config"]))["scenarios"]

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(
            "Benchmark suite YAML must define a non-empty 'scenarios' list or a supported 'scenario_source'."
        )
    return scenarios


def _method_metrics_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods = summary.get("methods", {})
    quantum_methods = methods.get("quantum_reinforce", {})
    quantum_actor_critic_methods = methods.get("quantum_actor_critic", {})
    resource_efficiency = summary.get("resource_efficiency", {})
    quantum_efficiency = resource_efficiency.get("quantum", {})
    classical_efficiency = resource_efficiency.get("classical", {})

    random_summary = methods.get("random", summary.get("random_baseline", {})) or {}
    tabular_summary = methods.get("tabular_reinforce", summary.get("tabular_baseline", {})) or {}
    mlp_summary = methods.get("mlp_reinforce", summary.get("mlp_baseline", {})) or {}
    mlp_actor_critic_summary = methods.get("mlp_actor_critic", summary.get("mlp_actor_critic", {})) or {}

    def classical_entry(
        summary_block: dict[str, Any],
        efficiency_key: str,
        label: str,
    ) -> dict[str, Any]:
        efficiency = classical_efficiency.get(efficiency_key, {})
        return {
            "label": label,
            "eval_success": summary_block.get("eval_success_mean", summary_block.get("success_rate_mean")),
            "eval_reward": summary_block.get("eval_reward_mean", summary_block.get("avg_reward_mean")),
            "reward_auc": summary_block.get("reward_auc_mean"),
            "success_auc": summary_block.get("success_auc_mean"),
            "eval_success_ci95": summary_block.get("eval_success_ci95"),
            "total_runtime_mean": summary_block.get("total_runtime_mean"),
            "eval_success_per_runtime_sec": efficiency.get("eval_success_per_runtime_sec"),
            "eval_success_per_million_shots": efficiency.get("eval_success_per_million_shots"),
            "estimated_total_shots_per_seed": efficiency.get("estimated_total_shots_per_seed"),
        }

    def quantum_entry(
        block: dict[str, Any],
        efficiency_family: str,
        mode: str,
        label: str,
    ) -> dict[str, Any]:
        efficiency = quantum_efficiency.get(efficiency_family, {}).get(mode, {})
        return {
            "label": label,
            "eval_success": block.get("eval_success_mean"),
            "eval_reward": block.get("eval_reward_mean"),
            "reward_auc": block.get("reward_auc_mean"),
            "success_auc": block.get("success_auc_mean"),
            "eval_success_ci95": block.get("eval_success_ci95"),
            "total_runtime_mean": block.get("total_runtime_mean"),
            "eval_success_per_runtime_sec": efficiency.get("eval_success_per_runtime_sec"),
            "eval_success_per_million_shots": efficiency.get("eval_success_per_million_shots"),
            "estimated_total_shots_per_seed": efficiency.get("estimated_total_shots_per_seed"),
        }

    return {
        "random": classical_entry(random_summary, "random", "Random baseline"),
        "tabular_reinforce": classical_entry(tabular_summary, "tabular_reinforce", "Tabular REINFORCE"),
        "mlp_reinforce": classical_entry(mlp_summary, "mlp_reinforce", "MLP REINFORCE"),
        "mlp_actor_critic": classical_entry(
            mlp_actor_critic_summary,
            "mlp_actor_critic",
            "MLP Actor-Critic",
        ),
        "quantum_reinforce_ideal": quantum_entry(
            quantum_methods.get("ideal", {}),
            "quantum_reinforce",
            "ideal",
            "Quantum REINFORCE ideal",
        ),
        "quantum_reinforce_noisy": quantum_entry(
            quantum_methods.get("noisy", {}),
            "quantum_reinforce",
            "noisy",
            "Quantum REINFORCE noisy",
        ),
        "quantum_reinforce_mitigated": quantum_entry(
            quantum_methods.get("mitigated", {}),
            "quantum_reinforce",
            "mitigated",
            "Quantum REINFORCE mitigated",
        ),
        "quantum_actor_critic_ideal": quantum_entry(
            quantum_actor_critic_methods.get("ideal", {}),
            "quantum_actor_critic",
            "ideal",
            "Quantum Actor-Critic ideal",
        ),
        "quantum_actor_critic_noisy": quantum_entry(
            quantum_actor_critic_methods.get("noisy", {}),
            "quantum_actor_critic",
            "noisy",
            "Quantum Actor-Critic noisy",
        ),
        "quantum_actor_critic_mitigated": quantum_entry(
            quantum_actor_critic_methods.get("mitigated", {}),
            "quantum_actor_critic",
            "mitigated",
            "Quantum Actor-Critic mitigated",
        ),
    }


def _rank_methods(method_metrics: dict[str, dict[str, Any]]) -> dict[str, int]:
    ranked = sorted(
        (
            (method_name, metrics["eval_success"])
            for method_name, metrics in method_metrics.items()
            if metrics.get("eval_success") is not None
        ),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return {method_name: rank for rank, (method_name, _) in enumerate(ranked, start=1)}


def _metric_delta(
    method_metrics: dict[str, dict[str, Any]],
    *,
    left: str,
    right: str,
    key: str,
) -> float | None:
    left_value = method_metrics.get(left, {}).get(key)
    right_value = method_metrics.get(right, {}).get(key)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _noise_profile(method_metrics: dict[str, dict[str, Any]], family: str) -> dict[str, float | None]:
    return {
        "noise_drop_eval_success": _metric_delta(
            method_metrics,
            left=f"{family}_ideal",
            right=f"{family}_noisy",
            key="eval_success",
        ),
        "mitigation_recovery_eval_success": _metric_delta(
            method_metrics,
            left=f"{family}_mitigated",
            right=f"{family}_noisy",
            key="eval_success",
        ),
        "noise_drop_eval_reward": _metric_delta(
            method_metrics,
            left=f"{family}_ideal",
            right=f"{family}_noisy",
            key="eval_reward",
        ),
        "mitigation_recovery_eval_reward": _metric_delta(
            method_metrics,
            left=f"{family}_mitigated",
            right=f"{family}_noisy",
            key="eval_reward",
        ),
    }


def _scenario_result(
    *,
    scenario_index: int,
    scenario: dict[str, Any],
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    method_metrics = _method_metrics_from_summary(summary)
    ranks = _rank_methods(method_metrics)
    ranked_methods = [
        {
            "method": method_name,
            "label": method_metrics[method_name]["label"],
            "eval_success": method_metrics[method_name]["eval_success"],
            "rank": ranks.get(method_name),
        }
        for method_name, _ in METHOD_SPECS
        if method_metrics.get(method_name, {}).get("eval_success") is not None
    ]
    ranked_methods.sort(key=lambda item: (item["rank"], item["method"]))
    leader = ranked_methods[0] if ranked_methods else None

    return {
        "scenario_index": scenario_index,
        "name": str(scenario["name"]),
        "description": str(scenario.get("description", "")).strip(),
        "output_dir": str(output_dir.resolve()),
        "summary_path": str((output_dir / "summary.json").resolve()),
        "environment": summary.get("environment", {}),
        "method_metrics": method_metrics,
        "method_ranks": ranks,
        "leader": leader,
        "noise_profile": {
            "quantum_reinforce": _noise_profile(method_metrics, "quantum_reinforce"),
            "quantum_actor_critic": _noise_profile(method_metrics, "quantum_actor_critic"),
        },
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _aggregate_suite_results(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_methods: dict[str, dict[str, Any]] = {}
    for method_name, label in METHOD_SPECS:
        eval_success_values = [
            scenario["method_metrics"].get(method_name, {}).get("eval_success") for scenario in scenarios
        ]
        eval_reward_values = [
            scenario["method_metrics"].get(method_name, {}).get("eval_reward") for scenario in scenarios
        ]
        runtime_eff_values = [
            scenario["method_metrics"].get(method_name, {}).get("eval_success_per_runtime_sec")
            for scenario in scenarios
        ]
        shot_eff_values = [
            scenario["method_metrics"].get(method_name, {}).get("eval_success_per_million_shots")
            for scenario in scenarios
        ]
        ranks = [scenario["method_ranks"].get(method_name) for scenario in scenarios if method_name in scenario["method_ranks"]]
        aggregate_methods[method_name] = {
            "label": label,
            "scenario_count": int(len([value for value in eval_success_values if value is not None])),
            "average_eval_success": _mean_or_none(eval_success_values),
            "average_eval_reward": _mean_or_none(eval_reward_values),
            "average_rank": _mean_or_none(ranks),
            "scenario_wins": int(sum(1 for scenario in scenarios if scenario.get("leader", {}).get("method") == method_name)),
            "average_eval_success_per_runtime_sec": _mean_or_none(runtime_eff_values),
            "average_eval_success_per_million_shots": _mean_or_none(shot_eff_values),
        }

    ranked_methods = sorted(
        aggregate_methods.items(),
        key=lambda item: (
            float("inf") if item[1]["average_rank"] is None else float(item[1]["average_rank"]),
            item[0],
        ),
    )

    noise_summary = {
        family: {
            "average_noise_drop_eval_success": _mean_or_none(
                [scenario["noise_profile"][family]["noise_drop_eval_success"] for scenario in scenarios]
            ),
            "average_mitigation_recovery_eval_success": _mean_or_none(
                [scenario["noise_profile"][family]["mitigation_recovery_eval_success"] for scenario in scenarios]
            ),
        }
        for family in ("quantum_reinforce", "quantum_actor_critic")
    }

    def _winner(metric_key: str, minimize: bool = False) -> str | None:
        candidates = [
            (method_name, metrics)
            for method_name, metrics in aggregate_methods.items()
            if metrics.get(metric_key) is not None
        ]
        if not candidates:
            return None
        if minimize:
            best_name = min(candidates, key=lambda item: (float(item[1][metric_key]), item[0]))[0]
        else:
            best_name = max(candidates, key=lambda item: (float(item[1][metric_key]), item[0]))[0]
        return aggregate_methods[best_name]["label"]

    findings = [
        f"Best average rank: {_winner('average_rank', minimize=True)}.",
        f"Best raw held-out success: {_winner('average_eval_success')}.",
        f"Best runtime-normalized efficiency: {_winner('average_eval_success_per_runtime_sec')}.",
        f"Best shot-normalized efficiency: {_winner('average_eval_success_per_million_shots')}.",
    ]

    return {
        "method_leaderboard": [metrics | {"method": method_name} for method_name, metrics in ranked_methods],
        "noise_summary": noise_summary,
        "headline_answers": {
            "average_rank_winner": _winner("average_rank", minimize=True),
            "raw_performance_winner": _winner("average_eval_success"),
            "runtime_efficiency_winner": _winner("average_eval_success_per_runtime_sec"),
            "shot_efficiency_winner": _winner("average_eval_success_per_million_shots"),
        },
        "finding_summary": [finding for finding in findings if finding],
        "meta": {
            "scenario_count": len(scenarios),
            "method_count": len(METHOD_SPECS),
        },
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _render_markdown_report(
    *,
    suite_name: str,
    description: str,
    base_config_path: Path,
    suite_config_path: Path,
    aggregate: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {suite_name}",
        "",
        description.strip() or "Benchmark report for the QRL research suite.",
        "",
        "## Provenance",
        "",
        f"- Suite config: `{path_relative_to_project(suite_config_path)}`",
        f"- Base config: `{path_relative_to_project(base_config_path)}`",
        f"- Scenario count: `{len(scenarios)}`",
        "",
        "## Aggregate Leaderboard",
        "",
        "| Method | Avg eval success | Avg rank | Wins | Success / runtime sec | Success / million shots |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for entry in aggregate["method_leaderboard"]:
        lines.append(
            "| "
            f"{entry['label']} | "
            f"{_format_float(entry.get('average_eval_success'))} | "
            f"{_format_float(entry.get('average_rank'))} | "
            f"{int(entry.get('scenario_wins', 0))} | "
            f"{_format_float(entry.get('average_eval_success_per_runtime_sec'))} | "
            f"{_format_float(entry.get('average_eval_success_per_million_shots'))} |"
        )

    lines.extend(["", "## Headline Answers", ""])
    for key, value in aggregate.get("headline_answers", {}).items():
        lines.append(f"- `{key}`: {value or 'NA'}")

    lines.extend(["", "## Quantum Noise Summary", ""])
    for family, metrics in aggregate.get("noise_summary", {}).items():
        lines.append(
            f"- `{family}`: average noise drop = "
            f"{_format_float(metrics.get('average_noise_drop_eval_success'))}, "
            f"average mitigation recovery = "
            f"{_format_float(metrics.get('average_mitigation_recovery_eval_success'))}"
        )

    lines.extend(
        [
            "",
            "## Scenario Summary",
            "",
            "| Scenario | Leader | QRE noise drop | QRE mitigation recovery | QA2C noise drop | QA2C mitigation recovery |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        leader_label = scenario.get("leader", {}).get("label", "NA")
        qre = scenario["noise_profile"]["quantum_reinforce"]
        qa2c = scenario["noise_profile"]["quantum_actor_critic"]
        lines.append(
            "| "
            f"{scenario['name']} | "
            f"{leader_label} | "
            f"{_format_float(qre.get('noise_drop_eval_success'))} | "
            f"{_format_float(qre.get('mitigation_recovery_eval_success'))} | "
            f"{_format_float(qa2c.get('noise_drop_eval_success'))} | "
            f"{_format_float(qa2c.get('mitigation_recovery_eval_success'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark_suite(
    *,
    suite_path: str | Path,
    output_root: str | Path | None = None,
    max_scenarios: int | None = None,
    log_level: str | None = None,
) -> dict[str, Any]:
    suite_path = resolve_project_path(suite_path)
    suite_payload = _load_suite_definition(suite_path)
    base_config_path = resolve_project_path(suite_payload["base_config"])
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("Base config YAML must load to a dictionary.")
    scenario_payloads = _resolve_scenarios(suite_payload)
    if max_scenarios is not None:
        scenario_payloads = scenario_payloads[: max(0, int(max_scenarios))]

    output_root_path = ensure_dir(
        resolve_project_path(output_root or suite_payload.get("output_root", "results/benchmark_suite"))
    )
    suite_log_level = log_level or str(suite_payload.get("log_level", "INFO"))

    scenario_results: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenario_payloads, start=1):
        scenario_name = str(scenario.get("name", f"scenario_{scenario_index:02d}"))
        scenario_description = str(scenario.get("description", "")).strip()
        scenario_config = _deep_merge(base_config, suite_payload.get("experiment_overrides", {}))
        scenario_config = _deep_merge(scenario_config, scenario.get("overrides", {}))
        _sync_episode_limits(scenario_config)
        scenario_output_dir = output_root_path / scenario_name
        scenario_config.setdefault("results", {})
        scenario_config["results"]["output_dir"] = str(scenario_output_dir)
        scenario_config["results"]["log_level"] = suite_log_level

        summary = run_training_pipeline(scenario_config)
        scenario_results.append(
            _scenario_result(
                scenario_index=scenario_index,
                scenario={"name": scenario_name, "description": scenario_description},
                output_dir=scenario_output_dir,
                summary=summary,
            )
        )

    aggregate = _aggregate_suite_results(scenario_results)
    report_model = BenchmarkSuiteResult(
        suite_name=str(suite_payload.get("suite_name", "QRL Benchmark Suite")),
        description=str(suite_payload.get("description", "")).strip(),
        suite_config_path=str(suite_path.resolve()),
        base_config_path=str(base_config_path.resolve()),
        output_root=str(output_root_path.resolve()),
        scenarios=scenario_results,
        aggregate=aggregate,
    )
    report = report_model.model_dump(mode="json")
    save_json(output_root_path / "benchmark_report.json", report)
    markdown_report = _render_markdown_report(
        suite_name=report["suite_name"],
        description=report["description"],
        base_config_path=base_config_path,
        suite_config_path=suite_path,
        aggregate=aggregate,
        scenarios=scenario_results,
    )
    (output_root_path / "benchmark_report.md").write_text(markdown_report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-scenario QRL benchmark suite")
    parser.add_argument(
        "--suite",
        type=str,
        default="config/benchmark_suite.yaml",
        help="Path to the benchmark suite YAML definition.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for the suite output directory.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on the number of scenarios to run, useful for smoke verification.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for the suite runner.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    report = run_benchmark_suite(
        suite_path=args.suite,
        output_root=args.output_root,
        max_scenarios=args.max_scenarios,
        log_level=args.log_level,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
