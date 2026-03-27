"""Versioned schemas for benchmark, run, and report artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_SCHEMA_VERSION = "0.3.0"


class ScenarioSpec(BaseModel):
    """Scenario-level benchmark definition."""

    name: str
    description: str = ""
    seeds: list[int] = Field(default_factory=list)
    n_eval_episodes: int = 0
    environment: dict[str, Any] = Field(default_factory=dict)
    quantum_policy: dict[str, Any] = Field(default_factory=dict)


class MethodSpec(BaseModel):
    """Stable public descriptor for a train/eval method."""

    name: str
    family: Literal["quantum", "classical", "random"]
    training_algorithm: str
    execution_mode: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Evaluation summary for a frozen checkpoint or policy."""

    avg_reward: float
    std_reward: float
    success_rate: float
    avg_length: float


class ResourceMetrics(BaseModel):
    """Compute and shot-cost metrics for one method/mode."""

    avg_episode_length_estimate: float | None = None
    parameter_bindings_per_timestep: float | None = None
    training_parameter_bindings_per_seed: float | None = None
    evaluation_parameter_bindings_per_seed: float | None = None
    total_parameter_bindings_per_seed: float | None = None
    estimated_circuit_executions_per_logical_query: float | None = None
    estimated_total_shots_per_seed: float | None = None
    eval_success_per_runtime_sec: float | None = None
    reward_auc_per_runtime_sec: float | None = None
    eval_success_per_million_shots: float | None = None


class RunResult(BaseModel):
    """One method x scenario x seed run artifact."""

    schema_version: str = DEFAULT_SCHEMA_VERSION
    result_type: Literal["run_result"] = "run_result"
    scenario_spec: ScenarioSpec
    method_spec: MethodSpec
    seed: int
    output_dir: str
    training_log_path: str
    weights_path: str | None = None
    evaluation: EvalResult
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    resource_metrics: ResourceMetrics | None = None


class BenchmarkSuiteResult(BaseModel):
    """Top-level schema for a benchmark-suite aggregate."""

    schema_version: str = DEFAULT_SCHEMA_VERSION
    result_type: Literal["benchmark_suite_result"] = "benchmark_suite_result"
    suite_name: str
    description: str = ""
    suite_config_path: str
    base_config_path: str
    output_root: str
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)


class PaperReportBundle(BaseModel):
    """Schema for generated report bundles."""

    schema_version: str = DEFAULT_SCHEMA_VERSION
    result_type: Literal["paper_report_bundle"] = "paper_report_bundle"
    results_root: str
    benchmark_report_path: str
    markdown_report_path: str
    figure_paths: list[str] = Field(default_factory=list)
    table_paths: list[str] = Field(default_factory=list)
    appendix_paths: list[str] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)
