"""Typed record schemas for benchmark and study artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence


class RecordMixin:
    """Common helper for CSV/JSON-serializable dataclass records."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def to_serializable_records(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert dataclass records or dictionaries into plain dictionaries."""
    serialized: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "to_dict"):
            serialized.append(record.to_dict())
        elif isinstance(record, dict):
            serialized.append(dict(record))
        else:
            raise TypeError(f"Unsupported record type: {type(record)!r}")
    return serialized


@dataclass
class BenchmarkMetricRecord(RecordMixin):
    method: str
    depth: int
    expected_cut_value: float
    sampled_cut_value: float | None
    best_sampled_cut_value: float | None
    approximation_ratio: float
    minimization_objective: float
    reevaluated_minimization_objective: float | None
    objective_std: float | None
    objective_stderr: float | None
    n_evaluations: int | None
    runtime_sec: float
    representative_bitstring: str | None
    representative_probability: float | None
    best_sampled_bitstring: str | None
    analysis_mode: str
    diagnostics: str
    most_likely_bitstring: str | None = None


@dataclass
class HardwareFeasibilityRecord(RecordMixin):
    method: str
    depth: int
    logical_qubits: int
    logical_depth: int
    logical_size: int
    logical_two_qubit_gates: int
    transpiled_depth: int
    transpiled_size: int
    transpiled_two_qubit_gates: int
    entangling_gate_multiplier: float | None
    estimated_total_shots: int
    backend_name: str
    status: str
    issues: str


@dataclass
class StudyCandidateRecord(RecordMixin):
    depth: int
    n_initial_points: int
    maxiter: int
    split: str
    mean_ratio: float
    std_ratio: float
    ci_lower: float
    ci_upper: float
    n_instances: int


@dataclass
class StudyInstanceRecord(RecordMixin):
    split: str
    family: str
    seed: int
    method: str
    approximation_ratio: float
    cut_value: float
    optimal_value: float
    runtime_sec: float
    depth: int
    n_nodes: int
    n_edges: int
    objective_std: float
    objective_stderr: float
    n_initial_points: int | None = None
    maxiter: int | None = None
    n_objective_evaluations: int | None = None
    budget_reference: str | None = None


@dataclass
class StudyMethodSummaryRecord(RecordMixin):
    family: str
    method: str
    mean_ratio: float
    std_ratio: float
    sem_ratio: float
    ci_lower: float
    ci_upper: float
    mean_cut_value: float
    mean_runtime_sec: float
    mean_n_objective_evaluations: float | None
    n_instances: int
    budget_reference: str | None = None


@dataclass
class SignificanceRecord(RecordMixin):
    family: str
    method_a: str
    method_b: str
    mean_difference: float
    median_difference: float
    std_difference: float
    cohen_d: float
    probability_a_better: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_pairs: int
    p_value_holm: float | None = None


@dataclass
class PairwiseSummaryRecord(RecordMixin):
    family: str
    method_a: str
    method_b: str
    wins_a: int
    ties: int
    losses_a: int
    win_rate_a: float
    loss_rate_a: float
    mean_difference: float
    n_pairs: int


@dataclass
class StudyPositioningRecord(RecordMixin):
    contribution_type: str
    algorithmic_novelty: str
    research_insight: str
    tutorial_assessment: str
    workshop_fit: str
    publishable_as: str
    main_missing_components: str


@dataclass
class RobustnessRunRecord(RecordMixin):
    depth: int
    optimization_seed: int
    expected_cut_value: float
    sampled_cut_value: float | None
    best_sampled_cut_value: float | None
    approximation_ratio: float
    runtime_sec: float
    representative_bitstring: str | None
    representative_probability: float | None
    objective_std: float
    objective_stderr: float
    hit_iteration_budget: int
    plateau_warning: int
    diagnostics: str


@dataclass
class RobustnessSummaryRecord(RecordMixin):
    depth: int
    mean_ratio: float
    std_ratio: float
    sem_ratio: float
    ci_lower: float
    ci_upper: float
    mean_expected_cut_value: float
    mean_sampled_cut_value: float | None
    mean_sample_gap: float
    exact_sample_hit_rate: float
    mean_representative_probability: float | None
    iteration_budget_hit_rate: float
    plateau_warning_rate: float
    mean_runtime_sec: float
    n_runs: int
