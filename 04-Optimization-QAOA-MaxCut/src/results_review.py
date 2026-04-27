"""Robustness analysis and scientific-result review helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import networkx as nx

from .artifact_schema import RobustnessRunRecord, RobustnessSummaryRecord
from .evaluation_metrics import EvaluationMetrics


def _record_to_dict(record: Any) -> Dict[str, Any]:
    """Normalize dataclass or dictionary records into dictionaries."""
    if isinstance(record, dict):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "to_dict"):
        return record.to_dict()
    raise TypeError(f"Unsupported record type: {type(record)!r}")


@dataclass
class BenchmarkRobustnessRunner:
    """Run repeated seeded optimizations on the configured benchmark graph."""

    graph: nx.Graph
    exact_value: float
    depths: Sequence[int]
    optimization_seeds: Sequence[int]
    create_problem: Callable[[int, int], Any]
    create_optimizer: Callable[[int, int], Any]
    confidence: float = 0.95
    n_bootstrap: int = 1000

    def __post_init__(self) -> None:
        self.metrics = EvaluationMetrics()

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run the repeated-seed robustness study."""
        run_rows: List[RobustnessRunRecord] = []
        summary_rows: List[RobustnessSummaryRecord] = []

        for depth in self.depths:
            depth_rows: List[Dict[str, Any]] = []
            for seed in self.optimization_seeds:
                problem = self.create_problem(depth, int(seed))
                optimizer = self.create_optimizer(depth, int(seed))
                result = optimizer.optimize(
                    objective_function=problem.objective_function,
                    n_qubits=self.graph.number_of_nodes(),
                    graph=self.graph,
                    solution_decoder=problem.decode_solution,
                    selection_objective_function=problem.objective_function,
                )
                ratio = self.metrics.compute_approximation_ratio(
                    float(result.cut_value),
                    float(self.exact_value),
                )
                sampled_cut = float(result.sampled_cut_value) if result.sampled_cut_value is not None else None
                representative_probability = (
                    float(result.bitstring_probability) if result.bitstring_probability is not None else None
                )
                diagnostics = " | ".join(result.diagnostics)
                row = RobustnessRunRecord(
                    depth=depth,
                    optimization_seed=int(seed),
                    expected_cut_value=float(result.cut_value),
                    sampled_cut_value=sampled_cut,
                    best_sampled_cut_value=float(result.best_sampled_cut_value)
                    if result.best_sampled_cut_value is not None
                    else None,
                    approximation_ratio=ratio,
                    runtime_sec=float(result.runtime),
                    representative_bitstring=result.solution_bitstring,
                    representative_probability=representative_probability,
                    objective_std=float(result.objective_std or 0.0),
                    objective_stderr=float(result.objective_stderr or 0.0),
                    hit_iteration_budget=int(
                        any("iteration budget" in diagnostic.lower() for diagnostic in result.diagnostics)
                    ),
                    plateau_warning=int(
                        any(
                            ("plateau" in diagnostic.lower()) or ("stalled" in diagnostic.lower())
                            for diagnostic in result.diagnostics
                        )
                    ),
                    diagnostics=diagnostics,
                )
                run_rows.append(row)
                depth_rows.append(row)

            summary_rows.append(self._summarize_depth(depth_rows))

        return {
            "runs": run_rows,
            "summary": summary_rows,
        }

    def _summarize_depth(self, rows: List[RobustnessRunRecord]) -> RobustnessSummaryRecord:
        """Aggregate repeated-seed benchmark runs for one depth."""
        ratios = [float(row.approximation_ratio) for row in rows]
        expected_values = [float(row.expected_cut_value) for row in rows]
        sampled_values = [
            float(row.sampled_cut_value)
            for row in rows
            if row.sampled_cut_value is not None
        ]
        probabilities = [
            float(row.representative_probability)
            for row in rows
            if row.representative_probability is not None
        ]
        summary = self.metrics.summarize_values(
            ratios,
            confidence=self.confidence,
            n_bootstrap=self.n_bootstrap,
            seed=int(rows[0].optimization_seed),
        )
        sampled_gap = [
            float(row.sampled_cut_value) - float(row.expected_cut_value)
            for row in rows
            if row.sampled_cut_value is not None
        ]
        exact_hits = [
            int(float(row.sampled_cut_value) >= float(self.exact_value))
            for row in rows
            if row.sampled_cut_value is not None
        ]

        return RobustnessSummaryRecord(
            depth=int(rows[0].depth),
            mean_ratio=summary["mean"],
            std_ratio=summary["std"],
            sem_ratio=summary["sem"],
            ci_lower=summary["ci_lower"],
            ci_upper=summary["ci_upper"],
            mean_expected_cut_value=float(sum(expected_values) / len(expected_values)),
            mean_sampled_cut_value=float(sum(sampled_values) / len(sampled_values))
            if sampled_values
            else None,
            mean_sample_gap=float(sum(sampled_gap) / len(sampled_gap)) if sampled_gap else 0.0,
            exact_sample_hit_rate=float(sum(exact_hits) / len(exact_hits)) if exact_hits else 0.0,
            mean_representative_probability=float(sum(probabilities) / len(probabilities))
            if probabilities
            else None,
            iteration_budget_hit_rate=float(sum(int(row.hit_iteration_budget) for row in rows) / len(rows)),
            plateau_warning_rate=float(sum(int(row.plateau_warning) for row in rows) / len(rows)),
            mean_runtime_sec=float(sum(float(row.runtime_sec) for row in rows) / len(rows)),
            n_runs=len(rows),
        )


class ScientificResultsReviewer:
    """Produce an explicit scientific verdict from benchmark and study artifacts."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.alpha = float(self.config.get("alpha", 0.05))
        self.randomness_std_threshold = float(self.config.get("randomness_ratio_std_threshold", 0.05))
        self.sample_gap_threshold = float(self.config.get("representative_gap_threshold", 0.15))

    def review(
        self,
        benchmark_rows: List[Dict[str, Any]],
        benchmark_robustness_summary: List[Dict[str, Any]],
        study_summary_rows: List[Dict[str, Any]],
        significance_rows: List[Dict[str, Any]],
        pairwise_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess result strength, randomness risk, and overclaim risk."""
        if not benchmark_rows or not study_summary_rows:
            return {
                "overall_label": "invalid",
                "misleading_risk": "high",
                "reasoning": [
                    "Missing benchmark or held-out study artifacts prevents a scientific verdict.",
                ],
            }

        benchmark_dicts = [_record_to_dict(row) for row in benchmark_rows]
        study_summary_dicts = [_record_to_dict(row) for row in study_summary_rows]
        significance_dicts = [_record_to_dict(row) for row in significance_rows]
        pairwise_dicts = [_record_to_dict(row) for row in pairwise_rows]
        benchmark_robustness_dicts = [_record_to_dict(row) for row in benchmark_robustness_summary]

        qaoa_benchmark_rows = [row for row in benchmark_dicts if str(row.get("method", "")).startswith("qaoa_p")]
        qaoa_study_rows = [row for row in study_summary_dicts if row.get("method") == "qaoa_tuned"]
        baseline_study_rows = [row for row in study_summary_dicts if row.get("method") != "qaoa_tuned"]

        classical_outperformance = []
        for qaoa_row in qaoa_study_rows:
            family = qaoa_row["family"]
            qaoa_ratio = float(qaoa_row["mean_ratio"])
            better_baselines = [
                row["method"]
                for row in baseline_study_rows
                if row["family"] == family and float(row["mean_ratio"]) > qaoa_ratio
            ]
            if better_baselines:
                classical_outperformance.append(
                    {
                        "family": family,
                        "qaoa_mean_ratio": qaoa_ratio,
                        "better_baselines": better_baselines,
                    }
                )

        positive_significance = [
            row
            for row in significance_dicts
            if float(row["mean_difference"]) > 0
            and float(row.get("p_value_holm", row["p_value"])) <= self.alpha
        ]
        negative_significance = [
            row
            for row in significance_dicts
            if float(row["mean_difference"]) < 0
            and float(row.get("p_value_holm", row["p_value"])) <= self.alpha
        ]

        randomness_flags = []
        for row in benchmark_robustness_dicts:
            if float(row["std_ratio"]) >= self.randomness_std_threshold:
                randomness_flags.append(
                    f"Depth {row['depth']} has ratio std {float(row['std_ratio']):.4f}, above the configured robustness threshold."
                )
            if float(row["mean_sample_gap"]) >= self.sample_gap_threshold:
                randomness_flags.append(
                    f"Depth {row['depth']} shows a sampled-vs-expected gap of {float(row['mean_sample_gap']):.4f}, which makes single sampled outcomes look better than the optimized objective."
                )

        pairwise_losses = [
            row
            for row in pairwise_dicts
            if float(row["loss_rate_a"]) > 0.5
        ]

        benchmark_follow_method = []
        for row in qaoa_benchmark_rows:
            expected = row.get("expected_cut_value")
            sampled = row.get("sampled_cut_value")
            best_sampled = row.get("best_sampled_cut_value")
            representative_probability = row.get("representative_probability")

            consistent = expected is not None
            if (
                consistent
                and sampled is not None
                and best_sampled is not None
                and float(best_sampled) + 1e-9 < float(sampled)
            ):
                consistent = False
            if representative_probability is not None:
                probability = float(representative_probability)
                if probability < -1e-9 or probability > 1.0 + 1e-9:
                    consistent = False
            benchmark_follow_method.append(consistent)

        reasons: List[str] = []
        if all(benchmark_follow_method):
            reasons.append(
                "The benchmark outputs are internally consistent with the method: expected objectives are tracked separately from representative sampled bitstrings, and sampled summaries obey their own bookkeeping constraints."
            )
        else:
            reasons.append(
                "At least one benchmark row violates the expected-versus-sampled bookkeeping used by the project."
            )

        if classical_outperformance:
            reasons.append(
                "Classical baselines outperform tuned QAOA on the held-out study families."
            )
        if negative_significance:
            reasons.append(
                "The statistically significant effects in the held-out study favor classical baselines, not QAOA."
            )
        if randomness_flags:
            reasons.append(
                "Repeated benchmark runs show materially unstable behavior across optimizer and backend seeds."
            )
        if any("sampled-vs-expected gap" in flag for flag in randomness_flags):
            reasons.append(
                "Sampled headline outputs can look materially better than the optimized expected objective."
            )
        if not positive_significance:
            reasons.append(
                "There is no corrected statistically significant evidence that QAOA improves over the included classical baselines."
            )

        misleading_risk = "high" if randomness_flags and not positive_significance else "medium"
        if positive_significance and not classical_outperformance and not randomness_flags:
            overall_label = "strong"
        elif not all(benchmark_follow_method):
            overall_label = "invalid"
        else:
            overall_label = "weak"

        return {
            "overall_label": overall_label,
            "misleading_risk": misleading_risk,
            "classical_outperformance": classical_outperformance,
            "positive_significance": positive_significance,
            "negative_significance": negative_significance,
            "pairwise_losses": pairwise_losses,
            "randomness_flags": randomness_flags,
            "reasoning": reasons,
        }

    @staticmethod
    def render_markdown(review: Dict[str, Any]) -> str:
        """Render the scientific verdict as a short Markdown summary."""
        lines = [
            "# Scientific Results Verdict",
            "",
            f"- Overall label: `{review['overall_label']}`",
            f"- Misleading-risk level: `{review['misleading_risk']}`",
            "",
            "## Why",
        ]
        for reason in review.get("reasoning", []):
            lines.append(f"- {reason}")

        if review.get("randomness_flags"):
            lines.extend(["", "## Randomness Flags"])
            for flag in review["randomness_flags"]:
                lines.append(f"- {flag}")

        if review.get("classical_outperformance"):
            lines.extend(["", "## Classical Outperformance"])
            for row in review["classical_outperformance"]:
                lines.append(
                    f"- {row['family']}: QAOA mean ratio {row['qaoa_mean_ratio']:.4f}, better baselines: {', '.join(row['better_baselines'])}"
                )

        if review.get("negative_significance"):
            lines.extend(["", "## Significant Negative Results"])
            for row in review["negative_significance"]:
                corrected_p = row.get("p_value_holm", row["p_value"])
                lines.append(
                    f"- {row['family']}: QAOA vs {row['method_b']} mean difference {float(row['mean_difference']):.4f}, corrected p={float(corrected_p):.4f}"
                )

        return "\n".join(lines) + "\n"
