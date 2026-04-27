"""Experimental-study runner for reproducible QAOA benchmarking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .artifact_schema import (
    PairwiseSummaryRecord,
    SignificanceRecord,
    StudyCandidateRecord,
    StudyInstanceRecord,
    StudyMethodSummaryRecord,
    StudyPositioningRecord,
    to_serializable_records,
)
from .classical_solver import ApproximateSolver, ClassicalSolver
from .evaluation_metrics import EvaluationMetrics
from .graph_generator import GraphGenerator
from .qaoa_optimizer import MaxCutQAOAProblem, QAOAOptimizer
from .runtime_executor import RuntimeExecutor

logger = logging.getLogger(__name__)


@dataclass
class CandidateConfig:
    """QAOA hyperparameter candidate used during tuning."""

    depth: int
    n_initial_points: int
    maxiter: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "depth": self.depth,
            "n_initial_points": self.n_initial_points,
            "maxiter": self.maxiter,
        }


class ExperimentalStudyRunner:
    """Run a modest but scientifically defensible QAOA benchmark study."""

    def __init__(self, project_root: Path, config: Dict[str, Any]) -> None:
        self.project_root = project_root
        self.config = config
        self.study_cfg = config["study"]
        self.qaoa_cfg = config["qaoa"]
        self.metrics = EvaluationMetrics()
        self.graph_generator = GraphGenerator(seed=self.study_cfg.get("seed", 42))

    def run(self) -> Dict[str, Any]:
        """Run the full tune/evaluate study and persist artifacts."""
        tuning_instances = self._build_instances(self.study_cfg["tuning_seeds"])
        evaluation_instances = self._build_instances(self.study_cfg["evaluation_seeds"])
        candidates = self._candidate_configs()

        tuning_rows = []
        best_candidate = None
        best_ratio = float("-inf")
        for candidate in candidates:
            candidate_rows = []
            for instance in tuning_instances:
                candidate_rows.append(self._evaluate_qaoa_instance(instance, candidate))
            ratios = [row.approximation_ratio for row in candidate_rows]
            summary = self.metrics.summarize_values(
                ratios,
                confidence=self.study_cfg.get("confidence_level", 0.95),
                n_bootstrap=self.study_cfg.get("bootstrap_samples", 1000),
                seed=self.study_cfg.get("seed", 42),
            )
            tuning_rows.append(
                StudyCandidateRecord(
                    depth=candidate.depth,
                    n_initial_points=candidate.n_initial_points,
                    maxiter=candidate.maxiter,
                    split="tuning",
                    mean_ratio=summary["mean"],
                    std_ratio=summary["std"],
                    ci_lower=summary["ci_lower"],
                    ci_upper=summary["ci_upper"],
                    n_instances=summary["n"],
                )
            )
            if summary["mean"] > best_ratio:
                best_ratio = summary["mean"]
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("No QAOA candidate configuration was evaluated.")

        per_instance_rows: List[StudyInstanceRecord] = []
        for instance in evaluation_instances:
            optimal_row = self._evaluate_exact_baseline(instance)
            per_instance_rows.append(optimal_row)
            qaoa_row = self._evaluate_qaoa_instance(instance, best_candidate)
            per_instance_rows.append(qaoa_row)
            per_instance_rows.extend(
                self._evaluate_classical_baselines(
                    instance,
                    qaoa_evaluations=qaoa_row.n_objective_evaluations,
                )
            )

        summary_rows = self._summarize_methods(per_instance_rows)
        budget_summary_rows = self._summarize_budget_matched_methods(per_instance_rows)
        significance_rows = self._compute_significance(per_instance_rows)
        pairwise_rows = self._summarize_pairwise_outcomes(per_instance_rows)
        publication_position = self._build_publication_positioning(
            summary_rows=summary_rows,
            significance_rows=significance_rows,
            budget_summary_rows=budget_summary_rows,
        )
        manifest = {
            "selected_candidate": best_candidate.as_dict(),
            "candidate_search": to_serializable_records(tuning_rows),
            "tuning_seeds": list(self.study_cfg["tuning_seeds"]),
            "evaluation_seeds": list(self.study_cfg["evaluation_seeds"]),
            "graph_families": self.study_cfg["graph_families"],
            "study_backend_mode": self.study_cfg.get("executor_mode", "local"),
            "baseline_methods": self._study_baseline_methods(),
            "budget_matched_methods": self._study_budget_methods(),
        }

        results_dir = self.project_root / self.config["results"]["output_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        self._save_csv(tuning_rows, results_dir / "study_candidate_search.csv")
        self._save_csv(per_instance_rows, results_dir / "study_instance_metrics.csv")
        self._save_csv(summary_rows, results_dir / "study_method_summary.csv")
        self._save_csv(budget_summary_rows, results_dir / "study_budget_summary.csv")
        self._save_csv(significance_rows, results_dir / "study_significance.csv")
        self._save_csv(pairwise_rows, results_dir / "study_pairwise_summary.csv")
        (results_dir / "study_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (results_dir / "publication_positioning.json").write_text(
            json.dumps(publication_position, indent=2),
            encoding="utf-8",
        )
        (results_dir / "publication_positioning.md").write_text(
            self._render_publication_positioning_markdown(publication_position),
            encoding="utf-8",
        )

        return {
            "selected_candidate": best_candidate.as_dict(),
            "candidate_search": tuning_rows,
            "per_instance": per_instance_rows,
            "summary": summary_rows,
            "budget_summary": budget_summary_rows,
            "significance": significance_rows,
            "pairwise": pairwise_rows,
            "publication_positioning": publication_position,
        }

    def _candidate_configs(self) -> List[CandidateConfig]:
        """Build the candidate grid used for held-out tuning."""
        candidates = []
        for depth in self.study_cfg["candidate_depths"]:
            for n_initial_points in self.study_cfg["candidate_initial_points"]:
                candidates.append(
                    CandidateConfig(
                        depth=int(depth),
                        n_initial_points=int(n_initial_points),
                        maxiter=int(self.study_cfg["maxiter"]),
                    )
                )
        return candidates

    def _build_instances(self, seeds: List[int]) -> List[Dict[str, Any]]:
        """Generate all benchmark instances for the requested seeds and families."""
        instances = []
        for family_spec in self.study_cfg["graph_families"]:
            for seed in seeds:
                graph = self._build_graph(family_spec, seed)
                instances.append(
                    {
                        "family": family_spec["name"],
                        "seed": seed,
                        "graph": graph,
                        "exact_result": None,
                    }
                )
        return instances

    @staticmethod
    def _ensure_exact_result(instance: Dict[str, Any]) -> Any:
        """Cache the exact classical optimum for an instance."""
        if instance.get("exact_result") is None:
            solver = ClassicalSolver(seed=instance["seed"])
            instance["exact_result"] = solver.solve_exact(instance["graph"])
        return instance["exact_result"]

    def _build_graph(self, family_spec: Dict[str, Any], seed: int) -> nx.Graph:
        """Construct a benchmark graph from a family specification."""
        graph_type = family_spec["type"]
        if graph_type == "d_regular":
            return self.graph_generator.generate_d_regular_graph(
                n_nodes=family_spec["n_nodes"],
                degree=family_spec["degree"],
                seed=seed,
            )
        if graph_type == "erdos_renyi":
            return self.graph_generator.generate_erdos_renyi_graph(
                n_nodes=family_spec["n_nodes"],
                probability=family_spec["edge_probability"],
                seed=seed,
            )
        if graph_type == "barabasi_albert":
            return self.graph_generator.generate_barabasi_albert_graph(
                n_nodes=family_spec["n_nodes"],
                m=family_spec["degree"],
                seed=seed,
            )
        if graph_type == "communication_mesh":
            return self.graph_generator.generate_communication_mesh_graph(
                n_nodes=family_spec["n_nodes"],
                degree=family_spec["degree"],
                seed=seed,
                area_size=float(family_spec.get("area_size", 1.0)),
                reliability_scale=float(family_spec.get("reliability_scale", 0.35)),
            )
        raise ValueError(f"Unsupported study graph type: {graph_type}")

    def _evaluate_exact_baseline(self, instance: Dict[str, Any]) -> StudyInstanceRecord:
        """Evaluate the exact classical optimum for one instance."""
        result = self._ensure_exact_result(instance)
        return StudyInstanceRecord(
            split="evaluation",
            family=instance["family"],
            seed=instance["seed"],
            method="exact",
            approximation_ratio=1.0,
            cut_value=float(result.optimal_value),
            optimal_value=float(result.optimal_value),
            runtime_sec=float(result.runtime),
            depth=0,
            n_nodes=instance["graph"].number_of_nodes(),
            n_edges=instance["graph"].number_of_edges(),
            objective_std=0.0,
            objective_stderr=0.0,
            n_objective_evaluations=result.n_objective_evaluations,
            budget_reference="classical_exact",
        )

    def _evaluate_classical_baselines(
        self,
        instance: Dict[str, Any],
        qaoa_evaluations: Optional[int],
    ) -> List[StudyInstanceRecord]:
        """Evaluate the configured classical baselines for one instance."""
        graph = instance["graph"]
        seed = instance["seed"]
        exact_value = self._ensure_exact_result(instance).optimal_value
        solver = ClassicalSolver(seed=seed)

        local_search_start = None
        local_search_runtime = None
        local_search_value = None
        local_search_evals = None

        rows: List[StudyInstanceRecord] = []
        for method in self._study_baseline_methods():
            cut_value: float
            runtime_sec: float
            n_objective_evaluations: Optional[int] = None
            budget_reference: Optional[str] = None

            if method == "greedy":
                result = solver.solve_greedy(graph)
                cut_value = float(result.optimal_value)
                runtime_sec = float(result.runtime)
                n_objective_evaluations = result.n_objective_evaluations
                budget_reference = "deterministic_greedy"
            elif method == "random_cut":
                result = solver.solve_random(
                    graph,
                    n_trials=self.study_cfg.get("random_cut_trials", 64),
                )
                cut_value = float(result.optimal_value)
                runtime_sec = float(result.runtime)
                n_objective_evaluations = result.n_objective_evaluations
                budget_reference = "fixed_random_trials"
            elif method == "local_search":
                local_search_start = time.perf_counter()
                local_search_value, _ = ApproximateSolver.solve_local_search(
                    graph,
                    max_iterations=self.study_cfg.get("local_search_iterations", 200),
                    seed=seed,
                )
                local_search_runtime = time.perf_counter() - local_search_start
                cut_value = float(local_search_value)
                runtime_sec = float(local_search_runtime)
                n_objective_evaluations = local_search_evals
                budget_reference = "fixed_local_search_iterations"
            elif method == "goemans_williamson":
                try:
                    result = solver.solve_goemans_williamson(
                        graph,
                        num_trials=self.study_cfg.get("gw_num_trials", 64),
                    )
                except ImportError:
                    logger.warning(
                        "Skipping Goemans-Williamson baseline for %s seed %s because cvxpy is unavailable.",
                        instance["family"],
                        seed,
                    )
                    continue
                cut_value = float(result.optimal_value)
                runtime_sec = float(result.runtime)
                budget_reference = "sdp_rounding"
            elif method == "random_budget_matched":
                if not qaoa_evaluations:
                    continue
                result = solver.solve_random(graph, n_trials=int(qaoa_evaluations))
                cut_value = float(result.optimal_value)
                runtime_sec = float(result.runtime)
                n_objective_evaluations = int(qaoa_evaluations)
                budget_reference = "qaoa_objective_evaluations"
            elif method == "hill_climb_budget_matched":
                if not qaoa_evaluations:
                    continue
                result = solver.solve_budgeted_hill_climb(
                    graph,
                    evaluation_budget=int(qaoa_evaluations),
                    n_restarts=int(self.study_cfg.get("budgeted_hill_climb_restarts", 4)),
                )
                cut_value = float(result.optimal_value)
                runtime_sec = float(result.runtime)
                n_objective_evaluations = result.n_objective_evaluations
                budget_reference = "qaoa_objective_evaluations"
            else:
                raise ValueError(f"Unsupported study baseline method: {method}")

            rows.append(
                StudyInstanceRecord(
                    split="evaluation",
                    family=instance["family"],
                    seed=seed,
                    method=method,
                    approximation_ratio=self.metrics.compute_approximation_ratio(
                        float(cut_value),
                        float(exact_value),
                    ),
                    cut_value=float(cut_value),
                    optimal_value=float(exact_value),
                    runtime_sec=float(runtime_sec),
                    depth=0,
                    n_nodes=graph.number_of_nodes(),
                    n_edges=graph.number_of_edges(),
                    objective_std=0.0,
                    objective_stderr=0.0,
                    n_objective_evaluations=n_objective_evaluations,
                    budget_reference=budget_reference,
                )
            )
        return rows

    def _evaluate_qaoa_instance(
        self,
        instance: Dict[str, Any],
        candidate: CandidateConfig,
    ) -> StudyInstanceRecord:
        """Run one QAOA configuration on one instance."""
        graph = instance["graph"]
        seed = instance["seed"]
        exact_value = self._ensure_exact_result(instance).optimal_value

        executor = RuntimeExecutor(
            mode=self.study_cfg.get("executor_mode", "local"),
            backend_name=self.study_cfg.get("backend_name"),
            shots=self.study_cfg.get("shots", 0),
            seed=seed,
            simulate_noise=self.study_cfg.get("simulate_noise", False),
        )
        problem = MaxCutQAOAProblem(
            graph=graph,
            p=candidate.depth,
            executor=executor,
            seed=seed,
            analysis_shots=self.study_cfg.get("analysis_shots", 0),
            analysis_mode=self.study_cfg.get("analysis_mode", "none"),
            objective_repetitions=self.study_cfg.get("objective_repetitions", 1),
            report_repetitions=self.study_cfg.get("report_repetitions", 1),
            objective_mode=self.study_cfg.get("objective_mode", "expected"),
            cvar_alpha=float(self.study_cfg.get("cvar_alpha", 1.0)),
        )
        optimizer = QAOAOptimizer(
            p=candidate.depth,
            optimizer_type=self.study_cfg.get("optimizer_type", "COBYLA"),
            maxiter=candidate.maxiter,
            tol=float(self.study_cfg.get("tolerance", 1e-4)),
            seed=seed,
            n_initial_points=candidate.n_initial_points,
        )
        result = optimizer.optimize(
            objective_function=problem.objective_function,
            n_qubits=graph.number_of_nodes(),
            graph=graph,
            solution_decoder=problem.decode_solution,
            selection_objective_function=problem.objective_function,
        )
        return StudyInstanceRecord(
            split="tuning" if instance["seed"] in self.study_cfg["tuning_seeds"] else "evaluation",
            family=instance["family"],
            seed=seed,
            method="qaoa_tuned",
            approximation_ratio=self.metrics.compute_approximation_ratio(
                float(result.cut_value),
                float(exact_value),
            ),
            cut_value=float(result.cut_value),
            optimal_value=float(exact_value),
            runtime_sec=float(result.runtime),
            depth=candidate.depth,
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            objective_std=float(result.objective_std or 0.0),
            objective_stderr=float(result.objective_stderr or 0.0),
            n_initial_points=candidate.n_initial_points,
            maxiter=candidate.maxiter,
            n_objective_evaluations=int(result.n_evaluations),
            budget_reference="qaoa_objective_evaluations",
        )

    def _summarize_methods(self, rows: List[StudyInstanceRecord]) -> List[StudyMethodSummaryRecord]:
        """Aggregate evaluation rows into family/method summaries."""
        evaluation_rows = [row for row in rows if row.split == "evaluation" and row.method != "exact"]
        grouped: Dict[Tuple[str, str], List[StudyInstanceRecord]] = {}
        for row in evaluation_rows:
            grouped.setdefault((row.family, row.method), []).append(row)

        summary_rows = []
        for (family, method), group_rows in grouped.items():
            ratios = [row.approximation_ratio for row in group_rows]
            runtimes = [row.runtime_sec for row in group_rows]
            cut_values = [row.cut_value for row in group_rows]
            objective_evaluations = [
                int(row.n_objective_evaluations)
                for row in group_rows
                if row.n_objective_evaluations is not None
            ]
            budget_references = {
                row.budget_reference
                for row in group_rows
                if row.budget_reference is not None
            }
            summary = self.metrics.summarize_values(
                ratios,
                confidence=self.study_cfg.get("confidence_level", 0.95),
                n_bootstrap=self.study_cfg.get("bootstrap_samples", 1000),
                seed=self.study_cfg.get("seed", 42),
            )
            summary_rows.append(
                StudyMethodSummaryRecord(
                    family=family,
                    method=method,
                    mean_ratio=summary["mean"],
                    std_ratio=summary["std"],
                    sem_ratio=summary["sem"],
                    ci_lower=summary["ci_lower"],
                    ci_upper=summary["ci_upper"],
                    mean_cut_value=float(sum(cut_values) / len(cut_values)),
                    mean_runtime_sec=float(sum(runtimes) / len(runtimes)),
                    mean_n_objective_evaluations=float(sum(objective_evaluations) / len(objective_evaluations))
                    if objective_evaluations
                    else None,
                    n_instances=summary["n"],
                    budget_reference=budget_references.pop() if len(budget_references) == 1 else None,
                )
            )
        return sorted(summary_rows, key=lambda row: (row.family, row.method))

    def _summarize_budget_matched_methods(
        self,
        rows: List[StudyInstanceRecord],
    ) -> List[StudyMethodSummaryRecord]:
        """Summarize only QAOA and budget-matched black-box baselines."""
        budget_methods = {"qaoa_tuned", *self._study_budget_methods()}
        filtered = [row for row in rows if row.method in budget_methods]
        return self._summarize_methods(filtered)

    def _compute_significance(self, rows: List[StudyInstanceRecord]) -> List[SignificanceRecord]:
        """Compute paired significance tests for QAOA against classical baselines."""
        evaluation_rows = [row for row in rows if row.split == "evaluation"]
        by_family_seed: Dict[Tuple[str, int], Dict[str, float]] = {}
        for row in evaluation_rows:
            key = (row.family, row.seed)
            by_family_seed.setdefault(key, {})[row.method] = row.approximation_ratio

        significance_rows = []
        baseline_methods = sorted({row.method for row in evaluation_rows if row.method != "qaoa_tuned" and row.method != "exact"})
        for family in sorted({row.family for row in evaluation_rows}):
            family_pairs = {
                seed: methods
                for (row_family, seed), methods in by_family_seed.items()
                if row_family == family and "qaoa_tuned" in methods
            }
            for baseline in baseline_methods:
                qaoa_values = []
                baseline_values = []
                for _, methods in sorted(family_pairs.items()):
                    if baseline not in methods:
                        continue
                    qaoa_values.append(methods["qaoa_tuned"])
                    baseline_values.append(methods[baseline])
                if not qaoa_values:
                    continue
                test = self.metrics.paired_method_test(
                    qaoa_values,
                    baseline_values,
                    n_resamples=self.study_cfg.get("permutation_samples", 2000),
                    confidence=self.study_cfg.get("confidence_level", 0.95),
                    seed=self.study_cfg.get("seed", 42),
                )
                significance_rows.append(
                    SignificanceRecord(
                        family=family,
                        method_a="qaoa_tuned",
                        method_b=baseline,
                        mean_difference=float(test["mean_difference"]),
                        median_difference=float(test["median_difference"]),
                        std_difference=float(test["std_difference"]),
                        cohen_d=float(test["cohen_d"]),
                        probability_a_better=float(test["probability_a_better"]),
                        ci_lower=float(test["ci_lower"]),
                        ci_upper=float(test["ci_upper"]),
                        p_value=float(test["p_value"]),
                        n_pairs=int(test["n_pairs"]),
                    )
                )
        if significance_rows:
            adjusted = self.metrics.holm_bonferroni_correction(
                [float(row.p_value) for row in significance_rows]
            )
            for row, adjusted_p in zip(significance_rows, adjusted):
                row.p_value_holm = adjusted_p
        return significance_rows

    def _summarize_pairwise_outcomes(self, rows: List[StudyInstanceRecord]) -> List[PairwiseSummaryRecord]:
        """Summarize per-instance wins and losses for QAOA against each baseline."""
        evaluation_rows = [row for row in rows if row.split == "evaluation"]
        by_family_seed: Dict[Tuple[str, int], Dict[str, float]] = {}
        for row in evaluation_rows:
            key = (row.family, row.seed)
            by_family_seed.setdefault(key, {})[row.method] = row.approximation_ratio

        summary_rows = []
        baseline_methods = sorted({row.method for row in evaluation_rows if row.method not in {"qaoa_tuned", "exact"}})
        tolerance = float(self.study_cfg.get("pairwise_tie_tolerance", 1e-9))
        for family in sorted({row.family for row in evaluation_rows}):
            family_pairs = {
                seed: methods
                for (row_family, seed), methods in by_family_seed.items()
                if row_family == family and "qaoa_tuned" in methods
            }
            for baseline in baseline_methods:
                wins = 0
                ties = 0
                losses = 0
                diffs = []
                for _, methods in sorted(family_pairs.items()):
                    if baseline not in methods:
                        continue
                    diff = float(methods["qaoa_tuned"] - methods[baseline])
                    diffs.append(diff)
                    if diff > tolerance:
                        wins += 1
                    elif diff < -tolerance:
                        losses += 1
                    else:
                        ties += 1
                total = wins + ties + losses
                if total == 0:
                    continue
                summary_rows.append(
                    PairwiseSummaryRecord(
                        family=family,
                        method_a="qaoa_tuned",
                        method_b=baseline,
                        wins_a=wins,
                        ties=ties,
                        losses_a=losses,
                        win_rate_a=float(wins / total),
                        loss_rate_a=float(losses / total),
                        mean_difference=float(sum(diffs) / len(diffs)),
                        n_pairs=total,
                    )
                )
        return summary_rows

    def _build_publication_positioning(
        self,
        summary_rows: List[StudyMethodSummaryRecord],
        significance_rows: List[SignificanceRecord],
        budget_summary_rows: List[StudyMethodSummaryRecord],
    ) -> Dict[str, Any]:
        """Frame the current project honestly as a research artifact."""
        qaoa_rows = [row for row in summary_rows if row.method == "qaoa_tuned"]
        baseline_rows = [row for row in summary_rows if row.method not in {"qaoa_tuned", "exact"}]
        stronger_baselines = []
        for qaoa_row in qaoa_rows:
            better = [
                row.method
                for row in baseline_rows
                if row.family == qaoa_row.family and float(row.mean_ratio) > float(qaoa_row.mean_ratio)
            ]
            if better:
                stronger_baselines.append({"family": qaoa_row.family, "methods": better})

        budget_rows = [row for row in budget_summary_rows if row.method != "qaoa_tuned"]
        budget_failures = [
            row.method
            for row in budget_rows
            if any(
                qaoa_row.family == row.family and float(row.mean_ratio) >= float(qaoa_row.mean_ratio)
                for qaoa_row in qaoa_rows
            )
        ]
        negative_significance = [
            row for row in significance_rows if float(row.mean_difference) < 0 and float(row.p_value_holm or row.p_value) <= 0.05
        ]

        missing_components = [
            "A genuine algorithmic novelty claim beyond standard QAOA/RQAOA benchmarking.",
            "Larger and harder instance families, including weighted graphs and medium-scale regimes.",
            "Live hardware evidence or broader noisy studies beyond the configured proxy benchmark.",
            "Budget-matched comparisons against even stronger classical methods across larger scales.",
            "Either theoretical analysis or decisive empirical advantages, rather than only negative or neutral results.",
        ]

        contribution_type = "benchmark_negative_results_artifact"
        algorithmic_novelty = "none"
        research_insight = (
            "The repo now contributes a reproducible and honest QAOA Max-Cut benchmark showing "
            "that sampled outputs can overstate expected performance and that classical baselines remain stronger on the current study."
        )
        tutorial_assessment = "No longer just a tutorial, but still not an algorithmic research contribution."
        workshop_fit = "Plausible as a benchmark, software-artifact, or negative-results workshop paper."
        publishable_as = "benchmark_artifact_or_negative_results_workshop_paper"
        if not stronger_baselines and not negative_significance:
            publishable_as = "exploratory_workshop_benchmark"

        record = StudyPositioningRecord(
            contribution_type=contribution_type,
            algorithmic_novelty=algorithmic_novelty,
            research_insight=research_insight,
            tutorial_assessment=tutorial_assessment,
            workshop_fit=workshop_fit,
            publishable_as=publishable_as,
            main_missing_components=" | ".join(missing_components),
        )
        return {
            **record.to_dict(),
            "families_where_classical_is_stronger": stronger_baselines,
            "budget_matched_failures": sorted(set(budget_failures)),
            "negative_significance_count": len(negative_significance),
        }

    @staticmethod
    def _render_publication_positioning_markdown(positioning: Dict[str, Any]) -> str:
        """Render a concise publication-positioning note for the study."""
        lines = [
            "# Publication Positioning",
            "",
            f"- Contribution type: `{positioning['contribution_type']}`",
            f"- Algorithmic novelty: `{positioning['algorithmic_novelty']}`",
            f"- Publishable as: `{positioning['publishable_as']}`",
            f"- Workshop fit: {positioning['workshop_fit']}",
            "",
            "## Assessment",
            f"- {positioning['tutorial_assessment']}",
            f"- {positioning['research_insight']}",
            "",
            "## Missing For Stronger Publication",
        ]
        for item in str(positioning["main_missing_components"]).split(" | "):
            lines.append(f"- {item}")

        if positioning.get("families_where_classical_is_stronger"):
            lines.extend(["", "## Where Classical Methods Still Win"])
            for row in positioning["families_where_classical_is_stronger"]:
                lines.append(f"- {row['family']}: {', '.join(row['methods'])}")

        if positioning.get("budget_matched_failures"):
            lines.extend(["", "## Budget-Matched Concerns"])
            lines.append(
                f"- Budget-matched baselines matching QAOA objective-evaluation counts still remain competitive or stronger: {', '.join(positioning['budget_matched_failures'])}"
            )

        return "\n".join(lines) + "\n"

    def _study_baseline_methods(self) -> List[str]:
        """Return the configured list of baseline methods."""
        return list(
            self.study_cfg.get(
                "baseline_methods",
                [
                    "greedy",
                    "local_search",
                    "random_cut",
                    "goemans_williamson",
                    "random_budget_matched",
                    "hill_climb_budget_matched",
                ],
            )
        )

    def _study_budget_methods(self) -> List[str]:
        """Return only the explicitly budget-matched baseline names."""
        return [
            method
            for method in self._study_baseline_methods()
            if method.endswith("_budget_matched")
        ]

    @staticmethod
    def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
        """Save rows to CSV."""
        from .visualization import save_metrics_csv

        save_metrics_csv(to_serializable_records(rows), str(path))
