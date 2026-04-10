"""Experimental-study runner for reproducible QAOA benchmarking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

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
            ratios = [row["approximation_ratio"] for row in candidate_rows]
            summary = self.metrics.summarize_values(
                ratios,
                confidence=self.study_cfg.get("confidence_level", 0.95),
                n_bootstrap=self.study_cfg.get("bootstrap_samples", 1000),
                seed=self.study_cfg.get("seed", 42),
            )
            tuning_rows.append(
                {
                    **candidate.as_dict(),
                    "split": "tuning",
                    "mean_ratio": summary["mean"],
                    "std_ratio": summary["std"],
                    "ci_lower": summary["ci_lower"],
                    "ci_upper": summary["ci_upper"],
                    "n_instances": summary["n"],
                }
            )
            if summary["mean"] > best_ratio:
                best_ratio = summary["mean"]
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("No QAOA candidate configuration was evaluated.")

        per_instance_rows: List[Dict[str, Any]] = []
        for instance in evaluation_instances:
            optimal_row = self._evaluate_exact_baseline(instance)
            per_instance_rows.append(optimal_row)
            per_instance_rows.extend(self._evaluate_classical_baselines(instance))
            per_instance_rows.append(self._evaluate_qaoa_instance(instance, best_candidate))

        summary_rows = self._summarize_methods(per_instance_rows)
        significance_rows = self._compute_significance(per_instance_rows)
        manifest = {
            "selected_candidate": best_candidate.as_dict(),
            "candidate_search": tuning_rows,
            "tuning_seeds": list(self.study_cfg["tuning_seeds"]),
            "evaluation_seeds": list(self.study_cfg["evaluation_seeds"]),
            "graph_families": self.study_cfg["graph_families"],
            "study_backend_mode": self.study_cfg.get("executor_mode", "local"),
        }

        results_dir = self.project_root / self.config["results"]["output_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        self._save_csv(tuning_rows, results_dir / "study_candidate_search.csv")
        self._save_csv(per_instance_rows, results_dir / "study_instance_metrics.csv")
        self._save_csv(summary_rows, results_dir / "study_method_summary.csv")
        self._save_csv(significance_rows, results_dir / "study_significance.csv")
        (results_dir / "study_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return {
            "selected_candidate": best_candidate.as_dict(),
            "candidate_search": tuning_rows,
            "per_instance": per_instance_rows,
            "summary": summary_rows,
            "significance": significance_rows,
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
        raise ValueError(f"Unsupported study graph type: {graph_type}")

    def _evaluate_exact_baseline(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the exact classical optimum for one instance."""
        result = self._ensure_exact_result(instance)
        return {
            "split": "evaluation",
            "family": instance["family"],
            "seed": instance["seed"],
            "method": "exact",
            "approximation_ratio": 1.0,
            "cut_value": float(result.optimal_value),
            "optimal_value": float(result.optimal_value),
            "runtime_sec": float(result.runtime),
            "depth": 0,
            "n_nodes": instance["graph"].number_of_nodes(),
            "n_edges": instance["graph"].number_of_edges(),
            "objective_std": 0.0,
            "objective_stderr": 0.0,
        }

    def _evaluate_classical_baselines(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate simple classical heuristic baselines."""
        graph = instance["graph"]
        seed = instance["seed"]
        exact_value = self._ensure_exact_result(instance).optimal_value
        solver = ClassicalSolver(seed=seed)

        greedy = solver.solve_greedy(graph)
        random_result = solver.solve_random(
            graph,
            n_trials=self.study_cfg.get("random_cut_trials", 64),
        )
        local_search_start = time.perf_counter()
        local_search_value, local_search_bits = ApproximateSolver.solve_local_search(
            graph,
            max_iterations=self.study_cfg.get("local_search_iterations", 200),
            seed=seed,
        )
        local_search_runtime = time.perf_counter() - local_search_start

        baselines = [
            ("greedy", greedy.optimal_value, greedy.runtime),
            ("random_cut", random_result.optimal_value, random_result.runtime),
            ("local_search", float(local_search_value), local_search_runtime),
        ]

        rows = []
        for method, cut_value, runtime in baselines:
            rows.append(
                {
                    "split": "evaluation",
                    "family": instance["family"],
                    "seed": seed,
                    "method": method,
                    "approximation_ratio": self.metrics.compute_approximation_ratio(
                        float(cut_value),
                        float(exact_value),
                    ),
                    "cut_value": float(cut_value),
                    "optimal_value": float(exact_value),
                    "runtime_sec": float(runtime),
                    "depth": 0,
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "objective_std": 0.0,
                    "objective_stderr": 0.0,
                }
            )
        return rows

    def _evaluate_qaoa_instance(
        self,
        instance: Dict[str, Any],
        candidate: CandidateConfig,
    ) -> Dict[str, Any]:
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
        return {
            "split": "tuning" if instance["seed"] in self.study_cfg["tuning_seeds"] else "evaluation",
            "family": instance["family"],
            "seed": seed,
            "method": "qaoa_tuned",
            "approximation_ratio": self.metrics.compute_approximation_ratio(
                float(result.cut_value),
                float(exact_value),
            ),
            "cut_value": float(result.cut_value),
            "optimal_value": float(exact_value),
            "runtime_sec": float(result.runtime),
            "depth": candidate.depth,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "objective_std": float(result.objective_std or 0.0),
            "objective_stderr": float(result.objective_stderr or 0.0),
            "n_initial_points": candidate.n_initial_points,
            "maxiter": candidate.maxiter,
        }

    def _summarize_methods(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate evaluation rows into family/method summaries."""
        evaluation_rows = [row for row in rows if row["split"] == "evaluation" and row["method"] != "exact"]
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in evaluation_rows:
            grouped.setdefault((row["family"], row["method"]), []).append(row)

        summary_rows = []
        for (family, method), group_rows in grouped.items():
            ratios = [row["approximation_ratio"] for row in group_rows]
            runtimes = [row["runtime_sec"] for row in group_rows]
            summary = self.metrics.summarize_values(
                ratios,
                confidence=self.study_cfg.get("confidence_level", 0.95),
                n_bootstrap=self.study_cfg.get("bootstrap_samples", 1000),
                seed=self.study_cfg.get("seed", 42),
            )
            summary_rows.append(
                {
                    "family": family,
                    "method": method,
                    "mean_ratio": summary["mean"],
                    "std_ratio": summary["std"],
                    "sem_ratio": summary["sem"],
                    "ci_lower": summary["ci_lower"],
                    "ci_upper": summary["ci_upper"],
                    "mean_runtime_sec": float(sum(runtimes) / len(runtimes)),
                    "n_instances": summary["n"],
                }
            )
        return sorted(summary_rows, key=lambda row: (row["family"], row["method"]))

    def _compute_significance(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute paired significance tests for QAOA against classical baselines."""
        evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
        by_family_seed: Dict[Tuple[str, int], Dict[str, float]] = {}
        for row in evaluation_rows:
            key = (row["family"], row["seed"])
            by_family_seed.setdefault(key, {})[row["method"]] = row["approximation_ratio"]

        significance_rows = []
        baseline_methods = ["greedy", "local_search", "random_cut"]
        for family in sorted({row["family"] for row in evaluation_rows}):
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
                    {
                        "family": family,
                        "method_a": "qaoa_tuned",
                        "method_b": baseline,
                        **test,
                    }
                )
        return significance_rows

    @staticmethod
    def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
        """Save rows to CSV."""
        from .visualization import save_metrics_csv

        save_metrics_csv(rows, str(path))
