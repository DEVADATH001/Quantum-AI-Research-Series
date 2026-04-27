"""Thin orchestration layer for benchmark and artifact generation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .artifact_manager import ArtifactManager
from .artifact_schema import BenchmarkMetricRecord, HardwareFeasibilityRecord, to_serializable_records
from .classical_solver import ClassicalSolver
from .evaluation_metrics import EvaluationMetrics
from .experimental_study import ExperimentalStudyRunner
from .graph_generator import GraphGenerator
from .hardware_analysis import HardwareFeasibilityAnalyzer, HardwareFeasibilityThresholds
from .provenance import collect_run_provenance
from .qaoa_circuit import QAOACircuitBuilder
from .qaoa_optimizer import MaxCutQAOAProblem, ParameterGridEvaluator, QAOAOptimizer
from .results_review import BenchmarkRobustnessRunner, ScientificResultsReviewer
from .rqaoa_engine import RQAOAEngine
from .runtime_executor import RuntimeExecutor
from .visualization import Visualizer


class ArtifactPipeline:
    """Generate benchmark, study, robustness, and review artifacts."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.config = self.load_config(project_root)
        self.visual_cfg = self.config["visualization"]
        self.qaoa_cfg = self.config["qaoa"]
        self.optimizer_cfg = self.config["optimizer"]
        self.artifacts = ArtifactManager(
            project_root=project_root,
            output_dir=self.config["results"]["output_dir"],
        )

    @staticmethod
    def load_config(project_root: Path) -> Dict:
        """Load the experiment configuration."""
        config_path = project_root / "config" / "experiment_config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def build_graph(self) -> Tuple[GraphGenerator, object]:
        """Build the configured graph instance."""
        graph_cfg = self.config["graph"]
        generator = GraphGenerator(seed=graph_cfg.get("seed", 42))
        graph_type = graph_cfg.get("type", "d_regular")

        if graph_type == "d_regular":
            graph = generator.generate_d_regular_graph(
                n_nodes=graph_cfg["n_nodes"],
                degree=graph_cfg["degree"],
                seed=graph_cfg.get("seed"),
            )
        elif graph_type == "erdos_renyi":
            graph = generator.generate_erdos_renyi_graph(
                n_nodes=graph_cfg["n_nodes"],
                probability=graph_cfg["edge_probability"],
                seed=graph_cfg.get("seed"),
            )
        elif graph_type == "barabasi_albert":
            graph = generator.generate_barabasi_albert_graph(
                n_nodes=graph_cfg["n_nodes"],
                m=graph_cfg["degree"],
                seed=graph_cfg.get("seed"),
            )
        elif graph_type == "communication_mesh":
            graph = generator.generate_communication_mesh_graph(
                n_nodes=graph_cfg["n_nodes"],
                degree=graph_cfg["degree"],
                seed=graph_cfg.get("seed"),
                area_size=float(graph_cfg.get("area_size", 1.0)),
                reliability_scale=float(graph_cfg.get("reliability_scale", 0.35)),
            )
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        return generator, graph

    def create_executor(self, seed: int) -> RuntimeExecutor:
        """Create the configured quantum execution backend."""
        quantum_cfg = self.config["quantum"]
        backend_name = quantum_cfg.get("backend_name")
        if quantum_cfg["mode"] == "local":
            backend_name = None
        return RuntimeExecutor(
            mode=quantum_cfg["mode"],
            backend_name=backend_name,
            shots=quantum_cfg.get("shots", 1024),
            resilience_level=quantum_cfg.get("resilience_level", 1),
            optimization_level=quantum_cfg.get("optimization_level", 1),
            seed=seed,
            simulate_noise=quantum_cfg.get("simulate_noise", True),
            noise_model_path=quantum_cfg.get("noise_model_path"),
        )

    def create_problem(self, graph, depth: int, seed: int) -> MaxCutQAOAProblem:
        """Create a backend-aware Max-Cut QAOA workflow for a given depth."""
        quantum_cfg = self.config["quantum"]
        return MaxCutQAOAProblem(
            graph=graph,
            p=depth,
            executor=self.create_executor(seed=seed),
            seed=seed,
            analysis_shots=quantum_cfg.get("shots", 1024),
            objective_repetitions=quantum_cfg.get("objective_repetitions", 1),
            report_repetitions=quantum_cfg.get("report_repetitions"),
            analysis_mode=quantum_cfg.get("analysis_mode", "auto"),
            objective_mode=quantum_cfg.get("objective_mode", "expected"),
            cvar_alpha=float(quantum_cfg.get("cvar_alpha", 1.0)),
        )

    def create_optimizer(self, depth: int, seed_override: int | None = None) -> QAOAOptimizer:
        """Create the configured classical optimizer for a QAOA depth."""
        return QAOAOptimizer(
            p=depth,
            optimizer_type=self.optimizer_cfg["type"],
            maxiter=self.optimizer_cfg["maxiter"],
            tol=float(self.qaoa_cfg["tolerance"]),
            seed=seed_override if seed_override is not None else self.optimizer_cfg.get("seed", 42),
            n_initial_points=self.qaoa_cfg.get("n_initial_points", 3),
            spsa_learning_rate=self.optimizer_cfg.get("spsa_step_size", 0.1),
            spsa_perturbation=self.optimizer_cfg.get("spsa_perturbation", 0.1),
            selection_repetitions=self.optimizer_cfg.get("selection_repetitions", 1),
            plateau_window=self.optimizer_cfg.get("plateau_window", 20),
            plateau_tolerance=float(self.optimizer_cfg.get("plateau_tolerance", 1e-4)),
        )

    def create_hardware_analyzer(self, seed: int) -> HardwareFeasibilityAnalyzer | None:
        """Create a target-backend analyzer for NISQ feasibility estimates."""
        hardware_cfg = self.config.get("hardware", {})
        backend_name = hardware_cfg.get("target_backend") or self.config["quantum"].get("backend_name")
        backend, _ = RuntimeExecutor.resolve_backend(backend_name)
        if backend is None:
            return None

        thresholds = HardwareFeasibilityThresholds(
            max_transpiled_depth=int(hardware_cfg.get("max_transpiled_depth", 200)),
            max_two_qubit_gates=int(hardware_cfg.get("max_two_qubit_gates", 120)),
            max_total_shots=int(hardware_cfg.get("max_total_shots", 150000)),
        )
        return HardwareFeasibilityAnalyzer(
            backend=backend,
            optimization_level=self.config["quantum"].get("optimization_level", 1),
            seed=seed,
            thresholds=thresholds,
        )

    def _save_figure(self, filename: str, render_callable) -> None:
        """Render a figure into the run directory and mirror it to root results."""
        run_path = self.artifacts.run_path(filename)
        render_callable(str(run_path))
        self.artifacts.mirror_existing(filename)

    def _copy_root_artifact_to_run(self, filename: str) -> None:
        """Copy a root-level artifact generated elsewhere into the run directory."""
        shutil.copy2(self.artifacts.root_path(filename), self.artifacts.run_path(filename))

    def run(self) -> Dict[str, object]:
        """Run the full artifact-generation workflow."""
        results_dir = self.artifacts.root_dir
        visualizer = Visualizer(
            style=self.visual_cfg.get("style", "seaborn-v0_8-darkgrid"),
            dpi=self.visual_cfg.get("dpi", 150),
        )
        metrics = EvaluationMetrics()
        _, graph = self.build_graph()

        logging.info(
            "Running benchmark on graph with %s nodes and %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        seed = self.optimizer_cfg.get("seed", 42)
        exact_result = ClassicalSolver(seed=seed).solve_exact(graph)
        logging.info("Exact cut value: %.4f", exact_result.optimal_value)

        qaoa_rows: List[BenchmarkMetricRecord] = []
        qaoa_results: List[Tuple[int, QAOAOptimizer, object]] = []
        hardware_rows: List[HardwareFeasibilityRecord] = []
        previous_params = None
        warm_start_enabled = bool(self.qaoa_cfg.get("warm_start_across_depths", True))
        hardware_analyzer = self.create_hardware_analyzer(seed=seed)

        for depth in self.qaoa_cfg["p_layers"]:
            problem = self.create_problem(graph, depth=depth, seed=seed)
            optimizer = self.create_optimizer(depth=depth)
            initial_params = None

            if warm_start_enabled and previous_params is not None:
                initial_params = optimizer.build_initial_points(
                    warm_start_params=optimizer.extend_parameters_for_next_depth(previous_params),
                    n_points=self.qaoa_cfg.get("n_initial_points", 3),
                )

            result = optimizer.optimize(
                objective_function=problem.objective_function,
                n_qubits=graph.number_of_nodes(),
                initial_params=initial_params,
                graph=graph,
                solution_decoder=problem.decode_solution,
                selection_objective_function=problem.objective_function,
            )
            previous_params = result.optimal_params
            ratio = metrics.compute_approximation_ratio(
                qaoa_value=float(result.cut_value),
                optimal_value=float(exact_result.optimal_value),
            )
            qaoa_rows.append(
                BenchmarkMetricRecord(
                    method=f"qaoa_p{depth}",
                    depth=depth,
                    expected_cut_value=float(result.cut_value),
                    sampled_cut_value=float(result.sampled_cut_value)
                    if result.sampled_cut_value is not None
                    else None,
                    best_sampled_cut_value=float(result.best_sampled_cut_value)
                    if result.best_sampled_cut_value is not None
                    else None,
                    approximation_ratio=ratio,
                    minimization_objective=float(result.optimal_value),
                    reevaluated_minimization_objective=float(result.reevaluated_optimal_value)
                    if result.reevaluated_optimal_value is not None
                    else None,
                    objective_std=float(result.objective_std) if result.objective_std is not None else None,
                    objective_stderr=float(result.objective_stderr)
                    if result.objective_stderr is not None
                    else None,
                    n_evaluations=result.n_evaluations,
                    runtime_sec=round(result.runtime, 4),
                    representative_bitstring=result.solution_bitstring,
                    representative_probability=float(result.bitstring_probability)
                    if result.bitstring_probability is not None
                    else None,
                    best_sampled_bitstring=result.best_sampled_bitstring,
                    analysis_mode=self.config["quantum"].get("analysis_mode", "auto"),
                    diagnostics=" | ".join(result.diagnostics),
                    most_likely_bitstring=result.most_likely_bitstring,
                )
            )
            qaoa_results.append((depth, optimizer, result))
            if hardware_analyzer is not None:
                circuit = QAOACircuitBuilder(
                    n_qubits=graph.number_of_nodes(),
                    p=depth,
                ).build_qaoa_circuit_multilayer(
                    graph,
                    gammas=result.optimal_params[0::2].tolist(),
                    betas=result.optimal_params[1::2].tolist(),
                )
                hardware_rows.append(
                    HardwareFeasibilityRecord(
                        method=f"qaoa_p{depth}",
                        depth=depth,
                        **hardware_analyzer.analyze(
                            circuit,
                            shots_per_evaluation=self.config["quantum"].get("shots", 1024),
                            n_evaluations=result.n_evaluations,
                            objective_repetitions=self.config["quantum"].get("objective_repetitions", 1),
                            report_repetitions=self.config["quantum"].get("report_repetitions", 1),
                        ),
                    )
                )
            logging.info(
                "QAOA p=%s expected_cut=%.4f ratio=%.4f representative_bitstring=%s diagnostics=%s",
                depth,
                result.cut_value,
                ratio,
                result.solution_bitstring,
                "; ".join(result.diagnostics) if result.diagnostics else "none",
            )

        rqaoa_result = None
        rqaoa_row = None
        if self.config["rqaoa"].get("enabled", True):
            rqaoa = RQAOAEngine(
                p=self.qaoa_cfg["p_layers"][0],
                n_eliminate_per_step=self.config["rqaoa"]["eliminate_per_step"],
                correlation_threshold=self.config["rqaoa"]["correlation_threshold"],
                min_problem_size=self.config["rqaoa"]["min_problem_size"],
                max_depth=self.config["rqaoa"]["max_depth"],
                force_fallback_elimination=True,
                executor=self.create_executor(seed=seed),
                analysis_shots=self.config["quantum"].get("shots", 1024),
                correlation_method=self.config["rqaoa"].get("correlation_method", "auto"),
            )
            rqaoa_result = rqaoa.solve(graph, optimal_value=exact_result.optimal_value)
            rqaoa_row = BenchmarkMetricRecord(
                method="rqaoa",
                depth=self.qaoa_cfg["p_layers"][0],
                expected_cut_value=float(rqaoa_result.cut_value),
                sampled_cut_value=float(rqaoa_result.cut_value),
                best_sampled_cut_value=float(rqaoa_result.cut_value),
                approximation_ratio=float(rqaoa_result.approximation_ratio or 0.0),
                minimization_objective=-float(rqaoa_result.cut_value),
                reevaluated_minimization_objective=-float(rqaoa_result.cut_value),
                objective_std=0.0,
                objective_stderr=0.0,
                n_evaluations=None,
                runtime_sec=round(rqaoa_result.runtime, 4),
                representative_bitstring=rqaoa_result.solution_bitstring,
                representative_probability=None,
                best_sampled_bitstring=rqaoa_result.solution_bitstring,
                analysis_mode="rqaoa_recursive",
                diagnostics="",
                most_likely_bitstring=rqaoa_result.solution_bitstring,
            )

        metrics_rows: List[BenchmarkMetricRecord] = [
            BenchmarkMetricRecord(
                method="exact",
                depth=0,
                expected_cut_value=float(exact_result.optimal_value),
                sampled_cut_value=float(exact_result.optimal_value),
                best_sampled_cut_value=float(exact_result.optimal_value),
                approximation_ratio=1.0,
                minimization_objective=-float(exact_result.optimal_value),
                reevaluated_minimization_objective=-float(exact_result.optimal_value),
                objective_std=0.0,
                objective_stderr=0.0,
                n_evaluations=0,
                runtime_sec=round(exact_result.runtime, 4),
                representative_bitstring=exact_result.optimal_bitstrings[0],
                representative_probability=1.0,
                best_sampled_bitstring=exact_result.optimal_bitstrings[0],
                analysis_mode="classical_exact",
                diagnostics="",
                most_likely_bitstring=exact_result.optimal_bitstrings[0],
            )
        ]
        metrics_rows.extend(qaoa_rows)
        if rqaoa_row is not None:
            metrics_rows.append(rqaoa_row)
        self.artifacts.write_csv("metrics.csv", metrics_rows)
        if hardware_rows:
            self.artifacts.write_csv("hardware_feasibility.csv", hardware_rows)

        best_depth, best_optimizer, best_result = max(qaoa_results, key=lambda item: item[2].cut_value or 0.0)
        representative_bitstring = (
            best_result.solution_bitstring
            or best_result.best_sampled_bitstring
            or best_result.most_likely_bitstring
            or exact_result.optimal_bitstrings[0]
        )
        partition = [index for index, bit in enumerate(representative_bitstring) if bit == "0"]
        cut_edges = ClassicalSolver().get_cut_edges(graph, representative_bitstring)
        self._save_figure(
            "graph_cut_visualization.png",
            lambda path: visualizer.plot_graph_cut(
                graph=graph,
                partition=partition,
                cut_edges=cut_edges,
                save_path=path,
                title=f"Representative QAOA Cut (p={best_depth})",
            ),
        )

        depths = [row.depth for row in qaoa_rows]
        ratios = [row.approximation_ratio for row in qaoa_rows]
        self._save_figure(
            "approximation_ratio.png",
            lambda path: visualizer.plot_approximation_ratio(
                depths=depths,
                ratios=ratios,
                save_path=path,
                title="QAOA Expected Approximation Ratio vs Depth",
            ),
        )

        landscape_problem = MaxCutQAOAProblem(
            graph=graph,
            p=1,
            executor=RuntimeExecutor(mode="local", shots=0, seed=seed),
            seed=seed,
            analysis_shots=0,
            analysis_mode="none",
        )
        objective_p1 = landscape_problem.objective_function
        gamma_grid, beta_grid, cost_grid = ParameterGridEvaluator(p=1).evaluate_grid(
            objective_function=objective_p1,
            n_points=15,
        )
        self._save_figure(
            "energy_landscape.png",
            lambda path: visualizer.plot_energy_landscape(
                gamma_grid=gamma_grid,
                beta_grid=beta_grid,
                cost_grid=cost_grid,
                save_path=path,
                title="QAOA Minimization Objective Landscape (p=1)",
            ),
        )

        iterations, values = best_optimizer.get_convergence_plot_data(best_result)
        if len(iterations) > 0:
            self._save_figure(
                "optimization_convergence.png",
                lambda path: visualizer.plot_optimization_convergence(
                    iterations=iterations,
                    values=values,
                    save_path=path,
                    title=f"QAOA Convergence (p={best_depth})",
                ),
            )

        comparison_labels = ["Exact", f"QAOA p={best_depth}"]
        comparison_values = [
            float(exact_result.optimal_value),
            float(best_result.cut_value or 0.0),
        ]
        if rqaoa_result is not None:
            comparison_labels.append("RQAOA")
            comparison_values.append(float(rqaoa_result.cut_value))
        self._save_figure(
            "method_comparison.png",
            lambda path: visualizer.plot_comparison_bar(
                labels=comparison_labels,
                values=comparison_values,
                reference=float(exact_result.optimal_value),
                save_path=path,
                title="Expected Objective Comparison on Configured Graph",
            ),
        )

        study_results = None
        if self.config.get("study", {}).get("enabled", False):
            logging.info("Running held-out experimental study")
            study_results = ExperimentalStudyRunner(project_root=self.project_root, config=self.config).run()
            self._save_figure(
                "study_significance_heatmap.png",
                lambda path: visualizer.plot_significance_heatmap(
                    significance_rows=to_serializable_records(study_results["significance"]),
                    save_path=path,
                    title="Held-Out QAOA Minus Baseline Mean-Ratio Differences",
                ),
            )
            self._save_figure(
                "study_budget_fairness.png",
                lambda path: visualizer.plot_budget_fairness(
                    budget_rows=to_serializable_records(study_results["budget_summary"]),
                    save_path=path,
                    title="Budget-Matched Mean Approximation Ratios",
                ),
            )
            for filename in (
                "study_budget_fairness.png",
                "study_significance_heatmap.png",
                "study_candidate_search.csv",
                "study_budget_summary.csv",
                "study_instance_metrics.csv",
                "study_method_summary.csv",
                "study_pairwise_summary.csv",
                "study_significance.csv",
                "study_manifest.json",
                "publication_positioning.json",
                "publication_positioning.md",
            ):
                self._copy_root_artifact_to_run(filename)

        review_cfg = self.config.get("results_review", {})
        robustness_results = None
        if review_cfg.get("enabled", True):
            logging.info("Running repeated-seed benchmark robustness analysis")
            robustness_runner = BenchmarkRobustnessRunner(
                graph=graph,
                exact_value=float(exact_result.optimal_value),
                depths=self.qaoa_cfg["p_layers"],
                optimization_seeds=review_cfg.get(
                    "benchmark_optimization_seeds",
                    [seed],
                ),
                create_problem=lambda depth, bench_seed: self.create_problem(graph, depth=depth, seed=bench_seed),
                create_optimizer=lambda depth, bench_seed: self.create_optimizer(depth=depth, seed_override=bench_seed),
                confidence=float(review_cfg.get("confidence_level", 0.95)),
                n_bootstrap=int(review_cfg.get("bootstrap_samples", 1000)),
            )
            robustness_results = robustness_runner.run()
            self.artifacts.write_csv("benchmark_robustness_runs.csv", robustness_results["runs"])
            self.artifacts.write_csv("benchmark_robustness_summary.csv", robustness_results["summary"])
            self._save_figure(
                "sample_gap_analysis.png",
                lambda path: visualizer.plot_sample_gap(
                    robustness_rows=to_serializable_records(robustness_results["runs"]),
                    save_path=path,
                    title="Repeated-Seed Sampled-vs-Expected Gap",
                ),
            )

            reviewer = ScientificResultsReviewer(review_cfg)
            review_payload = reviewer.review(
                benchmark_rows=metrics_rows,
                benchmark_robustness_summary=robustness_results["summary"],
                study_summary_rows=study_results["summary"] if study_results is not None else [],
                significance_rows=study_results["significance"] if study_results is not None else [],
                pairwise_rows=study_results["pairwise"] if study_results is not None else [],
            )
            self.artifacts.write_json("results_verdict.json", review_payload)
            self.artifacts.write_text("results_verdict.md", reviewer.render_markdown(review_payload))

        run_manifest = {
            "run_id": self.artifacts.run_id,
            "run_output_dir": str(self.artifacts.run_dir),
            "stable_output_dir": str(results_dir),
            "provenance": collect_run_provenance(
                project_root=self.project_root,
                config=self.config,
                run_id=self.artifacts.run_id,
            ),
            "artifacts": sorted(path.name for path in self.artifacts.run_dir.iterdir()),
        }
        self.artifacts.write_json("run_manifest.json", run_manifest)

        logging.info("Artifacts written to %s", results_dir)
        logging.info("Run-scoped artifacts written to %s", self.artifacts.run_dir)
        return {
            "metrics": metrics_rows,
            "study": study_results,
            "robustness": robustness_results,
            "run_manifest": run_manifest,
        }
