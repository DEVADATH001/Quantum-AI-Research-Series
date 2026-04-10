"""Generate benchmark artifacts for the QAOA Max-Cut project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.classical_solver import ClassicalSolver
from src.evaluation_metrics import EvaluationMetrics
from src.experimental_study import ExperimentalStudyRunner
from src.graph_generator import GraphGenerator
from src.hardware_analysis import HardwareFeasibilityAnalyzer, HardwareFeasibilityThresholds
from src.qaoa_circuit import QAOACircuitBuilder
from src.qaoa_optimizer import MaxCutQAOAProblem, ParameterGridEvaluator, QAOAOptimizer
from src.rqaoa_engine import RQAOAEngine
from src.runtime_executor import RuntimeExecutor
from src.visualization import Visualizer, save_metrics_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.WARNING)
logging.getLogger("stevedore").setLevel(logging.WARNING)


def load_config(project_root: Path) -> Dict:
    """Load the experiment configuration."""
    config_path = project_root / "config" / "experiment_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_graph(config: Dict) -> Tuple:
    """Build the configured graph instance."""
    graph_cfg = config["graph"]
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
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    return generator, graph


def create_executor(config: Dict, seed: int) -> RuntimeExecutor:
    """Create the configured quantum execution backend."""
    quantum_cfg = config["quantum"]
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


def create_problem(config: Dict, graph, depth: int, seed: int) -> MaxCutQAOAProblem:
    """Create a backend-aware Max-Cut QAOA workflow for a given depth."""
    quantum_cfg = config["quantum"]
    return MaxCutQAOAProblem(
        graph=graph,
        p=depth,
        executor=create_executor(config, seed=seed),
        seed=seed,
        analysis_shots=quantum_cfg.get("shots", 1024),
        objective_repetitions=quantum_cfg.get("objective_repetitions", 1),
        report_repetitions=quantum_cfg.get("report_repetitions"),
        analysis_mode=quantum_cfg.get("analysis_mode", "auto"),
    )


def create_optimizer(config: Dict, depth: int) -> QAOAOptimizer:
    """Create the configured classical optimizer for a QAOA depth."""
    qaoa_cfg = config["qaoa"]
    optimizer_cfg = config["optimizer"]
    return QAOAOptimizer(
        p=depth,
        optimizer_type=optimizer_cfg["type"],
        maxiter=optimizer_cfg["maxiter"],
        tol=float(qaoa_cfg["tolerance"]),
        seed=optimizer_cfg.get("seed", 42),
        n_initial_points=qaoa_cfg.get("n_initial_points", 3),
        spsa_learning_rate=optimizer_cfg.get("spsa_step_size", 0.1),
        spsa_perturbation=optimizer_cfg.get("spsa_perturbation", 0.1),
        selection_repetitions=optimizer_cfg.get("selection_repetitions", 1),
        plateau_window=optimizer_cfg.get("plateau_window", 20),
        plateau_tolerance=float(optimizer_cfg.get("plateau_tolerance", 1e-4)),
    )


def create_hardware_analyzer(config: Dict, seed: int) -> HardwareFeasibilityAnalyzer | None:
    """Create a target-backend analyzer for NISQ feasibility estimates."""
    hardware_cfg = config.get("hardware", {})
    backend_name = hardware_cfg.get("target_backend") or config["quantum"].get("backend_name")
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
        optimization_level=config["quantum"].get("optimization_level", 1),
        seed=seed,
        thresholds=thresholds,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root)
    visual_cfg = config["visualization"]
    qaoa_cfg = config["qaoa"]
    optimizer_cfg = config["optimizer"]
    results_dir = project_root / config["results"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    _, graph = build_graph(config)
    metrics = EvaluationMetrics()
    visualizer = Visualizer(
        style=visual_cfg.get("style", "seaborn-v0_8-darkgrid"),
        dpi=visual_cfg.get("dpi", 150),
    )

    LOGGER.info(
        "Running benchmark on graph with %s nodes and %s edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    exact_result = ClassicalSolver(seed=optimizer_cfg.get("seed", 42)).solve_exact(graph)
    LOGGER.info("Exact cut value: %.4f", exact_result.optimal_value)

    qaoa_rows: List[Dict] = []
    qaoa_results: List[Tuple[int, QAOAOptimizer, object]] = []
    hardware_rows: List[Dict] = []
    previous_params = None
    warm_start_enabled = bool(qaoa_cfg.get("warm_start_across_depths", True))
    hardware_analyzer = create_hardware_analyzer(config, seed=optimizer_cfg.get("seed", 42))

    for depth in qaoa_cfg["p_layers"]:
        problem = create_problem(config, graph, depth=depth, seed=optimizer_cfg.get("seed", 42))
        optimizer = create_optimizer(config, depth=depth)
        initial_params = None

        if warm_start_enabled and previous_params is not None:
            initial_params = optimizer.build_initial_points(
                warm_start_params=optimizer.extend_parameters_for_next_depth(previous_params),
                n_points=qaoa_cfg.get("n_initial_points", 3),
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
            {
                "method": f"qaoa_p{depth}",
                "depth": depth,
                "expected_cut_value": float(result.cut_value),
                "sampled_cut_value": float(result.sampled_cut_value)
                if result.sampled_cut_value is not None
                else None,
                "best_sampled_cut_value": float(result.best_sampled_cut_value)
                if result.best_sampled_cut_value is not None
                else None,
                "approximation_ratio": ratio,
                "minimization_objective": float(result.optimal_value),
                "reevaluated_minimization_objective": float(result.reevaluated_optimal_value)
                if result.reevaluated_optimal_value is not None
                else None,
                "objective_std": float(result.objective_std) if result.objective_std is not None else None,
                "objective_stderr": float(result.objective_stderr)
                if result.objective_stderr is not None
                else None,
                "n_evaluations": result.n_evaluations,
                "runtime_sec": round(result.runtime, 4),
                "representative_bitstring": result.solution_bitstring,
                "representative_probability": float(result.bitstring_probability)
                if result.bitstring_probability is not None
                else None,
                "most_likely_bitstring": result.most_likely_bitstring,
                "best_sampled_bitstring": result.best_sampled_bitstring,
                "analysis_mode": config["quantum"].get("analysis_mode", "auto"),
                "diagnostics": " | ".join(result.diagnostics),
            }
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
                {
                    "method": f"qaoa_p{depth}",
                    "depth": depth,
                    **hardware_analyzer.analyze(
                        circuit,
                        shots_per_evaluation=config["quantum"].get("shots", 1024),
                        n_evaluations=result.n_evaluations,
                        objective_repetitions=config["quantum"].get("objective_repetitions", 1),
                        report_repetitions=config["quantum"].get("report_repetitions", 1),
                    ),
                }
            )
        LOGGER.info(
            "QAOA p=%s expected_cut=%.4f ratio=%.4f representative_bitstring=%s diagnostics=%s",
            depth,
            result.cut_value,
            ratio,
            result.solution_bitstring,
            "; ".join(result.diagnostics) if result.diagnostics else "none",
        )

    rqaoa_result = None
    rqaoa_row = None
    if config["rqaoa"].get("enabled", True):
        rqaoa = RQAOAEngine(
            p=qaoa_cfg["p_layers"][0],
            n_eliminate_per_step=config["rqaoa"]["eliminate_per_step"],
            correlation_threshold=config["rqaoa"]["correlation_threshold"],
            min_problem_size=config["rqaoa"]["min_problem_size"],
            max_depth=config["rqaoa"]["max_depth"],
            force_fallback_elimination=True,
            executor=create_executor(config, seed=optimizer_cfg.get("seed", 42)),
            analysis_shots=config["quantum"].get("shots", 1024),
            correlation_method=config["rqaoa"].get("correlation_method", "auto"),
        )
        rqaoa_result = rqaoa.solve(graph, optimal_value=exact_result.optimal_value)
        rqaoa_row = {
            "method": "rqaoa",
            "depth": qaoa_cfg["p_layers"][0],
            "expected_cut_value": float(rqaoa_result.cut_value),
            "sampled_cut_value": float(rqaoa_result.cut_value),
            "best_sampled_cut_value": float(rqaoa_result.cut_value),
            "approximation_ratio": float(rqaoa_result.approximation_ratio or 0.0),
            "minimization_objective": -float(rqaoa_result.cut_value),
            "reevaluated_minimization_objective": -float(rqaoa_result.cut_value),
            "objective_std": 0.0,
            "objective_stderr": 0.0,
            "n_evaluations": None,
            "runtime_sec": round(rqaoa_result.runtime, 4),
            "representative_bitstring": rqaoa_result.solution_bitstring,
            "representative_probability": None,
            "most_likely_bitstring": rqaoa_result.solution_bitstring,
            "best_sampled_bitstring": rqaoa_result.solution_bitstring,
            "analysis_mode": "rqaoa_recursive",
            "diagnostics": "",
            "n_levels": rqaoa_result.n_levels,
        }
        LOGGER.info(
            "RQAOA cut=%.4f ratio=%.4f levels=%s",
            rqaoa_result.cut_value,
            rqaoa_result.approximation_ratio or 0.0,
            rqaoa_result.n_levels,
        )

    metrics_rows = [
        {
            "method": "exact",
            "depth": 0,
            "expected_cut_value": float(exact_result.optimal_value),
            "sampled_cut_value": float(exact_result.optimal_value),
            "best_sampled_cut_value": float(exact_result.optimal_value),
            "approximation_ratio": 1.0,
            "minimization_objective": -float(exact_result.optimal_value),
            "reevaluated_minimization_objective": -float(exact_result.optimal_value),
            "objective_std": 0.0,
            "objective_stderr": 0.0,
            "n_evaluations": 0,
            "runtime_sec": round(exact_result.runtime, 4),
            "representative_bitstring": exact_result.optimal_bitstrings[0],
            "representative_probability": 1.0,
            "best_sampled_bitstring": exact_result.optimal_bitstrings[0],
            "analysis_mode": "classical_exact",
            "diagnostics": "",
        }
    ]
    metrics_rows.extend(qaoa_rows)
    if rqaoa_row is not None:
        metrics_rows.append(rqaoa_row)
    save_metrics_csv(metrics_rows, str(results_dir / "metrics.csv"))
    if hardware_rows:
        save_metrics_csv(hardware_rows, str(results_dir / "hardware_feasibility.csv"))

    best_depth, best_optimizer, best_result = max(qaoa_results, key=lambda item: item[2].cut_value or 0.0)
    representative_bitstring = (
        best_result.solution_bitstring
        or best_result.best_sampled_bitstring
        or best_result.most_likely_bitstring
        or exact_result.optimal_bitstrings[0]
    )
    partition = [index for index, bit in enumerate(representative_bitstring) if bit == "0"]
    cut_edges = ClassicalSolver().get_cut_edges(graph, representative_bitstring)
    visualizer.plot_graph_cut(
        graph=graph,
        partition=partition,
        cut_edges=cut_edges,
        save_path=str(results_dir / "graph_cut_visualization.png"),
        title=f"Representative QAOA Cut (p={best_depth})",
    )

    depths = [row["depth"] for row in qaoa_rows]
    ratios = [row["approximation_ratio"] for row in qaoa_rows]
    visualizer.plot_approximation_ratio(
        depths=depths,
        ratios=ratios,
        save_path=str(results_dir / "approximation_ratio.png"),
        title="QAOA Expected Approximation Ratio vs Depth",
    )

    landscape_problem = MaxCutQAOAProblem(
        graph=graph,
        p=1,
        executor=RuntimeExecutor(mode="local", shots=0, seed=optimizer_cfg.get("seed", 42)),
        seed=optimizer_cfg.get("seed", 42),
        analysis_shots=0,
        analysis_mode="none",
    )
    objective_p1 = landscape_problem.objective_function
    gamma_grid, beta_grid, cost_grid = ParameterGridEvaluator(p=1).evaluate_grid(
        objective_function=objective_p1,
        n_points=15,
    )
    visualizer.plot_energy_landscape(
        gamma_grid=gamma_grid,
        beta_grid=beta_grid,
        cost_grid=cost_grid,
        save_path=str(results_dir / "energy_landscape.png"),
        title="QAOA Minimization Objective Landscape (p=1)",
    )

    iterations, values = best_optimizer.get_convergence_plot_data(best_result)
    if len(iterations) > 0:
        visualizer.plot_optimization_convergence(
            iterations=iterations,
            values=values,
            save_path=str(results_dir / "optimization_convergence.png"),
            title=f"QAOA Convergence (p={best_depth})",
        )

    comparison_labels = ["Exact", f"QAOA p={best_depth}"]
    comparison_values = [
        float(exact_result.optimal_value),
        float(best_result.cut_value or 0.0),
    ]
    if rqaoa_result is not None:
        comparison_labels.append("RQAOA")
        comparison_values.append(float(rqaoa_result.cut_value))

    visualizer.plot_comparison_bar(
        labels=comparison_labels,
        values=comparison_values,
        reference=float(exact_result.optimal_value),
        save_path=str(results_dir / "method_comparison.png"),
        title="Expected Objective Comparison on Configured Graph",
    )

    if config.get("study", {}).get("enabled", False):
        LOGGER.info("Running held-out experimental study")
        ExperimentalStudyRunner(project_root=project_root, config=config).run()

    LOGGER.info("Artifacts written to %s", results_dir)


if __name__ == "__main__":
    main()
