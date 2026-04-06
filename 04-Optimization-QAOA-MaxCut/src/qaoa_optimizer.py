"""Classical optimization and workflow helpers for Max-Cut QAOA."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .hamiltonian_builder import HamiltonianBuilder
from .qaoa_circuit import QAOACircuitBuilder
from .runtime_executor import RuntimeExecutor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    optimal_params: np.ndarray
    optimal_value: float
    n_evaluations: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    runtime: float = 0.0
    converged: bool = False

    # Primary algorithmic output: expected Max-Cut value of the optimized state.
    cut_value: Optional[float] = None
    expected_interaction_value: Optional[float] = None
    objective_std: Optional[float] = None
    objective_stderr: Optional[float] = None

    # Representative sampled / decoded bitstring information.
    # ``solution_bitstring`` is the most likely observed sample, while
    # ``best_sampled_*`` tracks the best cut seen in the sampled output.
    solution_bitstring: Optional[str] = None
    sampled_cut_value: Optional[float] = None
    most_likely_bitstring: Optional[str] = None
    most_likely_cut_value: Optional[float] = None
    best_sampled_bitstring: Optional[str] = None
    best_sampled_cut_value: Optional[float] = None
    bitstring_probability: Optional[float] = None
    measurement_counts: Optional[Dict[str, int]] = None

    # Optimization diagnostics.
    reevaluated_optimal_value: Optional[float] = None
    diagnostics: List[str] = field(default_factory=list)


class MaxCutQAOAProblem:
    """
    Shared Max-Cut QAOA workflow object.

    This class owns the graph, cost operator, circuit construction, backend
    execution, and final-state decoding so the classical optimizer and reported
    quantum outputs stay aligned.
    """

    def __init__(
        self,
        graph: nx.Graph,
        p: int,
        executor: Optional[RuntimeExecutor] = None,
        analysis_executor: Optional[RuntimeExecutor] = None,
        seed: Optional[int] = 42,
        analysis_shots: int = 1024,
        objective_repetitions: int = 1,
        report_repetitions: Optional[int] = None,
        analysis_mode: str = "auto",
    ) -> None:
        self.graph = self._relabel_if_needed(graph)
        self.p = p
        self.seed = seed
        self.analysis_mode = analysis_mode
        self.objective_repetitions = max(1, int(objective_repetitions))
        self.report_repetitions = max(
            self.objective_repetitions,
            int(report_repetitions) if report_repetitions is not None else self.objective_repetitions,
        )

        self.hamiltonian, self.offset = HamiltonianBuilder().build_maxcut_hamiltonian(self.graph)
        self.circuit_builder = QAOACircuitBuilder(
            n_qubits=self.graph.number_of_nodes(),
            p=p,
        )

        base_executor = executor or RuntimeExecutor(
            mode="local",
            shots=analysis_shots,
            seed=seed,
        )
        self.objective_executor = self._create_objective_executor(base_executor)
        self.analysis_executor = self._create_analysis_executor(
            base_executor=base_executor,
            analysis_executor=analysis_executor,
            analysis_shots=analysis_shots,
        )

    def build_circuit(self, params: np.ndarray):
        """Build the QAOA circuit for the supplied parameter vector."""
        params_array = np.asarray(params, dtype=float)
        return self.circuit_builder.build_qaoa_circuit_multilayer(
            self.graph,
            gammas=params_array[0::2].tolist(),
            betas=params_array[1::2].tolist(),
        )

    def objective_function(self, params: np.ndarray) -> float:
        """Return the minimization objective ``-E[C]``."""
        stats = self.evaluate_objective_stats(params)
        return -float(stats["objective_value"])

    def evaluate_objective_stats(
        self,
        params: np.ndarray,
        repetitions: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate the expected objective, optionally averaging repeated calls."""
        reps = max(1, int(repetitions if repetitions is not None else self.objective_repetitions))
        circuit = self.build_circuit(params)

        objective_values = []
        interaction_values = []
        variances = []
        for _ in range(reps):
            result = self.objective_executor.execute_circuit(
                circuit,
                self.hamiltonian,
                offset=self.offset,
            )
            objective_values.append(float(result.objective_value))
            interaction_values.append(float(result.expectation_value))
            variances.append(float(result.variance))

        objective_array = np.asarray(objective_values, dtype=float)
        interaction_array = np.asarray(interaction_values, dtype=float)

        return {
            "objective_value": float(np.mean(objective_array)),
            "interaction_value": float(np.mean(interaction_array)),
            "objective_std": float(np.std(objective_array)),
            "objective_stderr": float(np.std(objective_array) / np.sqrt(reps)),
            "mean_variance": float(np.mean(variances)),
            "repetitions": reps,
        }

    def decode_solution(
        self,
        params: np.ndarray,
        graph: Optional[nx.Graph] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the optimized state.

        The primary reported value is the expected Max-Cut objective. A single
        bitstring is returned only as a representative sampled artifact.
        """
        del graph
        stats = self.evaluate_objective_stats(params, repetitions=self.report_repetitions)
        circuit = self.build_circuit(params)

        counts: Dict[str, int] = {}
        if self.analysis_executor is not None:
            analysis_result = self.analysis_executor.execute_circuit(
                circuit,
                self.hamiltonian,
                offset=self.offset,
            )
            counts = analysis_result.measurement_counts or {}
        elif self.objective_executor.mode == "ibm_hardware":
            logger.warning(
                "No measurement-capable analysis backend is configured for hardware mode. "
                "Representative bitstrings will be unavailable."
            )

        most_likely_bitstring, most_likely_cut_value, bitstring_probability = self._most_likely_bitstring(counts)
        best_sampled_bitstring, best_sampled_cut_value = self._best_sampled_bitstring(counts)
        solution_bitstring = most_likely_bitstring
        sampled_cut_value = most_likely_cut_value

        return {
            "cut_value": float(stats["objective_value"]),
            "interaction_value": float(stats["interaction_value"]),
            "objective_std": float(stats["objective_std"]),
            "objective_stderr": float(stats["objective_stderr"]),
            "bitstring": solution_bitstring,
            "sampled_cut_value": sampled_cut_value,
            "most_likely_bitstring": most_likely_bitstring,
            "most_likely_cut_value": most_likely_cut_value,
            "best_sampled_bitstring": best_sampled_bitstring,
            "best_sampled_cut_value": best_sampled_cut_value,
            "probability": bitstring_probability,
            "measurement_counts": counts if counts else None,
        }

    def _create_objective_executor(self, base_executor: RuntimeExecutor) -> RuntimeExecutor:
        """Create the executor used inside the optimization loop."""
        if base_executor.mode == "local":
            return RuntimeExecutor(
                mode="local",
                shots=0,
                seed=self.seed,
            )
        return base_executor

    def _create_analysis_executor(
        self,
        base_executor: RuntimeExecutor,
        analysis_executor: Optional[RuntimeExecutor],
        analysis_shots: int,
    ) -> Optional[RuntimeExecutor]:
        """Create the executor used for representative bitstring analysis."""
        if analysis_executor is not None:
            return analysis_executor

        mode = self.analysis_mode.lower()
        if mode == "auto":
            if base_executor.mode in {"local", "noisy_simulator"}:
                return base_executor
            return None
        if mode == "same_backend":
            if base_executor.mode == "ibm_hardware":
                logger.warning(
                    "analysis_mode='same_backend' cannot produce counts with the current "
                    "Estimator-only hardware path. Representative bitstrings will be unavailable."
                )
                return None
            return base_executor
        if mode == "none":
            return None
        if mode == "local_exact":
            logger.warning(
                "analysis_mode='local_exact' decodes representative bitstrings from a local "
                "exact model. This is useful for debugging but not a faithful hardware readout."
            )
            return RuntimeExecutor(
                mode="local",
                shots=analysis_shots,
                seed=self.seed,
            )

        raise ValueError(f"Unknown analysis_mode: {self.analysis_mode}")

    @staticmethod
    def _relabel_if_needed(graph: nx.Graph) -> nx.Graph:
        """Relabel a graph to consecutive integer nodes using sorted node order."""
        nodes = list(sorted(graph.nodes()))
        if nodes == list(range(len(nodes))):
            return graph

        mapping = {node: idx for idx, node in enumerate(nodes)}
        return nx.relabel_nodes(graph, mapping, copy=True)

    def _best_sampled_bitstring(
        self,
        counts: Dict[str, int],
    ) -> tuple[Optional[str], Optional[float]]:
        """Return the highest-cut bitstring that appears in the sampled counts."""
        if not counts:
            return None, None

        ranked = []
        for bitstring, count in counts.items():
            ranked.append(
                (
                    self._calculate_cut(self.graph, bitstring),
                    count,
                    bitstring,
                )
            )

        best_cut, _, best_bitstring = max(ranked, key=lambda item: (item[0], item[1], item[2]))
        return best_bitstring, float(best_cut)

    def _most_likely_bitstring(
        self,
        counts: Dict[str, int],
    ) -> tuple[Optional[str], Optional[float], Optional[float]]:
        """Return the most frequently observed bitstring and its cut value."""
        if not counts:
            return None, None, None

        total = sum(counts.values())
        bitstring, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        probability = float(count / total) if total > 0 else None
        return bitstring, self._calculate_cut(self.graph, bitstring), probability

    @staticmethod
    def _calculate_cut(graph: nx.Graph, bitstring: str) -> float:
        """Calculate the weighted cut value for a bitstring."""
        cut_value = 0.0
        for u, v in graph.edges():
            if u < len(bitstring) and v < len(bitstring):
                if bitstring[u] != bitstring[v]:
                    cut_value += float(graph[u][v].get("weight", 1.0))
        return float(cut_value)


class QAOAOptimizer:
    """
    Classical optimizer for QAOA parameters.

    The optimizer records every objective evaluation and reports the expected
    Max-Cut value of the optimized state as its primary quantum output.
    """

    def __init__(
        self,
        p: int = 1,
        optimizer_type: str = "COBYLA",
        maxiter: int = 500,
        tol: float = 1e-6,
        seed: Optional[int] = 42,
        n_initial_points: int = 3,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
        spsa_learning_rate: float = 0.1,
        spsa_perturbation: float = 0.1,
        selection_repetitions: int = 1,
        plateau_window: int = 20,
        plateau_tolerance: float = 1e-4,
    ) -> None:
        self.p = p
        self.optimizer_type = optimizer_type.upper()
        self.maxiter = maxiter
        self.tol = tol
        self.seed = seed
        self.n_initial_points = n_initial_points
        self.callback = callback
        self.spsa_learning_rate = spsa_learning_rate
        self.spsa_perturbation = spsa_perturbation
        self.selection_repetitions = max(1, int(selection_repetitions))
        self.plateau_window = max(2, int(plateau_window))
        self.plateau_tolerance = float(plateau_tolerance)
        self.rng = np.random.default_rng(seed)

        logger.info(
            "QAOAOptimizer initialized: p=%s, optimizer=%s",
            p,
            self.optimizer_type,
        )

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_qubits: int,
        initial_params: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[List[float], List[float]]] = None,
        graph: Optional[nx.Graph] = None,
        solution_decoder: Optional[
            Callable[[np.ndarray, Optional[nx.Graph]], Dict[str, Any]]
        ] = None,
        selection_objective_function: Optional[Callable[[np.ndarray], float]] = None,
    ) -> OptimizationResult:
        """Run the optimization loop and optionally decode the final state."""
        del n_qubits
        logger.info("Starting optimization with %s", self.optimizer_type)
        start_time = time.time()

        initial_points = self._normalize_initial_params(initial_params)
        if bounds is None:
            bounds = self._default_bounds()

        best_result: Optional[OptimizationResult] = None
        best_value = float("inf")

        for run_id, init_params in enumerate(initial_points):
            logger.info("Optimization run %s/%s", run_id + 1, len(initial_points))
            result = self._run_single_optimization(
                objective_function=objective_function,
                initial_params=np.asarray(init_params, dtype=float),
                bounds=bounds,
                run_id=run_id,
            )

            comparison_value = self._reevaluate_candidate(
                params=result.optimal_params,
                fallback_value=result.optimal_value,
                selection_objective_function=selection_objective_function,
            )
            result.reevaluated_optimal_value = comparison_value

            if comparison_value < best_value:
                best_value = comparison_value
                best_result = result

        if best_result is None:
            raise RuntimeError("Optimization did not produce any result.")

        best_result.optimal_value = float(
            best_result.reevaluated_optimal_value
            if best_result.reevaluated_optimal_value is not None
            else best_result.optimal_value
        )

        if graph is not None:
            decoder = solution_decoder or self._extract_solution
            solution = decoder(best_result.optimal_params, graph)
            best_result.cut_value = solution.get("cut_value")
            best_result.expected_interaction_value = solution.get("interaction_value")
            best_result.objective_std = solution.get("objective_std")
            best_result.objective_stderr = solution.get("objective_stderr")
            best_result.solution_bitstring = solution.get("bitstring")
            best_result.sampled_cut_value = solution.get("sampled_cut_value")
            best_result.most_likely_bitstring = solution.get("most_likely_bitstring")
            best_result.most_likely_cut_value = solution.get("most_likely_cut_value")
            best_result.best_sampled_bitstring = solution.get("best_sampled_bitstring")
            best_result.best_sampled_cut_value = solution.get("best_sampled_cut_value")
            best_result.bitstring_probability = solution.get("probability")
            best_result.measurement_counts = solution.get("measurement_counts")

        best_result.runtime = time.time() - start_time
        logger.info(
            "Optimization complete: value=%.6f, evaluations=%s, runtime=%.2fs",
            best_result.optimal_value,
            best_result.n_evaluations,
            best_result.runtime,
        )
        return best_result

    def build_initial_points(
        self,
        warm_start_params: Optional[np.ndarray] = None,
        n_points: Optional[int] = None,
    ) -> np.ndarray:
        """Build a restart stack with an optional warm-start point first."""
        total_points = self.n_initial_points if n_points is None else max(1, int(n_points))
        points: List[np.ndarray] = []

        if warm_start_params is not None:
            points.append(np.asarray(warm_start_params, dtype=float))
            total_points = max(0, total_points - 1)

        points.extend(self._generate_initial_params(total_points))
        return np.vstack(points) if len(points) > 1 else np.asarray(points[0], dtype=float)

    @staticmethod
    def extend_parameters_for_next_depth(previous_params: np.ndarray) -> np.ndarray:
        """Warm-start ``p+1`` by copying the last learned layer into the new slot."""
        params = np.asarray(previous_params, dtype=float)
        if params.size == 0:
            return np.zeros(2, dtype=float)
        if params.size % 2 != 0:
            raise ValueError("QAOA parameter vectors must have even length.")

        extended = np.zeros(params.size + 2, dtype=float)
        extended[:-2] = params
        extended[-2] = params[-2]
        extended[-1] = params[-1]
        return extended

    def _normalize_initial_params(
        self,
        initial_params: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Normalize user-provided initial parameters into a list of restarts."""
        if initial_params is None:
            return self._generate_initial_params(self.n_initial_points)

        params_array = np.asarray(initial_params, dtype=float)
        if params_array.ndim == 1:
            return [params_array]
        if params_array.ndim == 2:
            return [row for row in params_array]

        raise ValueError("initial_params must be a 1D vector or 2D array.")

    def _generate_initial_params(self, n_points: int) -> List[np.ndarray]:
        """Generate multiple initial parameter sets."""
        return [
            self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
            for _ in range(n_points)
        ]

    def _default_bounds(self) -> Tuple[List[float], List[float]]:
        """Get default parameter bounds."""
        lower = [0.0] * (2 * self.p)
        upper = [2 * np.pi] * (2 * self.p)
        return lower, upper

    def _run_single_optimization(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        bounds: Tuple[List[float], List[float]],
        run_id: int = 0,
    ) -> OptimizationResult:
        """Run optimization from a single initial point."""
        history: List[Dict[str, Any]] = []
        evaluation_count = 0

        def tracked_objective(params: np.ndarray) -> float:
            nonlocal evaluation_count
            params_array = np.asarray(params, dtype=float)
            value = float(objective_function(params_array))
            history.append(
                {
                    "iteration": evaluation_count,
                    "run_id": run_id,
                    "params": params_array.copy(),
                    "value": value,
                }
            )
            evaluation_count += 1
            if self.callback is not None:
                self.callback(evaluation_count, params_array.copy(), value)
            return value

        if self.optimizer_type == "COBYLA":
            from qiskit_algorithms.optimizers import COBYLA

            optimizer = COBYLA(maxiter=self.maxiter, tol=self.tol, rhobeg=1.0)
            result = optimizer.minimize(
                fun=tracked_objective,
                x0=initial_params,
                bounds=list(zip(bounds[0], bounds[1])),
            )
        elif self.optimizer_type == "SPSA":
            from qiskit_algorithms.optimizers import SPSA

            optimizer = SPSA(
                maxiter=self.maxiter,
                learning_rate=self.spsa_learning_rate,
                perturbation=self.spsa_perturbation,
            )
            result = optimizer.minimize(
                fun=tracked_objective,
                x0=initial_params,
                bounds=list(zip(bounds[0], bounds[1])),
            )
        elif self.optimizer_type in {"NELDER-MEAD", "L-BFGS-B"}:
            from scipy.optimize import minimize

            options: Dict[str, Any] = {"maxiter": self.maxiter}
            if self.optimizer_type == "NELDER-MEAD":
                options["xatol"] = self.tol
                options["fatol"] = self.tol

            result = minimize(
                tracked_objective,
                initial_params,
                method=self.optimizer_type,
                bounds=list(zip(bounds[0], bounds[1]))
                if self.optimizer_type == "L-BFGS-B"
                else None,
                options=options,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        optimal_params = np.asarray(result.x, dtype=float)
        optimal_value = float(result.fun)
        n_evaluations = int(getattr(result, "nfev", evaluation_count) or evaluation_count)
        converged = bool(getattr(result, "success", n_evaluations < self.maxiter))
        diagnostics = self._diagnose_history(history, converged, n_evaluations)

        return OptimizationResult(
            optimal_params=optimal_params,
            optimal_value=optimal_value,
            n_evaluations=n_evaluations,
            history=history,
            converged=converged,
            diagnostics=diagnostics,
        )

    def _reevaluate_candidate(
        self,
        params: np.ndarray,
        fallback_value: float,
        selection_objective_function: Optional[Callable[[np.ndarray], float]],
    ) -> float:
        """Re-evaluate a candidate solution to reduce noisy best-run selection bias."""
        if selection_objective_function is None or self.selection_repetitions <= 1:
            return float(fallback_value)

        values = [
            float(selection_objective_function(np.asarray(params, dtype=float)))
            for _ in range(self.selection_repetitions)
        ]
        return float(np.mean(values))

    def _diagnose_history(
        self,
        history: List[Dict[str, Any]],
        converged: bool,
        n_evaluations: int,
    ) -> List[str]:
        """Generate lightweight optimization diagnostics."""
        diagnostics: List[str] = []

        if not converged and n_evaluations >= self.maxiter:
            diagnostics.append("Optimizer hit the iteration budget before declaring convergence.")

        if len(history) >= self.plateau_window:
            recent_values = np.array([entry["value"] for entry in history[-self.plateau_window:]], dtype=float)
            improvement = float(recent_values[0] - np.min(recent_values))
            if improvement < self.plateau_tolerance:
                diagnostics.append(
                    "Objective improvement stalled over the recent optimization window; "
                    "this may indicate a plateau, poor initialization, or noisy evaluations."
                )

        return diagnostics

    def _extract_solution(
        self,
        params: np.ndarray,
        graph: Optional[nx.Graph],
    ) -> Dict[str, Any]:
        """Decode the optimized parameters using a local exact Max-Cut workflow."""
        if graph is None:
            return {
                "cut_value": None,
                "interaction_value": None,
                "objective_std": None,
                "objective_stderr": None,
                "bitstring": None,
                "sampled_cut_value": None,
                "most_likely_bitstring": None,
                "most_likely_cut_value": None,
                "best_sampled_bitstring": None,
                "best_sampled_cut_value": None,
                "probability": None,
                "measurement_counts": None,
            }

        problem = MaxCutQAOAProblem(
            graph=graph,
            p=len(np.asarray(params)[0::2]),
            seed=self.seed,
            analysis_mode="same_backend",
        )
        return problem.decode_solution(params)

    def get_convergence_plot_data(
        self,
        result: OptimizationResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract tracked objective values for convergence plotting."""
        iterations = np.array([h["iteration"] for h in result.history])
        values = np.array([h["value"] for h in result.history])
        return iterations, values


class ParameterGridEvaluator:
    """
    Evaluate the QAOA objective over a small parameter grid.

    For ``p > 1`` the grid fills only the first layer and keeps the rest of the
    parameters at zero. This is a visualization aid, not a full optimization
    routine.
    """

    def __init__(self, p: int = 1) -> None:
        self.p = p

    def evaluate_grid(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a 2D parameter grid."""
        gamma_range = np.linspace(0, np.pi, n_points)
        beta_range = np.linspace(0, np.pi, n_points)
        gamma_grid, beta_grid = np.meshgrid(gamma_range, beta_range)
        cost_grid = np.zeros_like(gamma_grid)

        for i in range(n_points):
            for j in range(n_points):
                params = np.zeros(2 * self.p)
                params[0] = gamma_grid[i, j]
                params[1] = beta_grid[i, j]
                cost_grid[i, j] = objective_function(params)

        return gamma_grid, beta_grid, cost_grid

    def find_optimal_region(
        self,
        cost_grid: np.ndarray,
        gamma_grid: np.ndarray,
        beta_grid: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, float]]:
        """Return the top-k lowest-cost grid points."""
        flat_indices = np.argsort(cost_grid.flatten())[:top_k]
        results: List[Dict[str, float]] = []

        for idx in flat_indices:
            i, j = np.unravel_index(idx, cost_grid.shape)
            results.append(
                {
                    "gamma": float(gamma_grid[i, j]),
                    "beta": float(beta_grid[i, j]),
                    "cost": float(cost_grid[i, j]),
                }
            )

        return results
