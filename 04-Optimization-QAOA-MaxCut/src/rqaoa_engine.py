"""Recursive QAOA utilities for Max-Cut benchmarks."""

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .classical_solver import ClassicalSolver
from .hamiltonian_builder import HamiltonianBuilder
from .qaoa_circuit import QAOACircuitBuilder
from .qaoa_optimizer import QAOAOptimizer

logger = logging.getLogger(__name__)


@dataclass
class RQAOAResult:
    """Container for RQAOA results."""

    solution_bitstring: str
    cut_value: float
    optimal_value: Optional[float] = None
    approximation_ratio: Optional[float] = None
    n_levels: int = 0
    original_size: int = 0
    reduced_size: int = 0
    reduction_constant: float = 0.0
    eliminated_vars: Optional[List[Dict[str, Any]]] = None
    runtime: float = 0.0

    def __post_init__(self) -> None:
        if self.eliminated_vars is None:
            self.eliminated_vars = []


class RQAOAEngine:
    """
    Recursive QAOA implementation for Max-Cut-style graph objectives.

    This implementation favors correctness and reproducibility over aggressive
    recursion. Opposite-spin eliminations accumulate an explicit constant term
    so that the reduced objective remains mathematically equivalent to the
    constrained original problem.
    """

    def __init__(
        self,
        p: int = 1,
        n_eliminate_per_step: int = 1,
        correlation_threshold: float = 0.8,
        min_problem_size: int = 4,
        max_depth: int = 5,
        optimizer: Optional[QAOAOptimizer] = None,
        qaoa_evaluator: Optional[Callable[[nx.Graph], np.ndarray]] = None,
        force_fallback_elimination: bool = True,
    ) -> None:
        self.p = p
        self.n_eliminate = n_eliminate_per_step
        self.correlation_threshold = correlation_threshold
        self.min_problem_size = min_problem_size
        self.max_depth = max_depth
        self.optimizer = optimizer
        self.qaoa_evaluator = qaoa_evaluator
        self.force_fallback_elimination = force_fallback_elimination

        logger.info(
            "RQAOA initialized: p=%s, eliminate=%s, threshold=%s",
            self.p,
            self.n_eliminate,
            self.correlation_threshold,
        )

    def solve(
        self,
        graph: nx.Graph,
        hamiltonian: Optional[Any] = None,
        optimal_value: Optional[float] = None,
    ) -> RQAOAResult:
        """Solve Max-Cut using recursive variable elimination."""
        del hamiltonian
        start_time = time.time()

        original_nodes = list(sorted(graph.nodes()))
        working_graph = copy.deepcopy(graph)
        working_graph.graph["constant_offset"] = float(
            working_graph.graph.get("constant_offset", 0.0)
        )
        elimination_history: List[Dict[str, Any]] = []
        n_levels = 0

        while (
            working_graph.number_of_nodes() > self.min_problem_size
            and n_levels < self.max_depth
        ):
            logger.info(
                "RQAOA level %s: solving graph with %s active nodes",
                n_levels,
                working_graph.number_of_nodes(),
            )
            qaoa_result = self._run_qaoa_on_subgraph(working_graph)
            correlations = self._analyze_correlations(qaoa_result)
            selected = self._select_eliminations(correlations)

            if not selected:
                logger.info("No eliminations selected; solving the current instance directly.")
                break

            elimination_history.extend(selected)
            working_graph = self._reduce_problem(working_graph, selected)
            n_levels += 1

        final_solution = self._solve_reduced_problem(working_graph)
        full_solution = self._reconstruct_solution(
            final_solution,
            elimination_history,
            original_nodes,
        )
        cut_value = self._calculate_cut(graph, full_solution, original_nodes)

        approximation_ratio = None
        if optimal_value is not None and optimal_value > 0:
            approximation_ratio = cut_value / optimal_value

        return RQAOAResult(
            solution_bitstring=full_solution,
            cut_value=cut_value,
            optimal_value=optimal_value,
            approximation_ratio=approximation_ratio,
            n_levels=n_levels,
            original_size=len(original_nodes),
            reduced_size=working_graph.number_of_nodes(),
            reduction_constant=float(working_graph.graph.get("constant_offset", 0.0)),
            eliminated_vars=elimination_history,
            runtime=time.time() - start_time,
        )

    def _run_qaoa_on_subgraph(self, graph: nx.Graph) -> Dict[str, Any]:
        """Optimize QAOA on the current working graph and estimate correlations."""
        relabeled_graph, node_order = self._prepare_contiguous_graph(graph)

        if self.qaoa_evaluator is not None:
            correlation_matrix = self.qaoa_evaluator(relabeled_graph)
            return {
                "correlations": correlation_matrix,
                "node_order": node_order,
                "solution_bitstring": None,
                "cut_value": None,
            }

        builder = HamiltonianBuilder()
        hamiltonian, offset = builder.build_maxcut_hamiltonian(relabeled_graph)
        total_offset = float(offset) + float(relabeled_graph.graph.get("constant_offset", 0.0))
        circuit_builder = QAOACircuitBuilder(
            n_qubits=relabeled_graph.number_of_nodes(),
            p=self.p,
        )

        def objective_function(params: np.ndarray) -> float:
            from qiskit.quantum_info import Statevector

            circuit = circuit_builder.build_qaoa_circuit_multilayer(
                relabeled_graph,
                gammas=params[0::2].tolist(),
                betas=params[1::2].tolist(),
            )
            expectation = float(
                np.real(Statevector.from_instruction(circuit).expectation_value(hamiltonian))
            )
            return -(expectation + total_offset)

        optimizer = self.optimizer or QAOAOptimizer(
            p=self.p,
            optimizer_type="COBYLA",
            maxiter=120,
            n_initial_points=2,
            seed=42,
        )
        opt_result = optimizer.optimize(
            objective_function=objective_function,
            n_qubits=relabeled_graph.number_of_nodes(),
            graph=relabeled_graph,
        )
        correlation_matrix = self._compute_pair_correlations(
            relabeled_graph,
            opt_result.optimal_params,
        )

        reduced_cut_value = None
        if opt_result.cut_value is not None:
            reduced_cut_value = float(opt_result.cut_value) + float(
                relabeled_graph.graph.get("constant_offset", 0.0)
            )

        return {
            "correlations": correlation_matrix,
            "node_order": node_order,
            "solution_bitstring": opt_result.solution_bitstring,
            "cut_value": reduced_cut_value,
            "optimal_params": opt_result.optimal_params,
        }

    def _compute_pair_correlations(self, graph: nx.Graph, params: np.ndarray) -> np.ndarray:
        """Compute ``<Z_i Z_j>`` correlations for the optimized QAOA state."""
        from qiskit.quantum_info import SparsePauliOp, Statevector

        n_qubits = graph.number_of_nodes()
        builder = QAOACircuitBuilder(n_qubits=n_qubits, p=self.p)
        circuit = builder.build_qaoa_circuit_multilayer(
            graph,
            gammas=params[0::2].tolist(),
            betas=params[1::2].tolist(),
        )
        statevector = Statevector.from_instruction(circuit)

        correlations = np.eye(n_qubits)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                pauli = ["I"] * n_qubits
                pauli[i] = "Z"
                pauli[j] = "Z"
                operator = SparsePauliOp(["".join(reversed(pauli))])
                corr = float(np.real(statevector.expectation_value(operator)))
                correlations[i, j] = corr
                correlations[j, i] = corr
        return correlations

    def _analyze_correlations(self, qaoa_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank candidate eliminations using absolute pair correlations."""
        correlations = qaoa_result["correlations"]
        node_order = qaoa_result["node_order"]
        n_qubits = correlations.shape[0]

        ranked: List[Dict[str, Any]] = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                corr = float(correlations[i, j])
                magnitude = abs(corr)
                if magnitude < self.correlation_threshold:
                    continue
                ranked.append(
                    {
                        "var1": node_order[i],
                        "var2": node_order[j],
                        "correlation": magnitude,
                        "signed_correlation": corr,
                        "relationship": "same" if corr >= 0 else "opposite",
                        "selection_reason": "threshold",
                    }
                )

        ranked.sort(key=lambda item: item["correlation"], reverse=True)
        if ranked:
            return ranked

        if not self.force_fallback_elimination or n_qubits < 2:
            return []

        i, j = np.unravel_index(
            np.argmax(np.abs(correlations - np.eye(n_qubits))),
            correlations.shape,
        )
        if i == j:
            return []

        corr = float(correlations[i, j])
        return [
            {
                "var1": node_order[min(i, j)],
                "var2": node_order[max(i, j)],
                "correlation": abs(corr),
                "signed_correlation": corr,
                "relationship": "same" if corr >= 0 else "opposite",
                "selection_reason": "fallback-strongest-pair",
            }
        ]

    def _select_eliminations(self, correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select non-overlapping eliminations from a ranked correlation list."""
        selected: List[Dict[str, Any]] = []
        used_nodes = set()

        for candidate in correlations:
            keep = candidate["var1"]
            remove = candidate["var2"]
            if keep in used_nodes or remove in used_nodes:
                continue
            selected.append(candidate)
            used_nodes.add(remove)
            if len(selected) >= self.n_eliminate:
                break

        return selected

    def _reduce_problem(
        self,
        graph: nx.Graph,
        eliminations: List[Dict[str, Any]],
    ) -> nx.Graph:
        """
        Reduce the graph by substituting eliminated variables into their keeper.

        ``same`` correlations transfer the incident edge weight directly.
        ``opposite`` correlations negate the transferred edge and add an
        assignment-independent constant term that is tracked in
        ``graph.graph["constant_offset"]``.
        """
        reduced = copy.deepcopy(graph)
        constant_offset = float(reduced.graph.get("constant_offset", 0.0))

        for elimination in eliminations:
            keep = elimination["var1"]
            remove = elimination["var2"]
            relationship = elimination["relationship"]

            if keep not in reduced or remove not in reduced:
                continue

            sign = 1.0 if relationship == "same" else -1.0
            for neighbor, edge_data in list(reduced[remove].items()):
                weight = float(edge_data.get("weight", 1.0))

                if relationship == "opposite":
                    constant_offset += weight

                if neighbor == keep:
                    continue

                delta = sign * weight
                if reduced.has_edge(keep, neighbor):
                    reduced[keep][neighbor]["weight"] = (
                        float(reduced[keep][neighbor].get("weight", 1.0)) + delta
                    )
                else:
                    reduced.add_edge(keep, neighbor, weight=delta)

                if (
                    reduced.has_edge(keep, neighbor)
                    and abs(reduced[keep][neighbor]["weight"]) < 1e-12
                ):
                    reduced.remove_edge(keep, neighbor)

            reduced.remove_node(remove)

        reduced.graph["constant_offset"] = constant_offset

        logger.info(
            "Reduced problem: %s -> %s active nodes",
            graph.number_of_nodes(),
            reduced.number_of_nodes(),
        )
        return reduced

    def _solve_reduced_problem(self, graph: nx.Graph) -> Dict[str, Any]:
        """Solve the reduced graph exactly when feasible, else with QAOA."""
        if graph.number_of_nodes() == 0:
            return {"assignments": {}, "bitstring": "", "cut_value": 0.0}

        relabeled_graph, node_order = self._prepare_contiguous_graph(graph)
        n_nodes = relabeled_graph.number_of_nodes()
        reduction_constant = float(relabeled_graph.graph.get("constant_offset", 0.0))

        if n_nodes <= 20:
            exact_result = ClassicalSolver().solve_exact(relabeled_graph)
            bitstring = exact_result.optimal_bitstrings[0]
            assignments = {
                node_order[index]: bitstring[index] for index in range(len(bitstring))
            }
            return {
                "assignments": assignments,
                "bitstring": bitstring,
                "cut_value": float(exact_result.optimal_value) + reduction_constant,
            }

        builder = HamiltonianBuilder()
        hamiltonian, offset = builder.build_maxcut_hamiltonian(relabeled_graph)
        total_offset = float(offset) + reduction_constant
        circuit_builder = QAOACircuitBuilder(n_qubits=n_nodes, p=self.p)

        def objective_function(params: np.ndarray) -> float:
            from qiskit.quantum_info import Statevector

            circuit = circuit_builder.build_qaoa_circuit_multilayer(
                relabeled_graph,
                gammas=params[0::2].tolist(),
                betas=params[1::2].tolist(),
            )
            expectation = float(
                np.real(Statevector.from_instruction(circuit).expectation_value(hamiltonian))
            )
            return -(expectation + total_offset)

        optimizer = self.optimizer or QAOAOptimizer(
            p=self.p,
            optimizer_type="COBYLA",
            maxiter=120,
            n_initial_points=2,
            seed=42,
        )
        qaoa_result = optimizer.optimize(
            objective_function=objective_function,
            n_qubits=n_nodes,
            graph=relabeled_graph,
        )
        assignments = {
            node_order[index]: qaoa_result.solution_bitstring[index]
            for index in range(len(qaoa_result.solution_bitstring or ""))
        }
        reduced_cut_value = None
        if qaoa_result.cut_value is not None:
            reduced_cut_value = float(qaoa_result.cut_value) + reduction_constant
        return {
            "assignments": assignments,
            "bitstring": qaoa_result.solution_bitstring,
            "cut_value": reduced_cut_value,
        }

    def _reconstruct_solution(
        self,
        final_solution: Dict[str, Any],
        eliminated_vars: List[Dict[str, Any]],
        original_nodes: List[int],
    ) -> str:
        """Reconstruct the original bitstring from the reduced solution."""
        assignment_map: Dict[int, str] = dict(final_solution.get("assignments", {}))

        for elimination in reversed(eliminated_vars):
            keep = elimination["var1"]
            remove = elimination["var2"]
            keep_value = assignment_map.get(keep, "0")
            assignment_map[remove] = (
                keep_value
                if elimination["relationship"] == "same"
                else self._flip_bit(keep_value)
            )

        for node in original_nodes:
            assignment_map.setdefault(node, "0")

        return "".join(assignment_map[node] for node in original_nodes)

    @staticmethod
    def _flip_bit(bit: str) -> str:
        return "1" if bit == "0" else "0"

    @staticmethod
    def _prepare_contiguous_graph(graph: nx.Graph) -> Tuple[nx.Graph, List[int]]:
        """Relabel a graph to consecutive integer nodes for Qiskit routines."""
        node_order = list(sorted(graph.nodes()))
        mapping = {node: index for index, node in enumerate(node_order)}
        relabeled = nx.relabel_nodes(graph, mapping, copy=True)
        relabeled.graph.update(copy.deepcopy(graph.graph))
        return relabeled, node_order

    @staticmethod
    def _calculate_cut(
        graph: nx.Graph,
        bitstring: str,
        node_order: Optional[List[int]] = None,
    ) -> float:
        """Calculate the weighted cut value on the original graph."""
        if node_order is None:
            node_order = list(sorted(graph.nodes()))
        index_map = {node: idx for idx, node in enumerate(node_order)}

        cut_value = 0.0
        for u, v in graph.edges():
            left = bitstring[index_map[u]]
            right = bitstring[index_map[v]]
            if left != right:
                cut_value += float(graph[u][v].get("weight", 1.0))
        return cut_value


class AdaptiveRQAOA(RQAOAEngine):
    """Adaptive-threshold variant of RQAOA."""

    def __init__(
        self,
        p: int = 1,
        correlation_threshold: float = 0.8,
        min_problem_size: int = 4,
        max_depth: int = 5,
    ) -> None:
        super().__init__(
            p=p,
            n_eliminate_per_step=1,
            correlation_threshold=correlation_threshold,
            min_problem_size=min_problem_size,
            max_depth=max_depth,
        )
        self.threshold_history = [correlation_threshold]

    def adapt_threshold(self, correlation_strength: float) -> None:
        """Adapt the threshold based on observed correlation strengths."""
        if correlation_strength < self.correlation_threshold * 0.5:
            self.correlation_threshold *= 0.8
        elif correlation_strength > self.correlation_threshold * 1.5:
            self.correlation_threshold *= 1.1

        self.threshold_history.append(self.correlation_threshold)
        logger.info("Adapted RQAOA threshold to %.3f", self.correlation_threshold)
