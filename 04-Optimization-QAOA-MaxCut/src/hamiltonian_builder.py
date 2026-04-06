"""Hamiltonian builders for Max-Cut and related helper utilities."""

import logging
from typing import Tuple

import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """
    Build Ising interaction operators for Max-Cut.

    For a weighted graph, the Max-Cut objective is

        C = sum_(i,j in E) w_ij (1 - Z_i Z_j) / 2
          = sum_(i,j in E) w_ij / 2 - sum_(i,j in E) w_ij Z_i Z_j / 2.

    This module returns the ZZ interaction term and the constant offset
    separately, so the cut value of a state is

        cut_value = offset + <interaction_term>.
    """

    def __init__(self) -> None:
        logger.info("HamiltonianBuilder initialized")

    def build_maxcut_hamiltonian(
        self,
        graph: nx.Graph,
    ) -> Tuple[SparsePauliOp, float]:
        """Build the ZZ interaction term and constant offset for Max-Cut."""
        n_qubits = graph.number_of_nodes()
        edges = list(graph.edges())
        weights = self._edge_weights(graph, edges)

        logger.info(
            "Building Max-Cut Hamiltonian for %s qubits and %s edges",
            n_qubits,
            len(edges),
        )

        pauli_list = []
        coeffs = []

        for i, j in edges:
            weight = weights[(i, j)]

            pauli_str = ["I"] * n_qubits
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"
            pauli_list.append("".join(reversed(pauli_str)))
            coeffs.append(-weight / 2.0)

        offset = sum(weights.values()) / 2.0

        if pauli_list:
            hamiltonian = SparsePauliOp(pauli_list, coeffs=coeffs).simplify()
        else:
            hamiltonian = SparsePauliOp(["I" * n_qubits], coeffs=[0.0])

        logger.info(
            "Hamiltonian built with %s terms and offset %.4f",
            len(hamiltonian),
            offset,
        )
        return hamiltonian, float(offset)

    def build_maxcut_hamiltonian_qubo(
        self,
        graph: nx.Graph,
    ) -> Tuple[SparsePauliOp, float]:
        """
        Build the Max-Cut Ising operator starting from the binary QUBO form.

        For ``x_i in {0, 1}``, one edge contributes

            x_i + x_j - 2 x_i x_j.

        Using ``z_i = 2 x_i - 1`` and ``x_i = (1 + z_i) / 2`` gives

            x_i + x_j - 2 x_i x_j = (1 - z_i z_j) / 2.

        The linear ``z_i`` terms cancel exactly, so the QUBO and Ising
        constructions reduce to the same constant-plus-ZZ operator.
        """
        logger.info("Building QUBO-derived Max-Cut Hamiltonian")
        return self.build_maxcut_hamiltonian(graph)

    def evaluate_expectation(
        self,
        hamiltonian: SparsePauliOp,
        statevector: np.ndarray,
    ) -> float:
        """Evaluate ``<psi|H|psi>`` for a supplied statevector."""
        return float(np.real(np.conj(statevector) @ hamiltonian.to_matrix() @ statevector))

    def get_cost_from_bitstring(
        self,
        graph: nx.Graph,
        bitstring: str,
    ) -> float:
        """Compute the weighted Max-Cut objective for a bitstring."""
        n = len(bitstring)
        cost = 0.0

        for i, j in graph.edges():
            if i < n and j < n and bitstring[i] != bitstring[j]:
                cost += float(graph[i][j].get("weight", 1.0))

        return cost

    def bitstring_to_partition(self, bitstring: str) -> Tuple[set, set]:
        """Convert a node-order bitstring into the two cut partitions."""
        partition_a = set()
        partition_b = set()

        for i, bit in enumerate(bitstring):
            if bit == "0":
                partition_a.add(i)
            else:
                partition_b.add(i)

        return partition_a, partition_b

    def create_mixer_hamiltonian(self, n_qubits: int) -> SparsePauliOp:
        """Create the standard X-mixer Hamiltonian ``sum_i X_i``."""
        logger.info("Creating mixer Hamiltonian for %s qubits", n_qubits)

        pauli_list = []
        coeffs = []

        for i in range(n_qubits):
            pauli_str = ["I"] * n_qubits
            pauli_str[i] = "X"
            pauli_list.append("".join(reversed(pauli_str)))
            coeffs.append(1.0)

        return SparsePauliOp(pauli_list, coeffs=coeffs)

    def compute_classical_cost(
        self,
        graph: nx.Graph,
        x: np.ndarray,
    ) -> float:
        """Compute the weighted Max-Cut objective for binary variables."""
        z = 2 * np.asarray(x, dtype=float) - 1

        cost = 0.0
        for i, j in graph.edges():
            weight = float(graph[i][j].get("weight", 1.0))
            cost += weight * (1.0 - z[i] * z[j]) / 2.0

        return float(cost)

    @staticmethod
    def _edge_weights(
        graph: nx.Graph,
        edges: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """Return a normalized edge-weight map keyed by the graph edge tuples."""
        if nx.is_weighted(graph):
            return {
                edge: float(graph[edge[0]][edge[1]].get("weight", 1.0))
                for edge in edges
            }
        return {edge: 1.0 for edge in edges}
