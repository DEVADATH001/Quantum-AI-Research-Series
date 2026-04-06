"""QAOA circuit builders for Max-Cut."""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class QAOACircuitBuilder:
    """Build parameterized QAOA circuits for Max-Cut instances."""

    def __init__(self, n_qubits: int, p: int = 1) -> None:
        self.n_qubits = n_qubits
        self.p = p
        self.rng = np.random.default_rng()
        self.gammas = ParameterVector("gamma", p)
        self.betas = ParameterVector("beta", p)

        logger.info(
            "QAOACircuitBuilder initialized: n_qubits=%s, p=%s",
            n_qubits,
            p,
        )

    def build_qaoa_circuit(
        self,
        cost_hamiltonian: SparsePauliOp,
        initial_state: Optional[QuantumCircuit] = None,
    ) -> QuantumCircuit:
        """
        Build a parameterized QAOA circuit from an interaction Hamiltonian.

        ``HamiltonianBuilder.build_maxcut_hamiltonian`` returns only the
        non-constant ZZ interaction term. For an edge coefficient ``-w / 2``,
        exponentiating the interaction gives

            exp(-i gamma (-w / 2) Z_i Z_j) = exp(+i gamma w Z_i Z_j / 2).

        Qiskit's ``RZZ(theta)`` gate implements ``exp(-i theta Z_i Z_j / 2)``,
        so the correct Max-Cut angle is ``theta = -gamma * w``.
        """
        logger.info("Building QAOA circuit with p=%s layers", self.p)

        qc = QuantumCircuit(self.n_qubits)

        if initial_state is None:
            for qubit in range(self.n_qubits):
                qc.h(qubit)
        else:
            qc.compose(initial_state, inplace=True)

        for layer in range(self.p):
            gamma = self.gammas[layer]
            beta = self.betas[layer]

            qc.compose(self._build_cost_unitary(cost_hamiltonian, gamma), inplace=True)
            qc.compose(self._build_mixer_unitary(beta), inplace=True)

        logger.info(
            "QAOA circuit built with %s qubits and depth %s",
            qc.num_qubits,
            qc.depth(),
        )
        return qc

    def _build_cost_unitary(
        self,
        hamiltonian: SparsePauliOp,
        gamma: float,
    ) -> QuantumCircuit:
        """Build the ZZ interaction unitary for the supplied Hamiltonian."""
        qc = QuantumCircuit(self.n_qubits)

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            indices = [i for i, c in enumerate(reversed(pauli.to_label())) if c == "Z"]
            real_coeff = float(np.real(coeff))

            if len(indices) == 2:
                i, j = indices
                qc.rzz(2 * gamma * real_coeff, i, j)
            elif len(indices) == 1:
                i = indices[0]
                qc.rz(2 * gamma * real_coeff, i)

        return qc

    def _build_mixer_unitary(self, beta: float) -> QuantumCircuit:
        """Build the standard X-mixer ``exp(-i beta sum_i X_i)``."""
        qc = QuantumCircuit(self.n_qubits)
        for qubit in range(self.n_qubits):
            qc.rx(2 * beta, qubit)
        return qc

    def build_qaoa_circuit_simple(
        self,
        graph: nx.Graph,
        gamma: float,
        beta: float,
    ) -> QuantumCircuit:
        """Build a single-layer Max-Cut QAOA circuit directly from a graph."""
        qc = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):
            qc.h(qubit)

        for i, j in graph.edges():
            qc.rzz(self._maxcut_rzz_angle(graph, i, j, gamma), i, j)

        for qubit in range(self.n_qubits):
            qc.rx(2 * beta, qubit)

        return qc

    def build_qaoa_circuit_multilayer(
        self,
        graph: nx.Graph,
        gammas: List[float],
        betas: List[float],
    ) -> QuantumCircuit:
        """Build a multi-layer Max-Cut QAOA circuit directly from a graph."""
        if len(gammas) != len(betas):
            raise ValueError(
                f"gammas and betas must have same length, got {len(gammas)} and {len(betas)}"
            )

        qc = QuantumCircuit(self.n_qubits)
        for qubit in range(self.n_qubits):
            qc.h(qubit)

        for gamma, beta in zip(gammas, betas):
            for i, j in graph.edges():
                qc.rzz(self._maxcut_rzz_angle(graph, i, j, gamma), i, j)

            for qubit in range(self.n_qubits):
                qc.rx(2 * beta, qubit)

        return qc

    def get_initial_parameters(self, strategy: str = "random") -> np.ndarray:
        """Generate initial QAOA parameters."""
        params = np.zeros(2 * self.p)

        if strategy == "random":
            params = self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
        elif strategy == "linear":
            for i in range(self.p):
                params[2 * i] = np.pi / 2
                params[2 * i + 1] = np.pi / 4
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info("Generated initial parameters using strategy=%s", strategy)
        return params

    def parameter_bounds(self) -> Tuple[List[float], List[float]]:
        """Get default parameter bounds for optimization."""
        lower = [0.0] * (2 * self.p)
        upper = [2 * np.pi] * (2 * self.p)
        return lower, upper

    def circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """Export a circuit to QASM."""
        return circuit.qasm()

    def count_gates(self, circuit: QuantumCircuit) -> dict:
        """Count gate occurrences in a circuit."""
        from collections import Counter

        gate_counts = Counter()
        for instruction in circuit.data:
            gate_counts[instruction[0].name] += 1
        return dict(gate_counts)

    @staticmethod
    def _edge_weight(graph: nx.Graph, i: int, j: int) -> float:
        """Return the edge weight used by the Max-Cut objective."""
        return float(graph[i][j].get("weight", 1.0))

    @classmethod
    def _maxcut_rzz_angle(
        cls,
        graph: nx.Graph,
        i: int,
        j: int,
        gamma: float,
    ) -> float:
        """Map a Max-Cut edge term onto Qiskit's ``RZZ`` convention."""
        return -float(gamma) * cls._edge_weight(graph, i, j)


class QAOACircuitFactory:
    """Factory helpers for alternate QAOA-style circuit components."""

    @staticmethod
    def create_xy_mixer(
        n_qubits: int,
        graph: nx.Graph,
        beta: float = np.pi / 4,
    ) -> QuantumCircuit:
        """
        Create an XY-mixer circuit.

        The implemented unitary is

            exp[-i beta sum_(i,j in E) (X_i X_j + Y_i Y_j)].

        On a given edge, the XX and YY terms commute, so applying
        ``RXX(2 * beta)`` and ``RYY(2 * beta)`` sequentially gives the exact
        two-body XY evolution for that pair.
        """
        qc = QuantumCircuit(n_qubits)
        for i, j in graph.edges():
            qc.rxx(2 * beta, i, j)
            qc.ryy(2 * beta, i, j)
        return qc

    @staticmethod
    def create_hea_qaoa(
        n_qubits: int,
        p: int,
        layers: Optional[List[int]] = None,
    ) -> QuantumCircuit:
        """Create a simple hardware-efficient ansatz style circuit."""
        del layers
        qc = QuantumCircuit(n_qubits)

        for qubit in range(n_qubits):
            qc.ry(np.pi / 4, qubit)

        for _ in range(p):
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)

            for qubit in range(n_qubits):
                qc.ry(np.random.uniform(0, 2 * np.pi), qubit)

        return qc
