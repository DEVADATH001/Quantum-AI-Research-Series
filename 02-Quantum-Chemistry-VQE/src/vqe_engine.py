"""VQE execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA

from .optimizer_callbacks import VQECallback


@dataclass
class VQEResultRecord:
    """Serializable VQE result object."""

    energy: float
    optimal_point: List[float]
    evaluations: int
    iterations: int
    history: List[Dict[str, Any]]

    @property
    def total_energies(self) -> List[float]:
        """Compatibility property with legacy result handling."""
        return [self.energy]


class VQEEngine:
    """Reusable VQE runner."""

    def __init__(
        self,
        estimator: Any,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Any] = None,
        maxiter: int = 80,
    ) -> None:
        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer if optimizer is not None else SPSA(maxiter=maxiter)
        self.callback = VQECallback()
        self._vqe: Optional[VQE] = None
        if ansatz is not None:
            self.initialize_vqe(ansatz=ansatz, optimizer=self.optimizer)

    def initialize_vqe(self, ansatz: QuantumCircuit, optimizer: Optional[Any] = None) -> VQE:
        """Initialize VQE primitive."""
        if optimizer is None:
            optimizer = self.optimizer
        self.callback.clear()
        self.ansatz = ansatz
        self.optimizer = optimizer
        self._vqe = VQE(
            estimator=self.estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=self.callback,
        )
        return self._vqe

    def run_vqe_qubit(self, qubit_operator: SparsePauliOp, ansatz: Optional[QuantumCircuit] = None) -> VQEResultRecord:
        """Run VQE directly on a qubit Hamiltonian."""
        if ansatz is not None:
            self.initialize_vqe(ansatz=ansatz)
        if self._vqe is None:
            raise RuntimeError("VQE is not initialized. Provide an ansatz first.")

        result = self._vqe.compute_minimum_eigenvalue(qubit_operator)
        optimizer_result = getattr(result, "optimizer_result", None)

        if optimizer_result is None:
            evaluations = len(self.callback.history)
            iterations = len(self.callback.history)
        else:
            evaluations = int(getattr(optimizer_result, "nfev", len(self.callback.history)))
            iterations = int(getattr(optimizer_result, "nit", len(self.callback.history)))

        optimal_point = [float(x) for x in np.asarray(result.optimal_point, dtype=float)]
        return VQEResultRecord(
            energy=float(result.optimal_value),
            optimal_point=optimal_point,
            evaluations=evaluations,
            iterations=iterations,
            history=self.callback.get_history(),
        )

    def run_vqe(self, problem: Any, mapper: Any) -> VQEResultRecord:
        """Compatibility wrapper that maps from problem to qubit operator."""
        qubit_operator = mapper.map(problem.hamiltonian.second_q_op())
        return self.run_vqe_qubit(qubit_operator=qubit_operator)

    def collect_results(self) -> Dict[str, Any]:
        """Return callback trace in legacy format."""
        return {"history": self.callback.get_history()}
