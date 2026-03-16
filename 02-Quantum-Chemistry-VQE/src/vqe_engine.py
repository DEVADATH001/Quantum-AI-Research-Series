"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: VQE execution wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, SLSQP

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
    """Reusable VQE runner with hybrid system monitoring."""

    def __init__(
        self,
        estimator: Any,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Any] = None,
        maxiter: int = 80,
        energy_shift: float = 0.0,
    ) -> None:
        self.estimator = estimator
        self.ansatz = ansatz
        self.energy_shift = energy_shift
        # Default to SLSQP for stability in statevector simulations if not specified
        self.optimizer = optimizer if optimizer is not None else SLSQP(maxiter=maxiter)
        self.callback = VQECallback(energy_shift=energy_shift)
        self._vqe: Optional[VQE] = None
        if ansatz is not None:
            self.initialize_vqe(ansatz=ansatz, optimizer=self.optimizer)

    def initialize_vqe(self, ansatz: QuantumCircuit, optimizer: Optional[Any] = None, energy_shift: Optional[float] = None) -> VQE:
        """Initialize VQE primitive with optional updated energy shift for hybrid monitoring."""
        if optimizer is None:
            optimizer = self.optimizer
        if energy_shift is not None:
            self.energy_shift = energy_shift
            
        self.callback = VQECallback(energy_shift=self.energy_shift)
        self.ansatz = ansatz
        self.optimizer = optimizer
        self._vqe = VQE(
            estimator=self.estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=self.callback,
        )
        return self._vqe

    def run_vqe_qubit(
        self, 
        qubit_operator: SparsePauliOp, 
        ansatz: Optional[QuantumCircuit] = None,
        initial_point: Optional[List[float]] = None
    ) -> VQEResultRecord:
        """Run VQE with optional warm-start parameter support."""
        if ansatz is not None:
            self.initialize_vqe(ansatz=ansatz)
        if self._vqe is None:
            raise RuntimeError("VQE is not initialized. Provide an ansatz first.")

        # Update initial point if provided for research warm-starts
        if initial_point is not None:
            self._vqe.initial_point = initial_point

        result = self._vqe.compute_minimum_eigenvalue(qubit_operator)
        optimizer_result = getattr(result, "optimizer_result", None)

        if optimizer_result is None:
            evaluations = len(self.callback.history)
            iterations = len(self.callback.history)
        else:
            evaluations = int(getattr(optimizer_result, "nfev", len(self.callback.history)))
            iterations = int(getattr(optimizer_result, "nit", len(self.callback.history)))

        optimal_point = [float(x) for x in np.asarray(result.optimal_point, dtype=float)]
        # Add the energy shift to the final result for consistency
        total_energy = float(result.optimal_value) + self.energy_shift
        return VQEResultRecord(
            energy=total_energy,
            optimal_point=optimal_point,
            evaluations=evaluations,
            iterations=iterations,
            history=self.callback.get_history(),
        )

    def run_vqe(self, problem: Any, mapper: Any) -> VQEResultRecord:
        """Compatibility wrapper that maps from problem to qubit operator and adds constants."""
        qubit_operator = mapper.map(problem.hamiltonian.second_q_op())
        result = self.run_vqe_qubit(qubit_operator=qubit_operator)
        
        # Add all Hamiltonian constants (nuclear repulsion, etc.)
        total_constant = sum(problem.hamiltonian.constants.values())
        result.energy += total_constant
        return result

    def collect_results(self) -> Dict[str, Any]:
        """Return callback trace in legacy format."""
        return {"history": self.callback.get_history()}
