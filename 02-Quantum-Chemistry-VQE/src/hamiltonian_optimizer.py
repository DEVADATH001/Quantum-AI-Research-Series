"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Hamiltonian Optimization (Tapering & Symmetry Reduction)."""

import logging
from typing import Tuple

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import QubitMapper, TaperedQubitMapper

logger = logging.getLogger(__name__)

class HamiltonianOptimizer:
    """Applies symmetry reduction and tapering to qubit Hamiltonians."""
    
    @staticmethod
    def apply_z2_tapering(
        problem: ElectronicStructureProblem,
        mapper: QubitMapper
    ) -> Tuple[SparsePauliOp, QubitMapper]:
        """
        Discovers Z2 symmetries in the Hamiltonian and tapers off qubits.
        
        Args:
            problem: The electronic structure problem
            mapper: The base qubit mapper (e.g., ParityMapper)
            
        Returns:
            Tuple of (tapered_qubit_operator, tapered_mapper)
        """
        logger.info("Analyzing Hamiltonian for Z2 symmetries...")
        
        # We need the unmodified qubit operator to find symmetries
        qubit_op = mapper.map(problem.hamiltonian.second_q_op())
        original_qubits = qubit_op.num_qubits
        
        try:
            # Qiskit Nature 0.6+ approach
            tapered_mapper = problem.get_tapered_mapper(mapper)
            tapered_op = tapered_mapper.map(problem.hamiltonian.second_q_op())
            new_qubits = tapered_op.num_qubits
            
            logger.info(f"Tapering successful: Reduced from {original_qubits} to {new_qubits} qubits.")
            return tapered_op, tapered_mapper
        except Exception as e:
            logger.warning(f"Z2 Tapering failed or no symmetries found: {e}. Returning original.")
            return qubit_op, mapper
