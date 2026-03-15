"""Classical exact baselines."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


def get_exact_energy_from_qubit_operator(qubit_operator: SparsePauliOp) -> float:
    """Compute exact minimum eigenvalue of a qubit Hamiltonian."""
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(qubit_operator)
    return float(result.eigenvalue.real)


def get_exact_energy(problem: ElectronicStructureProblem) -> float:
    """Compatibility helper returning exact total energy (electronic + constants)."""
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_operator = mapper.map(problem.hamiltonian.second_q_op())
    electronic_energy = get_exact_energy_from_qubit_operator(qubit_operator)
    # Sum all constants (nuclear repulsion, frozen core shift, etc.)
    total_constant = sum(problem.hamiltonian.constants.values())
    return float(electronic_energy + total_constant)
