"""Fermion-to-qubit mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem


@dataclass
class HamiltonianMapping:
    """Mapped Hamiltonian and mapping diagnostics."""

    second_q_operator: SparseLabelOp
    qubit_operator: SparsePauliOp
    mapper: ParityMapper
    qubits_full: int
    qubits_reduced: int
    two_qubit_reduction_used: bool


def build_mapped_hamiltonian(
    problem: ElectronicStructureProblem, two_qubit_reduction: bool = True
) -> HamiltonianMapping:
    """Map second-quantized Hamiltonian with ParityMapper."""
    second_q_op = problem.hamiltonian.second_q_op()

    full_mapper = ParityMapper()
    full_qubit = full_mapper.map(second_q_op)

    if two_qubit_reduction:
        mapper = ParityMapper(num_particles=problem.num_particles)
        qubit_op = mapper.map(second_q_op)
    else:
        mapper = full_mapper
        qubit_op = full_qubit

    return HamiltonianMapping(
        second_q_operator=second_q_op,
        qubit_operator=qubit_op,
        mapper=mapper,
        qubits_full=full_qubit.num_qubits,
        qubits_reduced=qubit_op.num_qubits,
        two_qubit_reduction_used=two_qubit_reduction,
    )


def get_qubit_operator(problem: ElectronicStructureProblem) -> SparsePauliOp:
    """Compatibility wrapper returning mapped qubit operator."""
    return build_mapped_hamiltonian(problem).qubit_operator


def get_mapper(problem: ElectronicStructureProblem) -> ParityMapper:
    """Compatibility wrapper returning configured parity mapper."""
    return ParityMapper(num_particles=problem.num_particles)
