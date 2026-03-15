"""Fermion-to-qubit mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper, QubitMapper
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem


@dataclass
class HamiltonianMapping:
    """Mapped Hamiltonian and mapping diagnostics."""

    second_q_operator: SparseLabelOp
    qubit_operator: SparsePauliOp
    mapper: QubitMapper
    qubits_full: int
    qubits_reduced: int
    two_qubit_reduction_used: bool


def get_mapper_by_name(name: str, num_particles: Optional[Tuple[int, int]] = None) -> QubitMapper:
    """Return configured QubitMapper by name for research benchmarking."""
    name_norm = name.lower().strip()
    if name_norm == "parity":
        return ParityMapper(num_particles=num_particles)
    if name_norm == "jordan_wigner":
        return JordanWignerMapper()
    if name_norm == "bravyi_kitaev":
        return BravyiKitaevMapper()
    raise ValueError(f"Unknown mapping: {name}")


def build_mapped_hamiltonian(
    problem: ElectronicStructureProblem, two_qubit_reduction: bool = True, mapping_name: str = "parity"
) -> HamiltonianMapping:
    """Map second-quantized Hamiltonian with configurable mapper for research-level benchmarks."""
    second_q_op = problem.hamiltonian.second_q_op()

    # Get the mapper based on configuration
    mapper = get_mapper_by_name(mapping_name, num_particles=problem.num_particles if two_qubit_reduction else None)
    
    # Calculate full qubits for reference (using a non-reducing mapper)
    full_qubit = JordanWignerMapper().map(second_q_op)
    
    # Map with chosen mapper
    qubit_op = mapper.map(second_q_op)

    return HamiltonianMapping(
        second_q_operator=second_q_op,
        qubit_operator=qubit_op,
        mapper=mapper,
        qubits_full=full_qubit.num_qubits,
        qubits_reduced=qubit_op.num_qubits,
        two_qubit_reduction_used=two_qubit_reduction and mapping_name.lower() == "parity",
    )


def get_qubit_operator(problem: ElectronicStructureProblem) -> SparsePauliOp:
    """Compatibility wrapper returning mapped qubit operator."""
    return build_mapped_hamiltonian(problem).qubit_operator


def get_mapper(problem: ElectronicStructureProblem) -> ParityMapper:
    """Compatibility wrapper returning configured parity mapper."""
    return ParityMapper(num_particles=problem.num_particles)
