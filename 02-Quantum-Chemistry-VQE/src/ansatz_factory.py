"""Ansatz builders for VQE experiments."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from .interfaces import AbstractAnsatzFactory


def build_uccsd_ansatz(problem: ElectronicStructureProblem, mapper: QubitMapper) -> QuantumCircuit:
    """Build UCCSD with Hartree-Fock initial state."""
    hf = HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )
    return UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=hf,
    )


def build_hardware_efficient_ansatz(
    num_qubits: int, reps: int = 3, entanglement: str = "circular"
) -> QuantumCircuit:
    """Build EfficientSU2-style circuit with RY and CX blocks."""
    return efficient_su2(
        num_qubits=num_qubits,
        su2_gates=["ry"],
        entanglement=entanglement,
        reps=reps,
    )


def get_ansatz(
    name: str, problem: ElectronicStructureProblem, mapper: QubitMapper, **kwargs
) -> QuantumCircuit:
    """Factory wrapper returning the requested ansatz."""
    normalized = name.strip().upper()
    if normalized == "UCCSD":
        return build_uccsd_ansatz(problem, mapper)
    if normalized == "EFFICIENTSU2":
        num_qubits = mapper.map(problem.hamiltonian.second_q_op()).num_qubits
        reps = int(kwargs.get("reps", 3))
        entanglement = str(kwargs.get("entanglement", "circular"))
        return build_hardware_efficient_ansatz(num_qubits, reps=reps, entanglement=entanglement)
    raise ValueError(f"Unknown ansatz name: {name}")


class AnsatzFactory(AbstractAnsatzFactory):
    """Class-based ansatz factory for extension by inheritance."""

    def build(self, name: str, problem: ElectronicStructureProblem, mapper: QubitMapper, **kwargs) -> QuantumCircuit:
        return get_ansatz(name=name, problem=problem, mapper=mapper, **kwargs)
