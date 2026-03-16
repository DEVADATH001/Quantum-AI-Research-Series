"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Ansatz builders for VQE experiments."""

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
    num_qubits: int, 
    reps: int = 3, 
    entanglement: str = "circular", 
    su2_gates: List[str] = ["ry"],
    initial_state: Optional[QuantumCircuit] = None
) -> QuantumCircuit:
    """Build EfficientSU2-style circuit with configurable rotation gates and physical initial state."""
    su2 = efficient_su2(
        num_qubits=num_qubits,
        su2_gates=su2_gates,
        entanglement=entanglement,
        reps=reps,
    )
    if initial_state is not None:
        return initial_state.compose(su2)
    return su2

def get_ansatz(
    name: str, problem: ElectronicStructureProblem, mapper: QubitMapper, **kwargs
) -> QuantumCircuit:
    """Factory wrapper returning the requested ansatz with full configuration passthrough."""
    normalized = name.strip().upper()
    if normalized == "UCCSD":
        return build_uccsd_ansatz(problem, mapper)
    if normalized == "EFFICIENTSU2":
        # Always prepend Hartree-Fock to ensure the optimizer starts from a physical state
        hf = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )
        num_qubits = mapper.map(problem.hamiltonian.second_q_op()).num_qubits
        reps = int(kwargs.get("reps", 3))
        entanglement = str(kwargs.get("entanglement", "circular"))
        su2_gates = list(kwargs.get("su2_gates", ["ry"]))
        return build_hardware_efficient_ansatz(
            num_qubits, reps=reps, entanglement=entanglement, su2_gates=su2_gates, initial_state=hf
        )
    raise ValueError(f"Unknown ansatz name: {name}")

class AnsatzFactory(AbstractAnsatzFactory):
    """Class-based ansatz factory for extension by inheritance."""

    def build(self, name: str, problem: ElectronicStructureProblem, mapper: QubitMapper, **kwargs) -> QuantumCircuit:
        return get_ansatz(name=name, problem=problem, mapper=mapper, **kwargs)
