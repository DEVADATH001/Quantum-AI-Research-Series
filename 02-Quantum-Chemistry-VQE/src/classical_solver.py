"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Classical exact baselines."""

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

import importlib.util
HAS_PYSCF = importlib.util.find_spec("pyscf") is not None

def run_hartree_fock(molecule_name: str, bond_length: float, basis: str = "sto3g", charge: int = 0, spin: int = 0) -> float:
    if not HAS_PYSCF:
        return float('nan')
    import pyscf
    from .molecule_driver import _build_atom_string
    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, unit='angstrom')
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    return float(mf.kernel())

def run_cisd(molecule_name: str, bond_length: float, basis: str = "sto3g", charge: int = 0, spin: int = 0) -> float:
    if not HAS_PYSCF:
        return float('nan')
    import pyscf
    import pyscf.ci
    from .molecule_driver import _build_atom_string
    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, unit='angstrom')
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    myci = pyscf.ci.CISD(mf)
    myci.verbose = 0
    myci.kernel()
    return float(mf.e_tot + myci.e_corr)
