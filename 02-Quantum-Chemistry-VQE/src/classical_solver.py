"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Classical exact baselines (HF, CISD, FCI) via PySCF."""

from __future__ import annotations

import importlib.util

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

# ---------------------------------------------------------------------------
# Exact diagonalization (FCI equivalent on the qubit Hamiltonian)
# ---------------------------------------------------------------------------

def get_exact_energy_from_qubit_operator(qubit_operator: SparsePauliOp) -> float:
    """Compute exact minimum eigenvalue of a qubit Hamiltonian (FCI-equivalent)."""
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(qubit_operator)
    return float(result.eigenvalue.real)


def get_exact_energy(problem: ElectronicStructureProblem) -> float:
    """Compatibility helper returning exact total energy (electronic + constants)."""
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_operator = mapper.map(problem.hamiltonian.second_q_op())
    electronic_energy = get_exact_energy_from_qubit_operator(qubit_operator)
    total_constant = sum(problem.hamiltonian.constants.values())
    return float(electronic_energy + total_constant)


# ---------------------------------------------------------------------------
# PySCF-based classical methods
# ---------------------------------------------------------------------------

HAS_PYSCF = importlib.util.find_spec("pyscf") is not None


def run_hartree_fock(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
) -> float:
    """Compute Hartree-Fock energy via PySCF.

    Returns float('nan') if PySCF is unavailable.
    """
    if not HAS_PYSCF:
        return float("nan")
    import pyscf
    from .molecule_driver import _build_atom_string

    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(
        atom=atom_str, basis=basis, charge=charge, spin=spin, unit="angstrom", verbose=0
    )
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    return float(mf.kernel())


def run_cisd(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
) -> float:
    """Compute CISD energy (HF + singles/doubles correlation) via PySCF.

    Returns float('nan') if PySCF is unavailable.
    """
    if not HAS_PYSCF:
        return float("nan")
    import pyscf
    import pyscf.ci
    from .molecule_driver import _build_atom_string

    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(
        atom=atom_str, basis=basis, charge=charge, spin=spin, unit="angstrom", verbose=0
    )
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    myci = pyscf.ci.CISD(mf)
    myci.verbose = 0
    myci.kernel()
    return float(mf.e_tot + myci.e_corr)


def run_fci(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
) -> float:
    """Compute Full CI energy via PySCF (gold-standard classical reference).

    Returns float('nan') if PySCF is unavailable.
    This is the exact solution in the chosen basis set, equivalent to
    NumPy diagonalization of the second-quantized Hamiltonian.
    """
    if not HAS_PYSCF:
        return float("nan")
    import pyscf
    import pyscf.fci
    from .molecule_driver import _build_atom_string

    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(
        atom=atom_str, basis=basis, charge=charge, spin=spin, unit="angstrom", verbose=0
    )
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    cisolver = pyscf.fci.FCI(mf)
    cisolver.verbose = 0
    e_fci, _ = cisolver.kernel()
    return float(e_fci)


def run_mp2(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
):
    """Compute MP2 energy and t2 amplitudes for warm-starting VQE.
    
    Returns a tuple (mp2_energy, t2_amplitudes).
    Returns (float('nan'), None) if PySCF is unavailable.
    """
    if not HAS_PYSCF:
        return float("nan"), None
        
    import pyscf
    import pyscf.mp
    from .molecule_driver import _build_atom_string

    atom_str = _build_atom_string(molecule_name, bond_length)
    mol = pyscf.gto.M(
        atom=atom_str, basis=basis, charge=charge, spin=spin, unit="angstrom", verbose=0
    )
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    
    pt = pyscf.mp.MP2(mf)
    pt.verbose = 0
    e_mp2, t2 = pt.kernel()
    
    return float(mf.e_tot + e_mp2), t2
