"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Molecule construction utilities for quantum chemistry problems.

Supported molecules:
    H2   — 2 qubits after ParityMapper + two-qubit reduction
    LiH  — 4 qubits after active-space + freeze-core + reduction
    BeH2 — 6 qubits after active-space + freeze-core + reduction
    H2O  — stub for future extension (8 qubits)
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import List, Optional, Tuple

import numpy as np
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit_nature.units import DistanceUnit

from .interfaces import AbstractMoleculeDriver

HAS_PYSCF = importlib.util.find_spec("pyscf") is not None

if HAS_PYSCF:
    from qiskit_nature.second_q.drivers import PySCFDriver


@dataclass
class MoleculeMetadata:
    """Metadata describing how a molecular problem was produced."""

    molecule: str
    bond_length: float
    basis: str
    source: str
    freeze_core_applied: bool
    active_space_applied: bool


def generate_distances(start: float, end: float, step: float) -> List[float]:
    """Return a closed bond-length grid."""
    if step <= 0:
        raise ValueError("step must be > 0")
    values: List[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


# ---------------------------------------------------------------------------
# Atom string builders
# ---------------------------------------------------------------------------

def _build_atom_string(molecule_name: str, bond_length: float) -> str:
    """Return PySCF atom string for a molecule at a given bond length."""
    name = molecule_name.upper()
    if name == "H2":
        return f"H 0 0 0; H 0 0 {bond_length}"
    if name == "LIH":
        return f"Li 0 0 0; H 0 0 {bond_length}"
    if name == "BEH2":
        # Linear BeH2: H-Be-H along z-axis
        return f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 -{bond_length}"
    if name == "H2O":
        # Water: O at origin, H at ±104.5°/2 at given O-H distance
        import math
        angle = math.radians(104.5 / 2)
        hx = bond_length * math.sin(angle)
        hz = bond_length * math.cos(angle)
        return f"O 0 0 0; H {hx} 0 {hz}; H -{hx} 0 {hz}"
    raise ValueError(f"Unsupported molecule: {molecule_name}")


# ---------------------------------------------------------------------------
# Synthetic fallback Hamiltonians (smoke-testing only — NOT for research)
# ---------------------------------------------------------------------------

def _synthetic_problem(
    molecule_name: str, bond_length: float, num_particles: Tuple[int, int], num_spatial_orbitals: int
) -> ElectronicStructureProblem:
    """Build deterministic synthetic fallback Hamiltonians for smoke testing.

    WARNING: These are lightweight surrogate Hamiltonians used strictly for
    environment validation when PySCF is unavailable. They lack physical
    scaling properties (R-dependency) and are NOT intended for research-grade
    simulations or publication. Results from this path are invalid scientifically.
    """
    name = molecule_name.upper()
    if name == "H2":
        h1_mat = np.array([[-1.10, -0.15], [-0.15, -0.35]], dtype=float)
        h2_mat = np.zeros((2, 2, 2, 2), dtype=float)
        h2_mat[0, 0, 0, 0] = 0.72
        h2_mat[1, 1, 1, 1] = 0.66
        h2_mat[0, 1, 1, 0] = 0.61
        h2_mat[1, 0, 0, 1] = 0.61
        scale = 0.74 / bond_length
    elif name == "LIH":
        h1_mat = np.array([[-0.95, -0.10], [-0.10, -0.25]], dtype=float)
        h2_mat = np.zeros((2, 2, 2, 2), dtype=float)
        h2_mat[0, 0, 0, 0] = 0.80
        h2_mat[1, 1, 1, 1] = 0.58
        h2_mat[0, 1, 1, 0] = 0.47
        h2_mat[1, 0, 0, 1] = 0.47
        scale = 1.6 / bond_length
    elif name == "BEH2":
        # Approximate 2-orbital active space surrogate for BeH2
        h1_mat = np.array([[-0.80, -0.08], [-0.08, -0.20]], dtype=float)
        h2_mat = np.zeros((2, 2, 2, 2), dtype=float)
        h2_mat[0, 0, 0, 0] = 0.65
        h2_mat[1, 1, 1, 1] = 0.50
        h2_mat[0, 1, 1, 0] = 0.38
        h2_mat[1, 0, 0, 1] = 0.38
        scale = 1.33 / bond_length
    else:
        raise ValueError(f"No synthetic fallback for molecule: {molecule_name}")

    h1_mat = h1_mat * scale
    h2_mat = h2_mat * (scale ** 1.1)
    nuclear_repulsion = 0.529177 / max(bond_length, 1e-6)

    alpha_tensor = PolynomialTensor({"+-": h1_mat, "++--": h2_mat})
    beta_tensor = PolynomialTensor({"+-": h1_mat, "++--": h2_mat})
    beta_alpha_tensor = PolynomialTensor({"++--": h2_mat})

    integrals = ElectronicIntegrals(
        alpha=alpha_tensor,
        beta=beta_tensor,
        beta_alpha=beta_alpha_tensor,
    )

    hamiltonian = ElectronicEnergy(integrals, constants={"nuclear_repulsion_energy": nuclear_repulsion})
    problem = ElectronicStructureProblem(hamiltonian)
    problem.num_particles = num_particles
    problem.num_spatial_orbitals = num_spatial_orbitals
    return problem


# ---------------------------------------------------------------------------
# Active-space defaults per molecule
# ---------------------------------------------------------------------------

_MOLECULE_DEFAULTS: dict = {
    "H2": {
        "num_particles": (1, 1),
        "num_spatial_orbitals": 2,
        "freeze_core": False,
        "active_electrons": None,
        "active_spatial_orbitals": None,
    },
    "LIH": {
        "num_particles": (1, 1),
        "num_spatial_orbitals": 2,
        "freeze_core": True,
        "active_electrons": 2,
        "active_spatial_orbitals": 2,
    },
    "BEH2": {
        "num_particles": (2, 2),
        "num_spatial_orbitals": 3,
        "freeze_core": True,
        "active_electrons": 4,
        "active_spatial_orbitals": 3,
    },
}


def get_molecule_problem(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    freeze_core: bool = False,
    active_electrons: Optional[int] = None,
    active_spatial_orbitals: Optional[int] = None,
    allow_synthetic_fallback: bool = False,
) -> Tuple[ElectronicStructureProblem, MoleculeMetadata]:
    """Build an electronic-structure problem for a supported molecule.

    Args:
        molecule_name: One of H2, LiH, BeH2.
        bond_length: Bond length in Ångström.
        basis: Basis set string (default: sto-3g).
        charge: Molecular charge.
        spin: Spin multiplicity - 1 (0 = singlet).
        freeze_core: Apply FreezeCoreTransformer.
        active_electrons: Active electrons for ActiveSpaceTransformer.
        active_spatial_orbitals: Active spatial orbitals.
        allow_synthetic_fallback: If True, use surrogate Hamiltonian when PySCF
            is unavailable. Should be False for all research/publication runs.

    Returns:
        Tuple of (ElectronicStructureProblem, MoleculeMetadata).
    """
    molecule_upper = molecule_name.upper()
    supported = list(_MOLECULE_DEFAULTS.keys())
    if molecule_upper not in supported:
        raise ValueError(f"Unsupported molecule '{molecule_name}'. Supported: {supported}")

    if HAS_PYSCF:
        driver = PySCFDriver(
            atom=_build_atom_string(molecule_name, bond_length),
            basis=basis,
            charge=charge,
            spin=spin,
            unit=DistanceUnit.ANGSTROM,
        )
        problem = driver.run()
        freeze_core_applied = False
        active_space_applied = False

        if freeze_core:
            transformer = FreezeCoreTransformer(freeze_core=True)
            problem = transformer.transform(problem)
            freeze_core_applied = True

        if active_electrons is not None and active_spatial_orbitals is not None:
            transformer = ActiveSpaceTransformer(
                num_electrons=active_electrons,
                num_spatial_orbitals=active_spatial_orbitals,
            )
            problem = transformer.transform(problem)
            active_space_applied = True

        return problem, MoleculeMetadata(
            molecule=molecule_upper,
            bond_length=float(bond_length),
            basis=basis,
            source="pyscf",
            freeze_core_applied=freeze_core_applied,
            active_space_applied=active_space_applied,
        )

    if not allow_synthetic_fallback:
        raise RuntimeError(
            f"PySCF is not installed and allow_synthetic_fallback=False. "
            f"Install pyscf (Python 3.10–3.13 recommended) or enable fallback for smoke-testing only. "
            f"WARNING: synthetic results are NOT valid for research or publication."
        )

    defaults = _MOLECULE_DEFAULTS[molecule_upper]
    problem = _synthetic_problem(
        molecule_name,
        bond_length,
        num_particles=defaults["num_particles"],
        num_spatial_orbitals=defaults["num_spatial_orbitals"],
    )
    return problem, MoleculeMetadata(
        molecule=molecule_upper,
        bond_length=float(bond_length),
        basis=basis,
        source="synthetic_INVALID_FOR_RESEARCH",
        freeze_core_applied=bool(freeze_core),
        active_space_applied=active_electrons is not None and active_spatial_orbitals is not None,
    )


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def get_h2_problem(distance: float) -> ElectronicStructureProblem:
    """Compatibility wrapper for legacy code paths."""
    problem, _ = get_molecule_problem("H2", distance, allow_synthetic_fallback=True)
    return problem


def get_lih_problem(distance: float) -> ElectronicStructureProblem:
    """Compatibility wrapper for legacy code paths."""
    problem, _ = get_molecule_problem(
        "LiH",
        distance,
        freeze_core=True,
        active_electrons=2,
        active_spatial_orbitals=2,
        allow_synthetic_fallback=True,
    )
    return problem


class MoleculeDriver(AbstractMoleculeDriver):
    """Class-based molecule driver for extensibility."""

    def get_problem(self, molecule_name: str, bond_length: float, **kwargs):
        return get_molecule_problem(molecule_name=molecule_name, bond_length=bond_length, **kwargs)

    @staticmethod
    def distance_grid(start: float, end: float, step: float) -> List[float]:
        return generate_distances(start, end, step)
