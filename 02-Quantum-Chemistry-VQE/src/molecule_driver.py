"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Molecule construction utilities for quantum chemistry problems."""

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

def _synthetic_problem(
    molecule_name: str, bond_length: float, num_particles: Tuple[int, int], num_spatial_orbitals: int
) -> ElectronicStructureProblem:
    """Build deterministic synthetic fallback Hamiltonians for smoke testing.
    
    WARNING: These are lightweight surrogate Hamiltonians used strictly for 
    environment validation when PySCF is unavailable. They lack physical 
    scaling properties (R-dependency) and are not intended for research-grade 
    simulations or publication.
    """
    if molecule_name.upper() == "H2":
        h1_mat = np.array([[-1.10, -0.15], [-0.15, -0.35]], dtype=float)
        h2_mat = np.zeros((2, 2, 2, 2), dtype=float)
        h2_mat[0, 0, 0, 0] = 0.72
        h2_mat[1, 1, 1, 1] = 0.66
        h2_mat[0, 1, 1, 0] = 0.61
        h2_mat[1, 0, 0, 1] = 0.61
        scale = 0.74 / bond_length
    else:
        # Effective LiH-valence surrogate
        h1_mat = np.array([[-0.95, -0.10], [-0.10, -0.25]], dtype=float)
        h2_mat = np.zeros((2, 2, 2, 2), dtype=float)
        h2_mat[0, 0, 0, 0] = 0.80
        h2_mat[1, 1, 1, 1] = 0.58
        h2_mat[0, 1, 1, 0] = 0.47
        h2_mat[1, 0, 0, 1] = 0.47
        scale = 1.6 / bond_length

    h1_mat = h1_mat * scale
    h2_mat = h2_mat * (scale ** 1.1)
    nuclear_repulsion = 0.529177 / max(bond_length, 1e-6)

    # Correctly map spatial integrals to all required spin sectors (alpha, beta, etc.)
    # We combine one-body and two-body terms into PolynomialTensors for each spin sector.
    alpha_tensor = PolynomialTensor({"+-": h1_mat, "++--": h2_mat})
    beta_tensor = PolynomialTensor({"+-": h1_mat, "++--": h2_mat})
    beta_alpha_tensor = PolynomialTensor({"++--": h2_mat})
    
    integrals = ElectronicIntegrals(
        alpha=alpha_tensor, 
        beta=beta_tensor, 
        beta_alpha=beta_alpha_tensor
    )
    
    hamiltonian = ElectronicEnergy(integrals, constants={"nuclear_repulsion_energy": nuclear_repulsion})
    problem = ElectronicStructureProblem(hamiltonian)
    problem.num_particles = num_particles
    problem.num_spatial_orbitals = num_spatial_orbitals
    return problem

def _build_atom_string(molecule_name: str, bond_length: float) -> str:
    name = molecule_name.upper()
    if name == "H2":
        return f"H 0 0 0; H 0 0 {bond_length}"
    if name == "LIH":
        return f"Li 0 0 0; H 0 0 {bond_length}"
    raise ValueError(f"Unsupported molecule {molecule_name}")

def get_molecule_problem(
    molecule_name: str,
    bond_length: float,
    basis: str = "sto3g",
    charge: int = 0,
    spin: int = 0,
    freeze_core: bool = False,
    active_electrons: Optional[int] = None,
    active_spatial_orbitals: Optional[int] = None,
    allow_synthetic_fallback: bool = True,
) -> Tuple[ElectronicStructureProblem, MoleculeMetadata]:
    """Build an electronic-structure problem for H2 or LiH."""
    molecule_upper = molecule_name.upper()
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
            "PySCF is not installed and synthetic fallback is disabled. "
            "Install pyscf (Python 3.10-3.13 recommended) or enable fallback."
        )

    if molecule_upper == "H2":
        problem = _synthetic_problem("H2", bond_length, num_particles=(1, 1), num_spatial_orbitals=2)
    elif molecule_upper == "LIH":
        problem = _synthetic_problem("LiH", bond_length, num_particles=(1, 1), num_spatial_orbitals=2)
    else:
        raise ValueError(f"Unsupported molecule {molecule_name}")

    return problem, MoleculeMetadata(
        molecule=molecule_upper,
        bond_length=float(bond_length),
        basis=basis,
        source="synthetic",
        freeze_core_applied=bool(freeze_core),
        active_space_applied=active_electrons is not None and active_spatial_orbitals is not None,
    )

def get_h2_problem(distance: float) -> ElectronicStructureProblem:
    """Compatibility wrapper used by legacy code paths."""
    problem, _ = get_molecule_problem("H2", distance, allow_synthetic_fallback=True)
    return problem

def get_lih_problem(distance: float) -> ElectronicStructureProblem:
    """Compatibility wrapper used by legacy code paths."""
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
