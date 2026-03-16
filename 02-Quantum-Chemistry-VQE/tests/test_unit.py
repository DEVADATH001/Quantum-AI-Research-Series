"""Author: DEVADATH H K

Project: Quantum Chemistry VQE

Unit tests for quantum chemistry VQE stack."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ansatz_factory import AnsatzFactory, get_ansatz
from src.classical_solver import get_exact_energy_from_qubit_operator
from src.config_schema import validate_config
from src.molecule_driver import MoleculeDriver, generate_distances, get_molecule_problem
from src.pes_generator import PESGenerator
from src.problem_builder import build_mapped_hamiltonian
from src.vqe_engine import VQEEngine
import src.molecule_driver as molecule_driver_module

def _base_config() -> dict:
    return {
        "general": {"random_seed": 7, "allow_synthetic_fallback": True},
        "molecules": {
            "H2": {
                "distances": {"start": 0.6, "end": 0.8, "step": 0.2},
                "charge": 0,
                "spin": 0,
                "basis": "sto3g",
                "active_space": None,
            }
        },
        "vqe": {
            "ansatz": [
                {"name": "EfficientSU2", "reps": 1, "entanglement": "circular"},
            ],
            "optimizer": {"name": "SPSA", "maxiter": 3},
        },
        "runtime": {"backend": "local", "resilience_level": 1, "optimization_level": 1, "shots": 1024},
        "analysis": {"chemical_accuracy_mhartree": 1.6},
    }

class TestQuantumChemistryVQE(unittest.TestCase):
    def test_distance_generation(self):
        self.assertEqual(generate_distances(0.5, 0.9, 0.2), [0.5, 0.7, 0.9])

    def test_config_validation_valid(self):
        validated = validate_config(_base_config())
        self.assertIn("molecules", validated)
        self.assertIn("vqe", validated)

    def test_config_validation_invalid_distance_step(self):
        bad = _base_config()
        bad["molecules"]["H2"]["distances"]["step"] = 0
        with self.assertRaises(Exception):
            validate_config(bad)

    def test_problem_build_and_mapping(self):
        problem, metadata = get_molecule_problem("H2", 0.74, allow_synthetic_fallback=True)
        self.assertIn(metadata.source, {"pyscf", "synthetic"})
        mapping = build_mapped_hamiltonian(problem)
        self.assertGreater(mapping.qubit_operator.num_qubits, 0)

    def test_ansatz_factory_returns_expected_circuit_shape(self):
        problem, _ = get_molecule_problem("H2", 0.74, allow_synthetic_fallback=True)
        mapping = build_mapped_hamiltonian(problem)
        ansatz = get_ansatz("EfficientSU2", problem, mapping.mapper, reps=1)
        self.assertEqual(ansatz.num_qubits, mapping.qubit_operator.num_qubits)
        self.assertGreater(ansatz.num_parameters, 0)

    def test_class_based_ansatz_factory(self):
        problem, _ = get_molecule_problem("H2", 0.74, allow_synthetic_fallback=True)
        mapping = build_mapped_hamiltonian(problem)
        factory = AnsatzFactory()
        circuit = factory.build("UCCSD", problem, mapping.mapper)
        self.assertGreater(circuit.num_qubits, 0)

    def test_exact_solver_toy_operator(self):
        op = SparsePauliOp.from_list([("I", 0.5), ("Z", -1.0)])
        energy = get_exact_energy_from_qubit_operator(op)
        self.assertAlmostEqual(energy, -0.5, places=8)

    def test_vqe_engine_initialization_and_output_format(self):
        op = SparsePauliOp.from_list(
            [("II", -1.0), ("ZI", 0.2), ("IZ", 0.2), ("ZZ", 0.15), ("XX", 0.1)]
        )
        problem, _ = get_molecule_problem("H2", 0.74, allow_synthetic_fallback=True)
        mapping = build_mapped_hamiltonian(problem)
        ansatz = get_ansatz("EfficientSU2", problem, mapping.mapper, reps=1)
        engine = VQEEngine(estimator=StatevectorEstimator(seed=7), ansatz=ansatz, maxiter=4)
        self.assertIsNotNone(engine._vqe)
        result = engine.run_vqe_qubit(op)
        self.assertTrue(hasattr(result, "energy"))
        self.assertTrue(hasattr(result, "history"))
        self.assertIsInstance(result.total_energies, list)

    def test_molecule_driver_class_wrapper(self):
        driver = MoleculeDriver()
        problem, metadata = driver.get_problem("H2", 0.74, allow_synthetic_fallback=True)
        self.assertIsInstance(problem, ElectronicStructureProblem)
        self.assertIn(metadata.source, {"synthetic", "pyscf"})

    def test_mocked_pyscf_driver_path(self):
        class FakePySCFDriver:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self):
                return SimpleNamespace()

        with patch.object(molecule_driver_module, "HAS_PYSCF", True), patch.object(
            molecule_driver_module, "PySCFDriver", FakePySCFDriver, create=True
        ):
            problem, metadata = molecule_driver_module.get_molecule_problem(
                "H2",
                0.74,
                allow_synthetic_fallback=False,
                freeze_core=False,
                active_electrons=None,
                active_spatial_orbitals=None,
            )
            self.assertEqual(metadata.source, "pyscf")
            self.assertIsNotNone(problem)

    def test_pes_generator_h2(self):
        config = _base_config()
        generator = PESGenerator(copy.deepcopy(config))
        results = generator.run("H2")
        self.assertEqual(len(results["distances"]), 2)
        self.assertEqual(len(results["exact_energies"]), 2)
        self.assertIn("EfficientSU2", results["vqe_energies"])
        self.assertIn("failures", results)

    def test_pes_generator_handles_partial_failures(self):
        config = _base_config()
        config["vqe"]["ansatz"] = [
            {"name": "EfficientSU2", "reps": 1},
            {"name": "NonExistentAnsatz"},
        ]
        generator = PESGenerator(copy.deepcopy(config))
        results = generator.run("H2")
        self.assertEqual(len(results["vqe_energies"]["NonExistentAnsatz"]), len(results["distances"]))
        self.assertGreaterEqual(len(results["failures"]), 1)
        self.assertTrue(np.isnan(results["vqe_energies"]["NonExistentAnsatz"][0]))

if __name__ == "__main__":
    unittest.main()
