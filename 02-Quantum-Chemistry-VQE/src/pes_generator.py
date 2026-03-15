"""PES orchestration for exact and VQE energy scans."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from datetime import datetime
from pathlib import Path
import random
import warnings
from typing import Any, Dict, List

import numpy as np
from pydantic import ValidationError
from scipy.sparse import SparseEfficiencyWarning
import yaml

from qiskit_algorithms.optimizers import SLSQP, SPSA
from .ansatz_factory import get_ansatz
from .classical_solver import get_exact_energy_from_qubit_operator
from .config_schema import validate_config
from .data_processor import save_energy_table, save_results, save_qasm_circuit
from .molecule_driver import generate_distances, get_molecule_problem
from .plotting import plot_error, plot_pes_curve, plot_vqe_convergence
from .problem_builder import build_mapped_hamiltonian
from .runtime_executor import get_estimator
from .vqe_engine import VQEEngine

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


class PESGenerator:
    """Generates potential energy surfaces for configured molecules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_and_normalize(copy.deepcopy(config))
        self._set_seeds()

    @staticmethod
    def _validate_and_normalize(config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return validate_config(config)
        except ValidationError as exc:
            raise ValueError(f"Invalid simulation config:\n{exc}") from exc

    def _set_seeds(self) -> None:
        seed = int(self.config.get("general", {}).get("random_seed", 7))
        np.random.seed(seed)
        random.seed(seed)

    def generate_distances(self, mol_config: Dict[str, Any]) -> List[float]:
        """Distance helper compatible with legacy scripts."""
        start = float(mol_config["distances"]["start"])
        end = float(mol_config["distances"]["end"])
        step = float(mol_config["distances"]["step"])
        return generate_distances(start=start, end=end, step=step)

    def _runtime_context(self):
        runtime_cfg = self.config.get("runtime", {})
        backend = str(runtime_cfg.get("backend", "local"))
        return get_estimator(
            backend_name=backend,
            resilience_level=int(runtime_cfg.get("resilience_level", 1)),
            optimization_level=int(runtime_cfg.get("optimization_level", 1)),
            shots=int(runtime_cfg.get("shots", 4096)),
            seed=int(self.config.get("general", {}).get("random_seed", 7)),
        )

    def _get_molecule_options(self, molecule_name: str, mol_cfg: Dict[str, Any]) -> Dict[str, Any]:
        active = mol_cfg.get("active_space") or {}
        return {
            "molecule_name": molecule_name,
            "basis": str(mol_cfg.get("basis", "sto3g")),
            "charge": int(mol_cfg.get("charge", 0)),
            "spin": int(mol_cfg.get("spin", 0)),
            "freeze_core": bool(active.get("freeze_core", molecule_name.upper() == "LIH")),
            "active_electrons": active.get("active_electrons"),
            "active_spatial_orbitals": active.get("active_spatial_orbitals"),
            "allow_synthetic_fallback": bool(
                self.config.get("general", {}).get("allow_synthetic_fallback", True)
            ),
        }

    def run(self, molecule_name: str) -> Dict[str, Any]:
        """Run PES for one molecule."""
        molecule_key = molecule_name.upper()
        if molecule_key not in {key.upper() for key in self.config["molecules"].keys()}:
            raise ValueError(f"Molecule {molecule_name} not found in config.")

        original_key = next(key for key in self.config["molecules"] if key.upper() == molecule_key)
        mol_cfg = self.config["molecules"][original_key]
        distances = self.generate_distances(mol_cfg)
        runtime_context = self._runtime_context()
        optimizer_cfg = self.config.get("vqe", {}).get("optimizer", {})
        
        # Backend-aware optimizer selection for hybrid system stability
        if runtime_context.mode == "ibm_runtime":
            print("  IBM Runtime mode detected: Forcing noise-resilient SPSA optimizer.")
            opt_name = "SPSA"
            maxiter = int(optimizer_cfg.get("maxiter", 100))
        else:
            opt_name = str(optimizer_cfg.get("name", "SLSQP")).upper()
            maxiter = int(optimizer_cfg.get("maxiter", 100))
        
        if opt_name == "SLSQP":
            optimizer = SLSQP(maxiter=maxiter)
        elif opt_name == "SPSA":
            optimizer = SPSA(maxiter=maxiter)
        else:
            # Fallback for unrecognized names
            optimizer = SLSQP(maxiter=maxiter)

        ansatz_cfg = self.config.get("vqe", {}).get("ansatz", [])
        if not ansatz_cfg:
            raise ValueError("No ansatz configured in config.vqe.ansatz")

        print(f"Starting PES generation for {original_key} using mode={runtime_context.mode}")

        exact_energies: List[float] = []
        vqe_energies: Dict[str, List[float]] = {cfg["name"]: [] for cfg in ansatz_cfg}
        histories: Dict[str, Dict[str, Any]] = {cfg["name"]: {} for cfg in ansatz_cfg}
        mapping_stats: Dict[str, Dict[str, Any]] = {}
        source_info: Dict[str, str] = {}
        failures: List[Dict[str, Any]] = []

        engine = VQEEngine(estimator=runtime_context.estimator, optimizer=optimizer)
        molecule_opts = self._get_molecule_options(original_key, mol_cfg)
        threshold = float(self.config.get("analysis", {}).get("chemical_accuracy_mhartree", 1.6)) / 1000.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pes_{original_key}_{timestamp}.json"
        table_filename = f"pes_{original_key}_{timestamp}_table.csv"

        # Research Configuration: Mapping and Warm-starting
        vqe_config = self.config.get("vqe", {})
        mapping_name = str(vqe_config.get("mapping", "parity"))
        warm_start_enabled = bool(vqe_config.get("warm_start", True))
        
        # Track optimal points for warm-starting across distances
        last_optimal_points: Dict[str, List[float]] = {}

        for distance in distances:
            bond = float(round(distance, 3))
            print(f"  Distance {bond:.3f} Angstrom (Mapping: {mapping_name})")

            try:
                problem, metadata = get_molecule_problem(bond_length=bond, **molecule_opts)
                source_info[f"{bond:.3f}"] = metadata.source

                # Use configurable mapping
                mapping = build_mapped_hamiltonian(problem, two_qubit_reduction=True, mapping_name=mapping_name)
                mapping_stats[f"{bond:.3f}"] = {
                    "mapping": mapping_name,
                    "qubits_full": mapping.qubits_full,
                    "qubits_reduced": mapping.qubits_reduced,
                    "two_qubit_reduction_used": mapping.two_qubit_reduction_used,
                    "molecule_metadata": asdict(metadata),
                }

                # Sum electronic energy with all constants (nuclear, frozen-core, etc.)
                total_constant = sum(problem.hamiltonian.constants.values())
                exact = get_exact_energy_from_qubit_operator(mapping.qubit_operator) + total_constant
                exact_energies.append(exact)
                print(f"    Exact: {exact:.8f} Ha (Electronic + Constants)")
            except Exception as exc:
                error_text = str(exc)
                print(f"    ERROR at {bond:.3f} Angstrom during problem/exact stage: {error_text}")
                failures.append({"stage": "problem_or_exact", "bond_length": bond, "error": error_text})
                exact_energies.append(float("nan"))
                for ansatz_entry in ansatz_cfg:
                    ansatz_name = str(ansatz_entry["name"])
                    vqe_energies[ansatz_name].append(float("nan"))
                    histories[ansatz_name][f"{bond:.3f}"] = [{"error": error_text}]
                continue

            for ansatz_entry in ansatz_cfg:
                name = str(ansatz_entry["name"])
                kwargs = dict(ansatz_entry)
                kwargs.pop("name", None)
                try:
                    ansatz = get_ansatz(name, problem, mapping.mapper, **kwargs)
                    
                    # Ensure real-time monitoring uses physical energy (Electronic + Constants)
                    engine.initialize_vqe(ansatz=ansatz, energy_shift=total_constant)
                    
                    # Implementation of Parameter Transfer (Warm-starting)
                    initial_point = last_optimal_points.get(name) if warm_start_enabled else None
                    if initial_point and len(initial_point) != ansatz.num_parameters:
                        # Reset if parameter count changed (e.g. active space shift)
                        initial_point = None
                        
                    run_result = engine.run_vqe_qubit(mapping.qubit_operator, initial_point=initial_point)
                    
                    # Store optimal point for next distance (Transfer Learning)
                    if warm_start_enabled:
                        last_optimal_points[name] = run_result.optimal_point
                    
                    # Research Transparency: Save QASM for the first bond length after optimization
                    if bond == float(round(distances[0], 3)):
                        try:
                            # Bind optimal parameters to circuit for QASM export
                            bound_circuit = ansatz.assign_parameters(run_result.optimal_point)
                            save_qasm_circuit(bound_circuit, f"{original_key}_{name}_{bond}A")
                        except Exception as qasm_exc:
                            print(f"    Warning: Could not save QASM: {qasm_exc}")

                    total_vqe_energy = run_result.energy
                    vqe_energies[name].append(total_vqe_energy)
                    histories[name][f"{bond:.3f}"] = run_result.history

                    delta = abs(total_vqe_energy - exact)
                    meets = delta <= threshold
                    flag = "OK" if meets else "MISS"
                    print(f"    VQE {name}: {total_vqe_energy:.8f} Ha | delta={delta:.8f} Ha | {flag}")
                except Exception as exc:
                    error_text = str(exc)
                    print(f"    ERROR in VQE {name} at {bond:.3f} Angstrom: {error_text}")
                    failures.append(
                        {"stage": "vqe", "ansatz": name, "bond_length": bond, "error": error_text}
                    )
                    vqe_energies[name].append(float("nan"))
                    histories[name][f"{bond:.3f}"] = [{"error": error_text}]
            
            # Incremental Saving: Checkpoint results after every bond length
            self._save_checkpoint(
                original_key, runtime_context, distances[:len(exact_energies)],
                exact_energies, vqe_energies, histories, mapping_stats, 
                source_info, threshold, failures, output_filename
            )

        rows: List[Dict[str, Any]] = []
        for idx, bond in enumerate(distances):
            if idx >= len(exact_energies):
                break
            exact_value = float(exact_energies[idx])
            rows.append(
                {
                    "molecule": original_key,
                    "bond_length": float(bond),
                    "method": "Exact",
                    "energy_hartree": exact_value,
                    "delta_hartree": 0.0,
                    "chemical_accuracy": not np.isnan(exact_value),
                }
            )
            for name, energies in vqe_energies.items():
                if idx >= len(energies):
                    continue
                energy_value = float(energies[idx])
                if np.isnan(exact_value) or np.isnan(energy_value):
                    delta = float("nan")
                    chemical_accuracy = False
                else:
                    delta = abs(energy_value - exact_value)
                    chemical_accuracy = bool(delta <= threshold)
                rows.append(
                    {
                        "molecule": original_key,
                        "bond_length": float(bond),
                        "method": name,
                        "energy_hartree": energy_value,
                        "delta_hartree": float(delta),
                        "chemical_accuracy": chemical_accuracy,
                    }
                )

        save_energy_table(rows, table_filename)

        plot_pes_curve(
            distances=distances[:len(exact_energies)],
            exact_energies=exact_energies,
            vqe_energies={k: v for k, v in vqe_energies.items() if v},
            molecule_name=original_key,
        )
        plot_error(
            distances=distances[:len(exact_energies)],
            exact_energies=exact_energies,
            vqe_energies={k: v for k, v in vqe_energies.items() if v},
            molecule_name=original_key,
            chemical_accuracy=threshold,
        )

        first_ansatz_name = str(ansatz_cfg[0]["name"])
        closest_idx = min(range(len(exact_energies)), key=lambda i: abs(distances[i] - 1.4))
        closest_bond = distances[closest_idx]
        closest_key = f"{closest_bond:.3f}"
        if closest_key in histories.get(first_ansatz_name, {}):
            first_history = histories[first_ansatz_name][closest_key]
            if first_history and "iteration" in first_history[0]:
                plot_vqe_convergence(
                    history=first_history,
                    ansatz_name=first_ansatz_name,
                    bond_length=closest_bond,
                    molecule_name=original_key,
                )
        
        # Load final results from the last checkpoint
        return json.loads(json.dumps(
            self._get_results_dict(
                original_key, runtime_context, distances[:len(exact_energies)],
                exact_energies, vqe_energies, histories, mapping_stats, 
                source_info, threshold, failures
            ), cls=NumpyEncoder
        ))

    def _get_results_dict(
        self, original_key, runtime_context, distances, exact_energies, 
        vqe_energies, histories, mapping_stats, source_info, threshold, failures
    ) -> Dict[str, Any]:
        return {
            "molecule": original_key,
            "runtime": {
                "mode": runtime_context.mode,
                "backend": runtime_context.backend,
                "mitigation": runtime_context.mitigation,
            },
            "distances": [float(x) for x in distances],
            "exact_energies": [float(x) for x in exact_energies],
            "vqe_energies": {k: [float(v) for v in vals] for k, vals in vqe_energies.items()},
            "histories": histories,
            "mapping_stats": mapping_stats,
            "source_info": source_info,
            "chemical_accuracy_hartree": threshold,
            "failures": failures,
        }

    def _save_checkpoint(
        self, original_key, runtime_context, distances, exact_energies, 
        vqe_energies, histories, mapping_stats, source_info, threshold, failures, filename
    ) -> None:
        results = self._get_results_dict(
            original_key, runtime_context, distances, exact_energies, 
            vqe_energies, histories, mapping_stats, source_info, threshold, failures
        )
        save_results(results, filename)

from .data_processor import NumpyEncoder


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PES generation.")
    parser.add_argument("--config", type=Path, default=Path("config/simulation_config.yaml"))
    parser.add_argument("--molecule", type=str, default="H2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    generator = PESGenerator(config)
    results = generator.run(args.molecule)
    print(json.dumps({"molecule": results["molecule"], "status": "completed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
