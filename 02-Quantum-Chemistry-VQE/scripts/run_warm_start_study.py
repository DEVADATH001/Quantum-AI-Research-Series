"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Warm-start vs cold-start comparative analysis.

Research contribution:
    Measures whether parameter transfer (warm-starting) across bond lengths
    reduces VQE optimizer iterations and improves chemical-accuracy convergence
    rates for UCCSD and EfficientSU2 ansätze on H2, LiH, and BeH2.

Methodology:
    For each molecule and each seed:
      1. Run VQE with warm_start=False (cold): new random parameters every bond length.
      2. Run VQE with warm_start=True (warm): transfer optimal parameters to next bond.
    Then aggregate: iteration counts, energy errors, chem-acc rates.

Usage:
    python scripts/run_warm_start_study.py --molecule H2 --seeds 10
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processor import NumpyEncoder, save_results
from src.molecule_driver import generate_distances
from src.pes_generator import PESGenerator
from src.plotting import plot_warm_start_comparison
from src.statistical_analysis import compute_warm_start_speedup


# ---------------------------------------------------------------------------
# Seed schedule (same as run_multi_seed for reproducibility)
# ---------------------------------------------------------------------------

def _make_seeds(n_seeds: int, base_seed: int = 7) -> List[int]:
    rng = np.random.default_rng(base_seed)
    seeds = [base_seed] + rng.integers(1, 10_000, size=n_seeds - 1).tolist()
    return [int(s) for s in seeds[:n_seeds]]


def _run_condition(
    base_config: Dict[str, Any],
    mol_key: str,
    seeds: List[int],
    warm_start: bool,
    ansatz_names: List[str],
    n_distances: int,
) -> List[Dict[str, Any]]:
    """Run PES for all seeds under one warm_start condition."""
    label = "warm" if warm_start else "cold"
    results: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"  [{label}] Seed {seed} ({i+1}/{len(seeds)})")
        config = copy.deepcopy(base_config)
        config["general"]["random_seed"] = seed
        config["vqe"]["warm_start"] = warm_start
        try:
            gen = PESGenerator(config)
            res = gen.run(mol_key)
            res["seed"] = seed
            results.append(res)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append({
                "seed": seed,
                "exact_energies": [float("nan")] * n_distances,
                "vqe_energies": {n: [float("nan")] * n_distances for n in ansatz_names},
                "histories": {n: {} for n in ansatz_names},
            })
    return results


def run_warm_start_study(
    molecule: str = "H2",
    n_seeds: int = 10,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """Run paired warm/cold VQE study and return speedup statistics."""
    if config_path is None:
        config_path = ROOT / "config" / "simulation_config.yaml"

    with config_path.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    base_config.setdefault("general", {})["allow_synthetic_fallback"] = True
    mol_key = next(
        (k for k in base_config["molecules"] if k.upper() == molecule.upper()), None
    )
    if mol_key is None:
        raise ValueError(f"Molecule '{molecule}' not found in config.")

    mol_cfg = base_config["molecules"][mol_key]
    distances = generate_distances(
        float(mol_cfg["distances"]["start"]),
        float(mol_cfg["distances"]["end"]),
        float(mol_cfg["distances"]["step"]),
    )
    ansatz_names = [a["name"] for a in base_config["vqe"]["ansatz"]]
    seeds = _make_seeds(n_seeds)
    threshold = float(base_config.get("analysis", {}).get("chemical_accuracy_mhartree", 1.6)) / 1000.0

    print(f"\n{'='*60}")
    print(f" Warm-Start Study: {mol_key} | {n_seeds} seeds")
    print(f" Ansätze: {', '.join(ansatz_names)}")
    print(f"{'='*60}")

    print("\n[1/2] Running COLD starts...")
    cold_results = _run_condition(base_config, mol_key, seeds, False, ansatz_names, len(distances))

    print("\n[2/2] Running WARM starts...")
    warm_results = _run_condition(base_config, mol_key, seeds, True, ansatz_names, len(distances))

    # Compute speedup per ansatz
    speedups: Dict[str, Any] = {}
    for name in ansatz_names:
        sp = compute_warm_start_speedup(cold_results, warm_results, name, distances, threshold)
        speedups[name] = sp.to_dict()
        print(f"\n  {name}: mean speedup = {sp.mean_speedup():.2f}×")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output: Dict[str, Any] = {
        "molecule": mol_key.upper(),
        "n_seeds": n_seeds,
        "seeds": seeds,
        "distances": distances,
        "ansatz_names": ansatz_names,
        "speedup_by_ansatz": speedups,
        "threshold_hartree": threshold,
        "timestamp": timestamp,
    }

    filename = f"warm_start_study_{mol_key.upper()}_{timestamp}.json"
    save_results(output, filename)
    print(f"\nStudy results saved → results/raw_data/{filename}")

    # Generate comparison figures
    for name in ansatz_names:
        sp_dict = speedups[name]
        try:
            plot_warm_start_comparison(
                distances=distances,
                cold_iterations=sp_dict["cold_iterations"],
                warm_iterations=sp_dict["warm_iterations"],
                cold_chem_acc=sp_dict["cold_chem_acc_rate"],
                warm_chem_acc=sp_dict["warm_chem_acc_rate"],
                ansatz_name=name,
                molecule_name=mol_key.upper(),
                speedup_ratio=sp_dict["speedup_ratio"],
            )
        except Exception as exc:
            print(f"  Warning: figure failed for {name}: {exc}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-start vs cold-start VQE comparative study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--molecule", type=str, default="H2",
                        help="Molecule: H2, LiH, BeH2")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds per condition")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to simulation_config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = run_warm_start_study(
        molecule=args.molecule,
        n_seeds=args.seeds,
        config_path=args.config,
    )
    print(json.dumps({
        "molecule": results["molecule"],
        "n_seeds": results["n_seeds"],
        "status": "completed",
        "mean_speedups": {
            name: round(results["speedup_by_ansatz"][name]["mean_speedup"], 3)
            for name in results["ansatz_names"]
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
