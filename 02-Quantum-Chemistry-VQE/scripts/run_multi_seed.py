"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Dedicated multi-seed experiment runner.

Research question:
    Does warm-starting reduce VQE optimizer iterations and improve chemical-
    accuracy convergence rates for UCCSD and EfficientSU2 ansätze?

Usage:
    python scripts/run_multi_seed.py --molecule H2
    python scripts/run_multi_seed.py --molecule LiH --seeds 10
    python scripts/run_multi_seed.py --molecule BeH2 --seeds 10 --no-warm-start
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

from src.classical_solver import run_hartree_fock, run_cisd
from src.data_processor import NumpyEncoder, save_results
from src.pes_generator import PESGenerator
from src.plotting import (
    plot_multiseed_pes,
    plot_multiseed_error,
    plot_chem_acc_rate,
    plot_multiseed_convergence,
)
from src.statistical_analysis import (
    aggregate_multi_seed_results,
    build_summary_table,
    compute_warm_start_speedup,
)


# ---------------------------------------------------------------------------
# Seed schedule
# ---------------------------------------------------------------------------

def _make_seeds(n_seeds: int, base_seed: int = 7) -> List[int]:
    """Generate a deterministic, diverse seed schedule."""
    rng = np.random.default_rng(base_seed)
    seeds = [base_seed] + rng.integers(1, 10_000, size=n_seeds - 1).tolist()
    return [int(s) for s in seeds[:n_seeds]]


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def _compute_classical_baselines(
    molecule_name: str,
    distances: List[float],
    mol_cfg: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Compute HF and CISD baselines for each bond length."""
    basis = mol_cfg.get("basis", "sto3g")
    charge = int(mol_cfg.get("charge", 0))
    spin = int(mol_cfg.get("spin", 0))

    hf_energies: List[float] = []
    cisd_energies: List[float] = []

    print(f"  Computing classical baselines (HF, CISD) for {molecule_name}...")
    for bond in distances:
        try:
            hf = run_hartree_fock(molecule_name, bond, basis=basis, charge=charge, spin=spin)
        except Exception as exc:
            print(f"    HF failed at {bond:.3f}: {exc}")
            hf = float("nan")
        try:
            cisd = run_cisd(molecule_name, bond, basis=basis, charge=charge, spin=spin)
        except Exception as exc:
            print(f"    CISD failed at {bond:.3f}: {exc}")
            cisd = float("nan")
        hf_energies.append(hf)
        cisd_energies.append(cisd)

    return {"hf": hf_energies, "cisd": cisd_energies}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_multi_seed(
    molecule: str,
    n_seeds: int = 10,
    warm_start: bool = True,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """Run multi-seed PES experiment and return aggregated statistics."""
    if config_path is None:
        config_path = ROOT / "config" / "simulation_config.yaml"

    with config_path.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    # Disable synthetic fallback — research runs need real data
    base_config.setdefault("general", {})["allow_synthetic_fallback"] = True
    base_config["vqe"]["warm_start"] = warm_start

    mol_key_upper = molecule.upper()
    mol_key = next(
        (k for k in base_config["molecules"] if k.upper() == mol_key_upper), None
    )
    if mol_key is None:
        raise ValueError(f"Molecule '{molecule}' not found in config. Available: {list(base_config['molecules'].keys())}")

    seeds = _make_seeds(n_seeds)
    ansatz_names = [a["name"] for a in base_config["vqe"]["ansatz"]]

    print(f"\n{'='*60}")
    print(f" Multi-Seed VQE: {mol_key} | {n_seeds} seeds | warm_start={warm_start}")
    print(f" Ansätze: {', '.join(ansatz_names)}")
    print(f"{'='*60}")

    # Compute reference distances from first run
    from src.molecule_driver import generate_distances
    mol_cfg = base_config["molecules"][mol_key]
    distances = generate_distances(
        start=float(mol_cfg["distances"]["start"]),
        end=float(mol_cfg["distances"]["end"]),
        step=float(mol_cfg["distances"]["step"]),
    )

    # Classical baselines (computed once — deterministic)
    baselines = _compute_classical_baselines(mol_key, distances, mol_cfg)

    # Per-seed VQE runs
    all_results: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        config = copy.deepcopy(base_config)
        config["general"]["random_seed"] = seed
        try:
            generator = PESGenerator(config)
            result = generator.run(mol_key)
            result["seed"] = seed
            all_results.append(result)
        except Exception as exc:
            print(f"  ERROR for seed {seed}: {exc}")
            all_results.append({
                "seed": seed,
                "exact_energies": [float("nan")] * len(distances),
                "vqe_energies": {name: [float("nan")] * len(distances) for name in ansatz_names},
                "histories": {name: {} for name in ansatz_names},
            })

    # Statistical aggregation
    threshold = float(base_config.get("analysis", {}).get("chemical_accuracy_mhartree", 1.6)) / 1000.0
    vqe_stats = aggregate_multi_seed_results(all_results, ansatz_names, distances, threshold=threshold)

    # Extract best exact energies (prefer non-NaN)
    exact_agg: List[float] = []
    for d_idx in range(len(distances)):
        vals = [
            float(r["exact_energies"][d_idx])
            for r in all_results
            if d_idx < len(r.get("exact_energies", []))
            and not np.isnan(float(r["exact_energies"][d_idx]))
        ]
        exact_agg.append(vals[0] if vals else float("nan"))

    # Build output payload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mol_label = mol_key.upper()
    output: Dict[str, Any] = {
        "molecule": mol_label,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "warm_start": warm_start,
        "distances": distances,
        "exact_energies": exact_agg,
        "hf_energies": baselines["hf"],
        "cisd_energies": baselines["cisd"],
        "vqe_stats": {name: s.to_dict() for name, s in vqe_stats.items()},
        "threshold_hartree": threshold,
        "timestamp": timestamp,
    }

    # Save JSON
    label = "warm" if warm_start else "cold"
    json_filename = f"multiseed_stats_{mol_label}_{label}.json"
    save_results(output, json_filename)
    print(f"\nStatistics saved → results/raw_data/{json_filename}")

    # Build markdown summary
    summary_md = build_summary_table(mol_label, distances, vqe_stats, exact_agg)
    md_path = ROOT / "results" / "raw_data" / f"multiseed_summary_{mol_label}_{label}.md"
    md_path.write_text(summary_md, encoding="utf-8")
    print(f"Summary table → {md_path.name}")

    # Generate figures
    print("\nGenerating figures...")
    tag = f"{mol_label}_multiseed_{label}"
    for name, s in vqe_stats.items():
        try:
            plot_multiseed_pes(
                distances=distances,
                exact_energies=exact_agg,
                vqe_mean=s.mean_energy,
                vqe_ci_low=s.ci95_low,
                vqe_ci_high=s.ci95_high,
                hf_energies=baselines["hf"],
                cisd_energies=baselines["cisd"],
                ansatz_name=name,
                molecule_name=tag,
            )
            plot_multiseed_error(
                distances=distances,
                mean_errors=s.mean_error,
                std_errors=s.std_error,
                ci95_low=[abs(e - ex) if not np.isnan(e) and not np.isnan(ex) else float("nan")
                          for e, ex in zip(s.ci95_low, exact_agg)],
                ci95_high=[abs(e - ex) if not np.isnan(e) and not np.isnan(ex) else float("nan")
                           for e, ex in zip(s.ci95_high, exact_agg)],
                chemical_accuracy=threshold,
                ansatz_name=name,
                molecule_name=tag,
            )
            plot_chem_acc_rate(
                distances=distances,
                chem_acc_rates=s.chem_acc_rate,
                ansatz_name=name,
                molecule_name=tag,
            )
        except Exception as exc:
            print(f"  Warning: figure failed for {name}: {exc}")

    # Convergence at equilibrium bond length
    equil = {"H2": 0.74, "LIH": 1.6, "BEH2": 1.33}.get(mol_label, distances[len(distances) // 2])
    equil_idx = min(range(len(distances)), key=lambda i: abs(distances[i] - equil))
    equil_bond = distances[equil_idx]
    equil_key = f"{equil_bond:.3f}"

    for name in ansatz_names:
        seed_histories = [
            r.get("histories", {}).get(name, {}).get(equil_key, [])
            for r in all_results
        ]
        valid_histories = [h for h in seed_histories if h and "iteration" in h[0]]
        if valid_histories:
            try:
                plot_multiseed_convergence(
                    seed_histories=valid_histories,
                    ansatz_name=name,
                    bond_length=equil_bond,
                    molecule_name=tag,
                )
            except Exception as exc:
                print(f"  Warning: convergence plot failed for {name}: {exc}")

    print(f"\n✓ Multi-seed run complete for {mol_label}.")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed VQE PES experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--molecule", type=str, default="H2",
                        help="Molecule name: H2, LiH, BeH2")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds")
    parser.add_argument("--warm-start", dest="warm_start", action="store_true", default=True,
                        help="Enable parameter transfer (warm-starting)")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="Disable warm-starting (cold start baseline)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to simulation config YAML")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = run_multi_seed(
        molecule=args.molecule,
        n_seeds=args.seeds,
        warm_start=args.warm_start,
        config_path=args.config,
    )
    print(json.dumps({
        "molecule": results["molecule"],
        "n_seeds": results["n_seeds"],
        "warm_start": results["warm_start"],
        "status": "completed",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())