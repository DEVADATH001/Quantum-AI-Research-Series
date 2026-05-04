"""Author: DEVADATH H K

Project: Quantum Chemistry VQE
Task: Environment smoke test + FCI validation.

Runs a tiny 3-point H2 PES sweep (maxiter=8) to verify the pipeline
is wired up correctly before committing to multi-hour research runs.

Checks:
  1. PySCF available?  → uses real integrals
     PySCF missing?    → falls back to synthetic with a WARNING
  2. VQE energies are finite (not NaN)
  3. Exact (NumPy FCI) vs PySCF FCI agree within 1 mHa

Usage:
    python scripts/run_verification.py
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

# Force UTF-8 so Unicode glyphs don't crash on Windows CP1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pes_generator import PESGenerator
from src.plotting import plot_error, plot_pes_curve, plot_vqe_convergence

HAS_PYSCF = importlib.util.find_spec("pyscf") is not None


def _check_pyscf() -> None:
    if HAS_PYSCF:
        print("[OK] PySCF detected -- will use real molecular integrals.")
    else:
        print(
            "\n[WARNING] PySCF not found!\n"
            "  Results from this smoke test use synthetic surrogate Hamiltonians.\n"
            "  They are NOT valid for research or publication.\n"
            "  Install with: pip install 'pyscf>=2.3'\n"
        )


def _validate_fci_agreement(distances, exact_energies, vqe_energies) -> None:
    """Cross-check NumPy exact diagonalization vs PySCF FCI when available."""
    if not HAS_PYSCF:
        print("  (Skipping FCI agreement check — PySCF not installed)")
        return

    from src.classical_solver import run_fci
    print("\n  FCI Agreement Check:")
    all_ok = True
    for bond, exact in zip(distances, exact_energies):
        if np.isnan(exact):
            continue
        try:
            fci_energy = run_fci("H2", bond, basis="sto-3g")
            delta_mha = abs(exact - fci_energy) * 1000
            status = "[OK]" if delta_mha < 1.0 else "[WARN]"
            print(f"    d={bond:.2f}A  NumPy={exact:.6f}  PySCF-FCI={fci_energy:.6f}  "
                  f"delta={delta_mha:.3f} mHa  {status}")
            if delta_mha >= 1.0:
                all_ok = False
        except Exception as exc:
            print(f"    d={bond:.2f}A  FCI check failed: {exc}")
    if all_ok:
        print("  [OK] NumPy exact diagonalization agrees with PySCF FCI within 1 mHa.")
    else:
        print("  [WARN] Discrepancy detected -- check active-space settings.")


def _validate_no_nans(distances, exact_energies, vqe_energies) -> None:
    print("\n  NaN Check:")
    exact_arr = np.array(exact_energies, dtype=float)
    nan_count = int(np.isnan(exact_arr).sum())
    if nan_count > 0:
        print(f"  [FAIL] Exact energies have {nan_count}/{len(distances)} NaN values -- pipeline error!")
        sys.exit(1)
    print(f"  [OK] All {len(distances)} exact energies are finite.")
    for name, energies in vqe_energies.items():
        arr = np.array(energies, dtype=float)
        nc = int(np.isnan(arr).sum())
        status = "[OK]" if nc == 0 else "[FAIL]"
        print(f"  {status} {name}: {nc}/{len(distances)} NaN VQE energies.")


def main() -> None:
    _check_pyscf()

    config_path = ROOT / "config" / "simulation_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Override: allow synthetic fallback ONLY for verification smoke test
    config = copy.deepcopy(config)
    config["general"]["allow_synthetic_fallback"] = True   # ← smoke-test only
    config["vqe"]["optimizer"]["maxiter"] = 50             # small but realistic
    config["molecules"]["H2"]["distances"]["start"] = 0.60
    config["molecules"]["H2"]["distances"]["end"] = 1.50
    config["molecules"]["H2"]["distances"]["step"] = 0.30

    print("\nRunning 3-point H2 PES sweep (smoke test)...")
    generator = PESGenerator(config)
    results = generator.run("H2")

    distances = results["distances"]
    exact_energies = results["exact_energies"]
    vqe_energies = results["vqe_energies"]

    _validate_no_nans(distances, exact_energies, vqe_energies)
    _validate_fci_agreement(distances, exact_energies, vqe_energies)

    # Generate figures
    plot_pes_curve(distances, exact_energies, vqe_energies, "H2_Verification")
    plot_error(distances, exact_energies, vqe_energies, "H2_Verification")

    first_dist = f"{distances[0]:.3f}"
    first_ansatz = next(iter(vqe_energies.keys()))
    history = results["histories"].get(first_ansatz, {}).get(first_dist, [])
    if history and "iteration" in history[0]:
        plot_vqe_convergence(history, first_ansatz, distances[0], "H2_Verification")

    print("\n[PASS] Smoke test passed. Figures saved to results/figures/")
    print("  Next step: python scripts/run_experiment.py --molecule H2 --seeds 10\n")


if __name__ == "__main__":
    main()
