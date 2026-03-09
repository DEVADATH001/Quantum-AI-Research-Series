"""Quick verification runner for local development."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pes_generator import PESGenerator
from src.plotting import plot_error, plot_pes_curve, plot_vqe_convergence


def main() -> None:
    root = ROOT
    config_path = root / "config" / "simulation_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Small verification sweep for speed.
    config = copy.deepcopy(config)
    config["vqe"]["optimizer"]["maxiter"] = 8
    config["molecules"]["H2"]["distances"]["start"] = 0.6
    config["molecules"]["H2"]["distances"]["end"] = 1.2
    config["molecules"]["H2"]["distances"]["step"] = 0.3

    generator = PESGenerator(config)
    results = generator.run("H2")

    plot_pes_curve(
        results["distances"],
        results["exact_energies"],
        results["vqe_energies"],
        "H2_Verification",
    )
    plot_error(
        results["distances"],
        results["exact_energies"],
        results["vqe_energies"],
        "H2_Verification",
    )

    first_dist = f"{results['distances'][0]:.3f}"
    first_ansatz = next(iter(results["vqe_energies"].keys()))
    history = results["histories"][first_ansatz][first_dist]
    plot_vqe_convergence(history, first_ansatz, results["distances"][0], "H2_Verification")

    print("Verification complete. See results/raw_data and results/figures.")


if __name__ == "__main__":
    main()
