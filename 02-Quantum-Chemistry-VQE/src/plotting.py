"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Plotting utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_pes_curve(
    distances: List[float],
    exact_energies: List[float],
    vqe_energies: Dict[str, List[float]],
    molecule_name: str,
    output_dir: str = "results/figures",
) -> str:
    """Plot PES curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, exact_energies, "k-", linewidth=2, label="Exact")
    markers = ["o", "s", "^", "d"]
    for idx, (ansatz_name, energies) in enumerate(vqe_energies.items()):
        plt.plot(
            distances,
            energies,
            linestyle="--",
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            label=f"VQE ({ansatz_name})",
        )
    plt.xlabel("Bond Length (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"Potential Energy Surface: {molecule_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pes_curve_{molecule_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_vqe_convergence(
    history: List[Dict[str, Any]],
    ansatz_name: str,
    bond_length: float,
    molecule_name: str,
    output_dir: str = "results/figures",
) -> str:
    """Plot energy vs iteration."""
    iterations = [item["iteration"] for item in history]
    energies = [item["energy"] for item in history]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, "b-", linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"VQE Convergence: {molecule_name} {ansatz_name} at {bond_length:.2f} Angstrom")
    plt.grid(alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"convergence_{molecule_name}_{ansatz_name}_{bond_length:.2f}A.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_error(
    distances: List[float],
    exact_energies: List[float],
    vqe_energies: Dict[str, List[float]],
    molecule_name: str,
    chemical_accuracy: float = 0.0016,
    output_dir: str = "results/figures",
) -> str:
    """Plot absolute error against exact baseline."""
    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "d"]
    for idx, (ansatz_name, energies) in enumerate(vqe_energies.items()):
        errors = [abs(energy - exact_energies[i]) for i, energy in enumerate(energies)]
        plt.plot(
            distances,
            errors,
            marker=markers[idx % len(markers)],
            linewidth=1.6,
            label=f"{ansatz_name} error",
        )
    plt.axhline(chemical_accuracy, color="r", linestyle="--", label="Chemical accuracy")
    plt.xlabel("Bond Length (Angstrom)")
    plt.ylabel("Absolute Error (Hartree)")
    plt.yscale("log")
    plt.title(f"VQE Error Curve: {molecule_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"error_{molecule_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_pareto_front(
    errors: List[float],
    costs: List[int],
    labels: List[str],
    molecule_name: str,
    output_dir: str = "results/figures",
) -> str:
    """Plot Accuracy (Error) vs. Cost (CNOT count) Pareto front for research ablation."""
    plt.figure(figsize=(10, 6))
    plt.scatter(costs, errors, c='blue', marker='o', s=100, alpha=0.6)
    for i, label in enumerate(labels):
        plt.annotate(label, (costs[i], errors[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.axhline(0.0016, color='red', linestyle='--', label='Chemical Accuracy')
    plt.yscale('log')
    plt.xlabel('Hardware Cost (CNOT Count)')
    plt.ylabel('Absolute Error (Hartree)')
    plt.title(f"VQE Pareto Front: Accuracy vs. Complexity ({molecule_name})")
    plt.grid(alpha=0.3, which='both')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pareto_front_{molecule_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path
