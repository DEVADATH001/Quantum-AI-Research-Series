"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Plotting utilities — publication-grade figures with error bars and CI bands.

All figures are saved at 300 DPI to results/figures/.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_FIGURE_DIR = "results/figures"
_COLORS = {
    "exact": "#1a1a2e",
    "hf": "#457b9d",
    "cisd": "#2a9d8f",
    "fci": "#264653",
    "uccsd": "#e76f51",
    "efficientsu2": "#8338ec",
    "cold": "#e63946",
    "warm": "#2a9d8f",
    "chem_acc": "#e9c46a",
    "ci_band": 0.20,   # alpha for shaded CI bands
}
_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _ansatz_color(name: str) -> str:
    key = name.lower()
    for k, v in _COLORS.items():
        if k in key:
            return v
    return "#6c757d"


def _fig_path(filename: str, output_dir: str = _FIGURE_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _filter_nan(xs: List[float], *yss) -> tuple:
    """Remove indices where any of the arrays is NaN."""
    xs_arr = np.array(xs, dtype=float)
    mask = np.ones(len(xs_arr), dtype=bool)
    for ys in yss:
        if ys is not None:
            mask &= ~np.isnan(np.array(ys, dtype=float))
    return (xs_arr[mask],) + tuple(
        np.array(ys, dtype=float)[mask] if ys is not None else None
        for ys in yss
    )


# ---------------------------------------------------------------------------
# 1. Basic PES curve (single run)
# ---------------------------------------------------------------------------

def plot_pes_curve(
    distances: List[float],
    exact_energies: List[float],
    vqe_energies: Dict[str, List[float]],
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Plot PES curves for exact and VQE methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    d_arr = np.array(distances)
    e_arr = np.array(exact_energies, dtype=float)
    mask = ~np.isnan(e_arr)
    ax.plot(d_arr[mask], e_arr[mask], color=_COLORS["exact"], linewidth=2.5,
            label="Exact (FCI)", zorder=5)

    for idx, (name, energies) in enumerate(vqe_energies.items()):
        en = np.array(energies, dtype=float)
        vm = mask & ~np.isnan(en)
        ax.plot(d_arr[vm], en[vm],
                color=_ansatz_color(name),
                linestyle="--",
                marker=_MARKERS[idx % len(_MARKERS)],
                linewidth=1.8,
                markersize=5,
                label=f"VQE ({name})",
                zorder=4)

    ax.set_xlabel("Bond Length (Å)", fontsize=13)
    ax.set_ylabel("Energy (Hartree)", fontsize=13)
    ax.set_title(f"Potential Energy Surface: {molecule_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = _fig_path(f"pes_curve_{molecule_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Multi-seed PES curve with CI band and classical baselines
# ---------------------------------------------------------------------------

def plot_multiseed_pes(
    distances: List[float],
    exact_energies: List[float],
    vqe_mean: List[float],
    vqe_ci_low: List[float],
    vqe_ci_high: List[float],
    hf_energies: Optional[List[float]],
    cisd_energies: Optional[List[float]],
    ansatz_name: str,
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Multi-seed PES with 95% CI shading and classical baselines."""
    fig, ax = plt.subplots(figsize=(11, 6))
    d = np.array(distances)
    exact = np.array(exact_energies, dtype=float)
    mean = np.array(vqe_mean, dtype=float)
    lo = np.array(vqe_ci_low, dtype=float)
    hi = np.array(vqe_ci_high, dtype=float)

    e_mask = ~np.isnan(exact)
    ax.plot(d[e_mask], exact[e_mask], color=_COLORS["exact"], lw=2.5,
            label="Exact (FCI/NumPy)", zorder=6)

    if hf_energies is not None:
        hf = np.array(hf_energies, dtype=float)
        hm = ~np.isnan(hf)
        ax.plot(d[hm], hf[hm], color=_COLORS["hf"], lw=1.5, linestyle=":",
                label="Hartree-Fock", zorder=4)

    if cisd_energies is not None:
        ci = np.array(cisd_energies, dtype=float)
        cm = ~np.isnan(ci)
        ax.plot(d[cm], ci[cm], color=_COLORS["cisd"], lw=1.5, linestyle="-.",
                label="CISD", zorder=4)

    v_mask = ~np.isnan(mean)
    color = _ansatz_color(ansatz_name)
    ax.plot(d[v_mask], mean[v_mask], color=color, lw=2.0, linestyle="--",
            marker="o", markersize=4, label=f"VQE {ansatz_name} (mean, {molecule_name})", zorder=5)
    ci_mask = v_mask & ~np.isnan(lo) & ~np.isnan(hi)
    ax.fill_between(d[ci_mask], lo[ci_mask], hi[ci_mask],
                    alpha=_COLORS["ci_band"], color=color, label="95% CI")

    ax.set_xlabel("Bond Length (Å)", fontsize=13)
    ax.set_ylabel("Energy (Hartree)", fontsize=13)
    ax.set_title(
        f"Multi-Seed PES: {molecule_name} — {ansatz_name}\n(mean ± 95% bootstrap CI)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    path = _fig_path(f"pes_multiseed_{molecule_name}_{ansatz_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Error curve (single run)
# ---------------------------------------------------------------------------

def plot_error(
    distances: List[float],
    exact_energies: List[float],
    vqe_energies: Dict[str, List[float]],
    molecule_name: str,
    chemical_accuracy: float = 0.0016,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Plot absolute error against exact baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    d = np.array(distances)
    exact = np.array(exact_energies, dtype=float)

    for idx, (name, energies) in enumerate(vqe_energies.items()):
        en = np.array(energies, dtype=float)
        mask = ~np.isnan(exact) & ~np.isnan(en)
        errors = np.abs(en[mask] - exact[mask])
        ax.plot(d[mask], errors,
                color=_ansatz_color(name),
                marker=_MARKERS[idx % len(_MARKERS)],
                linewidth=1.8,
                markersize=5,
                label=f"{name}")

    ax.axhline(chemical_accuracy, color=_COLORS["chem_acc"], linestyle="--",
               linewidth=1.8, label=f"Chemical accuracy ({chemical_accuracy*1000:.1f} mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("Bond Length (Å)", fontsize=13)
    ax.set_ylabel("|E_VQE - E_exact| (Hartree)", fontsize=13)
    ax.set_title(f"VQE Error vs Exact: {molecule_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = _fig_path(f"error_{molecule_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Multi-seed error with CI band
# ---------------------------------------------------------------------------

def plot_multiseed_error(
    distances: List[float],
    mean_errors: List[float],
    std_errors: List[float],
    ci95_low: List[float],
    ci95_high: List[float],
    chemical_accuracy: float,
    ansatz_name: str,
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Multi-seed error plot with 95% CI shading."""
    fig, ax = plt.subplots(figsize=(10, 6))
    d = np.array(distances)
    mu = np.array(mean_errors, dtype=float)
    lo = np.array(ci95_low, dtype=float)
    hi = np.array(ci95_high, dtype=float)
    mask = ~np.isnan(mu)

    color = _ansatz_color(ansatz_name)
    ax.plot(d[mask], mu[mask], color=color, lw=2.0, marker="o", markersize=4,
            label=f"{ansatz_name} (mean error)")
    ci_mask = mask & ~np.isnan(lo) & ~np.isnan(hi)
    if ci_mask.any():
        ax.fill_between(d[ci_mask], lo[ci_mask], hi[ci_mask],
                        alpha=_COLORS["ci_band"], color=color, label="95% CI")

    ax.axhline(chemical_accuracy, color=_COLORS["chem_acc"], linestyle="--",
               linewidth=1.8, label=f"Chemical accuracy ({chemical_accuracy*1000:.1f} mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("Bond Length (Å)", fontsize=13)
    ax.set_ylabel("|E_VQE - E_exact| (Hartree)", fontsize=13)
    ax.set_title(
        f"Multi-Seed Error: {molecule_name} — {ansatz_name}\n(mean ± 95% bootstrap CI)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11)
    fig.tight_layout()
    path = _fig_path(f"error_multiseed_{molecule_name}_{ansatz_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Chemical accuracy rate
# ---------------------------------------------------------------------------

def plot_chem_acc_rate(
    distances: List[float],
    chem_acc_rates: List[float],
    ansatz_name: str,
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Bar chart of fraction of seeds achieving chemical accuracy per bond length."""
    fig, ax = plt.subplots(figsize=(10, 5))
    d = np.array(distances)
    rates = np.array(chem_acc_rates, dtype=float)
    mask = ~np.isnan(rates)

    colors = [_COLORS["warm"] if r >= 0.5 else _COLORS["cold"] for r in rates[mask]]
    ax.bar(d[mask], rates[mask] * 100, color=colors, width=np.diff(d).min() * 0.8 if len(d) > 1 else 0.05,
           edgecolor="white", linewidth=0.5)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="50% threshold")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Bond Length (Å)", fontsize=13)
    ax.set_ylabel("Chemical Accuracy Rate (%)", fontsize=13)
    ax.set_title(
        f"Chemical Accuracy Rate: {molecule_name} — {ansatz_name}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = _fig_path(f"chem_acc_rate_{molecule_name}_{ansatz_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. VQE convergence (single seed)
# ---------------------------------------------------------------------------

def plot_vqe_convergence(
    history: List[Dict[str, Any]],
    ansatz_name: str,
    bond_length: float,
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Plot energy vs iteration for a single VQE run."""
    if not history or "iteration" not in history[0]:
        return ""
    iters = [item["iteration"] for item in history]
    energies = [item["energy"] for item in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, energies, color=_ansatz_color(ansatz_name), lw=1.8)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Energy (Hartree)", fontsize=13)
    ax.set_title(
        f"VQE Convergence: {molecule_name} — {ansatz_name} at {bond_length:.2f} Å",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path(f"convergence_{molecule_name}_{ansatz_name}_{bond_length:.2f}A.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Multi-seed convergence overlay
# ---------------------------------------------------------------------------

def plot_multiseed_convergence(
    seed_histories: List[List[Dict[str, Any]]],
    ansatz_name: str,
    bond_length: float,
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Overlay all seed convergence traces with bold mean line."""
    if not seed_histories:
        return ""
    fig, ax = plt.subplots(figsize=(11, 6))
    color = _ansatz_color(ansatz_name)

    max_iters = max(h[-1]["iteration"] for h in seed_histories if h)
    all_energies: List[np.ndarray] = []

    for hist in seed_histories:
        if not hist or "iteration" not in hist[0]:
            continue
        iters = [item["iteration"] for item in hist]
        energies = [item["energy"] for item in hist]
        ax.plot(iters, energies, color=color, alpha=0.25, lw=1.0, zorder=2)
        # Interpolate to common grid for mean
        grid = np.arange(1, max_iters + 1)
        interp = np.interp(grid, iters, energies)
        all_energies.append(interp)

    if all_energies:
        mean_curve = np.nanmean(np.stack(all_energies), axis=0)
        grid = np.arange(1, max_iters + 1)
        ax.plot(grid, mean_curve, color=color, lw=2.5, zorder=5, label=f"Mean ({len(seed_histories)} seeds)")

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Energy (Hartree)", fontsize=13)
    ax.set_title(
        f"Multi-Seed Convergence: {molecule_name} — {ansatz_name} at {bond_length:.2f} Å",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_path(f"convergence_multiseed_{molecule_name}_{ansatz_name}_{bond_length:.2f}A.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. Warm-start comparison (the key research figure)
# ---------------------------------------------------------------------------

def plot_warm_start_comparison(
    distances: List[float],
    cold_iterations: List[float],
    warm_iterations: List[float],
    cold_chem_acc: List[float],
    warm_chem_acc: List[float],
    ansatz_name: str,
    molecule_name: str,
    speedup_ratio: Optional[List[float]] = None,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Two-panel figure: iteration counts (top) and chem-acc rates (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    d = np.array(distances)
    cold_it = np.array(cold_iterations, dtype=float)
    warm_it = np.array(warm_iterations, dtype=float)

    # --- Panel 1: Iterations ---
    mask = ~np.isnan(cold_it) & ~np.isnan(warm_it)
    ax1.plot(d[mask], cold_it[mask], color=_COLORS["cold"], lw=2.0, marker="o",
             markersize=5, label="Cold start (random init)")
    ax1.plot(d[mask], warm_it[mask], color=_COLORS["warm"], lw=2.0, marker="s",
             markersize=5, label="Warm start (parameter transfer)")
    ax1.set_ylabel("Mean Iterations to Convergence", fontsize=12)
    ax1.set_title(
        f"Warm-Start vs Cold-Start: {molecule_name} — {ansatz_name}",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    if speedup_ratio is not None:
        sp = np.array(speedup_ratio, dtype=float)
        ax1b = ax1.twinx()
        sp_mask = mask & ~np.isnan(sp)
        ax1b.plot(d[sp_mask], sp[sp_mask], color="#ff9f1c", lw=1.5,
                  linestyle=":", label="Speedup ratio (×)")
        ax1b.axhline(1.0, color="gray", linestyle="--", lw=0.8)
        ax1b.set_ylabel("Speedup Ratio (cold / warm)", fontsize=11, color="#ff9f1c")
        ax1b.tick_params(axis="y", labelcolor="#ff9f1c")
        ax1b.legend(fontsize=10, loc="upper left")

    # --- Panel 2: Chemical accuracy rates ---
    cold_ca = np.array(cold_chem_acc, dtype=float)
    warm_ca = np.array(warm_chem_acc, dtype=float)
    ca_mask = ~np.isnan(cold_ca) & ~np.isnan(warm_ca)
    width = np.diff(d).min() * 0.35 if len(d) > 1 else 0.04
    ax2.bar(d[ca_mask] - width / 2, cold_ca[ca_mask] * 100,
            width=width, color=_COLORS["cold"], alpha=0.8, label="Cold start")
    ax2.bar(d[ca_mask] + width / 2, warm_ca[ca_mask] * 100,
            width=width, color=_COLORS["warm"], alpha=0.8, label="Warm start")
    ax2.axhline(50, color="gray", linestyle="--", lw=1.0)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Bond Length (Å)", fontsize=12)
    ax2.set_ylabel("Chemical Accuracy Rate (%)", fontsize=12)
    ax2.set_title("Chemical Accuracy Rate by Starting Strategy", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = _fig_path(f"warm_start_comparison_{molecule_name}_{ansatz_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 9. Pareto front (accuracy vs circuit cost)
# ---------------------------------------------------------------------------

def plot_pareto_front(
    errors: List[float],
    costs: List[int],
    labels: List[str],
    molecule_name: str,
    output_dir: str = _FIGURE_DIR,
) -> str:
    """Scatter plot of energy error vs CNOT count with Pareto frontier overlay."""
    fig, ax = plt.subplots(figsize=(11, 7))

    err_arr = np.array(errors, dtype=float)
    cost_arr = np.array(costs)
    mask = ~np.isnan(err_arr)

    sc = ax.scatter(cost_arr[mask], err_arr[mask], c=err_arr[mask],
                    cmap="plasma_r", s=120, zorder=5, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="|E_VQE - E_exact| (Hartree)")

    for i in np.where(mask)[0]:
        ax.annotate(
            labels[i],
            (cost_arr[i], err_arr[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    # Compute and draw Pareto frontier
    pareto_mask = mask.copy()
    pareto_pts = sorted(
        [(cost_arr[i], err_arr[i]) for i in np.where(pareto_mask)[0]],
        key=lambda x: x[0]
    )
    pareto_front = []
    min_err = float("inf")
    for cost, err in pareto_pts:
        if err < min_err:
            pareto_front.append((cost, err))
            min_err = err
    if len(pareto_front) >= 2:
        px, py = zip(*pareto_front)
        ax.step(px, py, where="post", color=_COLORS["warm"], lw=2.0,
                linestyle="--", label="Pareto frontier", zorder=6)

    ax.axhline(0.0016, color=_COLORS["chem_acc"], linestyle="--",
               lw=1.8, label="Chemical accuracy (1.6 mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("Hardware Cost (CNOT Count)", fontsize=13)
    ax.set_ylabel("|E_VQE - E_exact| (Hartree)", fontsize=13)
    ax.set_title(
        f"Accuracy vs Complexity Pareto Front: {molecule_name}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = _fig_path(f"pareto_front_{molecule_name}.png", output_dir)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
