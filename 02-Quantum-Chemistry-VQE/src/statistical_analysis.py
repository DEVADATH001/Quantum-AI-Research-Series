"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Statistical analysis utilities for multi-seed VQE experiments.

Research context:
    This module provides the statistical backbone that transforms single-seed
    anecdotal results into publication-grade evidence. It computes:
      - Mean and standard deviation of VQE energies across seeds
      - 95% bootstrap confidence intervals
      - Chemical-accuracy convergence rates per distance
      - Warm-start speedup metrics (iterations saved vs cold start)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core aggregate statistics
# ---------------------------------------------------------------------------

def _safe_nanmean(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


def _safe_nanstd(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")


def bootstrap_ci_95(
    values: List[float],
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Return (low, high) 95% bootstrap confidence interval.

    Falls back to mean ± 2*std if fewer than 3 valid samples.
    """
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n == 0:
        return float("nan"), float("nan")
    if n < 3:
        mu = float(np.mean(valid))
        sd = float(np.std(valid, ddof=0))
        return mu - 2 * sd, mu + 2 * sd
    rng = rng or np.random.default_rng(42)
    boot_means = np.array(
        [rng.choice(valid, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def chemical_accuracy_rate(
    errors_per_seed: List[List[float]],
    threshold: float = 0.0016,
) -> List[float]:
    """Return fraction of seeds achieving chemical accuracy at each bond length.

    Args:
        errors_per_seed: List[n_seeds][n_distances] absolute errors.
        threshold: Chemical accuracy threshold in Hartree.

    Returns:
        List[n_distances] rates in [0, 1].
    """
    if not errors_per_seed:
        return []
    n_distances = len(errors_per_seed[0])
    rates: List[float] = []
    for d_idx in range(n_distances):
        vals = [
            errors_per_seed[s][d_idx]
            for s in range(len(errors_per_seed))
            if not np.isnan(errors_per_seed[s][d_idx])
        ]
        if not vals:
            rates.append(float("nan"))
        else:
            rates.append(float(sum(v <= threshold for v in vals) / len(vals)))
    return rates


# ---------------------------------------------------------------------------
# Per-ansatz statistics container
# ---------------------------------------------------------------------------

@dataclass
class AnsatzStats:
    """Statistical summary for one ansatz across seeds."""

    n_seeds: int
    mean_energy: List[float]
    std_energy: List[float]
    ci95_low: List[float]
    ci95_high: List[float]
    mean_error: List[float]
    std_error: List[float]
    chem_acc_rate: List[float]
    per_seed_energies: List[List[float]]
    per_seed_errors: List[List[float]]

    def to_dict(self) -> Dict:
        return {
            "n_seeds": self.n_seeds,
            "mean_energy": self.mean_energy,
            "std_energy": self.std_energy,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "chem_acc_rate": self.chem_acc_rate,
            "per_seed_energies": self.per_seed_energies,
            "per_seed_errors": self.per_seed_errors,
        }


def aggregate_multi_seed_results(
    seed_results: List[Dict],
    ansatz_names: List[str],
    distances: List[float],
    threshold: float = 0.0016,
) -> Dict[str, AnsatzStats]:
    """Aggregate a list of per-seed PESGenerator.run() result dicts.

    Args:
        seed_results: One dict per seed (each the return value of PESGenerator.run()).
        ansatz_names: Ordered ansatz names to process.
        distances: Bond-length grid.
        threshold: Chemical accuracy threshold in Hartree.

    Returns:
        Dict mapping ansatz_name → AnsatzStats.
    """
    n_distances = len(distances)
    stats: Dict[str, AnsatzStats] = {}

    for name in ansatz_names:
        per_seed_energies: List[List[float]] = []
        per_seed_errors: List[List[float]] = []

        for result in seed_results:
            energies = result.get("vqe_energies", {}).get(name, [float("nan")] * n_distances)
            exact = result.get("exact_energies", [float("nan")] * n_distances)
            padded_e: List[float] = [float(v) for v in energies[:n_distances]]
            padded_x: List[float] = [float(v) for v in exact[:n_distances]]
            # Pad if shorter
            while len(padded_e) < n_distances:
                padded_e.append(float("nan"))
            while len(padded_x) < n_distances:
                padded_x.append(float("nan"))

            errors = [
                abs(padded_e[i] - padded_x[i])
                if not (np.isnan(padded_e[i]) or np.isnan(padded_x[i]))
                else float("nan")
                for i in range(n_distances)
            ]
            per_seed_energies.append(padded_e)
            per_seed_errors.append(errors)

        # Per-distance aggregation
        mean_energy, std_energy, ci95_low, ci95_high = [], [], [], []
        mean_error, std_error = [], []
        rng = np.random.default_rng(42)

        for d_idx in range(n_distances):
            e_vals = [per_seed_energies[s][d_idx] for s in range(len(seed_results))]
            err_vals = [per_seed_errors[s][d_idx] for s in range(len(seed_results))]

            mean_energy.append(_safe_nanmean(e_vals))
            std_energy.append(_safe_nanstd(e_vals))
            lo, hi = bootstrap_ci_95(e_vals, rng=rng)
            ci95_low.append(lo)
            ci95_high.append(hi)
            mean_error.append(_safe_nanmean(err_vals))
            std_error.append(_safe_nanstd(err_vals))

        chem_acc = chemical_accuracy_rate(per_seed_errors, threshold=threshold)

        stats[name] = AnsatzStats(
            n_seeds=len(seed_results),
            mean_energy=mean_energy,
            std_energy=std_energy,
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            mean_error=mean_error,
            std_error=std_error,
            chem_acc_rate=chem_acc,
            per_seed_energies=per_seed_energies,
            per_seed_errors=per_seed_errors,
        )

    return stats


# ---------------------------------------------------------------------------
# Warm-start speedup metrics
# ---------------------------------------------------------------------------

@dataclass
class WarmStartSpeedup:
    """Metrics comparing warm vs cold VQE runs."""

    distances: List[float]
    ansatz: str
    cold_iterations: List[float]       # mean iterations without warm-start
    warm_iterations: List[float]       # mean iterations with warm-start
    speedup_ratio: List[float]         # cold / warm (>1 means warm is faster)
    cold_chem_acc_rate: List[float]
    warm_chem_acc_rate: List[float]

    def mean_speedup(self) -> float:
        vals = [v for v in self.speedup_ratio if not np.isnan(v) and v > 0]
        return float(np.mean(vals)) if vals else float("nan")

    def to_dict(self) -> Dict:
        return {
            "distances": self.distances,
            "ansatz": self.ansatz,
            "cold_iterations": self.cold_iterations,
            "warm_iterations": self.warm_iterations,
            "speedup_ratio": self.speedup_ratio,
            "cold_chem_acc_rate": self.cold_chem_acc_rate,
            "warm_chem_acc_rate": self.warm_chem_acc_rate,
            "mean_speedup": self.mean_speedup(),
        }


def compute_warm_start_speedup(
    cold_results: List[Dict],
    warm_results: List[Dict],
    ansatz: str,
    distances: List[float],
    threshold: float = 0.0016,
) -> WarmStartSpeedup:
    """Compute warm-start speedup from paired cold/warm run lists.

    Args:
        cold_results: Per-seed results with warm_start=False.
        warm_results: Per-seed results with warm_start=True.
        ansatz: Ansatz name to evaluate.
        distances: Bond-length grid.
        threshold: Chemical accuracy threshold.

    Returns:
        WarmStartSpeedup with per-distance stats.
    """
    n_d = len(distances)

    def _extract_iters(results: List[Dict], name: str) -> List[List[float]]:
        all_iters = []
        for res in results:
            histories = res.get("histories", {}).get(name, {})
            iters_per_d: List[float] = []
            for bond in distances:
                key = f"{float(bond):.3f}"
                hist = histories.get(key, [])
                if hist and isinstance(hist, list) and "iteration" in hist[-1]:
                    iters_per_d.append(float(hist[-1]["iteration"]))
                else:
                    iters_per_d.append(float("nan"))
            all_iters.append(iters_per_d)
        return all_iters

    def _extract_errors(results: List[Dict], name: str) -> List[List[float]]:
        all_errs = []
        for res in results:
            energies = res.get("vqe_energies", {}).get(name, [float("nan")] * n_d)
            exact = res.get("exact_energies", [float("nan")] * n_d)
            errs = [
                abs(float(energies[i]) - float(exact[i]))
                if i < len(energies) and i < len(exact)
                   and not (np.isnan(float(energies[i])) or np.isnan(float(exact[i])))
                else float("nan")
                for i in range(n_d)
            ]
            all_errs.append(errs)
        return all_errs

    cold_iters = _extract_iters(cold_results, ansatz)
    warm_iters = _extract_iters(warm_results, ansatz)
    cold_errors = _extract_errors(cold_results, ansatz)
    warm_errors = _extract_errors(warm_results, ansatz)

    cold_mean_iters, warm_mean_iters, speedup = [], [], []
    for d_idx in range(n_d):
        c = _safe_nanmean([cold_iters[s][d_idx] for s in range(len(cold_results))])
        w = _safe_nanmean([warm_iters[s][d_idx] for s in range(len(warm_results))])
        cold_mean_iters.append(c)
        warm_mean_iters.append(w)
        speedup.append(c / w if (not np.isnan(c) and not np.isnan(w) and w > 0) else float("nan"))

    return WarmStartSpeedup(
        distances=distances,
        ansatz=ansatz,
        cold_iterations=cold_mean_iters,
        warm_iterations=warm_mean_iters,
        speedup_ratio=speedup,
        cold_chem_acc_rate=chemical_accuracy_rate(cold_errors, threshold),
        warm_chem_acc_rate=chemical_accuracy_rate(warm_errors, threshold),
    )


# ---------------------------------------------------------------------------
# Summary table (human-readable markdown)
# ---------------------------------------------------------------------------

def build_summary_table(
    molecule: str,
    distances: List[float],
    stats: Dict[str, AnsatzStats],
    exact_energies: List[float],
) -> str:
    """Return a markdown summary table of mean errors and chem-acc rates."""
    lines = [
        f"# Multi-Seed Statistical Summary: {molecule}",
        "",
        f"Seeds: {next(iter(stats.values())).n_seeds if stats else 0}",
        "",
    ]

    try:
        from tabulate import tabulate  # type: ignore

        headers = ["Bond (Å)", "E_exact (Ha)"] + [
            f"{name} Err±σ (Ha) | CA%"
            for name in stats.keys()
        ]
        rows = []
        for i, d in enumerate(distances):
            row = [f"{d:.2f}", f"{exact_energies[i]:.6f}" if not np.isnan(exact_energies[i]) else "NaN"]
            for s in stats.values():
                mu = s.mean_error[i]
                sd = s.std_error[i]
                ca = s.chem_acc_rate[i]
                mu_s = f"{mu:.4f}" if not np.isnan(mu) else "NaN"
                sd_s = f"{sd:.4f}" if not np.isnan(sd) else "NaN"
                ca_s = f"{ca*100:.0f}%" if not np.isnan(ca) else "NaN"
                row.append(f"{mu_s}±{sd_s} | {ca_s}")
            rows.append(row)
        lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        lines.append("[tabulate not installed — install with: pip install tabulate]")
        for i, d in enumerate(distances):
            line = f"d={d:.2f}:"
            for name, s in stats.items():
                mu = s.mean_error[i]
                ca = s.chem_acc_rate[i]
                line += f"  {name}: err={mu:.4f} ca={ca*100:.0f}%"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)