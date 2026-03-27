"""Statistical utilities for experiment review and reporting."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any, Callable

import numpy as np


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 12345,
) -> list[float] | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if arr.size == 1:
        scalar = float(arr[0])
        return [scalar, scalar]

    alpha = max(0.0, min(0.5, (1.0 - float(confidence)) / 2.0))
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower = float(np.quantile(sample_means, alpha))
    upper = float(np.quantile(sample_means, 1.0 - alpha))
    return [lower, upper]


def summarize_scalar_distribution(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci95": None,
        }

    std = float(np.std(arr))
    sem = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": sem,
        "ci95": bootstrap_mean_ci(arr, confidence=confidence),
    }


def hodges_lehmann_estimate(diffs: list[float] | np.ndarray) -> float | None:
    """Return the paired-sample Hodges-Lehmann estimator."""

    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    walsh_averages = []
    for idx in range(arr.size):
        for jdx in range(idx, arr.size):
            walsh_averages.append((arr[idx] + arr[jdx]) / 2.0)
    return float(np.median(np.asarray(walsh_averages, dtype=float)))


def holm_correct(
    pvalues: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Apply the Holm step-down correction to a named family of p-values."""

    finite = [(name, float(value)) for name, value in pvalues.items() if value is not None]
    corrected: dict[str, float | None] = {name: None for name in pvalues}
    if not finite:
        return corrected

    ordered = sorted(finite, key=lambda item: item[1])
    m = len(ordered)
    running_max = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        adjusted = min(1.0, (m - index) * pvalue)
        running_max = max(running_max, adjusted)
        corrected[name] = float(running_max)
    return corrected


def _paired_seed_values(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    metric_fn: Callable[[dict[str, Any]], float | None],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    by_seed_a = {int(record["seed"]): record for record in records_a}
    by_seed_b = {int(record["seed"]): record for record in records_b}
    shared_seeds = sorted(set(by_seed_a).intersection(by_seed_b))
    values_a: list[float] = []
    values_b: list[float] = []
    retained_seeds: list[int] = []

    for seed in shared_seeds:
        value_a = metric_fn(by_seed_a[seed])
        value_b = metric_fn(by_seed_b[seed])
        if value_a is None or value_b is None:
            continue
        values_a.append(float(value_a))
        values_b.append(float(value_b))
        retained_seeds.append(seed)

    return retained_seeds, np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)


def sign_flip_pvalue(
    diffs: list[float] | np.ndarray,
    monte_carlo_samples: int = 20000,
    seed: int = 2026,
) -> float | None:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    observed = abs(float(np.mean(arr)))
    abs_arr = np.abs(arr)

    if arr.size <= 16:
        all_signs = np.array(list(product([-1.0, 1.0], repeat=arr.size)), dtype=float)
        null_stats = np.abs((all_signs * abs_arr).mean(axis=1))
    else:
        rng = np.random.default_rng(seed)
        random_signs = rng.choice(np.array([-1.0, 1.0]), size=(int(monte_carlo_samples), arr.size))
        null_stats = np.abs((random_signs * abs_arr).mean(axis=1))

    return float(np.mean(null_stats >= observed - 1e-12))


def paired_method_comparison(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    metric_name: str,
    metric_fn: Callable[[dict[str, Any]], float | None],
) -> dict[str, Any]:
    seeds, values_a, values_b = _paired_seed_values(records_a, records_b, metric_fn=metric_fn)
    if values_a.size == 0:
        return {
            "metric": metric_name,
            "paired_seed_count": 0,
            "shared_seeds": [],
            "mean_difference": None,
            "difference_ci95": None,
            "sign_flip_pvalue": None,
            "wins_a": 0,
            "wins_b": 0,
            "ties": 0,
        }

    diffs = values_a - values_b
    return {
        "metric": metric_name,
        "paired_seed_count": int(values_a.size),
        "shared_seeds": seeds,
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
        "hodges_lehmann_difference": hodges_lehmann_estimate(diffs),
        "difference_sem": float(np.std(diffs) / np.sqrt(diffs.size)) if diffs.size > 1 else 0.0,
        "difference_ci95": bootstrap_mean_ci(diffs),
        "sign_flip_pvalue": sign_flip_pvalue(diffs),
        "wins_a": int(np.sum(diffs > 1e-12)),
        "wins_b": int(np.sum(diffs < -1e-12)),
        "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
        "paired_differences": diffs.tolist(),
    }
