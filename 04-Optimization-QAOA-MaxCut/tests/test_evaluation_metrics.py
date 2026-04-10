"""Regression tests for evaluation metrics."""

import networkx as nx
import numpy as np

from src.evaluation_metrics import EvaluationMetrics


def test_energy_distribution_uses_energies_not_probabilities():
    graph = nx.Graph()
    graph.add_edge(0, 1)

    metrics = EvaluationMetrics()
    stats = metrics.compute_energy_distribution(
        samples=[("00", 0.9), ("01", 0.1)],
        graph=graph,
    )

    expected_mean = 0.9 * 0.0 + 0.1 * 1.0
    assert np.isclose(stats["mean"], expected_mean)
    assert stats["best_bitstring"] == "01"
    assert np.isclose(stats["best_energy"], 1.0)


def test_bootstrap_confidence_interval_contains_sample_mean():
    metrics = EvaluationMetrics()
    values = [0.7, 0.8, 0.9, 1.0]

    lower, upper = metrics.bootstrap_confidence_interval(values, n_bootstrap=500, seed=7)

    assert lower <= np.mean(values) <= upper


def test_paired_method_test_detects_positive_advantage():
    metrics = EvaluationMetrics()
    result = metrics.paired_method_test(
        [0.9, 0.92, 0.94, 0.96],
        [0.7, 0.72, 0.74, 0.76],
        n_resamples=1000,
        seed=7,
    )

    assert result["mean_difference"] > 0
    assert result["ci_lower"] > 0
