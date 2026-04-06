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
