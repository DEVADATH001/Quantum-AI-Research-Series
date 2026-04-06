"""Regression tests for QAOA parameter optimization."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.qaoa_optimizer import MaxCutQAOAProblem, QAOAOptimizer
from src.runtime_executor import RuntimeExecutor


def test_optimizer_reports_expected_value_and_representative_bitstring():
    graph = nx.Graph()
    graph.add_edge(0, 1)

    problem = MaxCutQAOAProblem(
        graph=graph,
        p=1,
        executor=RuntimeExecutor(mode="local", shots=256, seed=7),
        seed=7,
        analysis_shots=256,
    )

    optimizer = QAOAOptimizer(p=1, optimizer_type="COBYLA", maxiter=30, n_initial_points=2, seed=7)
    result = optimizer.optimize(
        objective_function=problem.objective_function,
        n_qubits=2,
        graph=graph,
        solution_decoder=problem.decode_solution,
    )

    assert result.n_evaluations > 0
    assert len(result.history) == result.n_evaluations
    assert result.cut_value is not None
    assert result.cut_value <= 1.0
    assert result.solution_bitstring in {"01", "10"}
    assert result.sampled_cut_value == 1.0
    assert result.most_likely_bitstring in {"01", "10"}
    assert result.best_sampled_bitstring in {"01", "10"}
    assert result.best_sampled_cut_value == 1.0
    assert result.bitstring_probability is not None


def test_analysis_mode_none_suppresses_representative_bitstrings():
    graph = nx.Graph()
    graph.add_edge(0, 1)

    problem = MaxCutQAOAProblem(
        graph=graph,
        p=1,
        executor=RuntimeExecutor(mode="local", shots=128, seed=3),
        seed=3,
        analysis_shots=128,
        analysis_mode="none",
    )

    solution = problem.decode_solution(np.array([0.2, 0.3]))

    assert solution["cut_value"] is not None
    assert solution["bitstring"] is None
    assert solution["sampled_cut_value"] is None
    assert solution["best_sampled_bitstring"] is None
    assert solution["measurement_counts"] is None


def test_build_initial_points_includes_warm_start_first():
    optimizer = QAOAOptimizer(p=2, n_initial_points=3, seed=11)
    warm_start = np.array([0.1, 0.2, 0.3, 0.4])

    initial_points = optimizer.build_initial_points(warm_start_params=warm_start, n_points=3)

    assert initial_points.shape == (3, 4)
    assert np.allclose(initial_points[0], warm_start)


def test_extend_parameters_for_next_depth_repeats_last_layer():
    extended = QAOAOptimizer.extend_parameters_for_next_depth(np.array([0.1, 0.2, 0.7, 0.8]))

    assert np.allclose(extended, np.array([0.1, 0.2, 0.7, 0.8, 0.7, 0.8]))


def test_spsa_optimizer_uses_configured_hyperparameters(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class DummySPSA:
        def __init__(self, maxiter, learning_rate, perturbation):
            captured["maxiter"] = maxiter
            captured["learning_rate"] = learning_rate
            captured["perturbation"] = perturbation

        def minimize(self, fun, x0, bounds=None):
            value = fun(np.asarray(x0, dtype=float))
            return SimpleNamespace(x=np.asarray(x0, dtype=float), fun=value, nfev=1, success=True)

    import qiskit_algorithms.optimizers as qiskit_optimizers

    monkeypatch.setattr(qiskit_optimizers, "SPSA", DummySPSA)

    optimizer = QAOAOptimizer(
        p=1,
        optimizer_type="SPSA",
        maxiter=7,
        seed=5,
        spsa_learning_rate=0.23,
        spsa_perturbation=0.17,
    )
    result = optimizer.optimize(
        objective_function=lambda params: float(np.sum(np.asarray(params, dtype=float) ** 2)),
        n_qubits=2,
        initial_params=np.array([0.4, 0.6]),
    )

    assert result.n_evaluations == 1
    assert captured["maxiter"] == 7
    assert captured["learning_rate"] == 0.23
    assert captured["perturbation"] == 0.17
