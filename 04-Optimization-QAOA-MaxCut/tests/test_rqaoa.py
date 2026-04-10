"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization"""

import pytest
import networkx as nx
from src.classical_solver import ClassicalSolver
from src.graph_generator import GraphGenerator
from src.rqaoa_engine import RQAOAEngine

def test_rqaoa_simple_graph():
    # Triangle graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    # Minimum problem size 2 so it forces at least one recursion step
    engine = RQAOAEngine(p=1, min_problem_size=2)
    result = engine.solve(G)
    
    # Triangle max cut is 2
    assert result.cut_value == 2
    assert len(result.solution_bitstring) == 3


def test_rqaoa_exact_fallback_on_regular_graph():
    graph = GraphGenerator(seed=42).generate_d_regular_graph(n_nodes=6, degree=3, seed=42)
    exact = ClassicalSolver().solve_exact(graph)

    engine = RQAOAEngine(
        p=1,
        correlation_threshold=0.99,
        min_problem_size=4,
        max_depth=2,
        force_fallback_elimination=False,
    )
    result = engine.solve(graph, optimal_value=exact.optimal_value)

    assert len(result.solution_bitstring) == 6
    assert result.cut_value == exact.optimal_value
    assert result.approximation_ratio == 1.0


def test_opposite_elimination_tracks_constant_offset_exactly():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])

    engine = RQAOAEngine()
    reduced = engine._reduce_problem(
        graph,
        [{"var1": 0, "var2": 1, "relationship": "opposite"}],
    )

    assert reduced.number_of_nodes() == 2
    assert reduced.has_edge(0, 2)
    assert reduced[0][2]["weight"] == -1.0
    assert reduced.graph["constant_offset"] == 2.0

    for reduced_bits in ("00", "01", "10", "11"):
        full_bits = reduced_bits[0] + ("1" if reduced_bits[0] == "0" else "0") + reduced_bits[1]
        original_value = ClassicalSolver().compute_cut_value(
            graph,
            [index for index, bit in enumerate(full_bits) if bit == "0"],
        )
        reduced_value = reduced.graph["constant_offset"] + ClassicalSolver().compute_cut_value(
            reduced,
            [node for node, bit in zip(sorted(reduced.nodes()), reduced_bits) if bit == "0"],
        )
        assert original_value == reduced_value


def test_count_based_rqaoa_correlations_are_symmetric():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    engine = RQAOAEngine(correlation_method="counts", analysis_shots=512)
    correlations = engine._compute_pair_correlations_from_counts(
        4,
        {
            "0000": 128,
            "0011": 128,
            "1100": 128,
            "1111": 128,
        },
    )

    assert correlations.shape == (4, 4)
    assert (correlations.diagonal() == 1.0).all()
    assert correlations[0, 1] == correlations[1, 0]
