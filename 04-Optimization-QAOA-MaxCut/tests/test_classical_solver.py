"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization"""

import pytest
import networkx as nx
from src.classical_solver import ClassicalSolver, ApproximateSolver

def test_exact_solver_triangle():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    solver = ClassicalSolver()
    result = solver.solve_exact(G)
    
    # Max cut of a triangle is 2 edges
    assert result.optimal_value == 2
    assert len(result.optimal_bitstrings) > 0
    # The bitstrings should have exactly one node in a different partition
    for bitstring in result.optimal_bitstrings:
        assert bitstring.count('1') == 1 or bitstring.count('0') == 1

def test_goemans_williamson():
    pytest.importorskip("cvxpy")
    G = nx.Graph()
    # A larger graph where GW can be tested
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    
    # Max cut is 4
    cut_value, partition = ApproximateSolver.goemans_williamson(G, num_trials=10)
    assert cut_value > 0
    assert len(partition) == 4

def test_local_search_is_reproducible_with_seed():
    G = nx.cycle_graph(6)

    value_a, bitstring_a = ApproximateSolver.solve_local_search(
        G,
        max_iterations=50,
        seed=123,
    )
    value_b, bitstring_b = ApproximateSolver.solve_local_search(
        G,
        max_iterations=50,
        seed=123,
    )

    assert value_a == value_b
    assert bitstring_a == bitstring_b
