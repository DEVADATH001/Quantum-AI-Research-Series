"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization"""

import pytest
import networkx as nx
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
