"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization"""

import pytest
import networkx as nx
import numpy as np
from src.hamiltonian_builder import HamiltonianBuilder

def test_hamiltonian_maxcut_simple():
    # Create simple 2-node graph with 1 edge
    G = nx.Graph()
    G.add_edge(0, 1)
    
    builder = HamiltonianBuilder()
    hamiltonian, offset = builder.build_maxcut_hamiltonian(G)
    
    # H = -0.5 * Z0 Z1, offset = 0.5
    assert len(hamiltonian.paulis) == 1
    assert hamiltonian.paulis[0].to_label() == 'ZZ'
    assert np.isclose(hamiltonian.coeffs[0], -0.5)
    assert np.isclose(offset, 0.5)

def test_hamiltonian_triangle():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    builder = HamiltonianBuilder()
    hamiltonian, offset = builder.build_maxcut_hamiltonian(G)
    
    assert len(hamiltonian.paulis) == 3
    labels = [p.to_label() for p in hamiltonian.paulis]
    # For n=3, the strings are IIZ, IZI, ZII etc. depending on edges
    assert 'IZZ' in labels or 'ZZI' in labels or 'ZIZ' in labels
    assert np.isclose(offset, 1.5)


def test_qubo_builder_matches_standard_maxcut_mapping():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=3.0)

    builder = HamiltonianBuilder()
    standard_hamiltonian, standard_offset = builder.build_maxcut_hamiltonian(G)
    qubo_hamiltonian, qubo_offset = builder.build_maxcut_hamiltonian_qubo(G)

    assert np.isclose(standard_offset, qubo_offset)
    assert standard_hamiltonian.equiv(qubo_hamiltonian)
