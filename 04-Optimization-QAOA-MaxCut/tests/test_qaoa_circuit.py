"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization"""

import pytest
import numpy as np
import networkx as nx
from src.qaoa_circuit import QAOACircuitBuilder
from src.hamiltonian_builder import HamiltonianBuilder

def test_circuit_builder():
    G = nx.Graph()
    G.add_edge(0, 1)
    
    h_builder = HamiltonianBuilder()
    hamiltonian, _ = h_builder.build_maxcut_hamiltonian(G)
    
    c_builder = QAOACircuitBuilder(n_qubits=2, p=1)
    circuit = c_builder.build_qaoa_circuit(hamiltonian)
    
    assert circuit.num_qubits == 2
    assert circuit.num_parameters == 2
    
def test_initial_parameters():
    c_builder = QAOACircuitBuilder(n_qubits=2, p=2)
    params = c_builder.get_initial_parameters("linear")
    
    assert len(params) == 4
    assert np.isclose(params[0], np.pi / 2)
    assert np.isclose(params[1], np.pi / 4)
