"""Regression tests for execution backends."""

import networkx as nx

from src.hamiltonian_builder import HamiltonianBuilder
from src.qaoa_circuit import QAOACircuitBuilder
from src.runtime_executor import RuntimeExecutor


def test_local_runtime_executor_handles_unmeasured_circuit():
    graph = nx.Graph()
    graph.add_edge(0, 1)

    hamiltonian, offset = HamiltonianBuilder().build_maxcut_hamiltonian(graph)
    circuit = QAOACircuitBuilder(n_qubits=2, p=1).build_qaoa_circuit_simple(
        graph,
        gamma=0.5,
        beta=0.2,
    )

    result = RuntimeExecutor(mode="local", shots=64, seed=7).execute_circuit(
        circuit,
        hamiltonian,
        offset=offset,
    )

    assert isinstance(result.expectation_value, float)
    assert isinstance(result.objective_value, float)
    assert result.objective_value >= result.expectation_value
    assert result.measurement_counts is not None
    assert sum(result.measurement_counts.values()) == 64
    assert result.sampled_bitstring is not None
    assert len(result.sampled_bitstring) == 2
