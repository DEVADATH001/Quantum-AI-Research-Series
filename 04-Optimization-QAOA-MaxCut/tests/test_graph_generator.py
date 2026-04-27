"""Tests for graph-generation utilities."""

import networkx as nx

from src.graph_generator import GraphGenerator


def test_communication_mesh_graph_is_connected_and_weighted():
    graph = GraphGenerator(seed=42).generate_communication_mesh_graph(
        n_nodes=8,
        degree=3,
        seed=42,
    )

    assert graph.graph["type"] == "communication_mesh"
    assert nx.is_connected(graph)
    assert graph.number_of_edges() >= 8
    for _, _, data in graph.edges(data=True):
        assert data["weight"] > 0.0
        assert 0.0 <= data["latency"] <= 1.0
        assert 0.0 <= data["interference"] <= 1.0
        assert 0.0 <= data["reliability"] <= 1.0
        assert data["bandwidth"] > 0.0
