"""Author: DEVADATH H K

Graph Generator Module

Generates graph structures for the Max-Cut problem, including:
- D-regular graphs (robot communication mesh networks)
- Erdős-Rényi random graphs
- Barabási-Albert scale-free graphs"""

import logging
from typing import Optional, Tuple, List
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

class GraphGenerator:
    """
    Generates various graph types for Max-Cut optimization.
    
    Supports D-regular graphs, Erdős-Rényi graphs, and Barabási-Albert graphs.
    Primarily used to generate robot communication mesh networks.
    """
    
    def __init__(self, seed: Optional[int] = 42) -> None:
        """
        Initialize the graph generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info(f"GraphGenerator initialized with seed={seed}")
    
    def generate_d_regular_graph(
        self, 
        n_nodes: int, 
        degree: int,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Generate a D-regular random graph.
        
        In a D-regular graph, each node is connected to exactly D neighbors.
        This simulates a robot communication mesh where each robot has
        a fixed number of communication links.
        
        Args:
            n_nodes: Number of nodes (robots)
            degree: Degree D (number of neighbors per node)
            seed: Random seed (overrides instance seed)
            
        Returns:
            NetworkX graph
            
        Raises:
            ValueError: If degree >= n_nodes (impossible to create D-regular)
        """
        if degree >= n_nodes:
            raise ValueError(
                f"Degree ({degree}) must be less than n_nodes ({n_nodes})"
            )
        
        if (n_nodes * degree) % 2 != 0:
            raise ValueError(
                f"n_nodes * degree must be even for D-regular graph. "
                f"Got {n_nodes} * {degree} = {n_nodes * degree}"
            )
        
        seed = seed if seed is not None else self.seed
        logger.info(f"Generating D-regular graph: n={n_nodes}, D={degree}")
        
        graph = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
        graph = self._add_metadata(graph, "d_regular", n_nodes, degree)
        
        logger.info(f"Generated graph with {graph.number_of_edges()} edges")
        return graph
    
    def generate_erdos_renyi_graph(
        self,
        n_nodes: int,
        probability: float,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Generate an Erdős-Rényi random graph.
        
        Each edge is included independently with probability p.
        
        Args:
            n_nodes: Number of nodes
            probability: Edge probability p ∈ [0, 1]
            seed: Random seed
            
        Returns:
            NetworkX graph
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        
        seed = seed if seed is not None else self.seed
        logger.info(
            f"Generating Erdős-Rényi graph: n={n_nodes}, p={probability}"
        )
        
        graph = nx.erdos_renyi_graph(n=n_nodes, p=probability, seed=seed)
        graph = self._add_metadata(graph, "erdos_renyi", n_nodes, probability)
        
        logger.info(f"Generated graph with {graph.number_of_edges()} edges")
        return graph
    
    def generate_barabasi_albert_graph(
        self,
        n_nodes: int,
        m: int,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Generate a Barabási-Albert scale-free graph.
        
        Uses preferential attachment to create scale-free networks.
        
        Args:
            n_nodes: Number of nodes
            m: Number of edges to attach from each new node
            seed: Random seed
            
        Returns:
            NetworkX graph
        """
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        
        seed = seed if seed is not None else self.seed
        logger.info(f"Generating Barabási-Albert graph: n={n_nodes}, m={m}")
        
        graph = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
        graph = self._add_metadata(graph, "barabasi_albert", n_nodes, m)
        
        logger.info(f"Generated graph with {graph.number_of_edges()} edges")
        return graph
    
    def generate_robot_mesh(
        self,
        n_robots: int,
        connectivity: int,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Generate a robot communication mesh network.
        
        This is the primary method for creating test problems.
        Uses D-regular graph as the underlying model.
        
        Args:
            n_robots: Number of robots (nodes)
            connectivity: Number of communication links per robot
            seed: Random seed
            
        Returns:
            NetworkX graph representing the mesh network
        """
        logger.info(
            f"Generating robot mesh: {n_robots} robots, "
            f"{connectivity} connections each"
        )
        
        graph = self.generate_d_regular_graph(
            n_nodes=n_robots,
            degree=connectivity,
            seed=seed
        )
        
        # Add robot-specific metadata
        graph.graph["type"] = "robot_mesh"
        graph.graph["description"] = (
            f"Robot communication mesh with {n_robots} nodes"
        )
        
        return graph

    def generate_communication_mesh_graph(
        self,
        n_nodes: int,
        degree: int,
        seed: Optional[int] = None,
        area_size: float = 1.0,
        reliability_scale: float = 0.35,
    ) -> nx.Graph:
        """
        Generate a weighted communication-style mesh graph.

        Nodes are embedded in 2D space and connected by a symmetric
        nearest-neighbor rule. Edge weights encode a communication-conflict
        score built from latency, local interference, and reliability loss.
        """
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if n_nodes < 2:
            raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")

        seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(seed)
        positions = rng.random((n_nodes, 2)) * float(area_size)
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)

        logger.info(
            "Generating communication mesh graph: n=%s, degree=%s, seed=%s",
            n_nodes,
            degree,
            seed,
        )

        graph = nx.Graph()
        for node in range(n_nodes):
            graph.add_node(node, position=tuple(float(value) for value in positions[node]))

        neighbor_count = max(1, min(degree, n_nodes - 1))
        for node in range(n_nodes):
            nearest_neighbors = np.argsort(distances[node])[:neighbor_count]
            for neighbor in nearest_neighbors:
                graph.add_edge(int(node), int(neighbor))

        self._connect_components_by_nearest_links(graph, distances)
        self._annotate_communication_edges(
            graph=graph,
            distances=distances,
            area_size=float(area_size),
            reliability_scale=float(reliability_scale),
        )

        graph.graph["type"] = "communication_mesh"
        graph.graph["param1"] = n_nodes
        graph.graph["param2"] = degree
        graph.graph["description"] = f"Weighted communication mesh with {n_nodes} nodes"
        graph.graph["weighted"] = True
        graph.graph["seed"] = seed

        logger.info(f"Generated communication mesh with {graph.number_of_edges()} edges")
        return graph
    
    def save_adjacency_list(
        self,
        graph: nx.Graph,
        filepath: str
    ) -> None:
        """
        Save graph as adjacency list format.
        
        Args:
            graph: NetworkX graph
            filepath: Output file path
        """
        nx.write_adjacency_list(graph, path=filepath)
        logger.info(f"Saved adjacency list to {filepath}")
    
    def load_adjacency_list(self, filepath: str) -> nx.Graph:
        """
        Load graph from adjacency list format.
        
        Args:
            filepath: Input file path
            
        Returns:
            NetworkX graph
        """
        graph = nx.read_adjlist(path=filepath, nodetype=int)
        logger.info(f"Loaded adjacency list from {filepath}")
        return graph
    
    def get_cut_value(
        self,
        graph: nx.Graph,
        partition: List[int]
    ) -> float:
        """
        Calculate the cut value for a given partition.
        
        Args:
            graph: NetworkX graph
            partition: List of node indices in one partition
            
        Returns:
            Number of edges crossing the cut
        """
        cut_set = set()
        other_nodes = set(graph.nodes()) - set(partition)
        
        for u, v in graph.edges():
            if (u in partition and v in other_nodes) or \
               (v in partition and u in other_nodes):
                cut_set.add((min(u, v), max(u, v)))
        
        cut_value = 0.0
        for u, v in cut_set:
            cut_value += float(graph[u][v].get("weight", 1.0))

        return cut_value
    
    def get_all_partitions(self, n_nodes: int) -> List[List[int]]:
        """
        Generate all possible partitions (for brute force).
        
        Args:
            n_nodes: Number of nodes
            
        Returns:
            List of partitions (each partition is a list of node indices)
        """
        partitions = []
        for i in range(1, 2**(n_nodes - 1)):
            partition = [
                j for j in range(n_nodes) if (i >> j) & 1
            ]
            partitions.append(partition)
        return partitions
    
    @staticmethod
    def _add_metadata(
        graph: nx.Graph,
        graph_type: str,
        param1: float,
        param2: float
    ) -> nx.Graph:
        """Add metadata to graph."""
        graph.graph["type"] = graph_type
        graph.graph["param1"] = param1
        graph.graph["param2"] = param2
        return graph

    @staticmethod
    def _connect_components_by_nearest_links(graph: nx.Graph, distances: np.ndarray) -> None:
        """Ensure the generated communication mesh is connected."""
        while graph.number_of_nodes() > 0 and not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            best_edge: Optional[Tuple[int, int]] = None
            best_distance = float("inf")
            for index, left_component in enumerate(components):
                for right_component in components[index + 1 :]:
                    for left in left_component:
                        for right in right_component:
                            distance = float(distances[left, right])
                            if distance < best_distance:
                                best_distance = distance
                                best_edge = (int(left), int(right))
            if best_edge is None:
                break
            graph.add_edge(*best_edge)

    @staticmethod
    def _annotate_communication_edges(
        graph: nx.Graph,
        distances: np.ndarray,
        area_size: float,
        reliability_scale: float,
    ) -> None:
        """Attach communication attributes and weighted Max-Cut edge costs."""
        finite_distances = distances[np.isfinite(distances)]
        max_distance = float(np.max(finite_distances)) if finite_distances.size else 1.0
        max_distance = max(max_distance, 1e-9)

        for u, v in graph.edges():
            distance = float(distances[u, v])
            latency = distance / max_distance
            shared_neighbors = len(set(graph.neighbors(u)).intersection(graph.neighbors(v)))
            interference = float(shared_neighbors / max(1, graph.number_of_nodes() - 2))
            reliability = float(np.exp(-distance / max(reliability_scale * area_size, 1e-9)))
            bandwidth = float(1.0 / (1.0 + latency + interference))
            conflict_weight = float(
                0.45 * latency
                + 0.35 * interference
                + 0.20 * (1.0 - reliability)
            )

            graph[u][v]["distance"] = distance
            graph[u][v]["latency"] = latency
            graph[u][v]["interference"] = interference
            graph[u][v]["reliability"] = reliability
            graph[u][v]["bandwidth"] = bandwidth
            graph[u][v]["weight"] = conflict_weight

