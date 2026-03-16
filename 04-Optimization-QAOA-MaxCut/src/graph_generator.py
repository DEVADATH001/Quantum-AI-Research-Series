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
        graph = nx.read_adjlist(path=filepath)
        logger.info(f"Loaded adjacency list from {filepath}")
        return graph
    
    def get_cut_value(
        self,
        graph: nx.Graph,
        partition: List[int]
    ) -> int:
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
        
        return len(cut_set)
    
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

