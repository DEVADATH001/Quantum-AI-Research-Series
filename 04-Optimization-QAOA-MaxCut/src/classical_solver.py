"""
Classical Solver Module

Implements exact classical solvers for Max-Cut problem.

Methods:
- Brute-force enumeration (for small graphs)
- Branch and bound
- Dynamic programming approaches

Used as baseline for QAOA performance comparison.

Author: Quantum AI Research Team
"""

import logging
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassicalResult:
    """
    Container for classical solver results.
    """
    # Optimal cut value
    optimal_value: int
    
    # Optimal bitstring(s)
    optimal_bitstrings: List[str]
    
    # Number of solutions found
    n_solutions: int
    
    # Computation time
    runtime: float
    
    # Problem size
    n_nodes: int
    
    # Number of edges
    n_edges: int


class ClassicalSolver:
    """
    Classical exact solvers for Max-Cut.
    
    Provides brute-force and approximation algorithms
    for benchmarking quantum solutions.
    """
    
    def __init__(self, seed: Optional[int] = 42) -> None:
        """
        Initialize classical solver.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        logger.info("ClassicalSolver initialized")
    
    def solve_exact(
        self,
        graph: nx.Graph
    ) -> ClassicalResult:
        """
        Solve Max-Cut exactly using brute-force enumeration.
        
        Only feasible for small graphs (n ≤ 20).
        
        Args:
            graph: NetworkX graph
            
        Returns:
            ClassicalResult with optimal solution
        """
        import time
        start_time = time.time()
        
        n_nodes = graph.number_of_nodes()
        
        logger.info(f"Running exact solver for {n_nodes} nodes")
        
        if n_nodes > 20:
            logger.warning(
                f"Brute-force for {n_nodes} nodes may be slow. "
                f"Consider using approximate methods."
            )
        
        # Try all 2^n possible partitions
        best_value = -1
        best_bitstrings = []
        
        for bits in itertools.product([0, 1], repeat=n_nodes):
            value = self._compute_cut(graph, bits)
            
            if value > best_value:
                best_value = value
                best_bitstrings = [''.join(map(str, bits))]
            elif value == best_value:
                best_bitstrings.append(''.join(map(str, bits)))
        
        runtime = time.time() - start_time
        
        logger.info(
            f"Exact solution found: value={best_value}, "
            f"runtime={runtime:.4f}s"
        )
        
        return ClassicalResult(
            optimal_value=best_value,
            optimal_bitstrings=best_bitstrings,
            n_solutions=len(best_bitstrings),
            runtime=runtime,
            n_nodes=n_nodes,
            n_edges=graph.number_of_edges()
        )
    
    def solve_branch_and_bound(
        self,
        graph: nx.Graph,
        upper_bound: Optional[int] = None
    ) -> ClassicalResult:
        """
        Solve Max-Cut using branch and bound.
        
        More efficient than brute-force for medium-sized graphs.
        
        Args:
            graph: NetworkX graph
            upper_bound: Optional upper bound for pruning
            
        Returns:
            ClassicalResult
        """
        import time
        start_time = time.time()
        
        n_nodes = graph.number_of_nodes()
        
        # Default upper bound: number of edges
        if upper_bound is None:
            upper_bound = graph.number_of_edges()
        
        logger.info(f"Running branch and bound for {n_nodes} nodes")
        
        # Track best solution
        best_value = 0
        best_bitstring = None
        
        # Current assignment
        current = [0] * n_nodes
        
        def bound(pos: int) -> int:
            """
            Compute upper bound for remaining unassigned nodes.
            
            Simplified: count remaining edges that could be cut.
            """
            # Count edges from assigned nodes
            assigned_cut = 0
            for i in range(pos):
                for j in range(i + 1, pos):
                    if graph.has_edge(i, j) and current[i] != current[j]:
                        assigned_cut += 1
            
            # Upper bound: assigned cut + all remaining edges
            remaining_edges = sum(
                1 for u, v in graph.edges()
                if u >= pos or v >= pos
            )
            
            return assigned_cut + remaining_edges
        
        def branch(pos: int) -> None:
            nonlocal best_value, best_bitstring
            
            if pos == n_nodes:
                # Evaluate complete assignment
                value = self._compute_cut_from_array(graph, current)
                
                if value > best_value:
                    best_value = value
                    best_bitstring = ''.join(map(str, current))
                return
            
            # Prune if upper bound <= best found
            if bound(pos) <= best_value:
                return
            
            # Try both assignments
            for val in [0, 1]:
                current[pos] = val
                branch(pos + 1)
        
        branch(0)
        
        runtime = time.time() - start_time
        
        return ClassicalResult(
            optimal_value=best_value,
            optimal_bitstrings=[best_bitstring] if best_bitstring else [],
            n_solutions=1 if best_bitstring else 0,
            runtime=runtime,
            n_nodes=n_nodes,
            n_edges=graph.number_of_edges()
        )
    
    def solve_greedy(
        self,
        graph: nx.Graph
    ) -> ClassicalResult:
        """
        Solve Max-Cut using greedy algorithm.
        
        O(V*E) complexity - fast but not optimal.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            ClassicalResult
        """
        import time
        start_time = time.time()
        
        n_nodes = graph.number_of_nodes()
        
        logger.info(f"Running greedy solver for {n_nodes} nodes")
        
        # Initialize all nodes to partition 0
        assignment = [0] * n_nodes
        
        # Greedy improvement
        improved = True
        while improved:
            improved = False
            
            for node in range(n_nodes):
                # Try flipping this node
                assignment[node] = 1 - assignment[node]
                
                new_value = self._compute_cut_from_array(graph, assignment)
                old_value = self._compute_cut_from_array_original(graph, assignment, node)
                
                if new_value > old_value:
                    improved = True
                else:
                    # Revert
                    assignment[node] = 1 - assignment[node]
        
        cut_value = self._compute_cut_from_array(graph, assignment)
        
        runtime = time.time() - start_time
        
        bitstring = ''.join(map(str, assignment))
        
        return ClassicalResult(
            optimal_value=cut_value,
            optimal_bitstrings=[bitstring],
            n_solutions=1,
            runtime=runtime,
            n_nodes=n_nodes,
            n_edges=graph.number_of_edges()
        )
    
    def compute_cut_value(
        self,
        graph: nx.Graph,
        partition: List[int]
    ) -> int:
        """
        Compute cut value for a given partition.
        
        Args:
            graph: NetworkX graph
            partition: List of node indices in partition A
            
        Returns:
            Cut value
        """
        partition_set = set(partition)
        
        cut = 0
        for u, v in graph.edges():
            if (u in partition_set) != (v in partition_set):
                cut += 1
        
        return cut
    
    def _compute_cut(
        self,
        graph: nx.Graph,
        bits: Tuple[int, ...]
    ) -> int:
        """
        Compute cut value for a bitstring.
        
        Args:
            graph: NetworkX graph
            bits: Tuple of 0/1 values
            
        Returns:
            Number of edges crossing the cut
        """
        cut = 0
        
        for u, v in graph.edges():
            if bits[u] != bits[v]:
                cut += 1
        
        return cut
    
    def _compute_cut_from_array(
        self,
        graph: nx.Graph,
        assignment: List[int]
    ) -> int:
        """
        Compute cut from array assignment.
        
        Args:
            graph: NetworkX graph
            assignment: List of 0/1 values
            
        Returns:
            Cut value
        """
        return self._compute_cut(graph, tuple(assignment))
    
    def _compute_cut_from_array_original(
        self,
        graph: nx.Graph,
        assignment: List[int],
        changed_node: int
    ) -> int:
        """
        Compute cut value before changing a node.
        
        Args:
            graph: NetworkX graph
            assignment: Current assignment
            changed_node: Node that was changed
            
        Returns:
            Cut value
        """
        original = assignment[changed_node]
        assignment[changed_node] = 1 - original
        value = self._compute_cut_from_array(graph, assignment)
        assignment[changed_node] = original
        return value
    
    def get_cut_edges(
        self,
        graph: nx.Graph,
        bitstring: str
    ) -> List[Tuple[int, int]]:
        """
        Get list of edges crossing the cut.
        
        Args:
            graph: NetworkX graph
            bitstring: Solution bitstring
            
        Returns:
            List of edges in the cut
        """
        cut_edges = []
        
        for u, v in graph.edges():
            if bitstring[u] != bitstring[v]:
                cut_edges.append((u, v))
        
        return cut_edges


class ApproximateSolver:
    """
    Classical approximate algorithms for Max-Cut.
    
    Provides guarantees for solution quality.
    """
    
    @staticmethod
    def solve_local_search(
        graph: nx.Graph,
        max_iterations: int = 1000
    ) -> Tuple[int, str]:
        """
        Local search algorithm for Max-Cut.
        
        Args:
            graph: NetworkX graph
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (cut_value, bitstring)
        """
        import random
        
        n = graph.number_of_nodes()
        
        # Random initial assignment
        assignment = [random.randint(0, 1) for _ in range(n)]
        
        for _ in range(max_iterations):
            improved = False
            
            for i in range(n):
                # Flip node i
                assignment[i] = 1 - assignment[i]
                
                new_cut = sum(
                    1 for u, v in graph.edges()
                    if assignment[u] != assignment[v]
                )
                
                old_cut = sum(
                    1 for u, v in graph.edges()
                    if (u == i and assignment[v] != assignment[i]) or
                       (v == i and assignment[u] != assignment[i])
                )
                
                if new_cut <= old_cut:
                    # Revert
                    assignment[i] = 1 - assignment[i]
                else:
                    improved = True
            
            if not improved:
                break
        
        cut_value = sum(
            1 for u, v in graph.edges()
            if assignment[u] != assignment[v]
        )
        
        return cut_value, ''.join(map(str, assignment))
    
    @staticmethod
    def goemans_williamson(
        graph: nx.Graph,
        num_trials: int = 100
    ) -> Tuple[float, List[int]]:
        """
        Goemans-Williamson algorithm using Semidefinite Programming (SDP).
        
        Provides 0.878-approximation guarantee.
        Requires `cvxpy` to be installed.
        
        Args:
            graph: NetworkX graph
            num_trials: Number of random hyperplanes to try
            
        Returns:
            Tuple of (approximate_cut, partition)
        """
        try:
            import cvxpy as cp
        except ImportError:
            import logging
            logging.getLogger(__name__).error("cvxpy is required for Goemans-Williamson.")
            return 0.0, [0] * graph.number_of_nodes()
            
        n = graph.number_of_nodes()
        
        # Build weight matrix
        W = np.zeros((n, n))
        for u, v in graph.edges():
            w = graph[u].get(v, {}).get('weight', 1.0)
            W[u, v] = W[v, u] = w
            
        # Define and solve the SDP relaxation
        # max 1/4 sum_ij W_ij (1 - X_ij)
        # s.t. X_ii = 1, X is positive semidefinite
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(n)]
        
        objective = cp.Maximize(cp.sum(cp.multiply(W, 1 - X)) / 4)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        
        if X.value is None:
            return 0.0, [0] * n
            
        # Cholesky decomposition X = V^T V
        # Add small identity to ensure positive definiteness due to numerical errors
        X_val = X.value + np.eye(n) * 1e-6
        try:
            V = np.linalg.cholesky(X_val).T
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(X_val)
            eigenvalues[eigenvalues < 0] = 0
            V = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            
        best_cut = 0
        best_partition = None
        
        # Random hyperplane rounding
        for _ in range(num_trials):
            r = np.random.randn(n)
            r = r / np.linalg.norm(r)
            
            partition = [1 if np.dot(V[:, i], r) > 0 else 0 for i in range(n)]
            
            cut = sum(
                W[u, v] for u, v in graph.edges()
                if partition[u] != partition[v]
            )
            
            if cut > best_cut:
                best_cut = cut
                best_partition = partition
                
        return best_cut, best_partition if best_partition else [0] * n

