"""Author: DEVADATH H K

Evaluation Metrics Module

Provides comprehensive metrics for evaluating QAOA performance:

Primary Metric:
- Approximation Ratio: r = QAOA_value / Optimal_value

Secondary Metrics:
- Solution quality
- Convergence behavior
- Energy gap
- Runtime performance

The approximation ratio is the primary benchmark metric."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """
    Container for evaluation metrics.
    """
    # Primary metrics
    approximation_ratio: float
    qaoa_value: float
    optimal_value: float
    
    # Additional metrics
    relative_error: float
    energy_gap: Optional[float] = None
    
    # Solution info
    solution_bitstring: Optional[str] = None
    optimal_bitstring: Optional[str] = None
    
    # Performance metrics
    runtime: float = 0.0
    n_function_evaluations: int = 0
    
    # Problem info
    n_nodes: int = 0
    n_edges: int = 0
    qaoa_depth: int = 0

class EvaluationMetrics:
    """
    Computes and tracks performance metrics for QAOA.
    
    Provides comprehensive evaluation including:
    - Approximation ratio
    - Solution quality
    - Convergence analysis
    - Comparison with classical methods
    """
    
    def __init__(self) -> None:
        """Initialize evaluation metrics."""
        self.history: List[EvaluationResult] = []
        logger.info("EvaluationMetrics initialized")
    
    def compute_approximation_ratio(
        self,
        qaoa_value: float,
        optimal_value: float
    ) -> float:
        """
        Compute approximation ratio.
        
        r = QAOA_value / Optimal_value
        
        Args:
            qaoa_value: Value obtained from QAOA
            optimal_value: Optimal/exact value
            
        Returns:
            Approximation ratio
            
        Raises:
            ValueError: If optimal_value is zero or negative
        """
        if optimal_value <= 0:
            raise ValueError(
                f"Optimal value must be positive, got {optimal_value}"
            )
        
        ratio = qaoa_value / optimal_value
        
        # Cap at 1.0 (can exceed due to sampling noise)
        ratio = min(ratio, 1.0)
        
        logger.info(f"Approximation ratio: {ratio:.4f}")
        
        return ratio
    
    def compute_relative_error(
        self,
        qaoa_value: float,
        optimal_value: float
    ) -> float:
        """
        Compute relative error.
        
        ε = |QAOA - Optimal| / Optimal = 1 - r
        
        Args:
            qaoa_value: QAOA solution value
            optimal_value: Optimal value
            
        Returns:
            Relative error
        """
        return 1.0 - self.compute_approximation_ratio(qaoa_value, optimal_value)
    
    def evaluate_solution(
        self,
        graph: nx.Graph,
        qaoa_bitstring: str,
        optimal_bitstring: Optional[str] = None,
        optimal_value: Optional[int] = None,
        runtime: float = 0.0,
        qaoa_depth: int = 1
    ) -> EvaluationResult:
        """
        Evaluate a complete QAOA solution.
        
        Args:
            graph: Problem graph
            qaoa_bitstring: QAOA solution
            optimal_bitstring: Optimal solution (optional)
            optimal_value: Optimal value (optional)
            runtime: Execution time
            qaoa_depth: QAOA depth p
            
        Returns:
            EvaluationResult
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        # Compute QAOA cut value
        qaoa_value = self._compute_cut(graph, qaoa_bitstring)
        
        # Get optimal if not provided
        if optimal_value is None and optimal_bitstring is not None:
            optimal_value = self._compute_cut(graph, optimal_bitstring)
        
        # If still not available, compute from QAOA bitstring (for ratio)
        if optimal_value is None:
            logger.warning("Optimal value not provided, using QAOA value as baseline")
            optimal_value = qaoa_value
        
        # Compute metrics
        approx_ratio = self.compute_approximation_ratio(
            float(qaoa_value),
            float(optimal_value)
        )
        
        rel_error = self.compute_relative_error(
            float(qaoa_value),
            float(optimal_value)
        )
        
        result = EvaluationResult(
            approximation_ratio=approx_ratio,
            qaoa_value=float(qaoa_value),
            optimal_value=float(optimal_value),
            relative_error=rel_error,
            solution_bitstring=qaoa_bitstring,
            optimal_bitstring=optimal_bitstring,
            runtime=runtime,
            n_nodes=n_nodes,
            n_edges=n_edges,
            qaoa_depth=qaoa_depth
        )
        
        self.history.append(result)
        
        return result
    
    def compare_depths(
        self,
        depth_results: Dict[int, float],
        optimal_value: float
    ) -> Dict[int, Dict]:
        """
        Compare QAOA performance across different depths.
        
        Args:
            depth_results: Dict mapping depth p to cut value
            optimal_value: Optimal value
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        for p, value in depth_results.items():
            ratio = self.compute_approximation_ratio(value, optimal_value)
            error = self.compute_relative_error(value, optimal_value)
            
            comparison[p] = {
                'cut_value': value,
                'approximation_ratio': ratio,
                'relative_error': error,
                'performance': self._rate_performance(ratio)
            }
        
        logger.info(f"Depth comparison: {comparison}")
        
        return comparison
    
    def compute_energy_distribution(
        self,
        samples: List[Tuple[str, float]],
        graph: Optional[nx.Graph] = None,
        energies: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Analyze energy distribution from measurement samples.
        
        Args:
            samples: List of ``(bitstring, probability)`` tuples.
            graph: Optional graph used to compute Max-Cut values directly.
            energies: Optional explicit mapping from bitstring to energy/value.
            
        Returns:
            Dictionary with distribution statistics
        """
        if not samples:
            raise ValueError("samples must not be empty")

        if graph is None and energies is None:
            raise ValueError("Provide either graph or energies to evaluate sample energies.")

        probability_sum = sum(probability for _, probability in samples)
        if probability_sum <= 0:
            raise ValueError("Sample probabilities must sum to a positive value.")

        energy_values = []
        normalized_probabilities = []
        for bitstring, probability in samples:
            if energies is not None:
                if bitstring not in energies:
                    raise KeyError(f"Missing energy for bitstring {bitstring!r}")
                energy = float(energies[bitstring])
            else:
                energy = float(self._compute_cut(graph, bitstring))

            energy_values.append(energy)
            normalized_probabilities.append(probability / probability_sum)

        energy_array = np.array(energy_values, dtype=float)
        probability_array = np.array(normalized_probabilities, dtype=float)
        weighted_mean = float(np.dot(probability_array, energy_array))
        weighted_second_moment = float(np.dot(probability_array, energy_array**2))
        weighted_std = float(np.sqrt(max(0.0, weighted_second_moment - weighted_mean**2)))

        stats = {
            'mean': weighted_mean,
            'std': weighted_std,
            'min': float(np.min(energy_array)),
            'max': float(np.max(energy_array)),
            'median': float(np.median(energy_array)),
            'n_samples': len(samples)
        }
        
        # Find best sample by energy/value, not by probability.
        best_idx = int(np.argmax(energy_array))
        best = samples[best_idx]
        stats['best_bitstring'] = best[0]
        stats['best_energy'] = float(energy_array[best_idx])
        stats['best_probability'] = float(probability_array[best_idx])
        
        return stats
    
    def compute_solution_statistics(
        self,
        graph: nx.Graph,
        solutions: List[str]
    ) -> Dict:
        """
        Compute statistics over multiple solution attempts.
        
        Args:
            graph: Problem graph
            solutions: List of solution bitstrings
            
        Returns:
            Dictionary with solution statistics
        """
        cut_values = [self._compute_cut(graph, sol) for sol in solutions]
        
        stats = {
            'mean': np.mean(cut_values),
            'std': np.std(cut_values),
            'min': np.min(cut_values),
            'max': np.max(cut_values),
            'median': np.median(cut_values),
            'n_solutions': len(solutions)
        }
        
        return stats
    
    def assess_quality(
        self,
        approximation_ratio: float
    ) -> str:
        """
        Assess solution quality based on approximation ratio.
        
        Args:
            approximation_ratio: r value
            
        Returns:
            Quality rating string
        """
        return self._rate_performance(approximation_ratio)
    
    def _rate_performance(
        self,
        ratio: float
    ) -> str:
        """
        Rate performance based on approximation ratio.
        
        Args:
            ratio: Approximation ratio
            
        Returns:
            Rating string
        """
        if ratio >= 0.95:
            return "Excellent"
        elif ratio >= 0.90:
            return "Very Good"
        elif ratio >= 0.80:
            return "Good"
        elif ratio >= 0.70:
            return "Fair"
        elif ratio >= 0.60:
            return "Poor"
        else:
            return "Very Poor"
    
    def _compute_cut(
        self,
        graph: nx.Graph,
        bitstring: str
    ) -> int:
        """
        Compute cut value for a bitstring.
        
        Args:
            graph: NetworkX graph
            bitstring: Binary string
            
        Returns:
            Cut value
        """
        cut = 0.0
        
        for u, v in graph.edges():
            if u < len(bitstring) and v < len(bitstring):
                if bitstring[u] != bitstring[v]:
                    cut += float(graph[u][v].get("weight", 1.0))

        return cut
    
    def get_summary(self) -> Dict:
        """
        Get summary of all evaluations.
        
        Returns:
            Summary dictionary
        """
        if not self.history:
            return {'n_evaluations': 0}
        
        ratios = [r.approximation_ratio for r in self.history]
        values = [r.qaoa_value for r in self.history]
        
        return {
            'n_evaluations': len(self.history),
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'min_ratio': np.min(ratios),
            'max_ratio': np.max(ratios),
            'mean_value': np.mean(values),
            'best_value': np.max(values)
        }

class BenchmarkSuite:
    """
    Comprehensive benchmark suite for QAOA evaluation.
    
    Tests on multiple graph instances and configurations.
    """
    
    def __init__(self) -> None:
        """Initialize benchmark suite."""
        self.results: List[EvaluationResult] = []
        logger.info("BenchmarkSuite initialized")
    
    def run_benchmark(
        self,
        graph_generator,
        maxcut_hamiltonian_builder,
        qaoa_solver,
        n_instances: int = 5,
        depths: List[int] = [1, 2, 3]
    ) -> Dict:
        """
        Run comprehensive benchmark.
        
        Args:
            graph_generator: GraphGenerator instance
            maxcut_hamiltonian_builder: HamiltonianBuilder instance
            qaoa_solver: QAOA solver function
            n_instances: Number of graph instances
            depths: List of QAOA depths to test
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark: {n_instances} instances, depths={depths}")
        
        all_results = []
        
        for instance_idx in range(n_instances):
            # Generate random graph
            graph = graph_generator.generate_d_regular_graph(
                n_nodes=10,
                degree=3,
                seed=instance_idx
            )
            
            # Get optimal solution
            from .classical_solver import ClassicalSolver
            solver = ClassicalSolver()
            exact_result = solver.solve_exact(graph)
            
            # Test each depth
            for p in depths:
                result = qaoa_solver(graph, p)
                
                eval_result = EvaluationMetrics().evaluate_solution(
                    graph=graph,
                    qaoa_bitstring=result['bitstring'],
                    optimal_value=exact_result.optimal_value,
                    qaoa_depth=p
                )
                
                all_results.append(eval_result)
        
        # Aggregate results
        return self._aggregate_results(all_results, depths)
    
    def _aggregate_results(
        self,
        results: List[EvaluationResult],
        depths: List[int]
    ) -> Dict:
        """
        Aggregate benchmark results.
        
        Args:
            results: List of evaluation results
            depths: Tested depths
            
        Returns:
            Aggregated results
        """
        aggregated = {}
        
        for p in depths:
            p_results = [r for r in results if r.qaoa_depth == p]
            
            if p_results:
                ratios = [r.approximation_ratio for r in p_results]
                
                aggregated[p] = {
                    'mean_ratio': np.mean(ratios),
                    'std_ratio': np.std(ratios),
                    'min_ratio': np.min(ratios),
                    'max_ratio': np.max(ratios),
                    'n_samples': len(p_results)
                }
        
        return aggregated

