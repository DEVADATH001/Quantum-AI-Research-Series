"""
QAOA Optimizer Module

Implements the classical optimization loop for QAOA.

The hybrid quantum-classical loop:
1. Initialize parameters (γ, β)
2. Prepare parameterized quantum circuit
3. Execute circuit and measure expectation value
4. Update parameters using classical optimizer
5. Repeat until convergence

Supports optimizers:
- COBYLA (Constrained Optimization BY Linear Approximations)
- SPSA (Simulated Parameter Setting Algorithm)
- Nelder-Mead
- L-BFGS-B

Author: Quantum AI Research Team
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Container for optimization results.
    """
    # Optimized parameters
    optimal_params: np.ndarray
    
    # Optimized cost value
    optimal_value: float
    
    # Number of function evaluations
    n_evaluations: int
    
    # Optimization history
    history: List[Dict] = field(default_factory=list)
    
    # Runtime in seconds
    runtime: float = 0.0
    
    # Convergence status
    converged: bool = False
    
    # Final bitstring solution
    solution_bitstring: Optional[str] = None
    
    # Cut value
    cut_value: Optional[int] = None


class QAOAOptimizer:
    """
    Classical optimizer for QAOA parameters.
    
    Implements the hybrid quantum-classical optimization loop
    to find optimal γ and β parameters.
    """
    
    def __init__(
        self,
        p: int = 1,
        optimizer_type: str = "COBYLA",
        maxiter: int = 500,
        tol: float = 1e-6,
        seed: Optional[int] = 42,
        n_initial_points: int = 3,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Initialize the QAOA optimizer.
        
        Args:
            p: Number of QAOA layers
            optimizer_type: Optimizer type ("COBYLA", "SPSA", "Nelder-Mead")
            maxiter: Maximum iterations
            tol: Convergence tolerance
            seed: Random seed
            n_initial_points: Number of random initial points
            callback: Optional callback function for logging
        """
        self.p = p
        self.optimizer_type = optimizer_type.upper()
        self.maxiter = maxiter
        self.tol = tol
        self.seed = seed
        self.n_initial_points = n_initial_points
        self.callback = callback
        
        self.rng = np.random.default_rng(seed)
        
        logger.info(
            f"QAOAOptimizer initialized: p={p}, optimizer={optimizer_type}"
        )
    
    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_qubits: int,
        initial_params: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[List[float], List[float]]] = None,
        graph: Optional = None
    ) -> OptimizationResult:
        """
        Run the optimization loop.
        
        Args:
            objective_function: Function to minimize
                Takes parameter array [γ0, β0, γ1, β1, ...] and returns cost
            n_qubits: Number of qubits in the problem
            initial_params: Initial parameters (optional)
            bounds: Parameter bounds as (lower, upper)
            graph: NetworkX graph (for solution extraction)
            
        Returns:
            OptimizationResult with optimal parameters and history
        """
        logger.info(f"Starting optimization with {self.optimizer_type}")
        
        start_time = time.time()
        
        # Generate initial parameters if not provided
        if initial_params is None:
            initial_params = self._generate_initial_params(
                self.n_initial_points
            )
        
        # Set default bounds
        if bounds is None:
            bounds = self._default_bounds()
        
        # Track best result
        best_result = None
        best_value = float('inf')
        
        # Try multiple initial points
        for i, init_params in enumerate(initial_params):
            logger.info(f"Optimization run {i+1}/{len(initial_params)}")
            
            result = self._run_single_optimization(
                objective_function=objective_function,
                initial_params=init_params,
                bounds=bounds,
                run_id=i
            )
            
            if result.optimal_value < best_value:
                best_value = result.optimal_value
                best_result = result
        
        # Extract solution if graph provided
        if graph is not None and best_result is not None:
            solution = self._extract_solution(
                best_result.optimal_params,
                graph
            )
            best_result.solution_bitstring = solution['bitstring']
            best_result.cut_value = solution['cut_value']
        
        best_result.runtime = time.time() - start_time
        
        logger.info(
            f"Optimization complete: value={best_value:.4f}, "
            f"runtime={best_result.runtime:.2f}s"
        )
        
        return best_result
    
    def _generate_initial_params(
        self,
        n_points: int
    ) -> List[np.ndarray]:
        """
        Generate multiple initial parameter sets.
        
        Args:
            n_points: Number of initial points
            
        Returns:
            List of parameter arrays
        """
        params_list = []
        
        for _ in range(n_points):
            # Random parameters in [0, 2π]
            params = self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
            params_list.append(params)
        
        return params_list
    
    def _default_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get default parameter bounds.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = [0.0] * (2 * self.p)
        upper = [2 * np.pi] * (2 * self.p)
        
        return lower, upper
    
    def _run_single_optimization(
        self,
        objective_function: Callable,
        initial_params: np.ndarray,
        bounds: Tuple[List[float], List[float]],
        run_id: int = 0
    ) -> OptimizationResult:
        """
        Run optimization from a single initial point.
        
        Args:
            objective_function: Cost function
            initial_params: Starting parameters
            bounds: Parameter bounds
            run_id: Run identifier
            
        Returns:
            OptimizationResult
        """
        history = []
        
        # Create optimizer
        if self.optimizer_type == "COBYLA":
            from qiskit_algorithms.optimizers import COBYLA
            
            optimizer = COBYLA(
                maxiter=self.maxiter,
                tol=self.tol,
                rhobeg=1.0  # Initial step size
            )
            
            # Custom callback to track history
            def cobyla_callback(xk):
                value = objective_function(xk)
                history.append({
                    'iteration': len(history),
                    'params': xk.copy(),
                    'value': value
                })
                if self.callback:
                    self.callback(len(history), xk, value)
            
            result = optimizer.minimize(
                fun=objective_function,
                x0=initial_params,
                bounds=list(zip(bounds[0], bounds[1])) if bounds else None
            )
            
            optimal_params = result.x
            optimal_value = result.fun
            n_evaluations = len(history)
            
        elif self.optimizer_type == "SPSA":
            from qiskit_algorithms.optimizers import SPSA
            
            optimizer = SPSA(
                maxiter=self.maxiter,
                learning_rate=0.1,
                perturbation=0.1,
            )
            
            # SPSA callback
            def spsa_callback(nfev, xk, fk, xp, fp, accept):
                history.append({
                    'iteration': nfev,
                    'params': xk.copy(),
                    'value': float(fk) if fk is not None else float('nan')
                })
            
            result = optimizer.minimize(
                fun=objective_function,
                x0=initial_params,
                bounds=list(zip(bounds[0], bounds[1])) if bounds else None
            )
            
            optimal_params = result.x
            optimal_value = result.fun
            n_evaluations = len(history)
            
        elif self.optimizer_type == "NELDER-MEAD":
            from scipy.optimize import minimize
            
            history = []
            
            def nm_callback(xk):
                value = objective_function(xk)
                history.append({
                    'iteration': len(history),
                    'params': xk.copy(),
                    'value': value
                })
            
            res = minimize(
                objective_function,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': self.maxiter,
                    'xatol': self.tol,
                    'fatol': self.tol,
                    'callback': nm_callback
                }
            )
            
            optimal_params = res.x
            optimal_value = res.fun
            n_evaluations = res.nfev
            
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Determine convergence
        converged = n_evaluations < self.maxiter
        
        return OptimizationResult(
            optimal_params=optimal_params,
            optimal_value=optimal_value,
            n_evaluations=n_evaluations,
            history=history,
            converged=converged
        )
    
    def _extract_solution(
        self,
        params: np.ndarray,
        graph
    ) -> Dict:
        """
        Extract solution bitstring from optimal parameters by evaluating
        the exact statevector and finding the most probable bitstring.
        
        Args:
            params: Optimal parameters
            graph: Problem graph
            
        Returns:
            Dictionary with bitstring and cut value
        """
        from qiskit.quantum_info import Statevector, SparsePauliOp
        from qiskit.circuit.library import QAOAAnsatz
        
        n = graph.number_of_nodes()
        edges = list(graph.edges())
        
        if not edges:
            bitstring = '0' * n
            return {'bitstring': bitstring, 'cut_value': 0}
            
        # Build Hamiltonian
        pauli_list = []
        for i, j in edges:
            pauli_str = ['I'] * n
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append(''.join(reversed(pauli_str)))
            
        hamiltonian = SparsePauliOp(pauli_list, coeffs=[-0.5] * len(edges))
        
        # Build ansatz and get statevector
        ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=self.p)
        qc = ansatz.assign_parameters(params)
        sv = Statevector(qc)
        
        # Find most probable bitstring
        probs = sv.probabilities_dict()
        bitstring = max(probs, key=probs.get)
        
        # Calculate cut value
        cut_value = self._calculate_cut(graph, bitstring)
        
        return {
            'bitstring': bitstring,
            'cut_value': cut_value
        }
    
    @staticmethod
    def _calculate_cut(graph, bitstring: str) -> int:
        """
        Calculate cut value for a bitstring.
        
        Args:
            graph: NetworkX graph
            bitstring: Binary string
            
        Returns:
            Number of edges crossing the cut
        """
        cut = 0
        for i, j in graph.edges():
            if i < len(bitstring) and j < len(bitstring):
                if bitstring[i] != bitstring[j]:
                    cut += 1
        return cut
    
    def get_convergence_plot_data(
        self,
        result: OptimizationResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract data for convergence plotting.
        
        Args:
            result: OptimizationResult
            
        Returns:
            Tuple of (iterations, values)
        """
        iterations = np.array([h['iteration'] for h in result.history])
        values = np.array([h['value'] for h in result.history])
        
        return iterations, values


class ParameterGridEvaluator:
    """
    Evaluates QAOA cost function over a grid of parameters.
    
    Used for:
    - Energy landscape visualization
    - Initial parameter selection
    - Parameter space analysis
    """
    
    def __init__(self, p: int = 1) -> None:
        """
        Initialize the grid evaluator.
        
        Args:
            p: Number of QAOA layers
        """
        self.p = p
    
    def evaluate_grid(
        self,
        objective_function: Callable,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate cost function over parameter grid.
        
        For p=1: γ ∈ [0, π], β ∈ [0, π]
        For p>1: Uses first layer parameters
        
        Args:
            objective_function: Cost function
            n_points: Number of grid points per dimension
            
        Returns:
            Tuple of (gamma_grid, beta_grid, cost_grid)
        """
        gamma_range = np.linspace(0, np.pi, n_points)
        beta_range = np.linspace(0, np.pi, n_points)
        
        gamma_grid, beta_grid = np.meshgrid(gamma_range, beta_range)
        
        # Flatten for evaluation
        cost_grid = np.zeros_like(gamma_grid)
        
        for i in range(n_points):
            for j in range(n_points):
                # Create parameter vector for p=1
                params = np.array([gamma_grid[i, j], beta_grid[i, j]])
                cost_grid[i, j] = objective_function(params)
        
        return gamma_grid, beta_grid, cost_grid
    
    def find_optimal_region(
        self,
        cost_grid: np.ndarray,
        gamma_grid: np.ndarray,
        beta_grid: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find top-k optimal regions in parameter space.
        
        Args:
            cost_grid: Cost values
            gamma_grid: Gamma values
            beta_grid: Beta values
            top_k: Number of top regions to return
            
        Returns:
            List of dictionaries with optimal parameters
        """
        flat_indices = np.argsort(cost_grid.flatten())[:top_k]
        
        results = []
        for idx in flat_indices:
            i, j = np.unravel_index(idx, cost_grid.shape)
            results.append({
                'gamma': gamma_grid[i, j],
                'beta': beta_grid[i, j],
                'cost': cost_grid[i, j]
            })
        
        return results

