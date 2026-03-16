"""Author: DEVADATH H K

QAOA Max-Cut Optimization Package

A production-grade quantum optimization research platform for solving
the Max-Cut problem using QAOA and Recursive QAOA algorithms.

Modules:
- graph_generator: Generate graph structures for optimization
- hamiltonian_builder: Build Ising Hamiltonians from graphs
- qaoa_circuit: Construct parameterized QAOA quantum circuits
- qaoa_optimizer: Classical optimization loop for QAOA
- rqaoa_engine: Recursive QAOA implementation
- classical_solver: Brute-force exact solver
- evaluation_metrics: Compute performance metrics
- runtime_executor: Qiskit Runtime execution engine
- visualization: Plotting and visualization utilities"""

__version__ = "1.0.0"
__author__ = "DEVADATH H K"

from .graph_generator import GraphGenerator
from .hamiltonian_builder import HamiltonianBuilder
from .qaoa_circuit import QAOACircuitBuilder
from .qaoa_optimizer import QAOAOptimizer
from .rqaoa_engine import RQAOAEngine
from .classical_solver import ClassicalSolver
from .evaluation_metrics import EvaluationMetrics
from .runtime_executor import RuntimeExecutor
from .visualization import Visualizer

__all__ = [
    "GraphGenerator",
    "HamiltonianBuilder",
    "QAOACircuitBuilder",
    "QAOAOptimizer",
    "RQAOAEngine",
    "ClassicalSolver",
    "EvaluationMetrics",
    "RuntimeExecutor",
    "Visualizer",
]

