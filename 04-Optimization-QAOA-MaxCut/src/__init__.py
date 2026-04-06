"""QAOA Max-Cut research toolkit.

This package contains benchmark-oriented components for studying Max-Cut with:
- exact classical solvers
- statevector-based QAOA
- reduction-based RQAOA
- local and noisy execution utilities
- plotting and evaluation helpers
"""

__version__ = "1.2.0"
__author__ = "DEVADATH H K"

from .classical_solver import ClassicalSolver
from .evaluation_metrics import EvaluationMetrics
from .graph_generator import GraphGenerator
from .hamiltonian_builder import HamiltonianBuilder
from .qaoa_circuit import QAOACircuitBuilder
from .qaoa_optimizer import MaxCutQAOAProblem, QAOAOptimizer
from .rqaoa_engine import RQAOAEngine
from .runtime_executor import RuntimeExecutor
from .visualization import Visualizer

__all__ = [
    "ClassicalSolver",
    "EvaluationMetrics",
    "GraphGenerator",
    "HamiltonianBuilder",
    "QAOACircuitBuilder",
    "MaxCutQAOAProblem",
    "QAOAOptimizer",
    "RQAOAEngine",
    "RuntimeExecutor",
    "Visualizer",
]
