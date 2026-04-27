"""QAOA Max-Cut research toolkit.

This package contains benchmark-oriented components for studying Max-Cut with:
- exact classical solvers
- statevector-based QAOA
- reduction-based RQAOA
- local and noisy execution utilities
- plotting and evaluation helpers
"""

__version__ = "1.7.0"
__author__ = "DEVADATH H K"

from .artifact_manager import ArtifactManager
from .artifact_pipeline import ArtifactPipeline
from .classical_solver import ClassicalSolver
from .evaluation_metrics import EvaluationMetrics
from .experimental_study import ExperimentalStudyRunner
from .graph_generator import GraphGenerator
from .hamiltonian_builder import HamiltonianBuilder
from .hardware_analysis import HardwareFeasibilityAnalyzer, HardwareFeasibilityThresholds
from .qaoa_circuit import QAOACircuitBuilder
from .qaoa_optimizer import MaxCutQAOAProblem, QAOAOptimizer
from .results_review import BenchmarkRobustnessRunner, ScientificResultsReviewer
from .provenance import collect_run_provenance
from .rqaoa_engine import RQAOAEngine
from .runtime_executor import RuntimeExecutor
from .visualization import Visualizer

__all__ = [
    "ArtifactManager",
    "ArtifactPipeline",
    "BenchmarkRobustnessRunner",
    "ClassicalSolver",
    "EvaluationMetrics",
    "ExperimentalStudyRunner",
    "GraphGenerator",
    "HamiltonianBuilder",
    "HardwareFeasibilityAnalyzer",
    "HardwareFeasibilityThresholds",
    "QAOACircuitBuilder",
    "MaxCutQAOAProblem",
    "QAOAOptimizer",
    "RQAOAEngine",
    "RuntimeExecutor",
    "ScientificResultsReviewer",
    "Visualizer",
    "collect_run_provenance",
]
