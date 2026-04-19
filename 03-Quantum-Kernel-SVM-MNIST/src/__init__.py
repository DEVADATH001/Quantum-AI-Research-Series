"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Research package initialization."""

__version__ = "2.0.0"
__author__ = "DEVADATH H K"

from importlib import import_module
from typing import Any

__all__ = [
    # Data
    "load_mnist_digits",
    "preprocess_data",
    # Classical
    "train_classical_svm",
    # Quantum feature maps
    "create_zz_feature_map",
    # Quantum kernel
    "create_quantum_kernel",
    "compute_kernel_target_alignment",
    "compute_centered_kta",        # NEW — Centered KTA (Cortes et al. 2012)
    "compute_geometric_difference",
    # Training
    "train_pegasos_qsvc",
    # Kernel learning
    "QuantumKernelLearner",        # NEW — QKL via parameter-shift
    # Hardware
    "HardwareBackendManager",      # NEW — hardware/simulation backend bridge
    # Evaluation
    "evaluate_models",
    "calculate_statistical_significance",
    # Visualization
    "plot_pca_scatter",
    "plot_pca_variance",
    "plot_class_distribution",
    "plot_kernel_heatmap",
    "plot_metrics_comparison",
    "plot_confusion_matrix",
    "plot_qkl_convergence",        # NEW — QKL convergence curve
    # Noise
    "simulate_noisy_kernel",
]

_EXPORT_MAP: dict[str, str] = {
    "load_mnist_digits": "src.data_loader",
    "preprocess_data": "src.preprocessing",
    "train_classical_svm": "src.classical_models",
    "create_zz_feature_map": "src.quantum_feature_maps",
    "create_quantum_kernel": "src.quantum_kernel_engine",
    "compute_kernel_target_alignment": "src.quantum_kernel_engine",
    "compute_centered_kta": "src.quantum_kernel_engine",
    "compute_geometric_difference": "src.quantum_kernel_engine",
    "train_pegasos_qsvc": "src.quantum_training",
    "QuantumKernelLearner": "src.kernel_learning",
    "HardwareBackendManager": "src.hardware_backend",
    "evaluate_models": "src.evaluation_metrics",
    "calculate_statistical_significance": "src.evaluation_metrics",
    "plot_pca_scatter": "src.visualization",
    "plot_pca_variance": "src.visualization",
    "plot_class_distribution": "src.visualization",
    "plot_kernel_heatmap": "src.visualization",
    "plot_metrics_comparison": "src.visualization",
    "plot_confusion_matrix": "src.visualization",
    "plot_qkl_convergence": "src.visualization",
    "simulate_noisy_kernel": "src.noise_simulation",
}

def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src' has no attribute '{name}'")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
