"""Quantum Kernel SVM for MNIST - Research Project.

This package provides tools for comparing classical RBF kernel SVM
against quantum kernel SVM with Pegasos optimization on MNIST digit
classification (4 vs 9).

Modules:
    - data_loader: MNIST dataset loading and filtering
    - preprocessing: Data normalization and PCA dimensionality reduction
    - classical_models: Classical SVM with RBF kernel
    - quantum_feature_maps: ZZFeatureMap for quantum encoding
    - quantum_kernel_engine: FidelityQuantumKernel computation
    - quantum_training: PegasosQSVC implementation
    - evaluation_metrics: Model evaluation metrics
    - visualization: Plotting utilities
    - noise_simulation: IBM noise model simulation
"""

__version__ = "1.0.0"
__author__ = "Quantum ML Research Lab"

from importlib import import_module
from typing import Any

__all__ = [
    "load_mnist_digits",
    "preprocess_data",
    "train_classical_svm",
    "create_zz_feature_map",
    "create_quantum_kernel",
    "train_pegasos_qsvc",
    "evaluate_models",
    "plot_pca_scatter",
    "plot_pca_variance",
    "plot_class_distribution",
    "plot_kernel_heatmap",
    "plot_metrics_comparison",
    "plot_confusion_matrix",
    "simulate_noisy_kernel",
]

_EXPORT_MAP: dict[str, str] = {
    "load_mnist_digits": "src.data_loader",
    "preprocess_data": "src.preprocessing",
    "train_classical_svm": "src.classical_models",
    "create_zz_feature_map": "src.quantum_feature_maps",
    "create_quantum_kernel": "src.quantum_kernel_engine",
    "train_pegasos_qsvc": "src.quantum_training",
    "evaluate_models": "src.evaluation_metrics",
    "plot_pca_scatter": "src.visualization",
    "plot_pca_variance": "src.visualization",
    "plot_class_distribution": "src.visualization",
    "plot_kernel_heatmap": "src.visualization",
    "plot_metrics_comparison": "src.visualization",
    "plot_confusion_matrix": "src.visualization",
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

