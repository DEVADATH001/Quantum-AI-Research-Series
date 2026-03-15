"""Quantum Utilities Module.

This module provides utility functions for quantum machine learning experiments.

Author: Quantum ML Research Lab
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Set Python hash seed
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def create_results_directory(base_dir: str = "results") -> Path:
    """Create results directory if it doesn't exist.

    Args:
        base_dir: Base directory name

    Returns:
        Path to results directory
    """
    results_dir = Path(base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")
    return results_dir


def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    import json
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"JSON saved to {filepath}")


def load_json(filepath: str) -> dict:
    """Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    import json
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    logger.info(f"JSON loaded from {filepath}")
    return data


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_qiskit_version() -> dict[str, str]:
    """Get versions of Qiskit packages.

    Returns:
        Dictionary of package versions
    """
    versions = {}
    
    packages = [
        "qiskit",
        "qiskit_aer",
        "qiskit_machine_learning",
        "qiskit_ibm_runtime",
    ]
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    
    return versions


def print_system_info() -> None:
    """Print system and package information."""
    import platform
    
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()
    
    versions = get_qiskit_version()
    print("Qiskit Packages:")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver}")
    
    print("=" * 60)


def validate_feature_map_parameters(
    feature_dimension: int,
    reps: int,
    entanglement: str,
) -> bool:
    """Validate feature map parameters.

    Args:
        feature_dimension: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement type

    Returns:
        True if valid

    Raises:
        ValueError: If parameters are invalid
    """
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be >= 1")
    
    if reps < 1:
        raise ValueError("reps must be >= 1")
    
    valid_entanglement = ["full", "linear", "reverse_linear", "circular", "sca"]
    if entanglement not in valid_entanglement:
        raise ValueError(f"entanglement must be one of {valid_entanglement}")
    
    return True


def estimate_memory_usage(n_samples: int, n_qubits: int) -> dict[str, float]:
    """Estimate memory usage for quantum kernel computation.

    Args:
        n_samples: Number of samples
        n_qubits: Number of qubits

    Returns:
        Dictionary with memory estimates in MB
    """
    # Kernel matrix size
    kernel_size = n_samples * n_samples * 8 / (1024**2)  # float64
    
    # Statevector size (2^n_qubits complex128)
    statevector_size = (2**n_qubits) * 16 / (1024**2)
    
    # Circuit parameters
    param_size = n_qubits * 3 * 8 / (1024**2)  # Rough estimate
    
    estimates = {
        "kernel_matrix_mb": kernel_size,
        "statevector_mb": statevector_size,
        "parameters_mb": param_size,
        "total_estimate_mb": kernel_size + statevector_size + param_size,
    }
    
    return estimates


def compare_kernel_matrices(
    K1: np.ndarray,
    K2: np.ndarray,
    labels: tuple[str, str] = ("Kernel 1", "Kernel 2"),
) -> dict[str, float]:
    """Compare two kernel matrices.

    Args:
        K1: First kernel matrix
        K2: Second kernel matrix
        labels: Labels for the two kernels

    Returns:
        Dictionary of comparison metrics
    """
    # Flatten for comparison
    k1_flat = K1.flatten()
    k2_flat = K2.flatten()
    
    # Basic statistics
    comparison = {
        f"{labels[0]}_mean": float(np.mean(k1_flat)),
        f"{labels[1]}_mean": float(np.mean(k2_flat)),
        f"{labels[0]}_std": float(np.std(k1_flat)),
        f"{labels[1]}_std": float(np.std(k2_flat)),
    }
    
    # Difference metrics
    diff = k1_flat - k2_flat
    comparison["mean_abs_diff"] = float(np.mean(np.abs(diff)))
    comparison["max_abs_diff"] = float(np.max(np.abs(diff)))
    comparison["rmse"] = float(np.sqrt(np.mean(diff**2)))
    
    # Correlation
    comparison["pearson_corr"] = float(np.corrcoef(k1_flat, k2_flat)[0, 1])
    
    return comparison


def get_entanglement_info(entanglement: str, n_qubits: int) -> dict[str, Any]:
    """Get information about entanglement pattern.

    Args:
        entanglement: Entanglement type
        n_qubits: Number of qubits

    Returns:
        Dictionary with entanglement information
    """
    # Number of CNOT gates for different entanglement patterns
    if entanglement == "full":
        n_cnots = n_qubits * (n_qubits - 1) // 2
    elif entanglement == "linear":
        n_cnots = n_qubits - 1
    elif entanglement == "reverse_linear":
        n_cnots = n_qubits - 1
    elif entanglement == "circular":
        n_cnots = n_qubits
    elif entanglement == "sca":
        n_cnots = n_qubits - 1
    else:
        n_cnots = n_qubits - 1
    
    info = {
        "entanglement_type": entanglement,
        "n_qubits": n_qubits,
        "estimated_cnots": n_cnots,
        "description": get_entanglement_description(entanglement),
    }
    
    return info


def get_entanglement_description(entanglement: str) -> str:
    """Get description of entanglement pattern.

    Args:
        entanglement: Entanglement type

    Returns:
        Description string
    """
    descriptions = {
        "full": "Full entanglement: every qubit is entangled with every other qubit",
        "linear": "Linear entanglement: each qubit is entangled with its neighbor",
        "reverse_linear": "Reverse linear: entanglement in reverse order",
        "circular": "Circular entanglement: linear with additional connection between ends",
        "sca": "SCA (shifted circular adjacency): scalable entanglement pattern",
    }
    
    return descriptions.get(entanglement, "Unknown entanglement type")

