"""Quantum-kernel construction and analysis utilities."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit

try:
    from qiskit.primitives import StatevectorSampler
except ImportError:  # pragma: no cover - older Qiskit
    StatevectorSampler = None

try:
    from qiskit.primitives import Sampler as LegacySampler
except ImportError:  # pragma: no cover - newer Qiskit
    LegacySampler = None

from qiskit_machine_learning.kernels import FidelityQuantumKernel

try:
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
except ImportError:  # pragma: no cover - older Qiskit ML
    ComputeUncompute = None

logger = logging.getLogger(__name__)


def _build_default_sampler() -> Any:
    """Create the best-available default sampler for current Qiskit."""
    if StatevectorSampler is not None:
        return StatevectorSampler()
    if LegacySampler is not None:
        return LegacySampler()
    raise RuntimeError("No compatible sampler primitive found in this Qiskit installation.")


def create_quantum_kernel(
    feature_map: QuantumCircuit,
    sampler: Optional[Any] = None,
    cache: bool = True,
    enforce_psd: bool = True,
) -> FidelityQuantumKernel:
    """Create a FidelityQuantumKernel compatible with old/new APIs."""
    del cache  # no-op for current Qiskit Machine Learning versions

    if sampler is None:
        sampler = _build_default_sampler()
        logger.info("Created default sampler: %s", type(sampler).__name__)

    signature = inspect.signature(FidelityQuantumKernel.__init__)
    params = signature.parameters

    if "fidelity" in params:
        if ComputeUncompute is None:
            raise RuntimeError(
                "This Qiskit Machine Learning version requires state fidelity helpers "
                "but they are unavailable."
            )
        fidelity = ComputeUncompute(sampler)
        kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=fidelity,
            enforce_psd=enforce_psd,
        )
    elif "sampler" in params:
        kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)
    else:
        kernel = FidelityQuantumKernel(feature_map=feature_map)

    logger.info("FidelityQuantumKernel created successfully")
    return kernel


def compute_kernel_matrix(
    quantum_kernel: FidelityQuantumKernel,
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a kernel matrix K(X, Y)."""
    if Y is None:
        logger.info("Computing kernel matrix for %s samples", X.shape[0])
        K = quantum_kernel.evaluate(x_vec=X)
    else:
        logger.info("Computing kernel matrix: %s x %s", X.shape[0], Y.shape[0])
        K = quantum_kernel.evaluate(x_vec=X, y_vec=Y)

    logger.info("Kernel matrix shape: %s", K.shape)
    return K


def analyze_kernel_properties(kernel_matrix: np.ndarray) -> dict:
    """Analyze symmetry, PSD, and numeric properties of a kernel matrix."""
    is_symmetric = np.allclose(kernel_matrix, kernel_matrix.T)
    eigenvalues = np.linalg.eigvalsh(kernel_matrix)
    is_psd = np.all(eigenvalues >= -1e-10)

    diagonal = np.diag(kernel_matrix)
    off_diagonal = kernel_matrix[np.triu_indices_from(kernel_matrix, k=1)]

    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))
    if abs(min_eig) < 1e-12:
        condition_number = float("inf")
    else:
        condition_number = float(max_eig / min_eig)

    properties = {
        "shape": kernel_matrix.shape,
        "is_symmetric": bool(is_symmetric),
        "is_positive_semidefinite": bool(is_psd),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": condition_number,
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)) if off_diagonal.size else 0.0,
        "off_diagonal_std": float(np.std(off_diagonal)) if off_diagonal.size else 0.0,
    }

    logger.info("Kernel properties: %s", properties)
    return properties


def regularize_kernel_matrix(kernel_matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project kernel matrix to the nearest PSD matrix if needed.
    
    This is mathematically essential for SVM stability, especially with noisy kernels.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
    
    # Clip negative eigenvalues to epsilon
    clipped_eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Reconstruct matrix: V * diag(Lambda_clipped) * V.T
    regularized_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    
    # Preserve symmetry
    regularized_matrix = (regularized_matrix + regularized_matrix.T) / 2
    
    # Renormalize diagonal to 1.0 (state fidelity property)
    d = np.diag(regularized_matrix)
    regularized_matrix = regularized_matrix / np.sqrt(np.outer(d, d))
    
    return regularized_matrix


def describe_kernel_theory() -> str:
    """Return a short theory note for the quantum kernel."""
    return (
        "\nQuantum Kernel Theory\n"
        "=====================\n\n"
        "The fidelity quantum kernel compares two embedded states:\n"
        "K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2\n"
    )


def compute_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute Frobenius kernel alignment between K1 and K2."""
    k1_flat = K1.flatten()
    k2_flat = K2.flatten()

    inner_product = float(np.dot(k1_flat, k2_flat))
    norm1 = float(np.linalg.norm(k1_flat))
    norm2 = float(np.linalg.norm(k2_flat))

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Kernel alignment is undefined for zero-norm kernel matrices.")

    alignment = inner_product / (norm1 * norm2)
    logger.info("Kernel alignment: %.4f", alignment)
    return float(alignment)


def compute_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Compute Kernel-Target Alignment (KTA) between K and labels y.
    
    KTA measures how well the kernel matrix matches the ideal label kernel Y = y * y.T.
    The labels y should be encoded as {-1, 1} for this calculation.
    """
    # Ensure labels are binary and in {-1, 1}
    unique_y = np.unique(y)
    if len(unique_y) != 2:
        logger.warning("KTA is primarily defined for binary classification.")
        return 0.0
    
    y_mapped = np.where(y == unique_y[0], -1, 1)
    Y = np.outer(y_mapped, y_mapped)
    
    return compute_kernel_alignment(K, Y)
