"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Quantum-kernel construction and analysis utilities.

Upgrades (v2):
  - compute_geometric_difference: g(K_Q, K_C) — necessary condition for quantum advantage
    (Huang et al. 2022, Nature Communications 13, 4468).
  - get_git_sha: captures commit hash for reproducible result stamping (Phase 1.2).
"""

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

def monitor_exponential_concentration(kernel_matrix: np.ndarray, variance_threshold: float = 1e-4) -> None:
    """Raises a critical warning if the kernel suffers from massive exponential concentration."""
    if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        return
    off_diagonal = kernel_matrix[np.triu_indices_from(kernel_matrix, k=1)]
    if off_diagonal.size == 0:
        return
    
    variance = float(np.var(off_diagonal))
    mean_val = float(np.mean(off_diagonal))
    
    if variance < variance_threshold:
        logger.warning(
            "CRITICAL: Exponential Concentration detected in Quantum Kernel matrix! "
            "The feature overlap variance (%.2e) has collapsed below threshold (%.2e). "
            "Expect SVM gradients to vanish as the geometry mimics an Identity mapping.",
            variance, variance_threshold
        )
    logger.info("Exponential Concentration Check: Variance=%.2e, Mean=%.2e", variance, mean_val)

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
    monitor_exponential_concentration(K)
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

def regularize_kernel_matrix(
    kernel_matrix: np.ndarray, 
    epsilon: float = 1e-10,
    strategy: str = "shift"
) -> np.ndarray:
    """Project kernel matrix to the nearest PSD matrix if needed.
    
    This is mathematically essential for SVM stability, especially with noisy kernels.
    Strategies:
    - "shift": Fast eigenvalue shifting K + |min_eig|I + eps*I. Recommended for large/noisy grids.
    - "clip": O(N^3) eigenvalue clipping and exact reconstruction.
    """
    if strategy == "shift":
        eigenvalues = np.linalg.eigvalsh(kernel_matrix)
        min_eig = np.min(eigenvalues)
        if min_eig < epsilon:
            shift_amount = abs(min_eig) + epsilon
            regularized_matrix = kernel_matrix + shift_amount * np.eye(kernel_matrix.shape[0])
        else:
            regularized_matrix = kernel_matrix.copy()
    elif strategy == "clip":
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
        clipped_eigenvalues = np.maximum(eigenvalues, epsilon)
        regularized_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Preserve symmetry
    regularized_matrix = (regularized_matrix + regularized_matrix.T) / 2
    
    # Renormalize diagonal to 1.0 (state fidelity property)
    d = np.diag(regularized_matrix)
    d = np.maximum(d, 1e-12)  # Avoid division by zero
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
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square kernel matrix.")
    if K.shape[0] != len(y):
        raise ValueError("K and y must contain the same number of samples.")

    # Binary labels are mapped to {-1, 1} to build the ideal target kernel.
    unique_y = np.unique(y)
    if len(unique_y) != 2:
        logger.warning("KTA is primarily defined for binary classification.")
        return 0.0

    y_mapped = np.where(y == unique_y[0], -1.0, 1.0)
    Y = np.outer(y_mapped, y_mapped)

    return compute_kernel_alignment(K, Y)


def compute_centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Compute **Centered** Kernel-Target Alignment (cKTA).

    Reference: Cortes, Mohri & Rostamizadeh (2012), "Algorithms for Learning
    Kernels Based on Centered Alignment", JMLR 13, 795–828.

    Vanilla KTA (Cristianini et al. 2002) is biased by the class marginals:
    if one class dominates, K is dragged toward a constant matrix, inflating
    the alignment artificially.  Centering eliminates this bias:

        H = I_n − (1/n) 1 1ᵀ          (centering matrix)
        K_c = H K H                     (doubly-centred kernel)
        Y_c = H Y H                     (doubly-centred target)

        cKTA(K, Y) = ⟨K_c, Y_c⟩_F / (‖K_c‖_F · ‖Y_c‖_F)

    where ⟨A, B⟩_F = Σᵢⱼ Aᵢⱼ Bᵢⱼ is the Frobenius inner product.

    Properties:
        - cKTA ∈ [−1, 1]
        - Invariant to positive scalar rescaling of K
        - Invariant to relabelling when classes are balanced at 50/50
        - Reduces to vanilla KTA when data are centred and classes balanced

    Args:
        K: Square kernel matrix of shape (n, n), assumed symmetric PSD.
        y: Label vector of length n (arbitrary binary labels).

    Returns:
        Scalar cKTA ∈ [−1, 1].  Returns 0.0 for non-binary or constant kernels.

    Raises:
        ValueError: If K is not square or lengths do not match.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")
    n = K.shape[0]
    if len(y) != n:
        raise ValueError(f"y length ({len(y)}) must match K dimension ({n}).")

    unique_y = np.unique(y)
    if len(unique_y) != 2:
        logger.warning("cKTA is only defined for binary classification. Returning 0.")
        return 0.0

    # Build target kernel Y = y_mapped · y_mappeᵀ in {-1, +1}
    y_mapped = np.where(y == unique_y[0], -1.0, 1.0)
    Y = np.outer(y_mapped, y_mapped)

    # Centering matrix H = I - (1/n) 11ᵀ
    H = np.eye(n) - np.ones((n, n)) / n

    # Doubly-centred matrices
    K_c = H @ K @ H
    Y_c = H @ Y @ H

    # Frobenius inner product and norms
    frob_inner = float(np.sum(K_c * Y_c))
    norm_K_c = float(np.linalg.norm(K_c, "fro"))
    norm_Y_c = float(np.linalg.norm(Y_c, "fro"))

    if norm_K_c < 1e-15 or norm_Y_c < 1e-15:
        logger.warning(
            "cKTA: zero-norm centred matrix — kernel may be constant (exponential "
            "concentration). Returning 0."
        )
        return 0.0

    ckta = frob_inner / (norm_K_c * norm_Y_c)
    logger.info("Centered KTA (cKTA) = %.4f", ckta)
    return float(ckta)


# ---------------------------------------------------------------------------
# Geometric difference (quantum advantage metric)  — Phase 2.4
# ---------------------------------------------------------------------------

def compute_geometric_difference(
    K_quantum: np.ndarray,
    K_classical: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute the geometric difference g(K_Q, K_C).

    Defined in Huang et al. (2022), Nature Communications 13, 4468:

        g(K_Q, K_C) = sqrt(||(K_C^{1/2}) K_Q^{-1} (K_C^{1/2})||_∞)

    where ||·||_∞ denotes the spectral norm (largest singular value).

    g > 1 is a *necessary* (not sufficient) condition for quantum
    kernel methods to outperform classical ones on the same task.

    Args:
        K_quantum: Quantum kernel matrix (n × n), assumed PSD.
        K_classical: Classical kernel matrix (n × n), assumed PSD.
        epsilon: Tikhonov regularisation added to K_Q before inversion
                 to handle near-singularity.

    Returns:
        Geometric difference g (positive float).
        Returns float('inf') if K_Q is numerically singular.
    """
    n = K_quantum.shape[0]
    if K_classical.shape != (n, n):
        raise ValueError(
            f"Kernel matrix shapes must match: {K_quantum.shape} vs {K_classical.shape}"
        )

    # Regularise K_Q for numerical stability
    K_q_reg = K_quantum + epsilon * np.eye(n)

    try:
        K_q_inv = np.linalg.inv(K_q_reg)
    except np.linalg.LinAlgError:
        logger.error("Quantum kernel matrix is singular — g is undefined (∞).")
        return float("inf")

    # K_C^{1/2} via eigendecomposition
    eigvals_c, eigvecs_c = np.linalg.eigh(K_classical)
    eigvals_c = np.maximum(eigvals_c, 0.0)  # clip negative numerical noise
    K_c_sqrt = eigvecs_c @ np.diag(np.sqrt(eigvals_c)) @ eigvecs_c.T

    # M = K_C^{1/2} K_Q^{-1} K_C^{1/2}
    M = K_c_sqrt @ K_q_inv @ K_c_sqrt

    # Spectral norm = largest singular value
    spectral_norm = float(np.linalg.norm(M, ord=2))
    g = float(np.sqrt(spectral_norm))

    logger.info(
        "Geometric difference g(K_Q, K_C) = %.4f  (%s)",
        g,
        "ADVANTAGE PRECONDITION MET" if g > 1.0 else "no advantage precondition",
    )
    return g


# ---------------------------------------------------------------------------
# Reproducibility helper  — Phase 1.2
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    """Return the short Git commit SHA for result provenance stamping.

    Returns:
        Short SHA string (7 chars), or 'unknown' if git is unavailable.
    """
    import subprocess  # noqa: PLC0415
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"
