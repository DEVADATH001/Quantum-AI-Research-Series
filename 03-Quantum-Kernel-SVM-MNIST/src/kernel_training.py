"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Trainable quantum kernel via Kernel Alignment Training (QKAT).

Reference: Hubregtsen et al. (2022). Training Quantum Embedding Kernels on
Near-Term Quantum Computers. Physical Review A, 106(4), 042431.
https://doi.org/10.1103/PhysRevA.106.042431
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KTA gradient utilities
# ---------------------------------------------------------------------------

def _compute_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centred Kernel-Target Alignment (Cortes et al. 2012).

    KTA(K, y) = <K, yy^T>_F / (||K||_F * n)

    Labels are mapped to {-1, +1}.
    """
    unique = np.unique(y)
    y_pm = np.where(y == unique[0], -1.0, 1.0)
    Y = np.outer(y_pm, y_pm)

    num = float(np.sum(K * Y))
    denom = float(np.linalg.norm(K, "fro") * len(y))
    if denom < 1e-14:
        return 0.0
    return num / denom


def _kta_gradient_fd(
    circuit: QuantumCircuit,
    param_values: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn: Any,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference gradient of KTA w.r.t. circuit parameters.

    Args:
        circuit: Parameterised feature map (parameters in sorted order).
        param_values: Current parameter values, shape (n_params,).
        X: Training data.
        y: Training labels.
        kernel_fn: Callable(circuit) → FidelityQuantumKernel.
        eps: Finite-difference step size.

    Returns:
        Gradient array of shape (n_params,).
    """
    grad = np.zeros_like(param_values)
    for i in range(len(param_values)):
        p_fwd = param_values.copy()
        p_bwd = param_values.copy()
        p_fwd[i] += eps
        p_bwd[i] -= eps

        K_fwd = _evaluate_kernel(circuit, p_fwd, X, kernel_fn)
        K_bwd = _evaluate_kernel(circuit, p_bwd, X, kernel_fn)

        kta_fwd = _compute_kta(K_fwd, y)
        kta_bwd = _compute_kta(K_bwd, y)

        grad[i] = (kta_fwd - kta_bwd) / (2 * eps)
    return grad


def _evaluate_kernel(
    circuit: QuantumCircuit,
    param_values: np.ndarray,
    X: np.ndarray,
    kernel_fn: Any,
) -> np.ndarray:
    """Bind parameters to circuit and compute the kernel matrix on X."""
    params = sorted(circuit.parameters, key=lambda p: p.name)
    bound = circuit.assign_parameters(dict(zip(params, param_values)))
    qk = kernel_fn(bound)
    return qk.evaluate(x_vec=X)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_quantum_kernel_alignment(
    circuit: QuantumCircuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_fn: Any,
    optimizer: str = "adam",
    max_iter: int = 50,
    learning_rate: float = 0.01,
    tol: float = 1e-5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train a parameterised feature map to maximise KTA.

    Uses gradient ascent (Adam or SGD) over circuit parameters θ:

        θ* = argmax_θ KTA(K_θ, y)
             where K_θ[i,j] = |<ψ(x_i;θ)|ψ(x_j;θ)>|²

    Args:
        circuit: Parameterised feature map (the parameters will be trained).
        X_train: Training data, shape (n, d).
        y_train: Training labels.
        kernel_fn: Factory Callable(circuit) → FidelityQuantumKernel.
        optimizer: "adam" or "sgd".
        max_iter: Maximum gradient ascent steps.
        learning_rate: Step size α.
        tol: Early stopping if |ΔKTA| < tol for 3 consecutive steps.
        seed: Random seed for parameter initialisation.
        verbose: Log progress every 10 steps.

    Returns:
        dict with keys:
            optimal_params: np.ndarray of trained parameter values.
            optimal_circuit: Circuit with optimal params bound (as ParameterVector).
            kta_history: list of KTA values per iteration.
            final_kta: float.
            n_iter: int.
            train_time: float (seconds).
    """
    rng = np.random.default_rng(seed)
    params = sorted(circuit.parameters, key=lambda p: p.name)
    n_params = len(params)

    if n_params == 0:
        logger.warning("Circuit has no trainable parameters — returning fixed kernel.")
        K0 = kernel_fn(circuit).evaluate(x_vec=X_train)
        kta0 = _compute_kta(K0, y_train)
        return {
            "optimal_params": np.array([]),
            "optimal_circuit": circuit,
            "kta_history": [kta0],
            "final_kta": kta0,
            "n_iter": 0,
            "train_time": 0.0,
        }

    # Initialise parameters randomly
    theta = rng.uniform(0, 2 * np.pi, size=n_params)
    kta_history = []
    stagnation = 0
    t0 = time.perf_counter()

    # Adam state
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    logger.info(
        "QKAT training: optimizer=%s lr=%.4f max_iter=%d n_params=%d n_train=%d",
        optimizer, learning_rate, max_iter, n_params, len(X_train),
    )

    for step in range(1, max_iter + 1):
        grad = _kta_gradient_fd(circuit, theta, X_train, y_train, kernel_fn)

        if optimizer == "adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            delta = learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
        else:  # SGD
            delta = learning_rate * grad

        theta += delta  # ascent

        K = _evaluate_kernel(circuit, theta, X_train, kernel_fn)
        kta = _compute_kta(K, y_train)
        kta_history.append(kta)

        if verbose and step % 10 == 0:
            logger.info("  QKAT step %3d/%d | KTA = %.4f | |grad| = %.4e",
                        step, max_iter, kta, float(np.linalg.norm(grad)))

        # Early stopping
        if len(kta_history) > 1 and abs(kta_history[-1] - kta_history[-2]) < tol:
            stagnation += 1
            if stagnation >= 3:
                logger.info("QKAT converged at step %d (|ΔKTA| < %.1e).", step, tol)
                break
        else:
            stagnation = 0

    train_time = time.perf_counter() - t0
    final_kta = kta_history[-1] if kta_history else 0.0

    logger.info(
        "QKAT done: final_KTA=%.4f | steps=%d | time=%.1fs",
        final_kta, len(kta_history), train_time,
    )

    # Build optimal circuit with bound parameters
    optimal_circuit = circuit.assign_parameters(dict(zip(params, theta)))

    return {
        "optimal_params": theta,
        "optimal_circuit": optimal_circuit,
        "kta_history": [float(k) for k in kta_history],
        "final_kta": float(final_kta),
        "n_iter": len(kta_history),
        "train_time": float(train_time),
    }
