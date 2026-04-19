"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Expressibility computation for quantum feature maps.

Reference: Sim et al. (2019). Expressibility and Entangling Capability of
Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms.
Advanced Quantum Technologies, 2(12), 1900070.
https://doi.org/10.1002/qute.201900070
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fidelity distribution sampling
# ---------------------------------------------------------------------------

def sample_fidelity_distribution(
    circuit: QuantumCircuit,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample pairwise state-fidelity |<ψ(θ)|ψ(φ)>|² for random θ, φ.

    Args:
        circuit: Parameterised quantum circuit (feature map or ansatz).
        n_samples: Number of random parameter pairs to sample.
        rng: Random number generator for reproducibility.

    Returns:
        Array of shape (n_samples,) with fidelity values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_params = circuit.num_parameters
    if n_params == 0:
        logger.warning("Circuit has no parameters — expressibility is undefined (returns 0).")
        return np.zeros(n_samples)

    params = list(circuit.parameters)
    fidelities = []

    for _ in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        phi   = rng.uniform(0, 2 * np.pi, size=n_params)

        bind_theta = dict(zip(params, theta))
        bind_phi   = dict(zip(params, phi))

        sv_theta = Statevector(circuit.assign_parameters(bind_theta))
        sv_phi   = Statevector(circuit.assign_parameters(bind_phi))

        fid = float(np.abs(sv_theta.inner(sv_phi)) ** 2)
        fidelities.append(fid)

    return np.array(fidelities, dtype=np.float64)


def haar_fidelity_distribution(n_qubits: int, n_samples: int = 10_000) -> np.ndarray:
    """Analytically sample fidelity distribution for Haar-random states.

    For n qubits, the Haar fidelity follows Beta(1, 2^n - 1).

    Args:
        n_qubits: Number of qubits.
        n_samples: Number of Monte-Carlo samples.

    Returns:
        Array of shape (n_samples,) with Haar fidelity values.
    """
    dim = 2 ** n_qubits
    return np.random.default_rng(0).beta(1, dim - 1, size=n_samples)


# ---------------------------------------------------------------------------
# KL divergence (expressibility)
# ---------------------------------------------------------------------------

def kl_divergence_histogram(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int = 75,
    epsilon: float = 1e-10,
) -> float:
    """Estimate KL(P ‖ Q) from empirical samples using histogram binning.

    Args:
        p_samples: Samples from distribution P (circuit fidelities).
        q_samples: Samples from distribution Q (Haar fidelities).
        n_bins: Number of histogram bins over [0, 1].
        epsilon: Smoothing constant to avoid log(0).

    Returns:
        KL divergence (non-negative float). Lower ⇒ more expressive.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    # Normalise + smooth
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    return kl


# ---------------------------------------------------------------------------
# High-level expressibility function
# ---------------------------------------------------------------------------

def compute_expressibility(
    circuit: QuantumCircuit,
    n_samples: int = 1000,
    n_bins: int = 75,
    seed: int = 42,
) -> dict:
    """Compute the expressibility ε = KL(P_PQC ‖ P_Haar).

    Lower ε means the circuit spans the Hilbert space more uniformly
    (more expressive). Haar-random circuits have ε = 0.

    Args:
        circuit: Parameterised quantum circuit.
        n_samples: Number of random fidelity samples for the empirical estimate.
        n_bins: Histogram bins for KL divergence.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys: expressibility (float), fidelity_mean, fidelity_std,
        haar_mean, haar_std, n_qubits, n_params, circuit_depth.
    """
    rng = np.random.default_rng(seed)
    n_qubits = circuit.num_qubits

    logger.info(
        "Computing expressibility: qubits=%d params=%d samples=%d",
        n_qubits, circuit.num_parameters, n_samples,
    )

    pqc_fids = sample_fidelity_distribution(circuit, n_samples=n_samples, rng=rng)
    haar_fids = haar_fidelity_distribution(n_qubits, n_samples=n_samples * 10)

    eps = kl_divergence_histogram(pqc_fids, haar_fids, n_bins=n_bins)

    result = {
        "expressibility": float(eps),
        "fidelity_mean": float(np.mean(pqc_fids)),
        "fidelity_std": float(np.std(pqc_fids)),
        "haar_mean": float(np.mean(haar_fids)),
        "haar_std": float(np.std(haar_fids)),
        "n_qubits": n_qubits,
        "n_params": circuit.num_parameters,
        "circuit_depth": circuit.depth(),
    }

    logger.info("Expressibility ε = %.4f (lower → more expressive / Haar-like)", eps)
    return result


# ---------------------------------------------------------------------------
# Entanglement capability
# ---------------------------------------------------------------------------

def compute_entanglement_capability(
    circuit: QuantumCircuit,
    n_samples: int = 500,
    seed: int = 42,
) -> float:
    """Estimate entanglement capability via Meyer-Wallach measure Q̄.

    Q̄ ∈ [0, 1]; Q̄ = 0 → fully separable, Q̄ = 1 → maximally entangled.

    Args:
        circuit: Parameterised circuit.
        n_samples: Number of random parameter samples.
        seed: RNG seed.

    Returns:
        Float Meyer-Wallach entanglement capability Q̄.
    """
    rng = np.random.default_rng(seed)
    n_params = circuit.num_parameters
    n_qubits = circuit.num_qubits

    if n_params == 0 or n_qubits < 2:
        logger.warning("Cannot compute entanglement capability for circuit with %d qubits / "
                       "%d params.", n_qubits, n_params)
        return 0.0

    params = list(circuit.parameters)
    Q_vals = []

    for _ in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        sv = Statevector(circuit.assign_parameters(dict(zip(params, theta))))
        state_vec = sv.data  # (2^n,) complex array
        state_vec = state_vec.reshape([2] * n_qubits)

        q = _meyer_wallach_Q(state_vec, n_qubits)
        Q_vals.append(q)

    Q_bar = float(np.mean(Q_vals))
    logger.info("Entanglement capability Q̄ = %.4f", Q_bar)
    return Q_bar


def _meyer_wallach_Q(state_tensor: np.ndarray, n_qubits: int) -> float:
    """Compute Meyer-Wallach Q for a single pure state tensor."""
    q_sum = 0.0
    for k in range(n_qubits):
        # Trace out all qubits except k
        axes = list(range(n_qubits))
        axes.remove(k)
        rho_k = np.tensordot(
            state_tensor, state_tensor.conj(), axes=(axes, axes)
        )
        # Linear entropy of single-qubit reduced state
        q_sum += 1.0 - float(np.real(np.trace(rho_k @ rho_k)))

    return (4.0 / n_qubits) * q_sum
