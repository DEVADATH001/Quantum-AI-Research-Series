"""Noise simulation helpers for quantum kernel experiments."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

try:
    from qiskit.primitives import BackendSamplerV2
except ImportError:  # pragma: no cover - older Qiskit
    BackendSamplerV2 = None

try:
    from qiskit_aer.primitives import Sampler as AerSamplerV1
except ImportError:  # pragma: no cover - unexpected environment
    AerSamplerV1 = None

from src.quantum_kernel_engine import create_quantum_kernel

logger = logging.getLogger(__name__)


def create_ibm_noise_model(
    backend_name: str = "ibm_brisbane",
    readout_error: float = 0.01,
    gate_error: float = 0.001,
) -> NoiseModel:
    """Create a lightweight IBM-like noise model with depolarizing and thermal relaxation errors."""
    logger.info("Creating IBM-style noise model (backend=%s)", backend_name)

    noise_model = NoiseModel()

    # Typical superconducting qubit parameters
    t1 = 50e-6
    t2 = 70e-6
    time_u1 = 0
    time_u2 = 50e-9
    time_u3 = 100e-9
    time_cx = 300e-9

    error_t1t2_u1 = thermal_relaxation_error(t1, t2, time_u1)
    error_t1t2_u2 = thermal_relaxation_error(t1, t2, time_u2)
    error_t1t2_u3 = thermal_relaxation_error(t1, t2, time_u3)
    error_t1t2_cx = thermal_relaxation_error(t1, t2, time_cx).expand(
        thermal_relaxation_error(t1, t2, time_cx)
    )

    error_1q = depolarizing_error(gate_error, 1)
    noise_model.add_all_qubit_quantum_error(error_1q.compose(error_t1t2_u2), ["u1", "u2", "rx", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(error_1q.compose(error_t1t2_u3), ["u3"])

    error_2q = depolarizing_error(gate_error * 3, 2)
    noise_model.add_all_qubit_quantum_error(error_2q.compose(error_t1t2_cx), ["cx", "cz", "crx", "cry", "crz"])

    readout_error_obj = ReadoutError(
        [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
    )
    noise_model.add_all_qubit_readout_error(readout_error_obj)

    logger.info("Noise model created: gate_error=%s readout_error=%s with T1/T2 thermal relaxation", gate_error, readout_error)
    return noise_model


def create_aer_simulator(
    noise_model: Optional[NoiseModel] = None,
    seed: int = 42,
) -> AerSimulator:
    """Create an Aer simulator with optional noise."""
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    logger.info("AerSimulator created (noise=%s)", noise_model is not None)
    return simulator


def create_noisy_sampler(
    noise_model: NoiseModel,
    shots: int = 1000,
    seed: int = 42,
):
    """Create a noisy sampler primitive for kernel evaluation."""
    simulator = create_aer_simulator(noise_model=noise_model, seed=seed)

    if BackendSamplerV2 is not None:
        return BackendSamplerV2(
            backend=simulator,
            options={"default_shots": shots, "seed_simulator": seed},
        )

    if AerSamplerV1 is not None:
        return AerSamplerV1(
            backend_options={"noise_model": noise_model, "seed_simulator": seed},
            run_options={"shots": shots},
        )

    raise RuntimeError("No compatible noisy sampler implementation found.")


def simulate_noisy_kernel(
    feature_map: QuantumCircuit,
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    backend: str = "ibm_brisbane",
    readout_error: float = 0.01,
    gate_error: float = 0.001,
    shots: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, NoiseModel]:
    """Evaluate a quantum kernel matrix under simulated hardware noise."""
    logger.info("Simulating noisy kernel with %s shots", shots)

    noise_model = create_ibm_noise_model(
        backend_name=backend,
        readout_error=readout_error,
        gate_error=gate_error,
    )
    sampler = create_noisy_sampler(noise_model=noise_model, shots=shots, seed=seed)

    # Aer backends may reject undecomposed library instructions.
    kernel = create_quantum_kernel(feature_map=feature_map.decompose(), sampler=sampler)

    if Y is None:
        K_noisy = kernel.evaluate(x_vec=X)
    else:
        K_noisy = kernel.evaluate(x_vec=X, y_vec=Y)

    logger.info("Noisy kernel computed: shape=%s", K_noisy.shape)
    return K_noisy, noise_model


def analyze_noise_effects(
    noiseless_kernel: np.ndarray,
    noisy_kernel: np.ndarray,
) -> dict[str, float]:
    """Quantify differences between noiseless and noisy kernels."""
    diff = noiseless_kernel - noisy_kernel

    analysis = {
        "mean_absolute_error": float(np.mean(np.abs(diff))),
        "max_absolute_error": float(np.max(np.abs(diff))),
        "root_mean_square_error": float(np.sqrt(np.mean(diff**2))),
        "frobenius_norm_diff": float(np.linalg.norm(diff, "fro")),
        "spectral_norm_diff": float(np.linalg.norm(diff, 2)),
        "correlation": float(np.corrcoef(noiseless_kernel.flatten(), noisy_kernel.flatten())[0, 1]),
    }

    eigvals_noiseless = np.linalg.eigvalsh(noiseless_kernel)
    eigvals_noisy = np.linalg.eigvalsh(noisy_kernel)

    analysis["eigvals_noiseless_min"] = float(np.min(eigvals_noiseless))
    analysis["eigvals_noisy_min"] = float(np.min(eigvals_noisy))
    analysis["condition_noiseless"] = float(
        np.max(eigvals_noiseless) / (np.min(eigvals_noiseless) + 1e-10)
    )
    analysis["condition_noisy"] = float(
        np.max(eigvals_noisy) / (np.min(eigvals_noisy) + 1e-10)
    )

    logger.info("Noise analysis: %s", analysis)
    return analysis


def describe_nisq_noise_effects() -> str:
    """Return a brief text description of NISQ noise effects."""
    return (
        "\nNISQ Noise Effects on Quantum Kernels\n"
        "=====================================\n\n"
        "Noise perturbs kernel entries, can break PSD, and generally degrades\n"
        "classification performance.\n"
    )


def create_noisy_kernel_comparison(
    feature_map: QuantumCircuit,
    X: np.ndarray,
    readout_error: float = 0.01,
    gate_error: float = 0.001,
    shots: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute noiseless and noisy kernels plus a comparison report."""
    noiseless_kernel = create_quantum_kernel(feature_map=feature_map)
    K_noiseless = noiseless_kernel.evaluate(x_vec=X)

    K_noisy, _ = simulate_noisy_kernel(
        feature_map=feature_map,
        X=X,
        readout_error=readout_error,
        gate_error=gate_error,
        shots=shots,
    )

    analysis = analyze_noise_effects(K_noiseless, K_noisy)
    analysis["noise_model"] = {
        "readout_error": readout_error,
        "gate_error": gate_error,
        "shots": shots,
    }
    return K_noiseless, K_noisy, analysis
