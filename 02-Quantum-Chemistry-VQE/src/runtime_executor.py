"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Runtime estimator factory.

Supported backends:
    local           — StatevectorEstimator (noiseless simulation)
    noisy_local     — AerEstimator with configurable depolarizing noise model
    <IBM backend>   — EstimatorV2 via qiskit_ibm_runtime (hardware/cloud)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from qiskit.primitives import StatevectorEstimator

try:
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except Exception:
    EstimatorV2 = None
    QiskitRuntimeService = None
    IBM_RUNTIME_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )
    AER_AVAILABLE = True
except Exception:
    AerSimulator = None
    AerEstimatorV2 = None
    NoiseModel = None
    depolarizing_error = None
    thermal_relaxation_error = None
    AER_AVAILABLE = False


@dataclass
class EstimatorContext:
    """Estimator plus runtime metadata."""

    estimator: Any
    mode: str
    backend: Optional[str]
    mitigation: str


# ---------------------------------------------------------------------------
# Noise model factory (NISQ-realistic parameters)
# ---------------------------------------------------------------------------

def _build_noise_model(level: str = "light") -> Any:
    """Build a depolarizing + T1/T2 noise model for local simulation.

    Args:
        level: "light" (T1/T2=50µs, gate_err=0.001) or
               "medium" (T1/T2=10µs, gate_err=0.005).

    Returns:
        qiskit_aer NoiseModel or None if Aer is unavailable.
    """
    if not AER_AVAILABLE:
        raise RuntimeError(
            "qiskit-aer is required for noisy_local backend. "
            "Install with: pip install qiskit-aer"
        )

    if level == "light":
        t1, t2 = 50e-6, 40e-6      # seconds
        gate_1q_err = 1e-3
        gate_2q_err = 5e-3
        gate_time_1q = 50e-9        # nanoseconds → seconds
        gate_time_2q = 200e-9
    elif level == "medium":
        t1, t2 = 10e-6, 8e-6
        gate_1q_err = 5e-3
        gate_2q_err = 2e-2
        gate_time_1q = 50e-9
        gate_time_2q = 300e-9
    else:
        raise ValueError(f"Unknown noise level: '{level}'. Choose 'light' or 'medium'.")

    nm = NoiseModel()

    # 1-qubit gate errors: depolarizing + T1/T2 relaxation
    err_1q = (
        depolarizing_error(gate_1q_err, 1)
        .compose(thermal_relaxation_error(t1, t2, gate_time_1q))
    )
    # 2-qubit gate errors
    err_2q = (
        depolarizing_error(gate_2q_err, 2)
        .expand(thermal_relaxation_error(t1, t2, gate_time_2q))
    )

    nm.add_all_qubit_quantum_error(err_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])

    return nm


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def get_estimator(
    backend_name: str = "local",
    resilience_level: int = 1,
    optimization_level: int = 1,
    shots: int = 4096,
    seed: int = 7,
    noise_level: str = "light",
) -> EstimatorContext:
    """Return an EstimatorContext for the requested backend.

    Args:
        backend_name: "local", "noisy_local", or IBM backend name.
        resilience_level: Error mitigation level (IBM Runtime only).
        optimization_level: Transpilation level (IBM Runtime only).
        shots: Number of shots (noisy/IBM modes only).
        seed: RNG seed for reproducible statevector simulation.
        noise_level: "light" or "medium" (noisy_local only).

    Returns:
        EstimatorContext with .estimator, .mode, .backend, .mitigation.
    """
    # --- Noiseless local simulation ---
    if backend_name == "local":
        return EstimatorContext(
            estimator=StatevectorEstimator(seed=seed),
            mode="local_statevector",
            backend=None,
            mitigation="Noiseless statevector simulation.",
        )

    # --- Noisy local simulation (Aer + depolarizing noise) ---
    if backend_name == "noisy_local":
        if not AER_AVAILABLE:
            raise RuntimeError(
                "qiskit-aer is required for noisy_local backend. "
                "Install with: pip install qiskit-aer"
            )
        noise_model = _build_noise_model(level=noise_level)
        aer_backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        estimator = AerEstimatorV2(
            backend=aer_backend,
            options={"default_shots": shots},
        )
        return EstimatorContext(
            estimator=estimator,
            mode="noisy_local",
            backend=f"AerSimulator[{noise_level}]",
            mitigation=f"Depolarizing+T1/T2 noise model ({noise_level}), {shots} shots, no mitigation.",
        )

    # --- IBM Quantum Runtime ---
    if not IBM_RUNTIME_AVAILABLE:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed but a hardware backend was requested. "
            "Install with: pip install qiskit-ibm-runtime"
        )

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    options = {
        "resilience_level": max(1, resilience_level),
        "optimization_level": max(1, optimization_level),
        "default_shots": max(1, shots),
    }

    mitigation_text = f"resilience_level={options['resilience_level']}"
    if options["resilience_level"] == 1:
        mitigation_text += " (Readout error mitigation / T-REX)"
    elif options["resilience_level"] >= 2:
        mitigation_text += " (Zero-Noise Extrapolation)"

    estimator = EstimatorV2(mode=backend, options=options)
    return EstimatorContext(
        estimator=estimator,
        mode="ibm_runtime",
        backend=backend_name,
        mitigation=mitigation_text,
    )
