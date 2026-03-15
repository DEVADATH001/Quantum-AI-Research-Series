"""Runtime estimator factory."""

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


@dataclass
class EstimatorContext:
    """Estimator plus runtime metadata."""

    estimator: Any
    mode: str
    backend: Optional[str]
    mitigation: str


def get_estimator(
    backend_name: str = "local",
    resilience_level: int = 1,
    optimization_level: int = 1,
    shots: int = 4096,
    seed: int = 7,
) -> EstimatorContext:
    """Return estimator context for local simulation or IBM Runtime."""
    if backend_name == "local":
        return EstimatorContext(
            estimator=StatevectorEstimator(seed=seed),
            mode="local_statevector",
            backend=None,
            mitigation="No sampling noise in statevector mode.",
        )

    if not IBM_RUNTIME_AVAILABLE:
        raise RuntimeError("qiskit_ibm_runtime is not installed but runtime backend was requested.")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    # Advanced NISQ hardware settings: 
    # resilience_level=2 enables Zero-Noise Extrapolation (ZNE)
    # optimization_level=3 enables aggressive gate-folding and swap reduction
    options = {
        "resilience_level": max(1, resilience_level),
        "optimization_level": max(1, optimization_level),
        "default_shots": max(1, shots),
    }
    
    mitigation_text = f"resilience_level={options['resilience_level']}"
    if options['resilience_level'] == 1:
        mitigation_text += " (Readout/T-REX)"
    elif options['resilience_level'] >= 2:
        mitigation_text += " (ZNE/Gate Error)"
        
    estimator = EstimatorV2(mode=backend, options=options)
    return EstimatorContext(
        estimator=estimator,
        mode="ibm_runtime",
        backend=backend_name,
        mitigation=mitigation_text,
    )
