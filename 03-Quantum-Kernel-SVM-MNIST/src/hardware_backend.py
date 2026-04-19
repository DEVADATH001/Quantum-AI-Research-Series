"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Hardware Backend Bridge — swap between AerSimulator (simulation) and
      QiskitRuntimeService (real IBM Quantum hardware) without touching any
      other module.

Design
------
The `HardwareBackendManager` follows the Strategy pattern: it exposes a single
`get_sampler()` method that returns either a `StatevectorSampler` (noiseless
simulation), an `AerSimulator`-backed sampler (noise simulation), or a
`SamplerV2` from `QiskitRuntimeService` (real hardware) — whichever is
appropriate given the configuration.

All downstream code passes the sampler into `create_quantum_kernel(sampler=...)`.
No other module needs to change.

IBM Quantum token
-----------------
The token is read with the following priority:
  1. Constructor argument `ibm_token`.
  2. Environment variable `IBM_QUANTUM_TOKEN`.
  3. Config file value (passed from YAML via the constructor).

Usage
-----
>>> from src.hardware_backend import HardwareBackendManager
>>> mgr = HardwareBackendManager(backend_name="ibm_brisbane",
...                               use_real_hardware=False)
>>> sampler = mgr.get_sampler()
>>> info = mgr.get_backend_info()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — each may not be present depending on Qiskit version
# ---------------------------------------------------------------------------

try:
    from qiskit.primitives import StatevectorSampler
    _HAS_STATEVECTOR_SAMPLER = True
except ImportError:
    StatevectorSampler = None  # type: ignore[assignment,misc]
    _HAS_STATEVECTOR_SAMPLER = False

try:
    from qiskit.primitives import Sampler as LegacySampler
    _HAS_LEGACY_SAMPLER = True
except ImportError:
    LegacySampler = None  # type: ignore[assignment,misc]
    _HAS_LEGACY_SAMPLER = False

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    _HAS_AER = True
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator  # type: ignore[no-redef]
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # type: ignore[no-redef]
        _HAS_AER = True
    except ImportError:
        AerSimulator = None  # type: ignore[assignment,misc]
        AerSamplerV2 = None  # type: ignore[assignment,misc]
        _HAS_AER = False

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSamplerV2
    _HAS_IBM_RUNTIME = True
except ImportError:
    QiskitRuntimeService = None  # type: ignore[assignment,misc]
    RuntimeSamplerV2 = None  # type: ignore[assignment,misc]
    _HAS_IBM_RUNTIME = False

try:
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    _HAS_FAKE_PROVIDER = True
except ImportError:
    FakeBrisbane = None  # type: ignore[assignment,misc]
    _HAS_FAKE_PROVIDER = False


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class HardwareBackendManager:
    """Select and configure the correct Qiskit sampler primitive.

    Parameters
    ----------
    backend_name:
        Name of the IBM Quantum backend (e.g. ``'ibm_brisbane'``).
        Used when ``use_real_hardware=True``.
    use_real_hardware:
        If True, attempt to connect to QiskitRuntimeService and use a real
        backend.  If False (default), use noise-free statevector simulation.
    shots:
        Number of measurement shots per circuit evaluation.  Only relevant
        for shot-based (non-statevector) samplers.
    ibm_token:
        IBM Quantum API token.  If None, falls back to the ``IBM_QUANTUM_TOKEN``
        environment variable.
    use_noise_model:
        If True (and use_real_hardware=False), apply the FakeBrisbane noise
        model via AerSimulator.  Ignored when use_real_hardware=True.
    """

    def __init__(
        self,
        backend_name: str = "ibm_brisbane",
        use_real_hardware: bool = False,
        shots: int = 4096,
        ibm_token: Optional[str] = None,
        use_noise_model: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.use_real_hardware = use_real_hardware
        self.shots = shots
        self._ibm_token: str | None = ibm_token or os.environ.get("IBM_QUANTUM_TOKEN")
        self.use_noise_model = use_noise_model

        self._service: Any = None  # QiskitRuntimeService instance (lazy)
        self._backend: Any = None  # IBM backend instance (lazy)

        logger.info(
            "HardwareBackendManager: backend=%s, real_hw=%s, shots=%d, noise_model=%s",
            backend_name, use_real_hardware, shots, use_noise_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_simulation(self) -> bool:
        """True if running in simulation mode (no real hardware)."""
        return not self.use_real_hardware

    def get_sampler(self) -> Any:
        """Return the appropriate sampler primitive for the current configuration.

        Priority:
        1. Real IBM Quantum hardware (use_real_hardware=True + valid token).
        2. Noisy AerSimulator with FakeBrisbane model (use_noise_model=True).
        3. Noiseless StatevectorSampler (fastest, most reproducible).

        Returns:
            A Qiskit sampler primitive compatible with FidelityQuantumKernel.
        """
        if self.use_real_hardware:
            return self._get_hardware_sampler()

        if self.use_noise_model:
            return self._get_noisy_aer_sampler()

        return self._get_statevector_sampler()

    def get_backend_info(self) -> dict[str, Any]:
        """Return metadata about the current backend configuration.

        Returns:
            Dict with keys: name, mode (simulation/hardware), shots,
            noise_model, n_qubits (if available), basis_gates (if available),
            t1_mean_us (if available from real backend or fake provider).
        """
        info: dict[str, Any] = {
            "name": self.backend_name,
            "mode": "hardware" if self.use_real_hardware else "simulation",
            "shots": self.shots,
            "noise_model": self.use_noise_model if not self.use_real_hardware else "device",
        }

        if self.use_real_hardware and self._backend is not None:
            try:
                cfg = self._backend.configuration()
                info["n_qubits"] = cfg.n_qubits
                info["basis_gates"] = cfg.basis_gates
            except Exception:
                pass
            try:
                props = self._backend.properties()
                t1_values = [
                    props.qubit_property(q, "T1")[0]
                    for q in range(self._backend.configuration().n_qubits)
                    if props.qubit_property(q, "T1") is not None
                ]
                if t1_values:
                    info["t1_mean_us"] = float(
                        sum(t1_values) / len(t1_values) * 1e6  # seconds → µs
                    )
            except Exception:
                pass

        if not self.use_real_hardware and self.use_noise_model and _HAS_FAKE_PROVIDER:
            try:
                fake = FakeBrisbane()
                cfg = fake.configuration()
                info["n_qubits"] = cfg.n_qubits
                info["basis_gates"] = cfg.basis_gates
                info["fake_provider"] = "FakeBrisbane"
            except Exception:
                pass

        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_statevector_sampler(self) -> Any:
        """Noiseless statevector sampler — fastest and fully reproducible."""
        if _HAS_STATEVECTOR_SAMPLER:
            sampler = StatevectorSampler()
            logger.info("Using StatevectorSampler (noiseless, exact).")
            return sampler
        if _HAS_LEGACY_SAMPLER:
            sampler = LegacySampler()
            logger.info("Using legacy Sampler primitive (noiseless).")
            return sampler
        raise RuntimeError(
            "No statevector sampler available. Install qiskit>=1.1."
        )

    def _get_noisy_aer_sampler(self) -> Any:
        """AerSimulator sampler with FakeBrisbane noise model."""
        if not _HAS_AER:
            logger.warning(
                "qiskit-aer not installed — falling back to noiseless statevector sampler."
            )
            return self._get_statevector_sampler()

        noise_model = None
        if _HAS_FAKE_PROVIDER:
            try:
                from qiskit_aer.noise import NoiseModel
                fake_backend = FakeBrisbane()
                noise_model = NoiseModel.from_backend(fake_backend)
                logger.info("FakeBrisbane noise model loaded successfully.")
            except Exception as exc:
                logger.warning("Could not load FakeBrisbane noise model: %s", exc)

        backend = AerSimulator(noise_model=noise_model)
        sampler = AerSamplerV2.from_backend(backend)
        sampler.options.default_shots = self.shots
        logger.info(
            "Using AerSamplerV2 with %s noise model, shots=%d.",
            "FakeBrisbane" if noise_model else "no",
            self.shots,
        )
        return sampler

    def _get_hardware_sampler(self) -> Any:
        """Real IBM Quantum hardware sampler via QiskitRuntimeService."""
        if not _HAS_IBM_RUNTIME:
            raise RuntimeError(
                "qiskit-ibm-runtime is not installed. "
                "Run: pip install qiskit-ibm-runtime"
            )
        if not self._ibm_token:
            raise ValueError(
                "IBM Quantum token is required for real hardware execution. "
                "Set ibm_token= or export IBM_QUANTUM_TOKEN=<token>."
            )

        if self._service is None:
            logger.info("Connecting to QiskitRuntimeService...")
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self._ibm_token,
            )
            logger.info("Connected to IBM Quantum (channel=ibm_quantum).")

        if self._backend is None:
            self._backend = self._service.backend(self.backend_name)
            logger.info("Acquired backend: %s", self.backend_name)

        from qiskit_ibm_runtime import SamplerV2 as _RuntimeSampler
        sampler = _RuntimeSampler(self._backend)
        sampler.options.default_shots = self.shots
        logger.info(
            "RuntimeSamplerV2 configured for %s, shots=%d.",
            self.backend_name, self.shots,
        )
        return sampler
