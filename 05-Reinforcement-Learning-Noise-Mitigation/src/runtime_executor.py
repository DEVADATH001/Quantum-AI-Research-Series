"""Execution engine for ideal, noisy, mitigated, and hardware QRL modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

from src.mitigation_engine import MitigationConfig, MitigationEngine, fold_circuit_for_noise_scaling
from src.noise_models import infer_readout_error_probability, load_ibm_noise_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime execution configuration."""

    mode: str = "ideal"  # ideal | noisy | mitigated | hardware
    shots: int = 1024
    backend_name: str = "ibm_osaka"
    optimization_level: int = 1
    resilience_level: int = 0
    seed: int = 42


class QuantumRuntimeExecutor:
    """Unified expectation-value executor for the QRL training loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        mitigation_engine: MitigationEngine | None = None,
    ) -> None:
        self.config = config
        self.mode = config.mode.lower()
        self.noise_model = None
        self._hardware_backend = None
        self._runtime_estimator = None
        self._runtime_session = None
        self._transpiled_circuits: dict[int, QuantumCircuit] = {}
        self._folded_circuits: dict[tuple[int, float], QuantumCircuit] = {}

        if self.mode not in {"ideal", "noisy", "mitigated", "hardware"}:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.mode == "ideal":
            self._aer_estimator = AerEstimatorV2(
                options={
                    "backend_options": {"seed_simulator": config.seed},
                    "run_options": {"shots": config.shots},
                }
            )
            self.mitigation_engine = mitigation_engine

        elif self.mode in {"noisy", "mitigated"}:
            self.noise_model = load_ibm_noise_model(config.backend_name)
            self._aer_estimator = AerEstimatorV2(
                options={
                    "backend_options": {
                        "seed_simulator": config.seed,
                        "noise_model": self.noise_model,
                    },
                    "run_options": {"shots": config.shots},
                }
            )
            readout_p = infer_readout_error_probability(self.noise_model)
            if mitigation_engine is None:
                mitigation_engine = MitigationEngine(
                    MitigationConfig(
                        resilience_level=max(2, config.resilience_level),
                        readout_error_probability=readout_p,
                    )
                )
            self.mitigation_engine = mitigation_engine

        else:
            self._aer_estimator = None
            self._configure_hardware_runtime()
            self.mitigation_engine = mitigation_engine

    def close(self) -> None:
        """Close any open hardware sessions."""
        if self._runtime_session is not None:
            self._runtime_session.close()
            logger.info("Closed IBM Runtime Session.")

    def _configure_hardware_runtime(self) -> None:
        """Configure IBM Runtime EstimatorV2 with Session support."""
        from qiskit_ibm_runtime import EstimatorOptions, EstimatorV2, QiskitRuntimeService, Session

        service = QiskitRuntimeService()
        backend = service.backend(self.config.backend_name)

        options = EstimatorOptions()
        options.default_shots = self.config.shots
        options.seed_estimator = self.config.seed
        options.resilience_level = self.config.resilience_level

        if self.config.resilience_level >= 2:
            options.twirling.enable_measure = True
            options.resilience.measure_mitigation = True
            options.resilience.zne_mitigation = True
            options.resilience.zne.noise_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
            options.resilience.zne.extrapolator = "exponential"

        self._hardware_backend = backend
        self._runtime_session = Session(service=service, backend=backend)
        self._runtime_estimator = EstimatorV2(mode=self._runtime_session, options=options)
        logger.info("Opened IBM Runtime Session on backend %s", backend.name)

    def _prepare_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile to apply the requested optimization level consistently."""
        circuit_id = id(circuit)
        if circuit_id in self._transpiled_circuits:
            return self._transpiled_circuits[circuit_id]
            
        if self.mode != "hardware":
            transpiled = circuit
        elif self._hardware_backend is not None:
            transpiled = transpile(
                circuit,
                backend=self._hardware_backend,
                optimization_level=self.config.optimization_level,
                seed_transpiler=self.config.seed,
            )
        else:
            transpiled = transpile(circuit, optimization_level=self.config.optimization_level)
            
        self._transpiled_circuits[circuit_id] = transpiled
        return transpiled

    def _run_estimator(
        self,
        circuit: QuantumCircuit,
        observables: Sequence[SparsePauliOp],
        parameter_values: np.ndarray | None,
    ) -> np.ndarray:
        circuit_to_run = self._prepare_circuit(circuit)
        if parameter_values is None:
            pub = (circuit_to_run, list(observables))
        else:
            pv = np.asarray(parameter_values, dtype=float)
            if pv.ndim == 2:
                # Reshape to (N, 1, P) so it broadcasts with observables shape (M,)
                # resulting in batch shape (N, M)
                pv = pv.reshape((pv.shape[0], 1, pv.shape[1]))
            pub = (circuit_to_run, list(observables), pv)

        if self.mode == "hardware":
            if self._runtime_estimator is None:
                raise RuntimeError("Runtime estimator is not initialized.")
            result = self._runtime_estimator.run([pub]).result()
            return np.asarray(result[0].data.evs, dtype=float)

        if self._aer_estimator is None:
            raise RuntimeError("Aer estimator is not initialized.")
        result = self._aer_estimator.run([pub]).result()
        return np.asarray(result[0].data.evs, dtype=float)

    def estimate_expectations(
        self,
        circuit: QuantumCircuit,
        observables: Sequence[SparsePauliOp],
        parameter_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Estimate expectation values according to the configured execution mode."""
        if self.mode in {"ideal", "noisy", "hardware"}:
            return self._run_estimator(
                circuit=circuit,
                observables=observables,
                parameter_values=parameter_values,
            )

        def base_eval() -> np.ndarray:
            return self._run_estimator(
                circuit=circuit,
                observables=observables,
                parameter_values=parameter_values,
            )

        def scaled_eval(scale: float) -> np.ndarray:
            cache_key = (id(circuit), scale)
            if cache_key not in self._folded_circuits:
                self._folded_circuits[cache_key] = fold_circuit_for_noise_scaling(circuit=circuit, scale_factor=scale)
            folded = self._folded_circuits[cache_key]
            return self._run_estimator(
                circuit=folded,
                observables=observables,
                parameter_values=parameter_values,
            )

        return self.mitigation_engine.mitigate(base_eval=base_eval, scaled_eval=scaled_eval)
