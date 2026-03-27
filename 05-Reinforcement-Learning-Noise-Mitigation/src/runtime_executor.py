"""Execution engine for measurement-based quantum policy evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from src.mitigation_engine import MitigationConfig, MitigationEngine, fold_circuit_for_noise_scaling
from src.noise_models import (
    infer_readout_error_rates_by_qubit,
    load_ibm_noise_model,
    resolve_fake_backend,
)
from utils.qiskit_helpers import project_probabilities

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime execution configuration."""

    mode: str = "ideal"  # ideal | noisy | mitigated | hardware
    shots: int = 512
    backend_name: str = "ibm_osaka"
    optimization_level: int = 1
    resilience_level: int = 0
    seed: int = 42
    compact_noise_model: bool = False


class QuantumRuntimeExecutor:
    """Unified sampler-based executor for the QRL training loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        mitigation_engine: MitigationEngine | None = None,
    ) -> None:
        self.config = config
        self.mode = config.mode.lower()
        self.noise_model = None
        self._hardware_backend = None
        self._transpile_backend = None
        self._runtime_sampler = None
        self._runtime_session = None
        self._sampler = None
        self._transpiled_circuits: dict[tuple[int, str], QuantumCircuit] = {}
        self._measurement_circuits: dict[tuple[int, tuple[int, ...], str], tuple[QuantumCircuit, tuple[int, ...]]] = {}
        self._folded_circuits: dict[tuple[int, float], tuple[QuantumCircuit, float]] = {}
        self._readout_error_rates_by_qubit: dict[int, tuple[float, float]] = {}

        if self.mode not in {"ideal", "noisy", "mitigated", "hardware"}:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.mode == "ideal":
            self._sampler = AerSamplerV2(
                default_shots=config.shots,
                seed=config.seed,
            )
            self.mitigation_engine = mitigation_engine

        elif self.mode in {"noisy", "mitigated"}:
            self._transpile_backend = resolve_fake_backend(config.backend_name)
            self.noise_model = load_ibm_noise_model(
                config.backend_name,
                compact=config.compact_noise_model,
            )
            self._sampler = AerSamplerV2(
                default_shots=config.shots,
                seed=config.seed,
                options={
                    "backend_options": {
                        "noise_model": self.noise_model,
                    }
                },
            )
            self._readout_error_rates_by_qubit = infer_readout_error_rates_by_qubit(self.noise_model)
            if mitigation_engine is None:
                mitigation_engine = MitigationEngine(
                    MitigationConfig(
                        resilience_level=max(2, config.resilience_level),
                    )
                )
            self.mitigation_engine = mitigation_engine

        else:
            self._configure_hardware_runtime()
            self.mitigation_engine = mitigation_engine

    def close(self) -> None:
        if self._runtime_session is not None:
            self._runtime_session.close()
            logger.info("Closed IBM Runtime Session.")

    def _configure_hardware_runtime(self) -> None:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerOptions, SamplerV2, Session
        
        service = QiskitRuntimeService()

        options = SamplerOptions()
        options.default_shots = self.config.shots
        options.simulator.seed_simulator = self.config.seed
        if self.config.resilience_level >= 2:
            options.twirling.enable_measure = True

        self._runtime_session = Session(service=service, backend=self.config.backend_name)
        self._runtime_sampler = SamplerV2(mode=self._runtime_session, options=options)
        self._hardware_backend = service.get_backend(self.config.backend_name)
        logger.info("Opened IBM Runtime Session on backend %s", self.config.backend_name)


    def _prepare_circuit(self, circuit: QuantumCircuit, cache_suffix: str = "base") -> QuantumCircuit:
        cache_key = (id(circuit), cache_suffix)
        if cache_key in self._transpiled_circuits:
            return self._transpiled_circuits[cache_key]

        optimization_level = 0 if cache_suffix.startswith("fold-") else self.config.optimization_level

        if self.mode == "hardware" and self._hardware_backend is not None:
            transpiled = transpile(
                circuit,
                backend=self._hardware_backend,
                optimization_level=optimization_level,
                seed_transpiler=self.config.seed,
            )
        elif self.mode in {"noisy", "mitigated"} and self._transpile_backend is not None:
            transpiled = transpile(
                circuit,
                backend=self._transpile_backend,
                optimization_level=optimization_level,
                seed_transpiler=self.config.seed,
            )
        else:
            transpiled = transpile(
                circuit,
                optimization_level=optimization_level,
                seed_transpiler=self.config.seed,
            )

        self._transpiled_circuits[cache_key] = transpiled
        return transpiled

    def _logical_to_physical_qubits(
        self,
        circuit_to_run: QuantumCircuit,
        logical_qubits: Sequence[int],
    ) -> tuple[int, ...]:
        layout = getattr(circuit_to_run, "layout", None)
        if layout is None:
            return tuple(int(qubit) for qubit in logical_qubits)

        try:
            mapping = layout.final_index_layout(filter_ancillas=False)
            return tuple(int(mapping[int(qubit)]) for qubit in logical_qubits)
        except Exception:
            return tuple(int(qubit) for qubit in logical_qubits)

    def _prepare_measurement_circuit(
        self,
        circuit: QuantumCircuit,
        action_qubits: Sequence[int],
        cache_suffix: str = "base",
    ) -> tuple[QuantumCircuit, tuple[int, ...]]:
        cache_key = (id(circuit), tuple(int(qubit) for qubit in action_qubits), cache_suffix)
        if cache_key in self._measurement_circuits:
            return self._measurement_circuits[cache_key]

        circuit_to_run = self._prepare_circuit(circuit, cache_suffix=cache_suffix)
        physical_action_qubits = self._logical_to_physical_qubits(circuit_to_run, action_qubits)

        measured = circuit_to_run.copy()
        classical_register = ClassicalRegister(len(action_qubits), "action")
        measured.add_register(classical_register)
        for cbit_idx, qubit_idx in enumerate(physical_action_qubits):
            measured.measure(qubit_idx, classical_register[cbit_idx])

        self._measurement_circuits[cache_key] = (measured, physical_action_qubits)
        return measured, physical_action_qubits

    def _extract_probability_array(
        self,
        sampler_result,
        n_action_qubits: int,
    ) -> np.ndarray:
        n_outcomes = 2**n_action_qubits
        data_bin = sampler_result[0].data
        bit_array = next(iter(data_bin.values()))

        if bit_array.shape == ():
            probabilities = np.zeros(n_outcomes, dtype=float)
            counts = bit_array.get_int_counts()
            total = max(1, int(sum(counts.values())))
            for outcome, count in counts.items():
                probabilities[int(outcome)] = float(count) / float(total)
            return probabilities

        probabilities = np.zeros((bit_array.size, n_outcomes), dtype=float)
        for batch_idx in range(bit_array.size):
            counts = bit_array.get_int_counts(batch_idx)
            total = max(1, int(sum(counts.values())))
            for outcome, count in counts.items():
                probabilities[batch_idx, int(outcome)] = float(count) / float(total)
        return probabilities

    def _run_sampler(
        self,
        circuit: QuantumCircuit,
        action_qubits: Sequence[int],
        parameter_values: np.ndarray | None,
        cache_suffix: str = "base",
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        measured_circuit, physical_action_qubits = self._prepare_measurement_circuit(
            circuit,
            action_qubits=action_qubits,
            cache_suffix=cache_suffix,
        )

        pub = (measured_circuit,) if parameter_values is None else (measured_circuit, np.asarray(parameter_values, dtype=float))
        if self.mode == "hardware":
            if self._runtime_sampler is None:
                raise RuntimeError("Runtime sampler is not initialized.")
            result = self._runtime_sampler.run([pub]).result()
        else:
            if self._sampler is None:
                raise RuntimeError("Aer sampler is not initialized.")
            result = self._sampler.run([pub]).result()
        
        probabilities = self._extract_probability_array(result, n_action_qubits=len(action_qubits))
        return probabilities, physical_action_qubits

    def _build_readout_confusion_matrix(self, physical_action_qubits: Sequence[int]) -> np.ndarray:
        confusion = np.array([[1.0]], dtype=float)
        default_rates = self._readout_error_rates_by_qubit.get(-1, (0.0, 0.0))

        for qubit_idx in reversed(tuple(int(qubit) for qubit in physical_action_qubits)):
            p01, p10 = self._readout_error_rates_by_qubit.get(qubit_idx, default_rates)
            single = np.array(
                [
                    [1.0 - p01, p10],
                    [p01, 1.0 - p10],
                ],
                dtype=float,
            )
            confusion = np.kron(confusion, single)
        return confusion

    def _apply_readout_correction(
        self,
        probabilities: np.ndarray,
        physical_action_qubits: Sequence[int],
    ) -> np.ndarray:
        if self.mitigation_engine is not None and not self.mitigation_engine.config.enable_readout_correction:
            return project_probabilities(probabilities)
        if not self._readout_error_rates_by_qubit:
            return project_probabilities(probabilities)

        values = np.asarray(probabilities, dtype=float)
        inverse_confusion = np.linalg.pinv(self._build_readout_confusion_matrix(physical_action_qubits))
        if values.ndim == 1:
            corrected = inverse_confusion @ values
        else:
            corrected = values @ inverse_confusion.T
        return project_probabilities(corrected)

    def estimate_action_probabilities(
        self,
        circuit: QuantumCircuit,
        action_qubits: Sequence[int],
        parameter_values: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.mode in {"ideal", "noisy", "hardware"}:
            probabilities, _ = self._run_sampler(
                circuit=circuit,
                action_qubits=action_qubits,
                parameter_values=parameter_values,
            )
            return project_probabilities(probabilities)

        def base_eval() -> np.ndarray:
            probabilities, physical_action_qubits = self._run_sampler(
                circuit=circuit,
                action_qubits=action_qubits,
                parameter_values=parameter_values,
            )
            return self._apply_readout_correction(probabilities, physical_action_qubits)

        def scaled_eval(scale: float) -> tuple[float, np.ndarray]:
            cache_key = (id(circuit), float(scale))
            if cache_key not in self._folded_circuits:
                self._folded_circuits[cache_key] = fold_circuit_for_noise_scaling(
                    circuit=circuit,
                    scale_factor=float(scale),
                )
            folded, achieved_scale = self._folded_circuits[cache_key]
            probabilities, physical_action_qubits = self._run_sampler(
                circuit=folded,
                action_qubits=action_qubits,
                parameter_values=parameter_values,
                cache_suffix=f"fold-{scale}",
            )
            corrected = self._apply_readout_correction(probabilities, physical_action_qubits)
            return achieved_scale, corrected

        mitigated = self.mitigation_engine.mitigate(base_eval=base_eval, scaled_eval=scaled_eval)
        return project_probabilities(mitigated)
