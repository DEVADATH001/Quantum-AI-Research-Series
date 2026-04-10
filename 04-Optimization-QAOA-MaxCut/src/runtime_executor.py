"""Author: DEVADATH H K

Runtime Executor Module

Provides a small execution layer for QAOA research workflows:
- ``local`` uses exact statevector evaluation for debugging and benchmarking
- ``noisy_simulator`` samples measured circuits on Aer
- ``ibm_hardware`` uses Runtime Estimator when credentials are available
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Container for quantum execution results."""

    expectation_value: float
    objective_value: float
    variance: float
    n_shots: int
    circuit_depth: int
    runtime: float
    backend_name: str
    sampled_bitstring: Optional[str] = None
    measurement_counts: Optional[Dict[str, int]] = None


class RuntimeExecutor:
    """Execution engine for local simulation, noisy simulation, and Runtime."""

    def __init__(
        self,
        mode: str = "local",
        backend_name: Optional[str] = None,
        shots: int = 1024,
        resilience_level: int = 1,
        optimization_level: int = 1,
        seed: Optional[int] = 42,
        simulate_noise: bool = True,
        noise_model_path: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.backend_name = backend_name
        self.shots = shots
        self.resilience_level = resilience_level
        self.optimization_level = optimization_level
        self.seed = seed
        self.simulate_noise = simulate_noise
        self.noise_model_path = noise_model_path
        self.rng = np.random.default_rng(seed)
        self.backend: Optional[Any] = None
        self.primitive: Optional[Any] = None
        self.target_backend: Optional[Any] = None
        self.backend_source: str = "none"
        self._initialize_backend()

        logger.info(
            "RuntimeExecutor initialized: mode=%s, backend=%s, shots=%s, simulate_noise=%s",
            self.mode,
            self.backend_name,
            self.shots,
            self.simulate_noise,
        )

    def _initialize_backend(self) -> None:
        """Initialize the backend needed for the configured mode."""
        if self.mode == "local":
            self.backend = "statevector"
            self.primitive = None
            return

        if self.mode == "noisy_simulator":
            from qiskit_aer import AerSimulator

            noise_model = self._load_noise_model(self.noise_model_path)
            self.target_backend, self.backend_source = self.resolve_backend(self.backend_name)

            if self.simulate_noise and self.target_backend is not None:
                try:
                    self.backend = AerSimulator.from_backend(self.target_backend)
                except Exception as exc:
                    logger.warning("Could not create Aer simulator from backend: %s", exc)
                    self.backend = None

            if self.backend is None:
                self.backend = AerSimulator(noise_model=noise_model)

            self.primitive = None
            return

        if self.mode == "ibm_hardware":
            if not self.backend_name:
                raise ValueError("backend_name is required for IBM hardware mode")

            from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

            service = QiskitRuntimeService()
            self.backend = service.backend(self.backend_name)
            self.target_backend = self.backend
            self.backend_source = "runtime_service"
            self.primitive = EstimatorV2(
                backend=self.backend,
                options={
                    "default_shots": self.shots,
                    "optimization_level": self.optimization_level,
                    "resilience_level": self.resilience_level,
                },
            )
            return

        raise ValueError(f"Unknown mode: {self.mode}")

    def execute_circuit(
        self,
        circuit,
        hamiltonian,
        parameter_values: Optional[np.ndarray] = None,
        offset: float = 0.0,
    ) -> ExecutionResult:
        """
        Execute a QAOA circuit and return both interaction and objective values.

        The returned bitstring is always normalized into node order, so
        ``bitstring[i]`` corresponds to qubit/node ``i``.

        Args:
            circuit: Circuit to execute.
            hamiltonian: Diagonal interaction Hamiltonian.
            parameter_values: Optional parameter values for binding.
            offset: Constant Max-Cut offset so that
                ``objective_value = offset + expectation_value``.
        """
        start_time = time.time()

        bound_circuit = circuit
        if parameter_values is not None and getattr(circuit, "parameters", None):
            bound_circuit = circuit.assign_parameters(parameter_values)

        if self.mode == "local":
            result = self._execute_local_exact(bound_circuit, hamiltonian, offset)
        elif self.mode == "noisy_simulator":
            result = self._execute_aer_sampled(bound_circuit, hamiltonian, offset)
        else:
            result = self._execute_estimator(bound_circuit, hamiltonian, offset)

        result.runtime = time.time() - start_time
        result.circuit_depth = bound_circuit.depth()
        return result

    def _execute_local_exact(self, circuit, hamiltonian, offset: float) -> ExecutionResult:
        """Evaluate an exact statevector expectation locally."""
        from qiskit.quantum_info import Statevector

        statevector_circuit = self._strip_final_measurements(circuit)
        statevector = Statevector.from_instruction(statevector_circuit)
        expectation_value = float(np.real(statevector.expectation_value(hamiltonian)))
        second_moment = float(
            np.real(statevector.expectation_value((hamiltonian @ hamiltonian).simplify()))
        )
        variance = max(0.0, second_moment - expectation_value**2)
        objective_value = float(expectation_value + offset)

        probabilities = statevector.probabilities()
        best_index = int(np.argmax(probabilities))
        raw_bitstring = format(best_index, f"0{circuit.num_qubits}b")
        sampled_bitstring = self._canonical_bitstring(raw_bitstring)
        counts = self._sample_counts_from_probabilities(probabilities, circuit.num_qubits)

        return ExecutionResult(
            expectation_value=expectation_value,
            objective_value=objective_value,
            variance=variance,
            n_shots=self.shots,
            circuit_depth=circuit.depth(),
            runtime=0.0,
            backend_name="local_statevector",
            sampled_bitstring=sampled_bitstring,
            measurement_counts=counts,
        )

    def _execute_aer_sampled(self, circuit, hamiltonian, offset: float) -> ExecutionResult:
        """Execute a sampled circuit on Aer, adding measurements if needed."""
        from qiskit import transpile

        measured_circuit = circuit if self._has_measurements(circuit) else circuit.copy()
        if not self._has_measurements(measured_circuit):
            measured_circuit.measure_all()

        transpiled_circuit = transpile(
            measured_circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )

        job = self.backend.run(
            transpiled_circuit,
            shots=self.shots,
            seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
        )
        result = job.result()
        raw_counts = result.get_counts()
        counts = {
            self._canonical_bitstring(bitstring): int(count)
            for bitstring, count in raw_counts.items()
        }
        expectation_value, variance = self._compute_moments_from_counts(counts, hamiltonian)
        objective_value = float(expectation_value + offset)
        sampled_bitstring = max(counts, key=counts.get) if counts else None

        return ExecutionResult(
            expectation_value=expectation_value,
            objective_value=objective_value,
            variance=variance,
            n_shots=self.shots,
            circuit_depth=transpiled_circuit.depth(),
            runtime=0.0,
            backend_name=self._backend_label(self.backend),
            sampled_bitstring=sampled_bitstring,
            measurement_counts=counts,
        )

    def _execute_estimator(self, circuit, hamiltonian, offset: float) -> ExecutionResult:
        """Execute using Qiskit Runtime EstimatorV2."""
        if self.primitive is None:
            raise RuntimeError("Estimator primitive is not initialized.")

        job = self.primitive.run([(circuit, [hamiltonian])])
        result = job.result()[0]
        expectation_value = float(result.data.evs[0])
        objective_value = float(expectation_value + offset)

        variance = 0.0
        if hasattr(result.data, "stds") and result.data.stds is not None:
            variance = float(result.data.stds[0] ** 2)

        return ExecutionResult(
            expectation_value=expectation_value,
            objective_value=objective_value,
            variance=variance,
            n_shots=self.shots,
            circuit_depth=circuit.depth(),
            runtime=0.0,
            backend_name=self._backend_label(self.backend),
            sampled_bitstring=None,
            measurement_counts=None,
        )

    def _sample_counts_from_probabilities(
        self, probabilities: np.ndarray, n_qubits: int
    ) -> Dict[str, int]:
        """Sample counts from an exact probability vector for local reporting."""
        if self.shots <= 0:
            return {}

        draws = self.rng.multinomial(self.shots, probabilities)
        counts: Dict[str, int] = {}
        for index, count in enumerate(draws):
            if count <= 0:
                continue
            raw_bitstring = format(index, f"0{n_qubits}b")
            counts[self._canonical_bitstring(raw_bitstring)] = int(count)
        return counts

    def _compute_moments_from_counts(
        self,
        counts: Dict[str, int],
        hamiltonian,
    ) -> tuple[float, float]:
        """Compute the mean and variance of a diagonal Hamiltonian from counts."""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0, 0.0

        mean = 0.0
        mean_square = 0.0
        for bitstring, count in counts.items():
            value = self._evaluate_hamiltonian(hamiltonian, bitstring)
            weight = count / total_shots
            mean += weight * value
            mean_square += weight * value**2

        variance = max(0.0, mean_square - mean**2)
        return mean, variance

    @staticmethod
    def _evaluate_hamiltonian(hamiltonian, bitstring: str) -> float:
        """Evaluate a diagonal Hamiltonian on a canonical bitstring."""
        z_values = np.array([1.0 if bit == "0" else -1.0 for bit in bitstring])
        value = 0.0
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            term_value = 1.0
            for index, pauli_char in enumerate(reversed(pauli.to_label())):
                if pauli_char == "Z":
                    term_value *= z_values[index]
            value += float(np.real(coeff)) * term_value
        return value

    @staticmethod
    def _canonical_bitstring(raw_bitstring: str) -> str:
        """Convert Qiskit's big-endian bitstring labels into node order."""
        return raw_bitstring[::-1]

    @staticmethod
    def _has_measurements(circuit) -> bool:
        """Return True if the circuit already contains measurement operations."""
        return any(instruction.operation.name == "measure" for instruction in circuit.data)

    @staticmethod
    def _strip_final_measurements(circuit):
        """Remove final measurements before statevector simulation."""
        if hasattr(circuit, "remove_final_measurements"):
            return circuit.remove_final_measurements(inplace=False)
        return circuit

    @staticmethod
    def _backend_label(backend: Any) -> str:
        """Get a human-readable backend name."""
        if backend is None:
            return "unknown"
        name = getattr(backend, "name", None)
        if callable(name):
            return str(name())
        if name is not None:
            return str(name)
        return str(backend)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the currently configured execution backend."""
        info: Dict[str, Any] = {
            "mode": self.mode,
            "backend_name": self.backend_name,
            "shots": self.shots,
            "resilience_level": self.resilience_level,
            "optimization_level": self.optimization_level,
            "simulate_noise": self.simulate_noise,
            "noise_model_path": self.noise_model_path,
            "backend_source": self.backend_source,
        }

        if self.backend not in {None, "statevector"}:
            info.update(self._collect_backend_metadata(self.target_backend or self.backend))

        return info

    @classmethod
    def resolve_backend(cls, backend_name: Optional[str]) -> tuple[Optional[Any], str]:
        """Resolve a real IBM backend when possible, else fall back to a fake backend."""
        if not backend_name:
            return None, "unspecified"

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService()
            return service.backend(backend_name), "runtime_service"
        except Exception as exc:
            logger.info("Runtime backend resolution failed for %s: %s", backend_name, exc)

        fake_backend = cls._resolve_fake_backend(backend_name)
        if fake_backend is not None:
            return fake_backend, "fake_provider"

        return None, "unresolved"

    @staticmethod
    def _resolve_fake_backend(backend_name: str) -> Optional[Any]:
        """Resolve a local fake backend that matches the requested IBM backend name."""
        try:
            from qiskit_ibm_runtime import fake_provider
        except Exception as exc:
            logger.info("Fake backend provider unavailable: %s", exc)
            return None

        sanitized = backend_name.lower().replace("ibm_", "").replace("-", "_")
        class_name = "Fake" + "".join(part.capitalize() for part in sanitized.split("_"))
        backend_cls = getattr(fake_provider, class_name, None)
        if backend_cls is None:
            return None
        return backend_cls()

    @staticmethod
    def _load_noise_model(noise_model_path: Optional[str]):
        """Load a serialized Aer noise model from JSON if provided."""
        if not noise_model_path:
            return None

        from qiskit_aer.noise import NoiseModel

        path = Path(noise_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Noise model file not found: {noise_model_path}")

        with path.open("r", encoding="utf-8") as handle:
            return NoiseModel.from_dict(json.load(handle))

    @staticmethod
    def _collect_backend_metadata(backend: Any) -> Dict[str, Any]:
        """Extract useful metadata from a backend-like object."""
        info: Dict[str, Any] = {}

        for attr in ("num_qubits", "basis_gates"):
            value = getattr(backend, attr, None)
            if callable(value):
                value = value()
            if value is not None:
                info[attr] = value

        coupling_map = getattr(backend, "coupling_map", None)
        if coupling_map is not None:
            if hasattr(coupling_map, "get_edges"):
                info["coupling_map"] = list(coupling_map.get_edges())
            else:
                info["coupling_map"] = coupling_map

        backend_name = getattr(backend, "name", None)
        if callable(backend_name):
            backend_name = backend_name()
        if backend_name is not None:
            info["resolved_backend_name"] = str(backend_name)

        return info


class BatchExecutor:
    """Helper for parameter sweeps and small evaluation batches."""

    def __init__(self, executor: RuntimeExecutor) -> None:
        self.executor = executor
        logger.info("BatchExecutor initialized")

    def execute_parameter_sweep(
        self,
        circuit_factory,
        hamiltonian,
        parameter_grid: List[np.ndarray],
        offset: float = 0.0,
    ) -> List[float]:
        """Execute a parameter sweep and return expectation values."""
        results: List[float] = []
        for params in parameter_grid:
            circuit = circuit_factory(params)
            parameter_values = params if getattr(circuit, "parameters", None) else None
            result = self.executor.execute_circuit(
                circuit,
                hamiltonian,
                parameter_values=parameter_values,
                offset=offset,
            )
            results.append(result.expectation_value)
        return results

    def execute_batch(
        self,
        circuits: List[Any],
        hamiltonians: List[Any],
        offsets: Optional[List[float]] = None,
    ) -> List[ExecutionResult]:
        """Execute multiple circuits sequentially."""
        results: List[ExecutionResult] = []
        if offsets is None:
            offsets = [0.0] * len(circuits)

        for circuit, hamiltonian, offset in zip(circuits, hamiltonians, offsets):
            results.append(self.executor.execute_circuit(circuit, hamiltonian, offset=offset))
        return results


def create_executor(
    mode: str = "local",
    backend_name: Optional[str] = None,
    **kwargs,
) -> RuntimeExecutor:
    """Factory function for RuntimeExecutor."""
    return RuntimeExecutor(mode=mode, backend_name=backend_name, **kwargs)
