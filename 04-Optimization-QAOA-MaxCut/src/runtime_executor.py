"""Author: DEVADATH H K

Runtime Executor Module

Provides a small execution layer for QAOA research workflows:
- ``local`` uses exact statevector evaluation for debugging and benchmarking
- ``noisy_simulator`` samples measured circuits on Aer
- ``ibm_hardware`` uses Runtime Estimator when credentials are available
"""

import logging
import time
from dataclasses import dataclass
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
    ) -> None:
        self.mode = mode
        self.backend_name = backend_name
        self.shots = shots
        self.resilience_level = resilience_level
        self.optimization_level = optimization_level
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.backend: Optional[Any] = None
        self.primitive: Optional[Any] = None
        self._initialize_backend()

        logger.info(
            "RuntimeExecutor initialized: mode=%s, backend=%s, shots=%s",
            self.mode,
            self.backend_name,
            self.shots,
        )

    def _initialize_backend(self) -> None:
        """Initialize the backend needed for the configured mode."""
        if self.mode == "local":
            self.backend = "statevector"
            self.primitive = None
            return

        if self.mode == "noisy_simulator":
            from qiskit_aer import AerSimulator

            noise_model = None
            if self.backend_name:
                try:
                    from qiskit_aer.noise import NoiseModel
                    from qiskit_ibm_runtime import QiskitRuntimeService

                    service = QiskitRuntimeService()
                    real_backend = service.backend(self.backend_name)
                    noise_model = NoiseModel.from_backend(real_backend)
                except Exception as exc:  # pragma: no cover - network credentials
                    logger.warning("Could not load backend noise model: %s", exc)

            self.backend = AerSimulator(noise_model=noise_model)
            self.primitive = None
            return

        if self.mode == "ibm_hardware":
            if not self.backend_name:
                raise ValueError("backend_name is required for IBM hardware mode")

            from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

            service = QiskitRuntimeService()
            self.backend = service.backend(self.backend_name)
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
        measured_circuit = circuit if self._has_measurements(circuit) else circuit.copy()
        if not self._has_measurements(measured_circuit):
            measured_circuit.measure_all()

        job = self.backend.run(
            measured_circuit,
            shots=self.shots,
            seed_simulator=self.seed,
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
            circuit_depth=measured_circuit.depth(),
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
        }

        if self.backend not in {None, "statevector"}:
            for attr in ("num_qubits", "coupling_map"):
                if hasattr(self.backend, attr):
                    info[attr] = getattr(self.backend, attr)

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
