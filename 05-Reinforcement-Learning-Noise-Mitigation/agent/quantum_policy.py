"""Author: DEVADATH H K

Project: Quantum RL Noise Mitigation

Parameterized quantum policy network for REINFORCE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from utils.qiskit_helpers import build_state_angles, make_z_observables, softmax

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

    from src.runtime_executor import QuantumRuntimeExecutor

@dataclass(slots=True)
class PolicyConfig:
    """Configuration for the quantum policy network."""

    num_qubits: int = 2
    reps: int = 2
    entanglement: str = "full"
    ansatz: str = "RealAmplitudes"
    temperature: float = 1.0
    seed: int = 42

class QuantumPolicyNetwork:
    """Quantum policy with angle embedding and variational ansatz using Data Re-uploading."""

    def __init__(self, n_actions: int, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self.n_actions = n_actions
        if self.config.num_qubits < self.n_actions:
            raise ValueError(
                f"num_qubits ({self.config.num_qubits}) must be >= n_actions ({self.n_actions})."
            )
        self.rng = np.random.default_rng(self.config.seed)
        
        # Build Data Re-uploading Circuit
        self.state_params = ParameterVector("state", self.config.num_qubits * self.config.reps)
        self.ansatz_params = ParameterVector("theta", self._get_ansatz_param_count())
        
        self._parameterized_circuit = QuantumCircuit(self.config.num_qubits)
        
        param_idx = 0
        ansatz_param_idx = 0
        
        # Interleave state embedding with variational layers (Data Re-uploading)
        for r in range(self.config.reps):
            # State embedding layer
            for q in range(self.config.num_qubits):
                self._parameterized_circuit.rx(self.state_params[r * self.config.num_qubits + q], q)
                self._parameterized_circuit.ry(self.state_params[r * self.config.num_qubits + q], q)
            
            # Variational layer (simplified ansatz layer)
            # Use RY and CX for a standard heuristic ansatz
            for q in range(self.config.num_qubits):
                self._parameterized_circuit.ry(self.ansatz_params[ansatz_param_idx], q)
                ansatz_param_idx += 1
            
            if self.config.num_qubits > 1:
                for q in range(self.config.num_qubits):
                    self._parameterized_circuit.cx(q, (q + 1) % self.config.num_qubits)

        self.parameter_count = ansatz_param_idx
        self.observables: list[SparsePauliOp] = make_z_observables(
            num_qubits=self.config.num_qubits,
            num_actions=self.n_actions,
        )

        # Store parameter order for binding
        self._all_parameters = list(self._parameterized_circuit.parameters)
        self._state_indices = [self._all_parameters.index(p) for p in self.state_params]
        self._ansatz_indices = [self._all_parameters.index(p) for p in self.ansatz_params]

    def _get_ansatz_param_count(self) -> int:
        # Each rep has num_qubits RY gates
        return self.config.num_qubits * self.config.reps

    def initial_parameters(self, scale: float = 0.1) -> np.ndarray:
        """Return deterministic random initialization for policy parameters."""
        return self.rng.normal(loc=0.0, scale=scale, size=self.parameter_count)

    def get_combined_parameters(self, state: int, parameters: np.ndarray) -> np.ndarray:
        """Get flattened parameter array for parameterized circuit."""
        angles = build_state_angles(state=state, num_qubits=self.config.num_qubits)
        # Repeat angles for each re-uploading layer
        repeated_angles = np.tile(angles, self.config.reps)
        
        combined = np.zeros(len(self._all_parameters), dtype=float)
        combined[self._state_indices] = repeated_angles
        combined[self._ansatz_indices] = parameters
        return combined

    def get_batched_parameters(self, states: Sequence[int], parameters_batch: np.ndarray) -> np.ndarray:
        """Get 2D array of parameters for a batch of states and parameters."""
        if parameters_batch.ndim == 1:
            parameters_batch = np.tile(parameters_batch, (len(states), 1))
        
        batch_size = len(states)
        combined = np.zeros((batch_size, len(self._all_parameters)), dtype=float)
        for i, state in enumerate(states):
            angles = build_state_angles(state=state, num_qubits=self.config.num_qubits)
            repeated_angles = np.tile(angles, self.config.reps)
            combined[i, self._state_indices] = repeated_angles
        combined[:, self._ansatz_indices] = parameters_batch
        return combined

    def expectation_logits(
        self,
        state: int,
        parameters: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> np.ndarray:
        """Evaluate Pauli-Z expectation values used as action logits."""
        combined_params = self.get_combined_parameters(state, parameters)
        expectations = executor.estimate_expectations(
            circuit=self._parameterized_circuit,
            observables=self.observables,
            parameter_values=combined_params,
        )
        return np.asarray(expectations, dtype=float)

    def batched_expectation_logits(
        self,
        states: Sequence[int],
        parameters_batch: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> np.ndarray:
        """Batch evaluate Pauli-Z expectation values."""
        combined_params = self.get_batched_parameters(states, parameters_batch)
        expectations = executor.estimate_expectations(
            circuit=self._parameterized_circuit,
            observables=self.observables,
            parameter_values=combined_params,
        )
        return np.asarray(expectations, dtype=float)

    def action_probabilities(
        self,
        state: int,
        parameters: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> np.ndarray:
        """Compute policy distribution by applying softmax over expectation logits."""
        logits = self.expectation_logits(state=state, parameters=parameters, executor=executor)
        return softmax(logits, temperature=self.config.temperature)

    def log_prob(
        self,
        state: int,
        action: int,
        parameters: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> float:
        """Log probability of a selected action in a given state."""
        probs = self.action_probabilities(
            state=state,
            parameters=parameters,
            executor=executor,
        )
        return float(np.log(np.clip(probs[action], 1e-9, 1.0)))

    def sample_action(self, probs: np.ndarray) -> int:
        """Sample an action from probability distribution."""
        return int(self.rng.choice(self.n_actions, p=probs))
