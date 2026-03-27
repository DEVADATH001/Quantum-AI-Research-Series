"""Parameterized quantum policy network with a measurement-defined action register."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Sequence, TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import EfficientSU2, RealAmplitudes

from utils.qiskit_helpers import build_state_angles

if TYPE_CHECKING:
    from src.runtime_executor import QuantumRuntimeExecutor


@dataclass(slots=True)
class PolicyConfig:
    """Configuration for the quantum policy network."""

    num_qubits: int = 3
    reuploads: int = 1
    ansatz: str = "RealAmplitudes"
    ansatz_reps: int = 1
    entanglement: str = "linear"
    state_encoding: str = "hybrid"
    seed: int = 42


class QuantumPolicyNetwork:
    """Measurement-native variational quantum policy for discrete-action control."""

    def __init__(
        self,
        n_actions: int,
        n_observations: int,
        config: PolicyConfig | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.n_actions = int(n_actions)
        self.n_observations = int(n_observations)

        if self.n_actions < 2:
            raise ValueError("The measurement-defined policy requires at least two actions.")
        if self.n_actions & (self.n_actions - 1):
            raise ValueError(
                "This measurement-native policy currently requires the number of actions to be a power of two."
            )

        self.action_register_size = int(round(log2(self.n_actions)))
        if self.config.num_qubits < self.action_register_size:
            raise ValueError(
                f"num_qubits ({self.config.num_qubits}) must be >= log2(n_actions) "
                f"({self.action_register_size})."
            )

        min_qubits = max(1, ceil(log2(max(2, self.n_observations))))
        if self.config.num_qubits < min_qubits:
            raise ValueError(
                f"num_qubits ({self.config.num_qubits}) must be >= ceil(log2(n_observations)) "
                f"({min_qubits}) to encode {self.n_observations} states."
            )
        if self.config.reuploads < 1:
            raise ValueError("reuploads must be >= 1.")
        if self.config.ansatz_reps < 1:
            raise ValueError("ansatz_reps must be >= 1.")

        self.rng = np.random.default_rng(self.config.seed)
        self.action_qubits = tuple(range(self.action_register_size))
        self.latent_qubits = tuple(range(self.action_register_size, self.config.num_qubits))

        self.state_params = ParameterVector(
            "state",
            self.config.num_qubits * self.config.reuploads,
        )
        self._parameterized_circuit = QuantumCircuit(self.config.num_qubits)

        self.ansatz_params: list[Parameter] = []
        block_parameter_count = 0
        for block_idx in range(self.config.reuploads):
            state_offset = block_idx * self.config.num_qubits
            for qubit_idx in range(self.config.num_qubits):
                state_param = self.state_params[state_offset + qubit_idx]
                # Hybrid angle embedding: a Y rotation carries the main feature,
                # while a tied Z phase provides a second, state-dependent channel.
                self._parameterized_circuit.ry(state_param, qubit_idx)
                self._parameterized_circuit.rz(0.5 * state_param, qubit_idx)

            ansatz_template = self._build_ansatz_template()
            template_params = list(ansatz_template.parameters)
            block_parameter_count = len(template_params)
            block_params = ParameterVector(f"theta_block_{block_idx}", block_parameter_count)
            self.ansatz_params.extend(block_params)
            reassigned = ansatz_template.assign_parameters(
                {template_param: block_params[i] for i, template_param in enumerate(template_params)},
                inplace=False,
            )
            self._parameterized_circuit.compose(reassigned, inplace=True)

        self.parameter_count = len(self.ansatz_params)
        self.parameter_count_per_block = block_parameter_count

        self._all_parameters = list(self._parameterized_circuit.parameters)
        self._state_indices = [self._all_parameters.index(param) for param in self.state_params]
        self._ansatz_indices = [self._all_parameters.index(param) for param in self.ansatz_params]

    def _build_ansatz_template(self) -> QuantumCircuit:
        ansatz_name = self.config.ansatz.lower()
        if ansatz_name == "realamplitudes":
            return RealAmplitudes(
                num_qubits=self.config.num_qubits,
                reps=self.config.ansatz_reps,
                entanglement=self.config.entanglement,
                flatten=True,
            )
        if ansatz_name == "efficientsu2":
            return EfficientSU2(
                num_qubits=self.config.num_qubits,
                reps=self.config.ansatz_reps,
                entanglement=self.config.entanglement,
                flatten=True,
            )
        raise ValueError(f"Unsupported ansatz: {self.config.ansatz}")

    def initial_parameters(self, scale: float = 0.1) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=scale, size=self.parameter_count)

    def _state_embedding_angles(self, state: int) -> np.ndarray:
        return build_state_angles(
            state=state,
            num_qubits=self.config.num_qubits,
            n_states=self.n_observations,
            encoding=self.config.state_encoding,
        )

    def get_combined_parameters(self, state: int, parameters: np.ndarray) -> np.ndarray:
        angles = self._state_embedding_angles(state=state)
        combined = np.zeros(len(self._all_parameters), dtype=float)
        combined[self._state_indices] = np.tile(angles, self.config.reuploads)
        combined[self._ansatz_indices] = parameters
        return combined

    def get_batched_parameters(
        self,
        states: Sequence[int],
        parameters_batch: np.ndarray,
    ) -> np.ndarray:
        if parameters_batch.ndim == 1:
            parameters_batch = np.tile(parameters_batch, (len(states), 1))

        combined = np.zeros((len(states), len(self._all_parameters)), dtype=float)
        for row_idx, state in enumerate(states):
            angles = self._state_embedding_angles(state=state)
            combined[row_idx, self._state_indices] = np.tile(angles, self.config.reuploads)
        combined[:, self._ansatz_indices] = parameters_batch
        return combined

    def batched_action_probabilities(
        self,
        states: Sequence[int],
        parameters_batch: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> np.ndarray:
        combined_params = self.get_batched_parameters(states, parameters_batch)
        probabilities = executor.estimate_action_probabilities(
            circuit=self._parameterized_circuit,
            action_qubits=self.action_qubits,
            parameter_values=combined_params,
        )
        return np.asarray(probabilities, dtype=float)

    def action_probabilities(
        self,
        state: int,
        parameters: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> np.ndarray:
        combined_params = self.get_combined_parameters(state, parameters)
        probabilities = executor.estimate_action_probabilities(
            circuit=self._parameterized_circuit,
            action_qubits=self.action_qubits,
            parameter_values=combined_params,
        )
        return np.asarray(probabilities, dtype=float)

    def log_prob(
        self,
        state: int,
        action: int,
        parameters: np.ndarray,
        executor: QuantumRuntimeExecutor,
    ) -> float:
        probs = self.action_probabilities(
            state=state,
            parameters=parameters,
            executor=executor,
        )
        return float(np.log(np.clip(probs[action], 1e-9, 1.0)))

    def sample_action(self, probs: np.ndarray) -> int:
        return int(self.rng.choice(self.n_actions, p=probs))
