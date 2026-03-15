"""Error-mitigation engine implementing TREX-style correction and ZNE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MitigationConfig:
    """Configuration for mitigation behavior."""

    resilience_level: int = 2
    enable_trex: bool = True
    enable_zne: bool = True
    scale_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    extrapolation: str = "polynomial"
    polynomial_degree: int = 2
    readout_error_probability: float = 0.0


def fold_circuit_for_noise_scaling(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    """
    Deterministic local folding for approximate noise scaling.

    Each selected gate U is replaced with U * U^-1 * U, adding two extra gates.
    """
    if scale_factor <= 1.0:
        return circuit.copy()

    op_data = [
        (inst, qargs, cargs)
        for inst, qargs, cargs in circuit.data
        if inst.name not in {"measure", "barrier"}
    ]
    if not op_data:
        return circuit.copy()

    target_extra_ops = int(round((scale_factor - 1.0) * len(op_data)))
    folds = max(0, target_extra_ops // 2)

    folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    folded.global_phase = circuit.global_phase

    for idx, (inst, qargs, cargs) in enumerate(circuit.data):
        q_idx = [circuit.find_bit(qubit).index for qubit in qargs]
        c_idx = [circuit.find_bit(clbit).index for clbit in cargs]
        folded.append(inst, q_idx, c_idx)

        if inst.name in {"measure", "barrier"}:
            continue
        if idx < folds:
            folded.append(inst.inverse(), q_idx, c_idx)
            folded.append(inst, q_idx, c_idx)

    return folded


class MitigationEngine:
    """Apply TREX and ZNE mitigation to expectation-value estimates."""

    def __init__(self, config: MitigationConfig | None = None) -> None:
        self.config = config or MitigationConfig()

    def trex_correct(self, expectations: np.ndarray) -> np.ndarray:
        """Apply symmetric readout correction to expectation values."""
        if not self.config.enable_trex or self.config.resilience_level < 1:
            return expectations
        p = np.clip(self.config.readout_error_probability, 0.0, 0.49)
        denominator = max(1e-6, 1.0 - 2.0 * p)
        corrected = expectations / denominator
        return np.clip(corrected, -1.0, 1.0)

    def _extrapolate_zero_noise(
        self,
        scale_factors: np.ndarray,
        values: np.ndarray,
    ) -> float:
        method = self.config.extrapolation.lower()

        if method == "richardson":
            # Richardson extrapolation for zero noise
            n = len(scale_factors)
            R = np.zeros((n, n))
            R[:, 0] = values
            for j in range(1, n):
                for i in range(n - j):
                    # Using standard formula for arbitrary scale factors
                    R[i, j] = (scale_factors[i+j] * R[i, j-1] - scale_factors[i] * R[i+1, j-1]) / (scale_factors[i+j] - scale_factors[i])
            return float(R[0, n-1])

        if method == "polynomial" and self.config.polynomial_degree > 1:
            degree = min(self.config.polynomial_degree, scale_factors.size - 1)
            coeffs = np.polyfit(scale_factors, values, deg=degree)
            return float(np.poly1d(coeffs)(0.0))

        if method == "exponential" and np.all(values > 0.0):
            coeffs = np.polyfit(scale_factors, np.log(values), deg=1)
            return float(np.exp(np.poly1d(coeffs)(0.0)))

        # Default to linear extrapolation as it is more stable
        coeffs = np.polyfit(scale_factors, values, deg=1)
        return float(np.poly1d(coeffs)(0.0))

    def mitigate_with_zne(
        self,
        evaluate_fn: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Run ZNE by sampling multiple noise scales and extrapolating to zero noise."""
        scales = np.asarray(self.config.scale_factors, dtype=float)
        observations: list[np.ndarray] = []
        for scale in scales:
            noisy_eval = evaluate_fn(float(scale))
            observations.append(self.trex_correct(noisy_eval))
            
        obs_array = np.array(observations)
        original_shape = obs_array.shape[1:]
        obs_flat = obs_array.reshape(len(scales), -1)

        mitigated_flat = np.zeros(obs_flat.shape[1], dtype=float)
        for idx in range(obs_flat.shape[1]):
            mitigated_flat[idx] = self._extrapolate_zero_noise(scales, obs_flat[:, idx])
            
        return np.clip(mitigated_flat.reshape(original_shape), -1.0, 1.0)

    def mitigate(
        self,
        base_eval: Callable[[], np.ndarray],
        scaled_eval: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Apply configured mitigation strategy."""
        if self.config.resilience_level < 1:
            return base_eval()
        if self.config.enable_zne and self.config.resilience_level >= 2:
            return self.mitigate_with_zne(scaled_eval)
        return self.trex_correct(base_eval())

