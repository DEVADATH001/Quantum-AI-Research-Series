"""Error-mitigation utilities for expectation-value estimation."""

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
    enable_readout_correction: bool = True
    enable_zne: bool = True
    scale_factors: tuple[float, ...] = (1.0, 2.0, 3.0)
    extrapolation: str = "linear"
    polynomial_degree: int = 1


def fold_circuit_for_noise_scaling(circuit: QuantumCircuit, scale_factor: float) -> tuple[QuantumCircuit, float]:
    """
    Apply deterministic local folding to approximate a larger effective noise rate.

    The circuit is scaled by inserting U^-1 U pairs after selected operations.
    """

    if scale_factor <= 1.0:
        return circuit.copy(), 1.0

    op_data = [
        (inst, qargs, cargs)
        for inst, qargs, cargs in circuit.data
        if inst.name not in {"measure", "barrier"}
    ]
    total_ops = len(op_data)
    if total_ops == 0:
        return circuit.copy(), 1.0

    fold_budget = max(0, int(round((scale_factor - 1.0) * total_ops / 2.0)))
    base_folds, remainder = divmod(fold_budget, total_ops)
    fold_counts = np.full(total_ops, base_folds, dtype=int)
    if remainder:
        for idx in np.linspace(0, total_ops - 1, num=remainder, dtype=int):
            fold_counts[idx] += 1

    folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    folded.global_phase = circuit.global_phase

    op_idx = 0
    for inst, qargs, cargs in circuit.data:
        q_idx = [circuit.find_bit(qubit).index for qubit in qargs]
        c_idx = [circuit.find_bit(clbit).index for clbit in cargs]
        folded.append(inst, q_idx, c_idx)

        if inst.name in {"measure", "barrier"}:
            continue

        for _ in range(int(fold_counts[op_idx])):
            folded.append(inst.inverse(), q_idx, c_idx)
            folded.append(inst, q_idx, c_idx)
        op_idx += 1

    achieved_scale = 1.0 + (2.0 * float(np.sum(fold_counts)) / float(total_ops))
    return folded, achieved_scale


class MitigationEngine:
    """Apply zero-noise extrapolation to already corrected expectation values."""

    def __init__(self, config: MitigationConfig | None = None) -> None:
        self.config = config or MitigationConfig()

    def _extrapolate_zero_noise(
        self,
        scale_factors: np.ndarray,
        values: np.ndarray,
    ) -> float:
        method = self.config.extrapolation.lower()

        if method == "richardson":
            n = len(scale_factors)
            table = np.zeros((n, n))
            table[:, 0] = values
            for col_idx in range(1, n):
                for row_idx in range(n - col_idx):
                    numerator = (
                        scale_factors[row_idx + col_idx] * table[row_idx, col_idx - 1]
                        - scale_factors[row_idx] * table[row_idx + 1, col_idx - 1]
                    )
                    denominator = scale_factors[row_idx + col_idx] - scale_factors[row_idx]
                    table[row_idx, col_idx] = numerator / denominator
            return float(table[0, n - 1])

        if method == "polynomial" and self.config.polynomial_degree > 1:
            degree = min(self.config.polynomial_degree, scale_factors.size - 1)
            coeffs = np.polyfit(scale_factors, values, deg=degree)
            return float(np.poly1d(coeffs)(0.0))

        if method == "exponential" and np.all(values > 0.0):
            coeffs = np.polyfit(scale_factors, np.log(values), deg=1)
            return float(np.exp(np.poly1d(coeffs)(0.0)))

        coeffs = np.polyfit(scale_factors, values, deg=1)
        return float(np.poly1d(coeffs)(0.0))

    def mitigate_with_zne(
        self,
        evaluate_fn: Callable[[float], tuple[float, np.ndarray]],
    ) -> np.ndarray:
        aggregated: dict[float, list[np.ndarray]] = {}
        for requested_scale in self.config.scale_factors:
            actual_scale, corrected_eval = evaluate_fn(float(requested_scale))
            aggregated.setdefault(float(actual_scale), []).append(np.asarray(corrected_eval, dtype=float))

        ordered_scales = np.array(sorted(aggregated.keys()), dtype=float)
        observations = np.asarray(
            [np.mean(np.stack(aggregated[scale], axis=0), axis=0) for scale in ordered_scales],
            dtype=float,
        )
        if ordered_scales.size == 1:
            return np.clip(observations[0], -1.0, 1.0)

        original_shape = observations.shape[1:]
        obs_flat = observations.reshape(len(ordered_scales), -1)

        mitigated_flat = np.zeros(obs_flat.shape[1], dtype=float)
        for idx in range(obs_flat.shape[1]):
            mitigated_flat[idx] = self._extrapolate_zero_noise(ordered_scales, obs_flat[:, idx])

        return np.clip(mitigated_flat.reshape(original_shape), -1.0, 1.0)

    def mitigate(
        self,
        base_eval: Callable[[], np.ndarray],
        scaled_eval: Callable[[float], tuple[float, np.ndarray]],
    ) -> np.ndarray:
        if self.config.resilience_level < 1:
            return base_eval()
        if self.config.enable_zne and self.config.resilience_level >= 2:
            return self.mitigate_with_zne(scaled_eval)
        return base_eval()
