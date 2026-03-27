"""Noise-model utilities for ideal, noisy, and mitigated execution modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NoiseConfig:
    """Compact fallback noise configuration."""

    backend_name: str = "ibm_osaka"
    gate_error: float = 0.001
    readout_error: float = 0.02
    t1_ns: float = 100_000.0
    t2_ns: float = 50_000.0
    gate_time_ns: float = 100.0


def resolve_fake_backend(backend_name: str) -> Any | None:
    backend_name = backend_name.lower().replace("-", "_")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeOsaka
    except Exception:  # pragma: no cover - optional dependency guards
        return None

    backend_map = {
        "ibm_osaka": FakeOsaka,
        "osaka": FakeOsaka,
    }
    backend_cls = backend_map.get(backend_name)
    if backend_cls is None:
        return None
    return backend_cls()


def _derive_noise_config_from_backend(fake_backend: Any, backend_name: str) -> NoiseConfig:
    properties = fake_backend.properties()

    one_qubit_errors: list[float] = []
    two_qubit_errors: list[float] = []
    gate_lengths_ns: list[float] = []
    t1_ns: list[float] = []
    t2_ns: list[float] = []
    readout_errors: list[float] = []

    for gate in properties.gates:
        n_qubits = len(gate.qubits)
        gate_error = None
        gate_length = None
        gate_length_unit = "ns"
        for param in gate.parameters:
            if param.name == "gate_error":
                gate_error = float(param.value)
            if param.name == "gate_length":
                gate_length = float(param.value)
                gate_length_unit = getattr(param, "unit", "ns") or "ns"
        if gate_error is not None:
            if n_qubits == 1:
                one_qubit_errors.append(gate_error)
            elif n_qubits == 2:
                two_qubit_errors.append(gate_error)
        if gate_length is not None:
            multiplier = 1000.0 if gate_length_unit == "us" else 1.0
            gate_lengths_ns.append(gate_length * multiplier)

    for qubit_props in properties.qubits:
        values = {item.name: float(item.value) for item in qubit_props}
        t1_ns.append(values.get("T1", 100.0) * 1000.0)  # us -> ns
        t2_ns.append(values.get("T2", 80.0) * 1000.0)  # us -> ns
        readout_errors.append(values.get("readout_error", 0.02))

    gate_error = float(np.mean(one_qubit_errors)) if one_qubit_errors else 0.001
    if two_qubit_errors:
        gate_error = max(gate_error, float(np.mean(two_qubit_errors) / 5.0))

    return NoiseConfig(
        backend_name=backend_name,
        gate_error=float(np.clip(gate_error, 1e-4, 0.05)),
        readout_error=float(np.clip(np.mean(readout_errors), 0.001, 0.2)),
        t1_ns=float(np.clip(np.mean(t1_ns), 10_000.0, 1_000_000.0)),
        t2_ns=float(np.clip(np.mean(t2_ns), 10_000.0, 1_000_000.0)),
        gate_time_ns=float(np.clip(np.mean(gate_lengths_ns), 20.0, 1_000.0))
        if gate_lengths_ns
        else 100.0,
    )


def load_ibm_noise_model(
    backend_name: str = "ibm_osaka",
    compact: bool = False,
) -> NoiseModel:
    """
    Load a backend-derived noise model.

    When possible, the full fake-backend model is used. Compact mode is kept as
    a fallback for quick experiments or environments without fake-backend data.
    """

    fake_backend = resolve_fake_backend(backend_name)
    if fake_backend is not None:
        if compact:
            compact_config = _derive_noise_config_from_backend(fake_backend, backend_name)
            logger.info(
                "Loaded compact backend-derived noise model for %s "
                "(gate_error=%.4g, readout_error=%.4g)",
                backend_name,
                compact_config.gate_error,
                compact_config.readout_error,
            )
            return build_custom_noise_model(compact_config)
        logger.info("Loaded full fake-backend noise model for %s", backend_name)
        return NoiseModel.from_backend(fake_backend)

    logger.warning(
        "Could not resolve fake backend '%s'. Falling back to a compact custom model.",
        backend_name,
    )
    return build_custom_noise_model(NoiseConfig(backend_name=backend_name))


def build_custom_noise_model(config: NoiseConfig) -> NoiseModel:
    noise_model = NoiseModel()

    one_qubit_error = depolarizing_error(config.gate_error, 1)
    two_qubit_error = depolarizing_error(min(5.0 * config.gate_error, 0.2), 2)
    noise_model.add_all_qubit_quantum_error(one_qubit_error, ["rx", "ry", "rz", "x", "sx"])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cz"])

    tr_error_1 = thermal_relaxation_error(
        t1=config.t1_ns,
        t2=min(config.t2_ns, 2 * config.t1_ns),
        time=config.gate_time_ns,
    )
    noise_model.add_all_qubit_quantum_error(tr_error_1, ["u", "u1", "u2", "u3"])

    p10 = max(0.0, min(config.readout_error * 0.5, 0.49))  # P(meas=1 | true=0)
    p01 = max(0.0, min(config.readout_error * 1.5, 0.49))  # P(meas=0 | true=1)
    readout = ReadoutError([[1.0 - p10, p10], [p01, 1.0 - p01]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def infer_average_readout_error_rates(noise_model: NoiseModel | None) -> tuple[float, float]:
    """Estimate average asymmetric readout rates (p01, p10) from a noise model."""

    if noise_model is None:
        return 0.0, 0.0

    p01_values: list[float] = []
    p10_values: list[float] = []
    serialized = noise_model.to_dict()
    for err in serialized.get("errors", []):
        if err.get("type") != "roerror":
            continue
        matrix = err.get("probabilities", [])
        if len(matrix) == 2 and len(matrix[0]) == 2:
            p10_values.append(float(matrix[0][1]))
            p01_values.append(float(matrix[1][0]))

    if not p01_values or not p10_values:
        return 0.0, 0.0
    return float(np.mean(p01_values)), float(np.mean(p10_values))


def infer_readout_error_rates_by_qubit(
    noise_model: NoiseModel | None,
) -> dict[int, tuple[float, float]]:
    """
    Estimate per-qubit asymmetric readout rates from a noise model.

    Returns a dictionary mapping qubit index -> (p01, p10), where
    p01 = P(meas=0 | true=1) and p10 = P(meas=1 | true=0).
    """

    if noise_model is None:
        return {}

    serialized = noise_model.to_dict()
    per_qubit: dict[int, list[tuple[float, float]]] = {}
    fallback_rates: list[tuple[float, float]] = []

    for err in serialized.get("errors", []):
        if err.get("type") != "roerror":
            continue
        matrix = err.get("probabilities", [])
        if len(matrix) != 2 or len(matrix[0]) != 2:
            continue

        p10 = float(matrix[0][1])
        p01 = float(matrix[1][0])
        gate_qubits = err.get("gate_qubits", [])
        if gate_qubits:
            for qubit_group in gate_qubits:
                if len(qubit_group) != 1:
                    continue
                qubit_idx = int(qubit_group[0])
                per_qubit.setdefault(qubit_idx, []).append((p01, p10))
        else:
            fallback_rates.append((p01, p10))

    if per_qubit:
        return {
            qubit_idx: (
                float(np.mean([rates[0] for rates in samples])),
                float(np.mean([rates[1] for rates in samples])),
            )
            for qubit_idx, samples in per_qubit.items()
        }

    if fallback_rates:
        avg_p01 = float(np.mean([rates[0] for rates in fallback_rates]))
        avg_p10 = float(np.mean([rates[1] for rates in fallback_rates]))
        return {-1: (avg_p01, avg_p10)}

    return {}


def infer_readout_error_probability(noise_model: NoiseModel | None) -> float:
    """Backward-compatible average symmetric readout error estimate."""

    p01, p10 = infer_average_readout_error_rates(noise_model)
    return 0.5 * (p01 + p10)
