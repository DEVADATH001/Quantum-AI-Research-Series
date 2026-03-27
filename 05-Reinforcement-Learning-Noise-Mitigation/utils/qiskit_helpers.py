"""Author: DEVADATH H K

Project: Quantum RL Noise Mitigation

Utility helpers for the Quantum RL noise-mitigation project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from core.schemas import DEFAULT_SCHEMA_VERSION

def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging for scripts and notebooks."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def set_global_seed(seed: int) -> np.random.Generator:
    """Set deterministic seeds across random libraries."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

def build_state_angles(
    state: int,
    num_qubits: int,
    n_states: int | None = None,
    encoding: str = "hybrid",
) -> np.ndarray:
    """
    Map a discrete environment state to embedding rotations.

    Supported encodings:
    - ``binary``: binary bits mapped to 0 / pi.
    - ``phase``: normalized state index mapped to qubit-dependent phases.
    - ``hybrid``: binary structure plus a smooth phase ramp over the state index.
    """
    if num_qubits == 0:
        return np.array([], dtype=float)

    n_states = max(int(n_states or 2**num_qubits), 1)
    if state < 0 or state >= n_states:
        raise ValueError(f"State {state} is outside [0, {n_states}).")

    binary = format(state, f"0{num_qubits}b")[-num_qubits:]
    binary_angles = np.array([np.pi * float(bit) for bit in binary], dtype=float)

    encoding_name = encoding.lower()
    if encoding_name == "binary":
        return binary_angles

    normalized_state = 0.0 if n_states <= 1 else float(state) / float(n_states - 1)
    frequencies = np.arange(1, num_qubits + 1, dtype=float)
    phase_angles = np.pi * normalized_state * frequencies

    if encoding_name == "phase":
        return phase_angles
    if encoding_name == "hybrid":
        return (0.5 * binary_angles) + phase_angles
    raise ValueError(f"Unsupported state encoding: {encoding}")

def make_z_observables(num_qubits: int, num_actions: int) -> list[SparsePauliOp]:
    """Build Pauli-Z observables used as action logits."""
    if num_actions > num_qubits:
        raise ValueError(
            "num_actions cannot exceed num_qubits when using single-qubit Z readouts."
        )
    observables: list[SparsePauliOp] = []
    for qubit_idx in range(num_actions):
        pauli = ["I"] * num_qubits
        pauli[num_qubits - 1 - qubit_idx] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli), 1.0)]))
    return observables

def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(values, dtype=float) / max(temperature, 1e-8)
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    normalizer = np.sum(exp_values)
    if normalizer <= 0.0:
        raise ValueError("Softmax normalization failed because the partition function is non-positive.")
    return exp_values / normalizer

def project_probabilities(probabilities: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Project a vector or batch of vectors onto the probability simplex."""
    values = np.asarray(probabilities, dtype=float)
    clipped = np.clip(values, 0.0, None)
    floor = max(0.0, float(floor))
    if floor > 0.0:
        n_actions = max(1, clipped.shape[-1] if clipped.ndim > 0 else 1)
        clipped = clipped + min(floor, 0.999999 / n_actions)

    if clipped.ndim == 1:
        total = float(clipped.sum())
        if total <= 1e-12:
            return np.full_like(clipped, 1.0 / max(1, clipped.size))
        return clipped / total

    totals = clipped.sum(axis=-1, keepdims=True)
    zero_mask = totals <= 1e-12
    normalized = np.divide(
        clipped,
        np.where(zero_mask, 1.0, totals),
        out=np.zeros_like(clipped),
        where=np.ones_like(clipped, dtype=bool),
    )
    if np.any(zero_mask):
        normalized[zero_mask.squeeze(axis=-1)] = 1.0 / max(1, clipped.shape[-1])
    return normalized

def _json_default(value: Any) -> Any:
    """JSON serialization helper for numpy objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported value type for JSON serialization: {type(value)!r}")

def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save dictionary payload as JSON with numpy support."""
    serializable = dict(payload)
    serializable.setdefault("schema_version", DEFAULT_SCHEMA_VERSION)
    Path(path).write_text(json.dumps(serializable, indent=2, default=_json_default), encoding="utf-8")
