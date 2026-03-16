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

def build_state_angles(state: int, num_qubits: int) -> np.ndarray:
    """
    Map a discrete environment state to angle-embedding rotations.
    Uses binary encoding for scalability.
    """
    if num_qubits == 0:
        return np.array([], dtype=float)
    
    # Binary representation of state
    binary = format(state, f"0{num_qubits}b")
    # Map '0' -> 0.0, '1' -> pi
    angles = np.array([np.pi * float(bit) for bit in binary], dtype=float)
    
    if len(angles) > num_qubits:
        return angles[:num_qubits]
    if len(angles) < num_qubits:
        return np.pad(angles, (0, num_qubits - len(angles)), "constant")
    return angles

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
    probs = exp_values / np.sum(exp_values)
    return np.clip(probs, 1e-9, 1.0)

def _json_default(value: Any) -> Any:
    """JSON serialization helper for numpy objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported value type for JSON serialization: {type(value)!r}")

def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save dictionary payload as JSON with numpy support."""
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default))
