"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Plugin registry for quantum feature maps.

Allows any feature map to be registered by name and constructed
from config without modifying the core engine. New maps are added
by decorating the builder function with @register("MyMapName").
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

try:
    from qiskit.circuit.library import zz_feature_map as _zz_fn
except ImportError:  # pragma: no cover
    _zz_fn = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., QuantumCircuit]] = {}


def register(name: str) -> Callable:
    """Decorator — register a feature-map builder under *name*."""

    def decorator(fn: Callable[..., QuantumCircuit]) -> Callable[..., QuantumCircuit]:
        if name in _REGISTRY:
            logger.warning("Feature map '%s' already registered; overwriting.", name)
        _REGISTRY[name] = fn
        logger.debug("Registered feature map: %s → %s", name, fn.__name__)
        return fn

    return decorator


def build(name: str, **kwargs: Any) -> QuantumCircuit:
    """Instantiate a registered feature map by name.

    Args:
        name: Registered map name (case-sensitive).
        **kwargs: Forwarded to the builder (feature_dimension, reps, …).

    Returns:
        Constructed QuantumCircuit.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise ValueError(
            f"Unknown feature map '{name}'. Available: {available}"
        )
    circuit = _REGISTRY[name](**kwargs)
    logger.info(
        "Built feature map '%s': qubits=%d depth=%d params=%d",
        name,
        circuit.num_qubits,
        circuit.depth(),
        circuit.num_parameters,
    )
    return circuit


def list_available() -> list[str]:
    """Return sorted list of all registered feature-map names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

@register("ZZFeatureMap")
def _build_zz(
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "full",
    parameter_prefix: str = "x",
    **_: Any,
) -> QuantumCircuit:
    """ZZ-feature map (default; classically hard to simulate for d≥50 qubits)."""
    if _zz_fn is not None:
        return _zz_fn(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )
    return ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
        parameter_prefix=parameter_prefix,
    )


@register("ZFeatureMap")
def _build_z(
    feature_dimension: int = 4,
    reps: int = 1,
    parameter_prefix: str = "x",
    **_: Any,
) -> QuantumCircuit:
    """Single-qubit Z rotations only — separable, no entanglement."""
    return ZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        parameter_prefix=parameter_prefix,
    )


@register("PauliFeatureMap")
def _build_pauli(
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "full",
    parameter_prefix: str = "x",
    paulis: list[str] | None = None,
    **_: Any,
) -> QuantumCircuit:
    """Generalized Pauli feature map (configurable Pauli word list)."""
    if paulis is None:
        paulis = ["Z", "ZZ"]
    return PauliFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        paulis=paulis,
        entanglement=entanglement,
        parameter_prefix=parameter_prefix,
    )


@register("IQPFeatureMap")
def _build_iqp(
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "full",
    parameter_prefix: str = "x",
    **_: Any,
) -> QuantumCircuit:
    """Instantaneous Quantum Polynomial (IQP) feature map.

    Equivalent to PauliFeatureMap with paulis=['Z','ZZ'] using diagonal
    (commuting) unitaries. Classically hard to simulate (Wood & Bartlett 2015).
    """
    return PauliFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        paulis=["Z", "ZZ"],
        entanglement=entanglement,
        parameter_prefix=parameter_prefix,
    )


@register("LinearEntanglementZZ")
def _build_linear_zz(
    feature_dimension: int = 4,
    reps: int = 1,
    parameter_prefix: str = "x",
    **_: Any,
) -> QuantumCircuit:
    """ZZFeatureMap with linear (nearest-neighbour) entanglement."""
    if _zz_fn is not None:
        return _zz_fn(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement="linear",
            parameter_prefix=parameter_prefix,
        )
    return ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement="linear",
        parameter_prefix=parameter_prefix,
    )
