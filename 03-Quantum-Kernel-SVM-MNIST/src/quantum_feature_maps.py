"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Quantum feature-map helpers."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap

try:
    from qiskit.circuit.library import zz_feature_map
except ImportError:  # pragma: no cover - older Qiskit
    zz_feature_map = None

logger = logging.getLogger(__name__)

def create_feature_map(
    feature_map_type: str = "ZZFeatureMap",
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "full",
    parameter_prefix: str = "x",
    decompose: bool = False,
    paulis: Optional[list[str]] = None,
) -> QuantumCircuit:
    """Create a quantum feature map by type."""
    logger.info(
        "Creating %s: dim=%s reps=%s entanglement=%s",
        feature_map_type,
        feature_dimension,
        reps,
        entanglement,
    )

    if feature_map_type == "ZZFeatureMap":
        if zz_feature_map is not None:
            feature_map = zz_feature_map(
                feature_dimension=feature_dimension,
                reps=reps,
                entanglement=entanglement,
                parameter_prefix=parameter_prefix,
            )
        else:
            feature_map = ZZFeatureMap(
                feature_dimension=feature_dimension,
                reps=reps,
                entanglement=entanglement,
                parameter_prefix=parameter_prefix,
            )
    elif feature_map_type == "PauliFeatureMap":
        if paulis is None:
            paulis = ['Z', 'ZZ']
        feature_map = PauliFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            paulis=paulis,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )
    elif feature_map_type == "ZFeatureMap":
        feature_map = ZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            parameter_prefix=parameter_prefix,
        )
    else:
        raise ValueError(f"Unsupported feature map type: {feature_map_type}")

    if decompose:
        feature_map = feature_map.decompose()

    logger.info("Feature map created with %s qubits", feature_map.num_qubits)
    logger.info("Circuit depth: %s", feature_map.depth())
    logger.info("Number of parameters: %s", feature_map.num_parameters)
    return feature_map

def create_zz_feature_map(
    feature_dimension: int = 4,
    reps: int = 1,
    entanglement: str = "full",
    parameter_prefix: str = "x",
    decompose: bool = False,
) -> QuantumCircuit:
    """Create a ZZ feature map circuit compatible with multiple Qiskit versions."""
    logger.info(
        "Creating ZZ feature map: dim=%s reps=%s entanglement=%s",
        feature_dimension,
        reps,
        entanglement,
    )

    if zz_feature_map is not None:
        feature_map = zz_feature_map(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )
    else:
        feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )

    if decompose:
        feature_map = feature_map.decompose()

    logger.info("Feature map created with %s qubits", feature_map.num_qubits)
    logger.info("Circuit depth: %s", feature_map.depth())
    logger.info("Number of parameters: %s", feature_map.num_parameters)
    return feature_map

def get_feature_map_circuit_info(feature_map: QuantumCircuit) -> dict:
    """Get metadata for a feature-map circuit."""
    info = {
        "num_qubits": feature_map.num_qubits,
        "num_parameters": feature_map.num_parameters,
        "depth": feature_map.depth(),
        "gate_counts": feature_map.count_ops(),
        "reps": getattr(feature_map, "reps", None),
    }
    logger.info("Feature map info: %s", info)
    return info

def encode_features(feature_map: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """Validate feature shape against a feature map."""
    if X.shape[1] != feature_map.num_qubits:
        raise ValueError(
            f"Feature dimension {X.shape[1]} does not match feature map qubits "
            f"{feature_map.num_qubits}"
        )
    logger.info("Encoded %s samples with %s features", X.shape[0], X.shape[1])
    return X

def describe_hilbert_space_mapping(feature_dimension: int) -> str:
    """Return a plain-language description of Hilbert-space mapping."""
    hilbert_dim = 2 ** feature_dimension
    return (
        "\nHilbert Space Mapping Explanation\n"
        "=================================\n\n"
        "Classical Data -> Quantum State Encoding\n"
        "----------------------------------------\n"
        f"- Classical features: {feature_dimension} dimensions\n"
        f"- Quantum Hilbert space: {hilbert_dim} dimensions (2^{feature_dimension})\n\n"
        "The ZZ feature map encodes vectors into quantum states and defines\n"
        "a quantum kernel by state overlap:\n\n"
        "    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2\n"
    )

