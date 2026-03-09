"""Abstract interfaces for extension points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from qiskit import QuantumCircuit
from qiskit_nature.second_q.problems import ElectronicStructureProblem


class AbstractMoleculeDriver(ABC):
    """Interface for molecule problem providers."""

    @abstractmethod
    def get_problem(self, molecule_name: str, bond_length: float, **kwargs) -> Tuple[ElectronicStructureProblem, Any]:
        """Return electronic structure problem and metadata."""


class AbstractAnsatzFactory(ABC):
    """Interface for ansatz construction providers."""

    @abstractmethod
    def build(self, name: str, problem: ElectronicStructureProblem, mapper: Any, **kwargs) -> QuantumCircuit:
        """Return a quantum circuit ansatz."""
