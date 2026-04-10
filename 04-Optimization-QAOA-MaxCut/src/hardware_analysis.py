"""Hardware-feasibility utilities for NISQ-style QAOA studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from qiskit import transpile


@dataclass
class HardwareFeasibilityThresholds:
    """Heuristic thresholds for small-scale NISQ feasibility."""

    max_transpiled_depth: int = 200
    max_two_qubit_gates: int = 120
    max_total_shots: int = 150000


class HardwareFeasibilityAnalyzer:
    """Analyze logical and transpiled circuit cost on a target backend."""

    def __init__(
        self,
        backend: Any,
        optimization_level: int = 1,
        seed: Optional[int] = 42,
        thresholds: Optional[HardwareFeasibilityThresholds] = None,
    ) -> None:
        self.backend = backend
        self.optimization_level = optimization_level
        self.seed = seed
        self.thresholds = thresholds or HardwareFeasibilityThresholds()

    def analyze(
        self,
        circuit,
        shots_per_evaluation: int,
        n_evaluations: int,
        objective_repetitions: int = 1,
        report_repetitions: int = 1,
    ) -> Dict[str, Any]:
        """Build a heuristic hardware-feasibility report for one circuit."""
        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )

        logical_counts = circuit.count_ops()
        transpiled_counts = transpiled.count_ops()
        logical_two_qubit = self._count_two_qubit_gates(logical_counts)
        transpiled_two_qubit = self._count_two_qubit_gates(transpiled_counts)
        total_shots = int(max(1, shots_per_evaluation) * (n_evaluations * max(1, objective_repetitions) + max(1, report_repetitions)))

        issues = []
        backend_qubits = getattr(self.backend, "num_qubits", None)
        if backend_qubits is not None and circuit.num_qubits > int(backend_qubits):
            issues.append("requires_more_qubits_than_backend")
        if transpiled.depth() > self.thresholds.max_transpiled_depth:
            issues.append("transpiled_depth_too_high")
        if transpiled_two_qubit > self.thresholds.max_two_qubit_gates:
            issues.append("too_many_two_qubit_gates")
        if total_shots > self.thresholds.max_total_shots:
            issues.append("shot_budget_too_high")

        if issues:
            status = "unlikely_without_major_noise"
        elif transpiled.depth() > int(0.7 * self.thresholds.max_transpiled_depth):
            status = "possible_but_fragile"
        else:
            status = "small_scale_feasible"

        return {
            "logical_qubits": circuit.num_qubits,
            "logical_depth": circuit.depth(),
            "logical_size": circuit.size(),
            "logical_two_qubit_gates": logical_two_qubit,
            "transpiled_depth": transpiled.depth(),
            "transpiled_size": transpiled.size(),
            "transpiled_two_qubit_gates": transpiled_two_qubit,
            "entangling_gate_multiplier": float(transpiled_two_qubit / logical_two_qubit)
            if logical_two_qubit > 0
            else None,
            "estimated_total_shots": total_shots,
            "backend_name": self._backend_name(self.backend),
            "status": status,
            "issues": "|".join(issues),
        }

    @staticmethod
    def _count_two_qubit_gates(counts) -> int:
        """Count common two-qubit entangling gates in an op-count dictionary."""
        return int(
            sum(
                int(counts.get(name, 0))
                for name in ("cx", "cz", "ecr", "swap", "iswap", "rzz", "rxx", "ryy")
            )
        )

    @staticmethod
    def _backend_name(backend: Any) -> str:
        """Return a readable backend name."""
        name = getattr(backend, "name", None)
        if callable(name):
            return str(name())
        if name is not None:
            return str(name)
        return backend.__class__.__name__
