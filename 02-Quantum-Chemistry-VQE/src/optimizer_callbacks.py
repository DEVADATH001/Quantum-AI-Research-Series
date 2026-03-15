"""Optimizer callback storage."""

from __future__ import annotations

from typing import Any, Dict, List


class VQECallback:
    """Collects iteration-level metrics from Qiskit VQE callback."""

    def __init__(self, energy_shift: float = 0.0) -> None:
        self.history: List[Dict[str, Any]] = []
        self.energy_shift = energy_shift

    def __call__(self, eval_count: int, parameters: Any, mean: float, metadata: Dict[str, Any]) -> None:
        params = parameters.tolist() if hasattr(parameters, "tolist") else list(parameters)
        variance = metadata.get("variance")
        # Log both electronic and total physical energy for hybrid system monitoring
        electronic_energy = float(mean)
        total_energy = electronic_energy + self.energy_shift
        
        self.history.append(
            {
                "iteration": int(eval_count),
                "energy": total_energy,
                "electronic_energy": electronic_energy,
                "variance": float(variance) if variance is not None else None,
                "parameters": [float(x) for x in params],
                "metadata": dict(metadata),
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Return callback trace."""
        return self.history

    def clear(self) -> None:
        """Clear stored trace."""
        self.history = []
