"""Optimizer callback storage."""

from __future__ import annotations

from typing import Any, Dict, List


class VQECallback:
    """Collects iteration-level metrics from Qiskit VQE callback."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def __call__(self, eval_count: int, parameters: Any, mean: float, metadata: Dict[str, Any]) -> None:
        params = parameters.tolist() if hasattr(parameters, "tolist") else list(parameters)
        variance = metadata.get("variance")
        self.history.append(
            {
                "iteration": int(eval_count),
                "energy": float(mean),
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
