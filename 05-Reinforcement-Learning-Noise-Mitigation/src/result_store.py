"""Artifact writing helpers for experiment outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.qiskit_helpers import ensure_dir, save_json


@dataclass(slots=True)
class ExperimentResultStore:
    """Filesystem layout helper for experiment artifacts."""

    output_dir: Path
    quantum_root: Path = field(init=False)
    baseline_root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = ensure_dir(self.output_dir)
        self.quantum_root = ensure_dir(self.output_dir / "quantum")
        self.baseline_root = ensure_dir(self.output_dir / "baselines")

    def quantum_seed_dir(self, seed: int) -> Path:
        return ensure_dir(self.quantum_root / f"seed_{seed}")

    def baseline_seed_dir(self, seed: int) -> Path:
        return ensure_dir(self.baseline_root / f"seed_{seed}")

    def save_json(self, path: Path, payload: dict) -> None:
        save_json(path, payload)

    def save_numpy(self, path: Path, payload: np.ndarray) -> None:
        np.save(path, payload)

    def save_episode_csv(
        self,
        path: Path,
        rewards: list[float],
        successes: list[bool],
        runtimes: list[float],
        grad_norms: list[float] | None = None,
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "reward", "success", "runtime_sec", "grad_norm"])
            for idx, (reward, success, runtime_sec) in enumerate(zip(rewards, successes, runtimes), start=1):
                grad_norm = grad_norms[idx - 1] if grad_norms is not None and idx - 1 < len(grad_norms) else ""
                writer.writerow([idx, reward, int(success), runtime_sec, grad_norm])
