"""Run-scoped artifact output management."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .artifact_schema import to_serializable_records
from .visualization import save_metrics_csv


class ArtifactManager:
    """Write artifacts to a run-scoped directory and mirror stable outputs."""

    def __init__(
        self,
        project_root: Path,
        output_dir: str = "results",
        run_id: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.root_dir = project_root / output_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run_path(self, filename: str) -> Path:
        """Return the path for a run-scoped artifact."""
        return self.run_dir / filename

    def root_path(self, filename: str) -> Path:
        """Return the stable mirrored artifact path."""
        return self.root_dir / filename

    def write_csv(
        self,
        filename: str,
        records: Sequence[Any],
        mirror_to_root: bool = True,
    ) -> Path:
        """Write CSV records into the run directory and optionally mirror them."""
        run_path = self.run_path(filename)
        save_metrics_csv(to_serializable_records(records), str(run_path))
        if mirror_to_root:
            shutil.copy2(run_path, self.root_path(filename))
        return run_path

    def write_json(
        self,
        filename: str,
        payload: Any,
        mirror_to_root: bool = True,
    ) -> Path:
        """Write JSON into the run directory and optionally mirror it."""
        run_path = self.run_path(filename)
        run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if mirror_to_root:
            shutil.copy2(run_path, self.root_path(filename))
        return run_path

    def write_text(
        self,
        filename: str,
        text: str,
        mirror_to_root: bool = True,
    ) -> Path:
        """Write text into the run directory and optionally mirror it."""
        run_path = self.run_path(filename)
        run_path.write_text(text, encoding="utf-8")
        if mirror_to_root:
            shutil.copy2(run_path, self.root_path(filename))
        return run_path

    def mirror_existing(self, filename: str) -> Path:
        """Copy an already-generated run artifact into the stable root path."""
        source = self.run_path(filename)
        destination = self.root_path(filename)
        shutil.copy2(source, destination)
        return destination
