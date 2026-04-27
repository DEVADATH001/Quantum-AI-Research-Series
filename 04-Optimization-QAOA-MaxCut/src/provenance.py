"""Run provenance collection helpers for reproducible research artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict


CORE_PACKAGES = [
    "numpy",
    "scipy",
    "networkx",
    "pandas",
    "matplotlib",
    "seaborn",
    "PyYAML",
    "qiskit",
    "qiskit-aer",
    "qiskit-ibm-runtime",
]


def _safe_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def config_hash(config: Dict[str, Any]) -> str:
    """Compute a stable hash for a loaded configuration dictionary."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_run_provenance(
    project_root: Path,
    config: Dict[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    """Collect environment and configuration provenance for one run."""
    package_versions = {
        package_name: _safe_package_version(package_name)
        for package_name in CORE_PACKAGES
    }

    return {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(project_root),
        "config_hash": config_hash(config),
        "package_versions": package_versions,
    }
