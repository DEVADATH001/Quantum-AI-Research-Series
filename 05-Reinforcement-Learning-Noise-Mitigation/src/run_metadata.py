"""Run-manifest helpers for reproducible experiment artifacts."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.project_paths import PROJECT_ROOT, path_relative_to_project


def _run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = completed.stdout.strip()
    return output or None


def collect_git_metadata() -> dict[str, Any]:
    commit = _run_git_command(["rev-parse", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def build_run_manifest(
    *,
    config_source: str,
    resolved_config: dict[str, Any],
    output_dir: Path,
    command: list[str],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": config_source,
        "resolved_config": resolved_config,
        "output_dir": path_relative_to_project(output_dir),
        "command": command,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git": collect_git_metadata(),
    }
