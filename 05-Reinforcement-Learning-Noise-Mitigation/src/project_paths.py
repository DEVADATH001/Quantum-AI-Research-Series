"""Project-root path helpers shared by CLI entrypoints and experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(raw: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""

    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0].lower() == PROJECT_ROOT.name.lower():
        return (PROJECT_ROOT.parent / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def path_relative_to_project(path: str | Path) -> str:
    """Render a path relative to the project root when possible."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
