from __future__ import annotations

import unittest
from pathlib import Path

from src.project_paths import PROJECT_ROOT, path_relative_to_project, resolve_project_path


class ProjectPathTests(unittest.TestCase):
    def test_resolve_relative_project_path(self) -> None:
        resolved = resolve_project_path("config/training_config.yaml")
        self.assertEqual(resolved, PROJECT_ROOT / "config" / "training_config.yaml")

    def test_resolve_project_prefixed_path(self) -> None:
        resolved = resolve_project_path(f"{PROJECT_ROOT.name}/config/smoke_test.yaml")
        self.assertEqual(resolved, PROJECT_ROOT / "config" / "smoke_test.yaml")

    def test_path_relative_to_project(self) -> None:
        rel = path_relative_to_project(PROJECT_ROOT / "results" / "summary.json")
        self.assertEqual(rel, str(Path("results") / "summary.json"))


if __name__ == "__main__":
    unittest.main()
