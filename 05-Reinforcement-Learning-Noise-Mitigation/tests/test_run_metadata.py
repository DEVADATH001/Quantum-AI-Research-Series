from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

from src.run_metadata import build_run_manifest

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class RunMetadataTests(unittest.TestCase):
    def test_build_run_manifest_contains_reproducibility_fields(self) -> None:
        temp_dir = TEST_TMP_ROOT / f"run_metadata_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest = build_run_manifest(
                config_source="in_memory_mapping",
                resolved_config={"seed": 1},
                output_dir=temp_dir,
                command=["python", "-m", "src.training_pipeline"],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.assertEqual(manifest["config_source"], "in_memory_mapping")
        self.assertEqual(manifest["resolved_config"]["seed"], 1)
        self.assertIn("git", manifest)
        self.assertIn("python_version", manifest)


if __name__ == "__main__":
    unittest.main()
