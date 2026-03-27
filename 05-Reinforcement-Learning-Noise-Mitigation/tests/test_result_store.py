from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

import numpy as np

from src.result_store import ExperimentResultStore

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class ResultStoreTests(unittest.TestCase):
    def test_result_store_writes_expected_artifacts(self) -> None:
        temp_dir = TEST_TMP_ROOT / f"result_store_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            store = ExperimentResultStore(output_dir=temp_dir)
            quantum_dir = store.quantum_seed_dir(7)
            baseline_dir = store.baseline_seed_dir(7)
            self.assertTrue(quantum_dir.exists())
            self.assertTrue(baseline_dir.exists())

            json_path = quantum_dir / "sample.json"
            npy_path = baseline_dir / "weights.npy"
            csv_path = quantum_dir / "episodes.csv"

            store.save_json(json_path, {"ok": True})
            store.save_numpy(npy_path, np.array([1.0, 2.0]))
            store.save_episode_csv(
                csv_path,
                rewards=[1.0, 0.5],
                successes=[True, False],
                runtimes=[0.1, 0.2],
                grad_norms=[0.3, 0.4],
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(npy_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("schema_version", json_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
