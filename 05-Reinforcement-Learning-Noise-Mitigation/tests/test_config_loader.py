from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

from src.config_loader import AppConfig, dump_config, load_config

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class ConfigLoaderTests(unittest.TestCase):
    def test_load_config_accepts_mapping(self) -> None:
        config = load_config(
            {
                "seed": 7,
                "results": {"output_dir": "results_test"},
                "experiment": {"seeds": [7, 9]},
            }
        )
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.experiment.seeds, [7, 9])

    def test_load_config_accepts_existing_app_config(self) -> None:
        config = AppConfig(seed=11)
        self.assertIs(load_config(config), config)

    def test_dump_config_round_trips_yaml(self) -> None:
        source = AppConfig(seed=13, results={"output_dir": "results_roundtrip"})
        payload = dump_config(source)
        self.assertEqual(payload["seed"], 13)

        temp_dir = TEST_TMP_ROOT / f"config_loader_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = temp_dir / "config.yaml"
            path.write_text("seed: 21\nresults:\n  output_dir: results_tmp\n", encoding="utf-8")
            loaded = load_config(path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.assertEqual(loaded.seed, 21)
        self.assertEqual(loaded.results.output_dir, "results_tmp")


if __name__ == "__main__":
    unittest.main()
