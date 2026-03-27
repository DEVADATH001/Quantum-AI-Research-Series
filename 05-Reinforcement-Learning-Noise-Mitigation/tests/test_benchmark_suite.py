from __future__ import annotations

import unittest
from pathlib import Path

from src.benchmark_suite import _deep_merge, _load_suite_definition, _rank_methods, _sync_episode_limits
from src.project_paths import resolve_project_path


class BenchmarkSuiteTests(unittest.TestCase):
    def test_deep_merge_preserves_unmodified_branches(self) -> None:
        base = {
            "environment": {"n_positions": 4, "slip_probability": 0.05},
            "training": {"n_episodes": 40},
        }
        patch = {
            "environment": {"slip_probability": 0.15},
            "results": {"output_dir": "results/benchmark_suite/test"},
        }
        merged = _deep_merge(base, patch)
        self.assertEqual(merged["environment"]["n_positions"], 4)
        self.assertEqual(merged["environment"]["slip_probability"], 0.15)
        self.assertEqual(merged["training"]["n_episodes"], 40)
        self.assertEqual(merged["results"]["output_dir"], "results/benchmark_suite/test")

    def test_sync_episode_limits_updates_all_trainers(self) -> None:
        payload = {
            "environment": {"max_episode_steps": 10},
            "training": {},
            "baselines": {},
            "mlp_baseline": {},
            "mlp_actor_critic": {},
            "quantum_actor_critic": {},
        }
        _sync_episode_limits(payload)
        self.assertEqual(payload["training"]["max_episode_steps"], 10)
        self.assertEqual(payload["baselines"]["max_episode_steps"], 10)
        self.assertEqual(payload["mlp_baseline"]["max_episode_steps"], 10)
        self.assertEqual(payload["mlp_actor_critic"]["max_episode_steps"], 10)
        self.assertEqual(payload["quantum_actor_critic"]["max_episode_steps"], 10)

    def test_rank_methods_orders_by_eval_success_descending(self) -> None:
        ranks = _rank_methods(
            {
                "quantum_reinforce_ideal": {"eval_success": 0.8},
                "mlp_actor_critic": {"eval_success": 0.95},
                "tabular_reinforce": {"eval_success": 0.7},
            }
        )
        self.assertEqual(ranks["mlp_actor_critic"], 1)
        self.assertEqual(ranks["quantum_reinforce_ideal"], 2)
        self.assertEqual(ranks["tabular_reinforce"], 3)

    def test_smoke_suite_yaml_is_loadable(self) -> None:
        payload = _load_suite_definition(resolve_project_path("config/benchmark_suite_smoke.yaml"))
        self.assertEqual(payload["scenario_source"], "smoke_registry")
        self.assertTrue(Path(resolve_project_path(payload["base_config"])).exists())


if __name__ == "__main__":
    unittest.main()
