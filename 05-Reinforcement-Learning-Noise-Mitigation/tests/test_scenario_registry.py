from __future__ import annotations

import unittest

from benchmarks.scenario_registry import (
    make_research_benchmark_suite,
    make_smoke_benchmark_suite,
    research_benchmark_scenarios,
)


class ScenarioRegistryTests(unittest.TestCase):
    def test_research_registry_has_eight_named_scenarios(self) -> None:
        scenarios = research_benchmark_scenarios()
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(scenarios[0]["name"], "default_4pos")
        self.assertEqual(scenarios[-1]["name"], "sparse_high_slip_5pos")

    def test_suite_builders_use_expected_scenario_counts(self) -> None:
        research_suite = make_research_benchmark_suite()
        smoke_suite = make_smoke_benchmark_suite()
        self.assertEqual(len(research_suite["scenarios"]), 8)
        self.assertEqual(len(smoke_suite["scenarios"]), 2)


if __name__ == "__main__":
    unittest.main()
