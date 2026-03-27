from __future__ import annotations

import unittest

import numpy as np

from src.research_stats import hodges_lehmann_estimate, holm_correct, paired_method_comparison


class ResearchStatsTests(unittest.TestCase):
    def test_hodges_lehmann_estimate_is_finite(self) -> None:
        estimate = hodges_lehmann_estimate([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(estimate), 2.0)

    def test_holm_correction_is_monotone(self) -> None:
        corrected = holm_correct({"a": 0.01, "b": 0.02, "c": 0.2})
        self.assertLessEqual(corrected["a"], corrected["b"])
        self.assertLessEqual(corrected["b"], corrected["c"])

    def test_paired_method_comparison_reports_effect_sizes(self) -> None:
        left = [
            {"seed": 7, "evaluation": {"success_rate": 0.8}},
            {"seed": 21, "evaluation": {"success_rate": 0.9}},
        ]
        right = [
            {"seed": 7, "evaluation": {"success_rate": 0.6}},
            {"seed": 21, "evaluation": {"success_rate": 0.7}},
        ]
        comparison = paired_method_comparison(
            left,
            right,
            metric_name="eval_success",
            metric_fn=lambda record: record["evaluation"]["success_rate"],
        )
        self.assertTrue(np.isfinite(comparison["mean_difference"]))
        self.assertTrue(np.isfinite(comparison["hodges_lehmann_difference"]))
        self.assertEqual(comparison["wins_a"], 2)


if __name__ == "__main__":
    unittest.main()
