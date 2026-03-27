from __future__ import annotations

import unittest

import numpy as np

from src.rl_utils import (
    baseline_adjusted_returns,
    discounted_returns,
    generalized_advantage_estimation,
    update_timestep_baseline,
)


class RLUtilsTests(unittest.TestCase):
    def test_discounted_returns(self) -> None:
        returns = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
        np.testing.assert_allclose(returns, np.array([2.75, 3.5, 3.0]))

    def test_timestep_baseline_adjustment_and_update(self) -> None:
        baseline = np.array([1.0, 0.5, 0.0], dtype=float)
        returns = np.array([2.0, 1.5], dtype=float)
        advantages = baseline_adjusted_returns(baseline, returns, decay=0.9)
        np.testing.assert_allclose(advantages, np.array([1.0, 1.0]))

        update_timestep_baseline(baseline, returns, decay=0.5)
        np.testing.assert_allclose(baseline[:2], np.array([1.5, 1.0]))

    def test_no_baseline_decay_returns_copy(self) -> None:
        returns = np.array([0.2, 0.4], dtype=float)
        adjusted = baseline_adjusted_returns(np.zeros(4), returns, decay=None)
        np.testing.assert_allclose(adjusted, returns)

    def test_generalized_advantage_estimation(self) -> None:
        advantages, returns = generalized_advantage_estimation(
            rewards=[1.0, 0.5],
            values=[0.2, 0.1],
            gamma=0.9,
            gae_lambda=0.95,
            bootstrap_value=0.0,
        )
        np.testing.assert_allclose(returns, np.array([1.432, 0.5]), atol=1e-8)
        self.assertEqual(advantages.shape, returns.shape)
        self.assertTrue(np.all(np.isfinite(advantages)))


if __name__ == "__main__":
    unittest.main()
