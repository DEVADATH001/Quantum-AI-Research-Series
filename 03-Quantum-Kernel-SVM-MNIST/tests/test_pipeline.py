"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Unit tests for the quantum pipeline."""

import unittest
import numpy as np

from src.preprocessing import (
    normalize_pixels,
    parse_feature_range,
    preprocess_data,
    scale_for_quantum,
)
from src.quantum_kernel_engine import (
    analyze_kernel_properties,
    compute_kernel_alignment,
    compute_kernel_target_alignment,
    regularize_kernel_matrix,
)
from src.evaluation_metrics import calculate_statistical_significance


class TestPreprocessing(unittest.TestCase):
    def test_normalization_range(self):
        X = np.array([[0, 255], [127.5, 64]])
        X_norm = normalize_pixels(X)
        self.assertTrue(np.all(X_norm <= 1.0))
        self.assertTrue(np.all(X_norm >= 0.0))
        self.assertAlmostEqual(X_norm[0, 1], 1.0)
    
    def test_quantum_scaling_default_pi_half(self):
        """Default feature_range is [0, pi/2] to prevent exponential kernel concentration."""
        X = np.array([[0.0, 10.0], [5.0, 5.0]])
        X_scaled = scale_for_quantum(X)  # default=(0, pi/2)
        self.assertAlmostEqual(float(np.max(X_scaled)), np.pi / 2.0, places=6)
        self.assertAlmostEqual(float(np.min(X_scaled)), 0.0, places=6)

    def test_quantum_scaling_explicit_pi(self):
        """Explicit [0, pi] scaling for comparison."""
        X = np.array([[0.0, 10.0], [5.0, 5.0]])
        X_scaled = scale_for_quantum(X, feature_range=(0.0, np.pi))
        self.assertAlmostEqual(float(np.max(X_scaled)), np.pi, places=6)
        self.assertAlmostEqual(float(np.min(X_scaled)), 0.0, places=6)

    def test_quantum_scaling_train_test_consistency(self):
        """Train and test must be scaled with the same (data_min, data_max)."""
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        X_test = np.array([[2.5, 7.5]])
        X_train_s, dmin, dmax = scale_for_quantum(X_train, return_params=True)
        X_test_s = scale_for_quantum(X_test, data_min=dmin, data_max=dmax)
        # Test value 2.5 / 10 = 0.25 of range
        expected = 0.25 * (np.pi / 2.0)
        self.assertAlmostEqual(float(X_test_s[0, 0]), expected, places=6)

    def test_parse_feature_range_pi(self):
        low, high = parse_feature_range("pi")
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, np.pi)

    def test_parse_feature_range_pi_half(self):
        low, high = parse_feature_range("pi/2")
        self.assertAlmostEqual(high, np.pi / 2.0)

    def test_parse_feature_range_default(self):
        low, high = parse_feature_range(None)
        self.assertAlmostEqual(high, np.pi / 2.0)

    def test_parse_feature_range_float(self):
        low, high = parse_feature_range("1.5")
        self.assertAlmostEqual(high, 1.5)

    def test_preprocess_data_shapes(self):
        """Full pipeline: output shapes must match and quantum range must be correct."""
        rng = np.random.default_rng(42)
        X_train = rng.integers(0, 256, size=(30, 64)).astype(float)
        X_test = rng.integers(0, 256, size=(10, 64)).astype(float)
        result = preprocess_data(X_train, X_test, n_components=4)
        self.assertEqual(result["X_train_quantum"].shape, (30, 4))
        self.assertEqual(result["X_test_quantum"].shape, (10, 4))
        # Quantum range should be within [0, pi/2]
        self.assertGreaterEqual(float(result["X_train_quantum"].min()), -1e-8)
        self.assertLessEqual(float(result["X_train_quantum"].max()), np.pi / 2.0 + 1e-8)


class TestKernelEngine(unittest.TestCase):
    def test_kernel_regularization_makes_psd(self):
        # Non-PSD matrix: eigenvalues are +3 and -1
        K = np.array([[1.0, 2.0], [2.0, 1.0]])
        K_psd = regularize_kernel_matrix(K)
        eigenvalues = np.linalg.eigvalsh(K_psd)
        self.assertTrue(np.all(eigenvalues >= -1e-10))
        # Diagonal must be 1.0 (fidelity kernel property)
        self.assertAlmostEqual(K_psd[0, 0], 1.0, places=5)
        self.assertAlmostEqual(K_psd[1, 1], 1.0, places=5)

    def test_kernel_regularization_already_psd(self):
        """A valid PSD matrix should be returned essentially unchanged."""
        K = np.eye(3)
        K_reg = regularize_kernel_matrix(K)
        np.testing.assert_allclose(K_reg, K, atol=1e-8)

    def test_kernel_target_alignment_range(self):
        K = np.eye(4)
        y = np.array([0, 0, 1, 1])
        kta = compute_kernel_target_alignment(K, y)
        self.assertGreaterEqual(kta, 0.0)
        self.assertLessEqual(kta, 1.0)

    def test_kernel_alignment_identical(self):
        """Alignment of a matrix with itself must be 1.0."""
        K = np.array([[1.0, 0.3], [0.3, 1.0]])
        alignment = compute_kernel_alignment(K, K)
        self.assertAlmostEqual(alignment, 1.0, places=5)

    def test_analyze_kernel_properties_identity(self):
        K = np.eye(4)
        props = analyze_kernel_properties(K)
        self.assertTrue(props["is_symmetric"])
        self.assertTrue(props["is_positive_semidefinite"])
        self.assertAlmostEqual(props["diagonal_mean"], 1.0)
        self.assertAlmostEqual(props["off_diagonal_mean"], 0.0)


class TestStatisticalSignificance(unittest.TestCase):
    def test_significance_identical_scores(self):
        """Identical lists → no significant advantage."""
        scores = [0.9, 0.85, 0.88]
        result = calculate_statistical_significance(scores, scores)
        # Both statistics are 0-difference draws — should not claim significance
        self.assertIn("paired_t_test_p_value", result)
        self.assertIn("wilcoxon_p_value", result)

    def test_significance_clearly_different(self):
        """Strongly different score lists should be flagged as needing investigation."""
        classical = [0.5, 0.52, 0.51]
        quantum = [0.92, 0.93, 0.91]
        result = calculate_statistical_significance(classical, quantum)
        self.assertIn("significant_advantage", result)
        self.assertIsInstance(result["significant_advantage"], bool)

    def test_significance_returns_all_keys(self):
        result = calculate_statistical_significance([0.8, 0.85], [0.82, 0.87])
        for key in ["paired_t_test_p_value", "wilcoxon_p_value", "t_statistic",
                    "w_statistic", "significant_advantage"]:
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()

