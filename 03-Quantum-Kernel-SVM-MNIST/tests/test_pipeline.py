"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Unit tests for the quantum pipeline."""

import unittest
import numpy as np
from src.preprocessing import normalize_pixels, scale_for_quantum
from src.quantum_kernel_engine import regularize_kernel_matrix, compute_kernel_target_alignment

class TestQuantumPipeline(unittest.TestCase):
    def test_normalization(self):
        X = np.array([[0, 255], [127.5, 64]])
        X_norm = normalize_pixels(X)
        self.assertTrue(np.all(X_norm <= 1.0))
        self.assertTrue(np.all(X_norm >= 0.0))
        self.assertEqual(X_norm[0, 1], 1.0)

    def test_quantum_scaling(self):
        X = np.array([[0, 10], [5, 5]])
        X_scaled = scale_for_quantum(X)
        self.assertAlmostEqual(np.max(X_scaled), np.pi)
        self.assertAlmostEqual(np.min(X_scaled), 0.0)

    def test_kernel_regularization(self):
        # Create a non-PSD matrix
        K = np.array([[1.0, 2.0], [2.0, 1.0]]) 
        # Eigenvalues are 3 and -1
        K_psd = regularize_kernel_matrix(K)
        eigenvalues = np.linalg.eigvalsh(K_psd)
        self.assertTrue(np.all(eigenvalues >= 0))
        self.assertAlmostEqual(K_psd[0, 0], 1.0)

    def test_kta(self):
        K = np.eye(4)
        y = np.array([0, 0, 1, 1])
        kta = compute_kernel_target_alignment(K, y)
        self.assertGreaterEqual(kta, 0.0)
        self.assertLessEqual(kta, 1.0)

if __name__ == '__main__':
    unittest.main()
