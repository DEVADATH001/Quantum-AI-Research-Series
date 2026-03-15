"""Data preprocessing utilities for the MNIST kernel experiments."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range."""
    return X / 255.0


def standardize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> tuple:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


def apply_pca(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_components: int = 4,
    random_state: int = 42,
) -> tuple:
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    logger.info("PCA: Reduced to %s components", n_components)
    logger.info("Variance explained per component: %s", variance_explained)
    logger.info("Cumulative variance explained: %s", cumulative_variance)

    if X_test is not None:
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca, pca

    return X_train_pca, pca


def scale_for_quantum(
    X: np.ndarray,
    feature_range: tuple[float, float] = (0.0, np.pi),
    data_min: Optional[np.ndarray] = None,
    data_max: Optional[np.ndarray] = None,
    return_params: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale features for angle encoding in a quantum feature map.

    If `data_min`/`data_max` are provided, they are used directly so train/test
    scaling can stay consistent.
    """
    if data_min is None:
        data_min = X.min(axis=0)
    if data_max is None:
        data_max = X.max(axis=0)

    low, high = feature_range
    span = high - low
    X_scaled = low + span * (X - data_min) / (data_max - data_min + 1e-12)

    logger.info("Scaled features to range [%s, %s]", feature_range[0], feature_range[1])

    if return_params:
        return X_scaled, data_min, data_max
    return X_scaled


def preprocess_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 4,
    random_state: int = 42,
    quantum_scaling: bool = True,
) -> dict[str, object]:
    """Run the full preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline...")

    X_train_norm = normalize_pixels(X_train)
    X_test_norm = normalize_pixels(X_test)
    logger.info("Step 1: Normalized pixel values")

    X_train_std, X_test_std, scaler = standardize_features(X_train_norm, X_test_norm)
    logger.info("Step 2: Standardized features")

    X_train_pca, X_test_pca, pca = apply_pca(X_train_std, X_test_std, n_components, random_state)
    logger.info("Step 3: Applied PCA (n_components=%s)", n_components)

    if quantum_scaling:
        X_train_quantum, q_min, q_max = scale_for_quantum(X_train_pca, return_params=True)
        X_test_quantum = scale_for_quantum(X_test_pca, data_min=q_min, data_max=q_max)
        logger.info("Step 4: Scaled for quantum encoding [0, pi]")
    else:
        X_train_quantum = X_train_pca
        X_test_quantum = X_test_pca
        q_min = None
        q_max = None

    return {
        "X_train_processed": X_train_pca,
        "X_test_processed": X_test_pca,
        "X_train_quantum": X_train_quantum,
        "X_test_quantum": X_test_quantum,
        "scaler": scaler,
        "pca": pca,
        "variance_explained": pca.explained_variance_ratio_,
        "quantum_scale_min": q_min,
        "quantum_scale_max": q_max,
    }


def get_preprocessing_summary(preprocessed_data: dict) -> str:
    """Generate a formatted preprocessing summary."""
    variance = np.asarray(preprocessed_data["variance_explained"])
    cumulative_variance = np.cumsum(variance)

    variance_lines = []
    for idx, var in enumerate(variance, start=1):
        variance_lines.append(
            f"  - PC{idx}: {var:.4f} (cumulative: {cumulative_variance[idx - 1]:.4f})"
        )

    return (
        "\nPreprocessing Summary\n"
        "====================\n"
        f"PCA Components: {len(variance)}\n"
        "Variance Explained:\n"
        f"{chr(10).join(variance_lines)}\n"
        f"Total Variance Explained: {cumulative_variance[-1]:.4f}\n"
        f"Training samples: {len(preprocessed_data['X_train_processed'])}\n"
        f"Test samples: {len(preprocessed_data['X_test_processed'])}\n"
    )
