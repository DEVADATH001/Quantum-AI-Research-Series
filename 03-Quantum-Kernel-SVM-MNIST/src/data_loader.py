"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: MNIST Data Loader Module."""

from __future__ import annotations

import logging
import ssl
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Windows SSL workaround: create an unverified context for OpenML fetches.
# This is scoped only to the fetch call and does not affect global SSL behaviour.
_SSL_UNVERIFIED_CONTEXT: ssl.SSLContext | None = None
try:
    _SSL_UNVERIFIED_CONTEXT = ssl._create_unverified_context()  # noqa: SLF001
except AttributeError:
    pass

def load_mnist_digits(
    digits: Optional[list[int]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
    data_home: Optional[str] = None,
    fallback_to_sklearn_digits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset and filter for specified digits.

    This function loads the MNIST dataset using sklearn's fetch_openml,
    filters for the specified digits (default: 4 and 9), and splits
    into training and test sets.

    Args:
        digits: List of digits to keep for classification.
                Default is [4, 9] for challenging binary classification.
        test_size: Proportion of data to use for testing.
                   Default is 0.25 (25% test, 75% train).
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify split by class labels.
        data_home: Directory to store OpenML cache.
                   Defaults to a writable local project cache.
        fallback_to_sklearn_digits: If True, falls back to sklearn's built-in
                                    digits dataset when OpenML is unavailable.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test):
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels

    Example:
        >>> X_train, X_test, y_train, y_test = load_mnist_digits(
        ...     digits=[4, 9],
        ...     test_size=0.25,
        ...     random_state=42
        ... )
        >>> print(f"Training samples: {len(X_train)}")
        Training samples: 1128
    """
    if digits is None:
        digits = [4, 9]

    logger.info("Loading MNIST dataset...")

    if data_home is None:
        data_home = str(Path(".cache") / "sklearn")

    X: np.ndarray
    y: np.ndarray

    # ------------------------------------------------------------------
    # Fast path: offline / fallback mode — skip OpenML entirely
    # ------------------------------------------------------------------
    if fallback_to_sklearn_digits:
        logger.info("Fallback mode: loading sklearn built-in digits dataset (no network).")
        digits_dataset = load_digits()
        X = digits_dataset.data
        y = digits_dataset.target.astype(int)
        logger.info("Dataset source: sklearn load_digits (8x8 images)")
    else:
        # ------------------------------------------------------------------
        # Standard path: fetch full MNIST from OpenML
        # On Windows, SSL verification sometimes blocks the download; try with
        # an unverified context first (safe for read-only data fetching).
        # ------------------------------------------------------------------
        _fetch_kwargs = dict(
            name="mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
        try:
            if _SSL_UNVERIFIED_CONTEXT is not None:
                _old = getattr(ssl, "_create_default_https_context", None)
                ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
                try:
                    mnist = fetch_openml(**_fetch_kwargs)
                finally:
                    if _old is not None:
                        ssl._create_default_https_context = _old  # noqa: SLF001
            else:
                mnist = fetch_openml(**_fetch_kwargs)
        except Exception:
            # Retry without SSL override in case the override itself caused issues
            mnist = fetch_openml(**_fetch_kwargs)

        X = mnist.data
        y = mnist.target.astype(int)
        logger.info("Dataset source: OpenML mnist_784")

    logger.info(f"Total dataset samples: {len(X)}")
    
    # Filter for specified digits
    mask = np.isin(y, digits)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    logger.info(f"Filtered samples (digits {digits}): {len(X_filtered)}")
    logger.info(f"Class distribution: {np.bincount(y_filtered)}")
    
    # Split data
    stratify_param = y_filtered if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Training class distribution: {np.bincount(y_train)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def get_digit_statistics(
    X: np.ndarray,
    y: np.ndarray,
    digits: Optional[list[int]] = None,
) -> dict[str, float]:
    """Compute statistics for the dataset.

    Args:
        X: Feature matrix
        y: Label vector
        digits: List of digit classes

    Returns:
        Dictionary containing dataset statistics
    """
    if digits is None:
        digits = [4, 9]

    stats = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(digits),
        "classes": digits,
        "pixel_mean": float(np.mean(X)),
        "pixel_std": float(np.std(X)),
        "pixel_min": float(np.min(X)),
        "pixel_max": float(np.max(X)),
    }
    
    # Per-class statistics
    for digit in digits:
        mask = y == digit
