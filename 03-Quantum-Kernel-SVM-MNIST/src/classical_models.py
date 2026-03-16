"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Classical Model Module."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_kernels

logger = logging.getLogger(__name__)

def train_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    random_state: int = 42,
    grid_search: bool = False,
    cv: int = 3,
    param_grid: Optional[dict[str, list]] = None,
    n_jobs: int = -1,
) -> tuple[SVC, dict[str, Any]]:
    """Train a classical SVM classifier.

    This function trains a Support Vector Machine with RBF kernel,
    optionally using GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Kernel type (default: "rbf")
        C: Regularization parameter
        gamma: Kernel coefficient
        random_state: Random seed
        grid_search: Whether to use GridSearchCV
        cv: Number of cross-validation folds
        param_grid: Parameter grid for GridSearchCV
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Tuple of (trained_model, results_dict)

    Example:
        >>> clf, results = train_classical_svm(
        ...     X_train, y_train,
        ...     grid_search=True,
        ...     param_grid={"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
        ... )
    """
    logger.info("Training classical SVM...")
    
    if grid_search:
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.1, 0.01],
            }
        
        logger.info(f"GridSearchCV with param_grid: {param_grid}")
        
        base_clf = SVC(kernel=kernel, random_state=random_state)
        
        grid_search_cv = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=1,
        )

        start_time = time.time()
        try:
            grid_search_cv.fit(X_train, y_train)
        except PermissionError:
            # Some restricted environments block multiprocessing pipes.
            logger.warning(
                "GridSearchCV multiprocessing failed; retrying with n_jobs=1."
            )
            grid_search_cv = GridSearchCV(
                estimator=base_clf,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                verbose=1,
            )
            grid_search_cv.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_model = grid_search_cv.best_estimator_
        best_params = grid_search_cv.best_params_
        best_cv_score = grid_search_cv.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_cv_score:.4f}")
        logger.info(f"Training time: {train_time:.3f}s")
        
        results = {
            "model": best_model,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "train_time": train_time,
            "grid_search_results": grid_search_cv.cv_results_,
        }
        
        return best_model, results
    else:
        # Train with specified parameters
        clf = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
        )
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        logger.info(f"Training time: {train_time:.3f}s")
        
        results = {
            "model": clf,
            "params": {"C": C, "gamma": gamma, "kernel": kernel},
            "train_time": train_time,
        }
        
        return clf, results

def get_svm_decision_function(
    model: SVC,
    X: np.ndarray,
) -> np.ndarray:
    """Get the decision function values for samples.

    Args:
        trained_svm: Trained SVC model
        X: Input samples

    Returns:
        Decision function values
    """
    return model.decision_function(X)

def get_support_vectors(
    model: SVC,
) -> np.ndarray:
    """Get the support vectors from a trained SVM.

    Args:
        model: Trained SVC model

    Returns:
        Support vectors
    """
    return model.support_vectors_

def compute_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> np.ndarray:
    """Compute kernel matrix between two datasets.

    This is useful for comparing with quantum kernel matrices.

    Args:
        X1: First feature matrix
        X2: Second feature matrix
        kernel: Kernel type
        gamma: Kernel coefficient

    Returns:
        Kernel matrix K where K[i,j] = k(X1[i], X2[j])
    """
    params: dict[str, Any] = {}
    if kernel in {"rbf", "poly", "sigmoid"}:
        params["gamma"] = gamma

    return pairwise_kernels(X1, X2, metric=kernel, filter_params=True, **params)

