"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Training helpers for Pegasos quantum kernel SVM."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Optional

import numpy as np

try:
    from qiskit_machine_learning.algorithms import PegasosQSVC, QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QML_AVAILABLE = True
except ImportError as e:  # pragma: no cover - environment-specific
    QML_AVAILABLE = False
    PegasosQSVC = None  # type: ignore[assignment]
    QSVC = None  # type: ignore[assignment]
    FidelityQuantumKernel = None  # type: ignore[assignment]
    QML_IMPORT_ERROR = e

from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def create_pegasos_qsvc(
    quantum_kernel: FidelityQuantumKernel,
    lambda_param: float = 1.0,
    max_iter: int = 1000,
    batch_size: int = 100,
    num_samples: Optional[int] = None,
    random_state: int = 42,
    precomputed: bool = False,
) -> PegasosQSVC:
    """Create PegasosQSVC while handling API differences across Qiskit ML versions.

    When ``precomputed=True`` the current Qiskit ML API requires
    ``quantum_kernel=None`` — the kernel has already been evaluated and the
    model receives a plain numpy kernel matrix instead.
    """
    if not QML_AVAILABLE:
        raise ImportError(
            "Qiskit Machine Learning is required for PegasosQSVC. "
            f"Import error: {QML_IMPORT_ERROR}"
        )

    # Newer Qiskit ML: quantum_kernel must be None when precomputed=True.
    qk_arg: FidelityQuantumKernel | None = None if precomputed else quantum_kernel

    signature = inspect.signature(PegasosQSVC.__init__)
    params = signature.parameters

    logger.info(
        "Creating PegasosQSVC with lambda=%s max_iter=%s batch_size=%s precomputed=%s",
        lambda_param,
        max_iter,
        batch_size,
        precomputed,
    )

    if "lambda_param" in params:
        return PegasosQSVC(
            quantum_kernel=qk_arg,
            lambda_param=lambda_param,
            max_iter=max_iter,
            batch_size=batch_size,
            num_samples=num_samples,
            random_state=random_state,
            precomputed=precomputed,
        )

    if lambda_param <= 0:
        raise ValueError("lambda_param must be > 0")

    # Mathematically align Pegasos regularization parameter with SVM `C` parameter.
    if num_samples is not None and num_samples > 0:
        C = 1.0 / (lambda_param * num_samples)
    else:
        logger.warning(
            "num_samples not provided. Defaulting to unscaled C = 1 / lambda."
        )
        C = 1.0 / lambda_param

    if batch_size != 100 or num_samples is not None:
        logger.info(
            "Current PegasosQSVC API uses num_steps for iterations. "
            "Batch size and num_samples apply fundamentally to optimization scaling."
        )

    return PegasosQSVC(
        quantum_kernel=qk_arg,
        C=C,
        num_steps=max_iter,
        seed=random_state,
        precomputed=precomputed,
    )


def train_qsvc(
    quantum_kernel: FidelityQuantumKernel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
    grid_search: bool = False,
    cv: int = 3,
    n_jobs: int = -1,
) -> tuple[QSVC, dict[str, Any]]:
    """Train an exact QSVC and return model plus metadata."""
    if not QML_AVAILABLE:
        raise ImportError(
            "Qiskit Machine Learning is required. "
            f"Import error: {QML_IMPORT_ERROR}"
        )

    if grid_search:
        logger.info("Training exact QSVC with GridSearchCV on %s samples", len(X_train))
        param_grid = {"C": [0.1, 1, 10, 100]}
        base_model = QSVC(quantum_kernel=quantum_kernel, random_state=random_state)
        
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        model = search.best_estimator_
        results = {
            "model": model,
            "train_time": train_time,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "grid_search_results": search.cv_results_,
        }
        logger.info("QSVC GridSearchCV completed in %.3fs. Best params: %s", train_time, search.best_params_)
    else:
        logger.info("Training exact QSVC on %s samples with C=%s", len(X_train), C)

        model = QSVC(quantum_kernel=quantum_kernel, C=C, random_state=random_state)

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        results = {
            "model": model,
            "train_time": train_time,
            "params": {
                "C": C,
            },
        }

        logger.info("QSVC training completed in %.3fs", train_time)
        
    return model, results

def train_pegasos_qsvc(
    quantum_kernel: FidelityQuantumKernel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_param: float = 1.0,
    max_iter: int = 1000,
    batch_size: int = 100,
    random_state: int = 42,
    precomputed: bool = False,
) -> tuple[PegasosQSVC, dict[str, Any]]:
    """Train PegasosQSVC and return model plus metadata."""
    if not QML_AVAILABLE:
        raise ImportError(
            "Qiskit Machine Learning is required. "
            f"Import error: {QML_IMPORT_ERROR}"
        )

    logger.info("Training PegasosQSVC on %s samples (precomputed=%s)", len(X_train), precomputed)

    model = create_pegasos_qsvc(
        quantum_kernel=quantum_kernel,
        lambda_param=lambda_param,
        max_iter=max_iter,
        batch_size=batch_size,
        num_samples=len(X_train),
        random_state=random_state,
        precomputed=precomputed,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    n_support_vectors = (
        getattr(model, "number_of_support_vectors_", None)
        or getattr(model, "n_support_", None)
    )
    support_vector_indices = getattr(model, "support_vector_indices_", None)

    results = {
        "model": model,
        "train_time": train_time,
        "n_support_vectors": n_support_vectors,
        "support_vector_indices": support_vector_indices,
        "params": {
            "lambda_param": lambda_param,
            "max_iter": max_iter,
            "batch_size": batch_size,
        },
    }

    logger.info("PegasosQSVC training completed in %.3fs", train_time)
    return model, results

def predict_with_qsvc(model: PegasosQSVC, X_test: np.ndarray) -> np.ndarray:
    """Predict labels with a trained PegasosQSVC model."""
    return model.predict(X_test)

def get_qsvc_decision_scores(model: PegasosQSVC, X: np.ndarray) -> np.ndarray:
    """Return decision function scores."""
    return model.decision_function(X)

def describe_pegasos_algorithm() -> str:
    """Return a short algorithm note."""
    return (
        "\nPegasos Quantum SVM\n"
        "===================\n\n"
        "Pegasos is a stochastic optimizer for SVM objectives. For modern\n"
        "Qiskit ML versions this is exposed as PegasosQSVC(C, num_steps).\n"
    )

