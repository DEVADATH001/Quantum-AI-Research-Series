"""Evaluation Metrics Module.

This module provides comprehensive evaluation metrics for comparing
classical and quantum SVM models.

Author: Quantum ML Research Lab
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: Optional[int] = None,
) -> dict[str, float]:
    """Compute evaluation metrics for classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multiclass
        pos_label: Positive class label for binary classification.
                   If None, uses the largest observed class label.

    Returns:
        Dictionary of metrics
    """
    if pos_label is None and average == "binary":
        labels = np.unique(np.concatenate([y_true, y_pred]))
        pos_label = int(labels.max())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        ),
    }
    
    logger.info(f"Metrics computed: {metrics}")
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[int]] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names

    Returns:
        Confusion matrix
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    logger.info(f"Confusion matrix:\n{cm}")
    
    return cm


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
) -> str:
    """Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names

    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    
    logger.info(f"Classification Report:\n{report}")
    
    return report


def evaluate_models(
    classical_model_results: dict[str, Any],
    quantum_model_results: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[list[str]] = None,
    pos_label: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate both classical and quantum models.

    Args:
        classical_model_results: Results from classical model training
                                 Must contain 'model' with predict method
        quantum_model_results: Results from quantum model training
                               Must contain 'model' with predict method
        X_test: Test features for prediction
        y_test: True test labels
        target_names: Optional class names
        pos_label: Positive class label for binary metrics

    Returns:
        Dictionary containing all evaluation results
    """
    logger.info("Evaluating models...")
    
    # Get predictions from classical model
    classical_model = classical_model_results["model"]
    classical_X = classical_model_results.get("X_test", X_test)
    classical_pred = classical_model.predict(classical_X)
    
    # Get predictions from quantum model
    quantum_model = quantum_model_results["model"]
    quantum_X = quantum_model_results.get("X_test", X_test)
    quantum_pred = quantum_model.predict(quantum_X)
    
    # Compute metrics for classical model
    classical_metrics = compute_metrics(y_test, classical_pred, pos_label=pos_label)
    classical_cm = compute_confusion_matrix(y_test, classical_pred)
    
    # Compute metrics for quantum model
    quantum_metrics = compute_metrics(y_test, quantum_pred, pos_label=pos_label)
    quantum_cm = compute_confusion_matrix(y_test, quantum_pred)
    
    # Compile results
    results = {
        "classical": {
            "metrics": classical_metrics,
            "confusion_matrix": classical_cm,
            "predictions": classical_pred,
            "train_time": classical_model_results.get("train_time", None),
        },
        "quantum": {
            "metrics": quantum_metrics,
            "confusion_matrix": quantum_cm,
            "predictions": quantum_pred,
            "train_time": quantum_model_results.get("train_time", None),
        },
        "test_labels": y_test,
        "target_names": target_names,
    }
    
    logger.info("Model evaluation completed")
    
    return results


def create_metrics_dataframe(
    results: dict[str, Any],
) -> pd.DataFrame:
    """Create a pandas DataFrame from evaluation results.

    Args:
        results: Results from evaluate_models

    Returns:
        DataFrame with metrics comparison
    """
    metrics_data = {
        "Model": ["Classical RBF SVM", "Quantum Kernel SVM"],
        "Accuracy": [
            results["classical"]["metrics"]["accuracy"],
            results["quantum"]["metrics"]["accuracy"],
        ],
        "Precision": [
            results["classical"]["metrics"]["precision"],
            results["quantum"]["metrics"]["precision"],
        ],
        "Recall": [
            results["classical"]["metrics"]["recall"],
            results["quantum"]["metrics"]["recall"],
        ],
        "F1 Score": [
            results["classical"]["metrics"]["f1_score"],
            results["quantum"]["metrics"]["f1_score"],
        ],
    }
    
    df = pd.DataFrame(metrics_data)
    
    logger.info(f"Metrics DataFrame:\n{df}")
    
    return df


def save_metrics_to_csv(
    results: dict[str, Any],
    filepath: str,
) -> None:
    """Save metrics to CSV file.

    Args:
        results: Results from evaluate_models
        filepath: Path to save CSV file
    """
    df = create_metrics_dataframe(results)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Metrics saved to {filepath}")


def compute_metric_differences(
    results: dict[str, Any],
) -> dict[str, float]:
    """Compute differences between classical and quantum metrics.

    Args:
        results: Results from evaluate_models

    Returns:
        Dictionary of metric differences (quantum - classical)
    """
    classical = results["classical"]["metrics"]
    quantum = results["quantum"]["metrics"]
    
    differences = {
        "accuracy_diff": quantum["accuracy"] - classical["accuracy"],
        "precision_diff": quantum["precision"] - classical["precision"],
        "recall_diff": quantum["recall"] - classical["recall"],
        "f1_score_diff": quantum["f1_score"] - classical["f1_score"],
    }
    
    logger.info(f"Metric differences (Q - C): {differences}")
    
    return differences

