"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Evaluation Metrics Module — publication-grade statistics.

Upgrades (v2):
  - bootstrap_confidence_interval: 95% Bootstrap CI over multi-seed scores.
  - cohens_d: standardised effect size for paired comparisons.
  - bonferroni_correct: family-wise error rate correction for multiple tests.
  - calculate_statistical_significance: now includes d and CI alongside p-values.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils import resample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core per-trial metrics
# ---------------------------------------------------------------------------

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
    """Generate detailed classification report."""
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    logger.info(f"Classification Report:\n{report}")
    return report


# ---------------------------------------------------------------------------
# Multi-result dataframe / CSV helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def evaluate_models(
    classical_model_results: dict[str, Any],
    quantum_model_results: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[list[str]] = None,
    pos_label: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate both classical and quantum models."""
    logger.info("Evaluating models...")

    classical_model = classical_model_results["model"]
    classical_X = classical_model_results.get("X_test", X_test)
    classical_pred = classical_model.predict(classical_X)

    quantum_model = quantum_model_results["model"]
    quantum_X = quantum_model_results.get("X_test", X_test)
    quantum_pred = quantum_model.predict(quantum_X)

    classical_metrics = compute_metrics(y_test, classical_pred, pos_label=pos_label)
    classical_cm = compute_confusion_matrix(y_test, classical_pred)

    quantum_metrics = compute_metrics(y_test, quantum_pred, pos_label=pos_label)
    quantum_cm = compute_confusion_matrix(y_test, quantum_pred)

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


def create_metrics_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame from evaluate_models results."""
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


def save_metrics_to_csv(results: dict[str, Any], filepath: str) -> None:
    """Save metrics to CSV file."""
    df = create_metrics_dataframe(results)
    df.to_csv(filepath, index=False)
    logger.info(f"Metrics saved to {filepath}")


def compute_metric_differences(results: dict[str, Any]) -> dict[str, float]:
    """Compute quantum − classical metric differences."""
    classical = results["classical"]["metrics"]
    quantum = results["quantum"]["metrics"]
    return {
        "accuracy_diff": quantum["accuracy"] - classical["accuracy"],
        "precision_diff": quantum["precision"] - classical["precision"],
        "recall_diff": quantum["recall"] - classical["recall"],
        "f1_score_diff": quantum["f1_score"] - classical["f1_score"],
    }


# ---------------------------------------------------------------------------
# Publication-grade statistics (Phase 3)
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    scores: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean of *scores*.

    Args:
        scores: Observed metric values across seeds/folds.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return float(np.mean(arr)), lo, hi


def cohens_d(
    group_a: list[float] | np.ndarray,
    group_b: list[float] | np.ndarray,
) -> float:
    """Compute Cohen's d effect size: (mean_a − mean_b) / pooled_std.

    Convention: positive d means group_a > group_b.

    Interpretation:
        |d| < 0.2  → negligible
        |d| ≈ 0.5  → medium
        |d| ≥ 0.8  → large
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    pooled_var = (
        (n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)
    ) / (n_a + n_b - 2)
    pooled_std = float(np.sqrt(max(pooled_var, 1e-15)))
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Return a human-readable effect-size label for Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict:
    """Apply Bonferroni correction to a list of p-values.

    Args:
        p_values: Uncorrected p-values from m simultaneous tests.
        alpha: Family-wise error rate target.

    Returns:
        dict with keys:
            corrected_alpha: alpha / m.
            reject: list[bool] — True if the null is rejected after correction.
            n_significant: int.
            p_values: original list.
    """
    m = len(p_values)
    if m == 0:
        return {"corrected_alpha": alpha, "reject": [], "n_significant": 0, "p_values": []}

    corrected_alpha = alpha / m
    reject = [p < corrected_alpha for p in p_values]
    logger.info(
        "Bonferroni correction: m=%d tests, α_corrected=%.4f, "
        "significant=%d/%d",
        m, corrected_alpha, sum(reject), m,
    )
    return {
        "corrected_alpha": float(corrected_alpha),
        "reject": reject,
        "n_significant": int(sum(reject)),
        "p_values": [float(p) for p in p_values],
    }


def calculate_statistical_significance(
    classical_scores: list[float],
    quantum_scores: list[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Full statistical comparison between classical and quantum score lists.

    Includes:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d effect size with interpretation
    - Bootstrap 95% CI for both score sets
    - Significance flag (p < 0.05 on either test, uncorrected)

    Args:
        classical_scores: List of classical F1 scores across seeds.
        quantum_scores: List of corresponding quantum scores (same order).
        n_bootstrap: Bootstrap resamples for CI estimation.
        seed: RNG seed.

    Returns:
        Dictionary of statistical measures.
    """
    try:
        t_stat, t_p_value = stats.ttest_rel(classical_scores, quantum_scores)
    except Exception:
        t_stat, t_p_value = float("nan"), float("nan")

    try:
        w_stat, w_p_value = stats.wilcoxon(classical_scores, quantum_scores)
    except Exception:
        w_stat, w_p_value = float("nan"), float("nan")

    d = cohens_d(quantum_scores, classical_scores)  # quantum − classical

    c_mean, c_lo, c_hi = bootstrap_confidence_interval(
        classical_scores, n_bootstrap=n_bootstrap, seed=seed
    )
    q_mean, q_lo, q_hi = bootstrap_confidence_interval(
        quantum_scores, n_bootstrap=n_bootstrap, seed=seed
    )

    sig_flag = (
        bool(t_p_value < 0.05 or w_p_value < 0.05)
        if not np.isnan(t_p_value)
        else False
    )

    return {
        "paired_t_test_p_value": float(t_p_value),
        "wilcoxon_p_value": float(w_p_value),
        "t_statistic": float(t_stat),
        "w_statistic": float(w_stat),
        "significant_advantage": sig_flag,
        "cohens_d": float(d) if not np.isnan(d) else None,
        "effect_size_label": interpret_cohens_d(d) if not np.isnan(d) else "unknown",
        "classical_bootstrap_ci_95": [c_lo, c_hi],
        "quantum_bootstrap_ci_95": [q_lo, q_hi],
        "classical_mean": c_mean,
        "quantum_mean": q_mean,
    }
