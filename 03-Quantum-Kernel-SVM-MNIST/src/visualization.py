"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Visualization Module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import seaborn as sns
except ImportError:
    sns = None

logger = logging.getLogger(__name__)

def setup_plot_style(style: str = "seaborn-v0_8-darkgrid") -> None:
    """Setup matplotlib plot style.

    Args:
        style: Matplotlib style name
    """
    try:
        plt.style.use(style)
    except:
        plt.style.use("default")
    
    # Set default figure parameters
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

def _plot_heatmap(
    data: np.ndarray,
    ax: plt.Axes,
    cmap: str = "viridis",
    cbar_label: str = "Value",
    annot: bool = False,
    fmt: str = ".2f",
    xticklabels: Optional[list[str]] = None,
    yticklabels: Optional[list[str]] = None,
    square: bool = False,
) -> None:
    """Draw a heatmap using seaborn when available, else pure matplotlib."""
    if sns is not None:
        sns.heatmap(
            data,
            cmap=cmap,
            square=square,
            ax=ax,
            annot=annot,
            fmt=fmt,
            xticklabels=xticklabels if xticklabels is not None else "auto",
            yticklabels=yticklabels if yticklabels is not None else "auto",
            cbar_kws={"label": cbar_label},
        )
        return

    im = ax.imshow(data, cmap=cmap, aspect="equal" if square else "auto")
    plt.colorbar(im, ax=ax, label=cbar_label)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if annot:
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                ax.text(col, row, format(data[row, col], fmt), ha="center", va="center")

def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    pca: Optional[PCA] = None,
    title: str = "PCA Scatter Plot",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
    classes: Optional[list[int]] = None,
    class_names: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot PCA scatter plot of data.

    Args:
        X: Feature matrix
        y: Labels
        pca: Optional pre-fitted PCA (if X is already transformed)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        classes: List of class labels
        class_names: List of class names

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Apply PCA if not already done
    if pca is not None and X.shape[1] > 2:
        X_pca = pca.transform(X)
    else:
        X_pca = X
    
    # If only 2D, use as is; otherwise take first 2 components
    if X_pca.shape[1] >= 2:
        X_plot = X_pca[:, :2]
    else:
        X_plot = X_pca
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class
    if classes is None:
        classes = sorted(np.unique(y))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        mask = y == cls
        label = class_names[i] if class_names else str(cls)
        ax.scatter(
            X_plot[mask, 0],
            X_plot[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            edgecolors="black",
            s=50,
        )
    
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PCA scatter plot saved to {save_path}")
    
    return fig

def plot_pca_variance(
    pca: PCA,
    title: str = "PCA Variance Explained",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot PCA variance explained.

    Args:
        pca: Fitted PCA transformer
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    n_components = len(variance_ratio)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    axes[0].bar(range(1, n_components + 1), variance_ratio, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title("Individual Variance Explained")
    axes[0].set_xticks(range(1, n_components + 1))
    
    # Cumulative variance
    axes[1].plot(
        range(1, n_components + 1),
        cumulative_variance,
        "o-",
        color="steelblue",
        linewidth=2,
    )
    axes[1].axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    axes[1].fill_between(
        range(1, n_components + 1),
        cumulative_variance,
        alpha=0.3,
        color="steelblue",
    )
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance Explained")
    axes[1].set_title("Cumulative Variance Explained")
    axes[1].set_xticks(range(1, n_components + 1))
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"PCA variance plot saved to {save_path}")
    
    return fig

def plot_class_distribution(
    y: np.ndarray,
    classes: Optional[list[int]] = None,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
    class_names: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot class distribution bar chart.

    Args:
        y: Labels
        classes: List of class labels
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        class_names: Optional class names

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    if classes is None:
        classes = sorted(np.unique(y))
    
    counts = [np.sum(y == cls) for cls in classes]
    
    if class_names is None:
        labels = [str(cls) for cls in classes]
    else:
        labels = class_names
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(labels, counts, color=["#3498db", "#e74c3c"], edgecolor="black")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Class distribution plot saved to {save_path}")
    
    return fig

def plot_kernel_heatmap(
    kernel_matrix: np.ndarray,
    title: str = "Quantum Kernel Matrix",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    max_samples: int = 100,
) -> plt.Figure:
    """Plot kernel matrix heatmap.

    Args:
        kernel_matrix: Kernel matrix
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap name
        max_samples: Maximum samples to display

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Limit size for large matrices
    if kernel_matrix.shape[0] > max_samples:
        logger.warning(f"Kernel matrix too large ({kernel_matrix.shape[0]}), truncating to {max_samples}")
        K_plot = kernel_matrix[:max_samples, :max_samples]
    else:
        K_plot = kernel_matrix
    
    fig, ax = plt.subplots(figsize=figsize)
    
    _plot_heatmap(
        K_plot,
        ax=ax,
        cmap=cmap,
        cbar_label="Kernel Value",
        square=True,
    )
    
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Kernel heatmap saved to {save_path}")
    
    return fig

def plot_metrics_comparison(
    results: dict,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot metrics comparison bar chart.

    Args:
        results: Evaluation results dictionary
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    classical_values = [results["classical"]["metrics"][m] for m in metrics]
    quantum_values = [results["quantum"]["metrics"][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width / 2, classical_values, width, label="Classical RBF SVM", color="#3498db")
    bars2 = ax.bar(x + width / 2, quantum_values, width, label="Quantum Kernel SVM", color="#e74c3c")
    
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
    class_names: Optional[list[str]] = None,
    model_name: str = "Model",
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        class_names: Optional class names
        model_name: Model name for subtitle

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    _plot_heatmap(
        cm,
        ax=ax,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else None,
        yticklabels=class_names if class_names else None,
        cbar_label="Count",
    )
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{title} - {model_name}")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig

def plot_noisy_vs_noiseless_kernel(
    noiseless_kernel: np.ndarray,
    noisy_kernel: np.ndarray,
    title: str = "Noiseless vs Noisy Kernel Comparison",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot comparison between noiseless and noisy kernel matrices.

    Args:
        noiseless_kernel: Kernel without noise
        noisy_kernel: Kernel with noise simulation
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Limit size
    max_samples = min(50, noiseless_kernel.shape[0])
    k1 = noiseless_kernel[:max_samples, :max_samples]
    k2 = noisy_kernel[:max_samples, :max_samples]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Noiseless
    _plot_heatmap(k1, ax=axes[0], cmap="viridis", cbar_label="Kernel", square=True)
    axes[0].set_title("Noiseless Kernel")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")
    
    # Noisy
    _plot_heatmap(k2, ax=axes[1], cmap="viridis", cbar_label="Kernel", square=True)
    axes[1].set_title("Noisy Kernel")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Sample Index")
    
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Noise comparison plot saved to {save_path}")
    
    return fig

