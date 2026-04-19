"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Visualization Module.

Upgrades (v2 — Phase 5):
  - setup_plot_style: 300 DPI, serif font, publication-quality axis settings.
  - plot_kernel_geometry_tsne: t-SNE embedding of kernel matrix (Phase 5.2).
  - plot_noise_difference_heatmap: ΔK = K_noiseless − K_noisy diverging colormap (Phase 5.3).
  - plot_ablation_dashboard: 2×3 publication summary figure (Phase 5.4).
  - plot_geometric_difference: g(K_Q, K_C) vs PCA dim (Phase 5.5).
  - plot_expressibility_vs_kta: scatter ε vs KTA (Phase 5.6).
  - plot_noise_robustness_curve: KTA vs readout error sweep (Phase 4.2).
  - plot_scalability_curve: F1 + wall-clock time vs training size (Phase 4.3).
"""

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
    """Setup matplotlib with publication-quality defaults (Phase 5.1).

    Targets IEEE/ACM camera-ready standards:
    - 300 DPI raster output
    - Serif body font (falls back gracefully)
    - Heavier axis spines/ticks for printed figures
    """
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("default")

    plt.rcParams.update({
        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Font
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Axes
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

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
        plt.close(fig)
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
        plt.close(fig)
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
        plt.close(fig)
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
        plt.close(fig)
        logger.info(f"Kernel heatmap saved to {save_path}")
    
    return fig

def plot_metrics_comparison(
    results: dict,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot metrics comparison bar chart.

    Args:
        results: Evaluation results dictionary. Must have a ``"classical"`` and
                 a ``"quantum"`` key. An optional ``"pegasos"`` key adds a third
                 bar group so that Classical vs Exact QSVC vs Pegasos QSVC are
                 all shown side-by-side.
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
    has_pegasos = "pegasos" in results

    x = np.arange(len(metrics))

    if has_pegasos:
        pegasos_values = [results["pegasos"]["metrics"][m] for m in metrics]
        width = 0.25
        offsets = [-width, 0.0, width]
        bar_specs = [
            (classical_values, "Classical RBF SVM", "#3498db"),
            (quantum_values,   "Exact QSVC",        "#e74c3c"),
            (pegasos_values,   "Pegasos QSVC",      "#2ecc71"),
        ]
    else:
        width = 0.35
        offsets = [-width / 2, width / 2]
        bar_specs = [
            (classical_values, "Classical RBF SVM", "#3498db"),
            (quantum_values,   "Quantum Kernel SVM", "#e74c3c"),
        ]

    fig, ax = plt.subplots(figsize=figsize)

    for (values, label, color), offset in zip(bar_specs, offsets):
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
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
        plt.close(fig)
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
        plt.close(fig)
        logger.info(f"Noise comparison plot saved to {save_path}")
    
    return fig

def plot_ablation_scaling(
    scaling_results: list[dict],
    title: str = "PCA Dimension Ablation Study",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot scaling results across different PCA dimensions with standard deviations.
    
    Args:
        scaling_results: List of dictionaries containing ablation matrices metrics.
        title: Title of the chart.
        save_path: Location to save the image.
        figsize: Matplotlib figure tuple.
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    dimensions = [r["dimension"] for r in scaling_results]
    
    # Extract means and stds for classical and quantum
    c_means = [r["classical_mean_f1"] for r in scaling_results]
    c_stds = [r["classical_std_f1"] for r in scaling_results]
    
    q_means = [r["quantum_mean_f1"] for r in scaling_results]
    q_stds = [r["quantum_std_f1"] for r in scaling_results]
    
    ax.errorbar(dimensions, c_means, yerr=c_stds, fmt='-o', color="#3498db", label="Classical RBF SVM", capsize=5)
    ax.errorbar(dimensions, q_means, yerr=q_stds, fmt='-s', color="#e74c3c", label="Pegasos QSVC", capsize=5)
    
    ax.set_xlabel("PCA Components (Dimensions)")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_xticks(dimensions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Ablation scaling plot saved to {save_path}")

    return fig


def plot_depth_ablation(
    depth_results: list[dict],
    title: str = "Feature-Map Depth Ablation (reps vs Accuracy & KTA)",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot how circuit depth (reps) affects F1 score and Kernel-Target Alignment.

    This is the novel contribution plot: it shows the relationship between
    ZZFeatureMap repetitions, classification performance, and kernel geometry
    (KTA), providing the mechanistic explanation for why one reps value is
    better than another.

    Args:
        depth_results: List of dicts, each with keys:
            ``reps``, ``mean_f1``, ``std_f1``, ``mean_kta``, ``std_kta``.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size tuple.

    Returns:
        Matplotlib figure.
    """
    setup_plot_style()

    reps_vals = [r["reps"] for r in depth_results]
    f1_means = [r["mean_f1"] for r in depth_results]
    f1_stds = [r["std_f1"] for r in depth_results]
    kta_means = [r["mean_kta"] for r in depth_results]
    kta_stds = [r["std_kta"] for r in depth_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: F1 vs reps ---
    ax1.errorbar(
        reps_vals, f1_means, yerr=f1_stds,
        fmt="-o", color="#e74c3c", capsize=6, linewidth=2,
        label="Pegasos QSVC F1",
    )
    ax1.set_xlabel("Feature-Map Repetitions (reps)")
    ax1.set_ylabel("F1 Score (mean ± std across seeds)")
    ax1.set_title("Classification Performance vs Circuit Depth")
    ax1.set_xticks(reps_vals)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # --- Right: KTA vs reps ---
    ax2.errorbar(
        reps_vals, kta_means, yerr=kta_stds,
        fmt="-s", color="#9b59b6", capsize=6, linewidth=2,
        label="Kernel-Target Alignment",
    )
    ax2.set_xlabel("Feature-Map Repetitions (reps)")
    ax2.set_ylabel("KTA (mean ± std across seeds)")
    ax2.set_title("Kernel Geometry (KTA) vs Circuit Depth")
    ax2.set_xticks(reps_vals)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Depth ablation plot saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Phase 5.2 — Kernel geometry via t-SNE
# ---------------------------------------------------------------------------

def plot_kernel_geometry_tsne(
    kernel_matrix: np.ndarray,
    y: np.ndarray,
    title: str = "Kernel Geometry (t-SNE)",
    save_path: Optional[str] = None,
    class_names: Optional[list[str]] = None,
    reps: int = 1,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Embed the kernel matrix via t-SNE to visualise geometry (Phase 5.2).

    Uses D = 1 - K_norm as a dissimilarity matrix for t-SNE.
    Shows how circuit depth (reps) reshapes the data's quantum geometry.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn TSNE unavailable; skipping kernel geometry plot.")
        return None

    # Normalise K to [0,1] using diagonal
    d = np.diag(kernel_matrix)
    d = np.maximum(d, 1e-12)
    K_norm = kernel_matrix / np.sqrt(np.outer(d, d))
    np.fill_diagonal(K_norm, 1.0)
    K_norm = np.clip(K_norm, 0.0, 1.0)
    D = 1.0 - K_norm
    D = (D + D.T) / 2  # enforce symmetry
    np.fill_diagonal(D, 0.0)

    n = D.shape[0]
    perplexity = min(30, max(5, n // 4))

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perplexity,
        random_state=42,
        init="random",
        n_iter=1000,
    )
    embedding = tsne.fit_transform(D)

    unique_labels = np.unique(y)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=figsize)
    for idx, label in enumerate(unique_labels):
        mask = y == label
        name = class_names[idx] if class_names and idx < len(class_names) else str(label)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=colors[idx % len(colors)],
            label=name, alpha=0.75, edgecolors="k", linewidths=0.3, s=50,
        )

    ax.set_title(f"{title} (reps={reps})")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Kernel geometry t-SNE plot saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 5.3 — Noise difference heatmap ΔK
# ---------------------------------------------------------------------------

def plot_noise_difference_heatmap(
    K_noiseless: np.ndarray,
    K_noisy: np.ndarray,
    title: str = "ΔK = K_noiseless − K_noisy",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """Plot the element-wise difference ΔK between noiseless and noisy kernels.

    Uses a symmetric diverging colormap (RdBu) centred at 0, as recommended
    for difference matrices in colour-vision-deficiency-accessible figures.
    """
    delta = K_noiseless - K_noisy
    vmax = float(np.abs(delta).max())
    vmax = vmax if vmax > 1e-10 else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    if sns is not None:
        sns.heatmap(
            delta, cmap="RdBu", center=0, vmin=-vmax, vmax=vmax,
            ax=ax, square=True,
            cbar_kws={"label": "ΔK (noiseless − noisy)"},
        )
    else:
        im = ax.imshow(delta, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="equal")
        plt.colorbar(im, ax=ax, label="ΔK (noiseless − noisy)")

    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Noise difference heatmap saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 5.4 — 2×3 Ablation Dashboard
# ---------------------------------------------------------------------------

def plot_ablation_dashboard(
    ablation_stats: list[dict],
    depth_stats: list[dict],
    noise_curve: Optional[list[dict]] = None,
    scalability_curve: Optional[list[dict]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (18, 10),
) -> plt.Figure:
    """2×3 publication summary dashboard (Phase 5.4).

    Layout::

        | F1 vs PCA dim  | KTA vs PCA dim  | Train time vs dim |
        | F1 vs reps     | KTA vs reps     | Noise curve       |

    Args:
        ablation_stats: Output of Phase-1 dimension ablation.
        depth_stats: Output of Phase-2 depth ablation.
        noise_curve: Optional list of {readout_error, kta} from Phase 4.2.
        scalability_curve: Optional list of {n_train, f1_mean, train_time_mean}.
        save_path: Where to save the figure.
        figsize: Dashboard dimensions.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        "Quantum Kernel SVM — Publication Summary Dashboard",
        fontsize=16, fontweight="bold", y=1.01,
    )

    dims = [r["dimension"] for r in ablation_stats]
    c_f1 = [r["classical_mean_f1"] for r in ablation_stats]
    c_f1_std = [r["classical_std_f1"] for r in ablation_stats]
    q_f1 = [r["quantum_mean_f1"] for r in ablation_stats]
    q_f1_std = [r["quantum_std_f1"] for r in ablation_stats]
    kta_q = [r["mean_kta_quantum"] for r in ablation_stats]
    kta_q_std = [r["std_kta_quantum"] for r in ablation_stats]
    kta_c = [r["mean_kta_classical"] for r in ablation_stats]
    kta_c_std = [r.get("std_kta_classical", 0.0) for r in ablation_stats]

    reps_vals = [r["reps"] for r in depth_stats]
    d_f1 = [r["mean_f1"] for r in depth_stats]
    d_f1_std = [r["std_f1"] for r in depth_stats]
    d_kta = [r["mean_kta"] for r in depth_stats]
    d_kta_std = [r["std_kta"] for r in depth_stats]

    # --- [0,0] F1 vs PCA dim ---
    ax = axes[0, 0]
    ax.errorbar(dims, c_f1, yerr=c_f1_std, fmt="-o", color="#3498db",
                capsize=5, label="Classical RBF", linewidth=2)
    ax.errorbar(dims, q_f1, yerr=q_f1_std, fmt="-s", color="#e74c3c",
                capsize=5, label="Pegasos QSVC", linewidth=2)
    ax.set_xlabel("PCA Dimensions")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs PCA Dimension")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(dims)

    # --- [0,1] KTA vs PCA dim ---
    ax = axes[0, 1]
    ax.errorbar(dims, kta_q, yerr=kta_q_std, fmt="-o", color="#9b59b6",
                capsize=5, label="Quantum KTA", linewidth=2)
    ax.errorbar(dims, kta_c, yerr=kta_c_std, fmt="-s", color="#1abc9c",
                capsize=5, label="Classical RBF KTA", linewidth=2)
    ax.set_xlabel("PCA Dimensions")
    ax.set_ylabel("KTA")
    ax.set_title("Kernel-Target Alignment vs PCA Dimension")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks(dims)

    # --- [0,2] Scalability or training time ---
    ax = axes[0, 2]
    if scalability_curve:
        ns = [r["n_train"] for r in scalability_curve]
        tt = [r.get("train_time_mean", 0) for r in scalability_curve]
        ax.plot(ns, tt, "-o", color="#e67e22", linewidth=2, markersize=6)
        ax.set_xlabel("Training Set Size (n)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title("Scalability: Training Time vs n")
    else:
        ax.text(0.5, 0.5, "Scalability curve\n(run Phase 4.3)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("Scalability Analysis")

    # --- [1,0] F1 vs reps ---
    ax = axes[1, 0]
    ax.errorbar(reps_vals, d_f1, yerr=d_f1_std, fmt="-o", color="#e74c3c",
                capsize=5, label="Pegasos QSVC", linewidth=2)
    ax.set_xlabel("Feature-Map Repetitions (reps)")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Circuit Depth")
    ax.set_xticks(reps_vals)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    # --- [1,1] KTA vs reps ---
    ax = axes[1, 1]
    ax.errorbar(reps_vals, d_kta, yerr=d_kta_std, fmt="-s", color="#9b59b6",
                capsize=5, label="Quantum KTA", linewidth=2)
    ax.set_xlabel("Feature-Map Repetitions (reps)")
    ax.set_ylabel("KTA")
    ax.set_title("KTA vs Circuit Depth")
    ax.set_xticks(reps_vals)
    ax.legend(fontsize=9)

    # --- [1,2] Noise robustness curve ---
    ax = axes[1, 2]
    if noise_curve:
        errs = [r["readout_error"] for r in noise_curve]
        ktas = [r["kta"] for r in noise_curve]
        ax.plot(errs, ktas, "-o", color="#c0392b", linewidth=2, markersize=6)
        ax.axhline(ktas[0], color="gray", linestyle="--", linewidth=1,
                   label="Noiseless baseline")
        ax.set_xlabel("Readout Error Rate")
        ax.set_ylabel("Kernel-Target Alignment")
        ax.set_title("Noise Robustness: KTA vs Readout Error")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Noise curve\n(run Phase 4.2)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_title("Noise Robustness Curve")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Ablation dashboard saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 5.5 — Geometric difference vs PCA dimension
# ---------------------------------------------------------------------------

def plot_geometric_difference(
    dims: list[int],
    g_values: list[float],
    title: str = "Geometric Difference g(K_Q, K_C) vs PCA Dimension",
    save_path: Optional[str] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot g(K_Q, K_C) vs PCA dimension (Phase 5.5).

    Draws a horizontal dashed line at g=1.0 (quantum advantage precondition boundary).
    Points above the line satisfy the necessary condition for quantum advantage.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dims, g_values, "-o", color="#8e44ad", linewidth=2, markersize=8,
            label="g(K_Q, K_C)")
    ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="g = 1 (advantage threshold)")

    # Shade region above threshold
    ax.fill_between(dims, 1.0, g_values,
                    where=[g > 1.0 for g in g_values],
                    alpha=0.15, color="#8e44ad",
                    label="Advantage precondition satisfied")

    ax.set_xlabel("PCA Dimensions (d)")
    ax.set_ylabel("Geometric Difference g")
    ax.set_title(title)
    ax.set_xticks(dims)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Geometric difference plot saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 5.6 — Expressibility vs KTA scatter
# ---------------------------------------------------------------------------

def plot_expressibility_vs_kta(
    configs: list[dict],
    title: str = "Expressibility vs Kernel-Target Alignment",
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Scatter plot: expressibility ε vs KTA for each (reps, dim) config (Phase 5.6).

    Each point is annotated with its (dim, reps) label.
    A negative correlation (more expressive → lower KTA) is evidence for
    the barren-plateau effect.

    Args:
        configs: List of dicts, each with keys:
            ``expressibility`` (float), ``kta`` (float),
            ``dim`` (int), ``reps`` (int).
    """
    eps = [c["expressibility"] for c in configs]
    kta = [c["kta"] for c in configs]
    dims = [c["dim"] for c in configs]
    reps = [c["reps"] for c in configs]

    colors_map = {1: "#3498db", 2: "#e74c3c", 3: "#2ecc71"}
    colors = [colors_map.get(r, "#7f8c8d") for r in reps]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(eps, kta, c=colors, s=90, edgecolors="k", linewidths=0.5, zorder=3)

    for i, (e, k, d, r) in enumerate(zip(eps, kta, dims, reps)):
        ax.annotate(
            f"d={d},r={r}",
            (e, k),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7.5,
        )

    # Correlation annotation
    if len(eps) >= 3:
        import scipy.stats as sc_stats
        r_val, p_val = sc_stats.pearsonr(eps, kta)
        ax.annotate(
            f"r = {r_val:.2f}, p = {p_val:.3f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Legend for reps colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
               markersize=8, markeredgecolor="k", label=f"reps={r}")
        for r, col in colors_map.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("Expressibility ε = KL(P_PQC \u2225 P_Haar)  [lower → more expressive]")
    ax.set_ylabel("Kernel-Target Alignment (KTA)")
    ax.set_title(title)
    ax.invert_xaxis()  # higher KL = less expressive on the right

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Expressibility vs KTA scatter saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 4.2 — Noise robustness curve
# ---------------------------------------------------------------------------

def plot_noise_robustness_curve(
    noise_curve: list[dict],
    title: str = "Noise Robustness: KTA vs Readout Error",
    save_path: Optional[str] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot KTA as a function of readout error rate (Phase 4.2).

    Args:
        noise_curve: List of dicts {readout_error: float, kta: float}.
    """
    errs = [r["readout_error"] for r in noise_curve]
    ktas = [r["kta"] for r in noise_curve]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(errs, ktas, "-o", color="#c0392b", linewidth=2, markersize=7)
    if ktas:
        ax.axhline(ktas[0], color="gray", linestyle="--", linewidth=1.2,
                   label=f"Noiseless KTA = {ktas[0]:.4f}")
    ax.fill_between(errs, ktas, ktas[0] if ktas else 0,
                    alpha=0.12, color="#c0392b", label="KTA degradation")
    ax.set_xlabel("Readout Error Rate")
    ax.set_ylabel("Kernel-Target Alignment (KTA)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Noise robustness curve saved to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Phase 4.3 — Scalability curve
# ---------------------------------------------------------------------------

def plot_scalability_curve(
    scalability_data: list[dict],
    title: str = "Scalability: F1 Score and Training Time vs Dataset Size",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot F1 score and wall-clock training time vs training set size (Phase 4.3).

    Args:
        scalability_data: List of dicts with keys:
            n_train (int), f1_mean (float), f1_std (float),
            train_time_mean (float), train_time_std (float).
    """
    ns = [r["n_train"] for r in scalability_data]
    f1_means = [r["f1_mean"] for r in scalability_data]
    f1_stds = [r.get("f1_std", 0.0) for r in scalability_data]
    tt_means = [r["train_time_mean"] for r in scalability_data]
    tt_stds = [r.get("train_time_std", 0.0) for r in scalability_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.errorbar(ns, f1_means, yerr=f1_stds, fmt="-o", color="#3498db",
                 capsize=5, linewidth=2, label="Quantum Kernel SVM")
    ax1.set_xlabel("Training Set Size (n)")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Classification Performance vs n")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)

    ax2.errorbar(ns, tt_means, yerr=tt_stds, fmt="-o", color="#e67e22",
                 capsize=5, linewidth=2, label="Kernel matrix time")
    ax2.set_xlabel("Training Set Size (n)")
    ax2.set_ylabel("Training Time (s)")
    ax2.set_title("Wall-clock Time vs n  [O(n²)]")
    ax2.legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Scalability curve saved to %s", save_path)
    return fig


def plot_qkl_convergence(
    history: list[dict],
    save_path: Optional[str] = None,
    title: str = "Quantum Kernel Learning — Convergence",
    figsize: tuple[float, float] = (10, 4),
) -> "plt.Figure":
    """Plot cKTA objective and gradient norm over QKL training epochs.

    Produces a publication-quality dual-panel figure:
    - Left panel: cKTA vs epoch with horizontal dashed line at the initial value.
    - Right panel: gradient norm ‖∇cKTA‖₂ vs epoch on a log scale.

    This plot is required by reviewers to verify that:
    (a) the objective monotonically improves (or converges),
    (b) the gradient norm decays (confirming convergence, not oscillation).

    Args:
        history: List of per-epoch dicts with keys
            {"epoch", "ckta", "grad_norm", "wall_time_s"}.
        save_path: If given, saves the figure as PNG.
        title: Figure super-title.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    if not history:
        logger.warning("plot_qkl_convergence: empty history — nothing to plot.")
        fig, _ = plt.subplots(figsize=figsize)
        return fig

    epochs = [h["epoch"] for h in history]
    cktas = [h["ckta"] for h in history]
    grad_norms = [h["grad_norm"] for h in history]
    initial_ckta = cktas[0] if cktas else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: cKTA curve ──
    ax1.plot(epochs, cktas, "-o", color="#2980b9", linewidth=2.0,
             markersize=4, label="cKTA (batch)")
    ax1.axhline(
        y=initial_ckta, color="#95a5a6", linestyle="--", linewidth=1.2,
        label=f"Initial cKTA = {initial_ckta:.4f}",
    )
    best_ckta = max(cktas)
    best_epoch = epochs[cktas.index(best_ckta)]
    ax1.scatter([best_epoch], [best_ckta], color="#e74c3c", zorder=5, s=60,
                label=f"Best = {best_ckta:.4f} (epoch {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("cKTA")
    ax1.set_title("Objective: Centered KTA")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Right: gradient norm ──
    ax2.semilogy(epochs, grad_norms, "-s", color="#e67e22", linewidth=2.0,
                 markersize=4, label=r"$\|\nabla\, \mathrm{cKTA}\|_2$")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Gradient norm $\|\nabla\|$  [log scale]")
    ax2.set_title("Gradient Norm — Parameter-Shift Rule")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("QKL convergence plot saved to %s", save_path)
    return fig
