"""Author: DEVADATH H K

Quantum AI Research Series

Project 01: Classical vs Quantum Visualization
Task: Classical vs quantum Iris classification with runtime-safe boundary plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import ZZFeatureMap

try:
    from qiskit.circuit.library import zz_feature_map
except ImportError:
    zz_feature_map = None
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
except ImportError as exc:
    QSVC = None
    FidelityQuantumKernel = None
    QML_IMPORT_ERROR = exc
else:
    QML_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical vs Quantum Iris classification (PCA-2D).")
    parser.add_argument("--random-state", type=int, default=7, help="Random seed for train/test split.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio.")
    parser.add_argument(
        "--classical-grid",
        type=int,
        default=140,
        help="Decision boundary grid resolution for classical models.",
    )
    parser.add_argument(
        "--quantum-grid",
        type=int,
        default=None,
        help="Decision boundary grid resolution for QSVC. If omitted, chosen from budget.",
    )
    parser.add_argument(
        "--quantum-max-kernel-evals",
        type=int,
        default=120_000,
        help="Upper bound for estimated QSVC kernel evaluations during boundary plotting.",
    )
    parser.add_argument(
        "--output-plot",
        default="assets/classical_vs_quantum_boundaries.png",
        help="Output path for the boundary comparison figure.",
    )
    parser.add_argument(
        "--output-report",
        default="assets/qml_iris_report.json",
        help="Output path for a metrics report.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib window.")
    return parser.parse_args()

def estimate_kernel_evals(train_size: int, grid_resolution: int) -> int:
    grid_points = grid_resolution * grid_resolution
    return (train_size * train_size) + (grid_points * train_size)

def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

def build_zz_feature_map(feature_dimension: int, reps: int = 2, entanglement: str = "linear"):
    if zz_feature_map is not None:
        return zz_feature_map(feature_dimension=feature_dimension, reps=reps, entanglement=entanglement)
    return ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement=entanglement)

def choose_quantum_grid(train_size: int, max_kernel_evals: int, floor: int = 12, cap: int = 80) -> int:
    for resolution in range(cap, floor - 1, -1):
        if estimate_kernel_evals(train_size, resolution) <= max_kernel_evals:
            return resolution
    return floor

def plot_decision_surface(ax, model, x_plot, y_plot, title: str, resolution: int):
    x_min, x_max = x_plot[:, 0].min() - 0.5, x_plot[:, 0].max() + 0.5
    y_min, y_max = x_plot[:, 1].min() - 0.5, x_plot[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.30, cmap="coolwarm")
    scatter = ax.scatter(
        x_plot[:, 0],
        x_plot[:, 1],
        c=y_plot,
        cmap="coolwarm",
        edgecolor="black",
        s=36,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return scatter

def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    if QML_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing dependency qiskit-machine-learning. "
            "Install project requirements with: pip install -r requirements.txt"
        ) from QML_IMPORT_ERROR

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=iris.target,
    )

    pca = PCA(n_components=2, random_state=args.random_state)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    scaler = StandardScaler()
    x_train_pca = scaler.fit_transform(x_train_pca)
    x_test_pca = scaler.transform(x_test_pca)

    x_all_pca = np.vstack([x_train_pca, x_test_pca])
    y_all = np.concatenate([y_train, y_test])

    quantum_resolution = args.quantum_grid
    if quantum_resolution is None:
        quantum_resolution = choose_quantum_grid(
            train_size=len(x_train_pca),
            max_kernel_evals=args.quantum_max_kernel_evals,
        )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.random_state),
        "Classical RBF-SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=args.random_state),
        "Quantum SVC (ZZ-Map)": QSVC(
            quantum_kernel=FidelityQuantumKernel(
                feature_map=build_zz_feature_map(feature_dimension=2, reps=2, entanglement="linear")
            )
        ),
    }

    print("Training models...")
    metrics = {}
    for name, model in models.items():
        model.fit(x_train_pca, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test_pca))
        metrics[name] = {"test_accuracy": round(float(accuracy), 6)}
        print(f"{name} test accuracy: {accuracy:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (name, model) in zip(axes, models.items()):
        resolution = quantum_resolution if isinstance(model, QSVC) else args.classical_grid
        scatter = plot_decision_surface(
            ax=ax,
            model=model,
            x_plot=x_all_pca,
            y_plot=y_all,
            title=f"{name} (grid={resolution})",
            resolution=resolution,
        )

    handles, _ = scatter.legend_elements()
    axes[-1].legend(handles, iris.target_names, title="Class", loc="best")

    output_plot = resolve_output_path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {output_plot}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    quantum_kernel_evals = estimate_kernel_evals(len(x_train_pca), quantum_resolution)
    report = {
        "random_state": args.random_state,
        "train_size": len(x_train_pca),
        "test_size": len(x_test_pca),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "grid_resolution": {
            "classical": args.classical_grid,
            "quantum": quantum_resolution,
        },
        "quantum_kernel_eval_estimate": quantum_kernel_evals,
        "quantum_max_kernel_evals_budget": args.quantum_max_kernel_evals,
        "metrics": metrics,
        "plot_path": str(output_plot),
    }

    output_report = resolve_output_path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {output_report}")

if __name__ == "__main__":
    main()
