"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: End-to-end experiment runner for Classical vs Quantum Kernel SVM."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

from src.classical_models import train_classical_svm
from src.data_loader import load_mnist_digits
from src.evaluation_metrics import compute_confusion_matrix, compute_metrics
from src.noise_simulation import create_noisy_kernel_comparison
from src.preprocessing import preprocess_data
from src.quantum_feature_maps import create_feature_map
from src.quantum_kernel_engine import (
    analyze_kernel_properties,
    compute_kernel_alignment,
    compute_kernel_matrix,
    compute_kernel_target_alignment,
    create_quantum_kernel,
    regularize_kernel_matrix,
)
from src.quantum_training import train_pegasos_qsvc, train_qsvc
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_kernel_heatmap,
    plot_metrics_comparison,
    plot_noisy_vs_noiseless_kernel,
    plot_pca_scatter,
    plot_pca_variance,
    setup_plot_style,
)

from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("run_experiment")

def _stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=seed
    )
    return X_sub, y_sub

def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _save_json(path: Path, payload: dict) -> None:
    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run classical vs quantum kernel SVM experiment.")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_config.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-quantum-train", type=int, default=400)
    parser.add_argument("--max-kernel-samples", type=int, default=400)
    parser.add_argument("--max-noise-samples", type=int, default=50)
    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--quantum-steps", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    setup_plot_style()

    cfg = _load_config(args.config)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("random_seed", 42))
    np.random.seed(seed)

    dataset_cfg = cfg.get("dataset", {})
    digits = dataset_cfg.get("digits", [4, 9])
    test_size = float(dataset_cfg.get("test_size", 0.25))
    stratify = bool(dataset_cfg.get("stratify", True))

    LOGGER.info("Loading dataset...")
    X_train_raw, X_test_raw, y_train, y_test = load_mnist_digits(
        digits=digits,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        data_home=str(results_dir / ".cache" / "sklearn"),
        fallback_to_sklearn_digits=bool(dataset_cfg.get("fallback_to_sklearn_digits", False)),
    )
    dataset_source = "openml_mnist_784" if X_train_raw.shape[1] == 784 else "sklearn_digits_fallback"
    if dataset_source != "openml_mnist_784":
        LOGGER.warning(
            "OpenML MNIST unavailable in this environment; using sklearn digits fallback."
        )

    preprocess_cfg = cfg.get("preprocessing", {})
    pca_cfg = preprocess_cfg.get("pca", {})
    n_components = int(pca_cfg.get("n_components", 8))
    quantum_scaling = bool(preprocess_cfg.get("quantum_scaling", True))

    LOGGER.info("Preprocessing...")
    pre = preprocess_data(
        X_train=X_train_raw,
        X_test=X_test_raw,
        n_components=n_components,
        random_state=seed,
        quantum_scaling=quantum_scaling,
    )

    X_train_pca = pre["X_train_processed"]
    X_test_pca = pre["X_test_processed"]
    X_train_quantum = pre["X_train_quantum"]
    X_test_quantum = pre["X_test_quantum"]
    pca_transformer = pre["pca"]

    plot_class_distribution(
        y_train,
        classes=digits,
        title="MNIST Training Class Distribution",
        save_path=str(results_dir / "class_distribution.png"),
        class_names=[f"Digit {d}" for d in digits],
    )

    plot_pca_scatter(
        X_train_pca,
        y_train,
        title=f"PCA Scatter (MNIST {digits[0]} vs {digits[1]})",
        save_path=str(results_dir / "pca_scatter.png"),
        class_names=[f"Digit {d}" for d in digits],
    )
    plot_pca_variance(
        pca_transformer,
        title="PCA Variance Explanation",
        save_path=str(results_dir / "pca_variance.png"),
    )

    np.save(results_dir / "X_train_pca.npy", X_train_pca)
    np.save(results_dir / "X_test_pca.npy", X_test_pca)
    np.save(results_dir / "X_train_quantum.npy", X_train_quantum)
    np.save(results_dir / "X_test_quantum.npy", X_test_quantum)
    np.save(results_dir / "y_train.npy", y_train)
    np.save(results_dir / "y_test.npy", y_test)

    classical_cfg = cfg.get("classical_model", {})
    gs_cfg = classical_cfg.get("grid_search", {})
    LOGGER.info("Training classical baseline...")
    classical_model, classical_train = train_classical_svm(
        X_train=X_train_pca,
        y_train=y_train,
        kernel=classical_cfg.get("kernel", "rbf"),
        random_state=seed,
        grid_search=bool(gs_cfg.get("enabled", True)),
        cv=int(gs_cfg.get("cv", 3)),
        param_grid=gs_cfg.get("param_grid"),
        n_jobs=int(gs_cfg.get("n_jobs", -1)),
    )
    
    # Save classical model
    joblib.dump(classical_model, results_dir / "classical_svm_model.joblib")

    y_pred_classical = classical_model.predict(X_test_pca)
    
    is_binary = len(np.unique(y_test)) == 2
    pos_label = int(max(digits)) if is_binary else None
    average = "binary" if is_binary else "weighted"

    classical_metrics = compute_metrics(y_test, y_pred_classical, pos_label=pos_label, average=average)
    classical_cm = compute_confusion_matrix(y_test, y_pred_classical, labels=digits)

    _save_json(
        results_dir / "classical_svm_results.json",
        {
            "model": "Classical RBF SVM",
            "best_params": classical_train.get("best_params"),
            "best_cv_score": classical_train.get("best_cv_score"),
            "train_time": classical_train.get("train_time"),
            "metrics": classical_metrics,
            "confusion_matrix": classical_cm.tolist(),
            "classification_report": classification_report(y_test, y_pred_classical, zero_division=0),
        },
    )

    qfm_cfg = cfg.get("quantum_feature_map", {})
    feature_map = create_feature_map(
        feature_map_type=qfm_cfg.get("type", "ZZFeatureMap"),
        feature_dimension=int(qfm_cfg.get("feature_dimension", n_components)),
        reps=int(qfm_cfg.get("reps", 2)),
        entanglement=qfm_cfg.get("entanglement", "full"),
        parameter_prefix=qfm_cfg.get("parameter_prefix", "x"),
    )

    LOGGER.info("Computing quantum kernel and training exact QSVC...")
    X_q_train, y_q_train = _stratified_subset(
        X_train_quantum,
        y_train,
        max_samples=args.max_quantum_train,
        seed=seed,
    )

    quantum_kernel = create_quantum_kernel(feature_map=feature_map)
    
    qsvc_cfg = cfg.get("qsvc", {})
    quantum_model, quantum_train = train_qsvc(
        quantum_kernel=quantum_kernel,
        X_train=X_q_train,
        y_train=y_q_train,
        C=float(qsvc_cfg.get("C", 1.0)),
        random_state=seed,
        grid_search=bool(qsvc_cfg.get("grid_search", True)),
        cv=int(qsvc_cfg.get("cv", 3)),
        n_jobs=int(qsvc_cfg.get("n_jobs", 1)),
    )
    
    # Save quantum model (QSVC might need pickle or joblib)
    try:
        joblib.dump(quantum_model, results_dir / "quantum_qsvc_model.joblib")
    except Exception as e:
        LOGGER.warning("Could not save quantum model with joblib: %s. Using pickle.", e)
        with open(results_dir / "quantum_qsvc_model.pkl", "wb") as f:
            pickle.dump(quantum_model, f)

    y_pred_quantum = quantum_model.predict(X_test_quantum)
    quantum_metrics = compute_metrics(y_test, y_pred_quantum, pos_label=pos_label, average=average)
    quantum_cm = compute_confusion_matrix(y_test, y_pred_quantum, labels=digits)

    X_kernel, y_kernel = _stratified_subset(
        X_train_quantum,
        y_train,
        max_samples=args.max_kernel_samples,
        seed=seed,
    )
    K_train = compute_kernel_matrix(quantum_kernel, X_kernel)
    kernel_props = analyze_kernel_properties(K_train)
    kta = compute_kernel_target_alignment(K_train, y_kernel)

    plot_kernel_heatmap(
        K_train,
        title=f"Quantum Kernel Matrix ({len(X_kernel)} samples)",
        save_path=str(results_dir / "kernel_heatmap.png"),
        max_samples=args.max_kernel_samples,
    )
    plot_confusion_matrix(
        classical_cm,
        title="Confusion Matrix",
        save_path=str(results_dir / "confusion_matrix_classical.png"),
        class_names=[f"Digit {d}" for d in digits],
        model_name="Classical RBF SVM",
    )
    plot_confusion_matrix(
        quantum_cm,
        title="Confusion Matrix",
        save_path=str(results_dir / "confusion_matrix_quantum.png"),
        class_names=[f"Digit {d}" for d in digits],
        model_name="Quantum Kernel SVM",
    )

    comparison = {
        "classical": {"metrics": classical_metrics},
        "quantum": {"metrics": quantum_metrics},
    }
    plot_metrics_comparison(
        comparison,
        title="Classical vs Quantum Metrics",
        save_path=str(results_dir / "metrics_comparison.png"),
    )

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Classical RBF SVM",
                **classical_metrics,
                "train_time": classical_train.get("train_time"),
            },
            {
                "Model": "Quantum Kernel SVM (exact QSVC)",
                **quantum_metrics,
                "train_time": quantum_train.get("train_time"),
            },
        ]
    )
    comparison_df.to_csv(results_dir / "metrics_comparison.csv", index=False)

    noise_report: dict | None = None
    if not args.disable_noise:
        noise_cfg = cfg.get("noise_simulation", {})
        noise_enabled = bool(noise_cfg.get("enabled", True))
        if noise_enabled:
            X_noise, y_noise = _stratified_subset(
                X_train_quantum,
                y_train,
                max_samples=args.max_noise_samples,
                seed=seed,
            )
            K_noiseless, K_noisy, noise_analysis = create_noisy_kernel_comparison(
                feature_map=feature_map,
                X=X_noise,
                readout_error=float(noise_cfg.get("noise_model", {}).get("readout_error", 0.01)),
                gate_error=float(noise_cfg.get("noise_model", {}).get("gate_error", 0.001)),
                shots=int(noise_cfg.get("shots", 1000)),
            )
            
            # Mathematically correct the noisy kernel for analysis
            K_noisy_psd = regularize_kernel_matrix(K_noisy)
            noise_analysis["alignment_noiseless_vs_noisy"] = compute_kernel_alignment(K_noiseless, K_noisy)
            noise_analysis["kta_noiseless"] = compute_kernel_target_alignment(K_noiseless, y_noise)
            noise_analysis["kta_noisy"] = compute_kernel_target_alignment(K_noisy_psd, y_noise)

            plot_noisy_vs_noiseless_kernel(
                K_noiseless,
                K_noisy,
                save_path=str(results_dir / "noise_comparison.png"),
            )
            noise_report = {
                "analysis": noise_analysis,
            }

    classical_train_summary = {
        k: v
        for k, v in classical_train.items()
        if k not in {"model", "grid_search_results"}
    }
    quantum_train_summary = {
        k: v
        for k, v in quantum_train.items()
        if k not in {"model", "support_vector_indices", "grid_search_results"}
    }

    summary = {
        "config_path": str(args.config),
        "dataset": {
            "source": dataset_source,
            "digits": digits,
            "train_samples": int(len(X_train_raw)),
            "test_samples": int(len(X_test_raw)),
            "pca_components": int(n_components),
        },
        "classical": {
            "train_info": classical_train_summary,
            "metrics": classical_metrics,
        },
        "quantum": {
            "train_subset_size": int(len(X_q_train)),
            "train_info": quantum_train_summary,
            "metrics": quantum_metrics,
            "kernel_properties": {**kernel_props, "kta": kta},
        },
        "noise": noise_report,
    }
    _save_json(results_dir / "experiment_summary.json", summary)

    print("Experiment complete.")
    print(f"Results directory: {results_dir.resolve()}")
    print("Metrics:")
    print(comparison_df.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
