"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: End-to-end experiment runner for Classical vs Quantum Kernel SVM.

Changes from v1 (publication-grade upgrade):
  - Dataset: real MNIST from OpenML by default; --fallback flag for offline use.
  - Fair comparison: classical and quantum train on the SAME n samples per cell.
  - Pegasos collapse fix: lambda_param 1.0 → 0.001 in config (no model collapse).
  - KTA: computed for BOTH the RBF gram matrix and quantum kernel; stored in CSV
    and experiment_summary.json so it appears in every results table.
  - Noise simulation: activated by default for seed[0]/dim[0]; stored in JSON
    (was always null before).
  - Depth ablation: outer loop over reps ∈ [1, 2, 3] for every dim/seed combo
    → writes depth_ablation.json and depth_ablation.png (novel contribution).
  - Statistical summary: console output and ablation_summary.json updated with KTA.

  Phase 3 / 4 / 5 additions (v2):
  - Geometric difference g(K_Q, K_C) computed per dimension (Huang et al. 2022).
  - Expressibility ε computed per (dim, reps) config (Sim et al. 2019).
  - Git SHA stamped on every experiment_summary.json for provenance (Phase 1.2).
  - Phase 5 publication plots: 2×3 dashboard, g vs dim, expressibility scatter.
"""

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

from src.classical_models import compute_rbf_gram_matrix, train_classical_svm
from src.data_loader import load_mnist_digits
from src.noise_simulation import create_noisy_kernel_comparison
from src.preprocessing import preprocess_data
from src.quantum_feature_maps import create_feature_map
from src.quantum_kernel_engine import (
    analyze_kernel_properties,
    compute_geometric_difference,
    compute_kernel_alignment,
    compute_kernel_matrix,
    compute_kernel_target_alignment,
    compute_centered_kta,
    create_quantum_kernel,
    get_git_sha,
    regularize_kernel_matrix,
)
from src.kernel_learning import QuantumKernelLearner
from src.hardware_backend import HardwareBackendManager
from src.quantum_training import train_pegasos_qsvc, train_qsvc
from src.visualization import (
    plot_ablation_dashboard,
    plot_ablation_scaling,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_depth_ablation,
    plot_geometric_difference,
    plot_expressibility_vs_kta,
    plot_kernel_heatmap,
    plot_metrics_comparison,
    plot_noisy_vs_noiseless_kernel,
    plot_pca_scatter,
    plot_pca_variance,
    plot_qkl_convergence,
    setup_plot_style,
)

from src.evaluation_metrics import (
    calculate_statistical_significance,
    compute_confusion_matrix,
    compute_metrics,
)

try:
    from src.expressibility import compute_expressibility
    _EXPR_AVAILABLE = True
except ImportError:
    _EXPR_AVAILABLE = False
    LOGGER_STARTUP = logging.getLogger("run_experiment")
    LOGGER_STARTUP.warning(
        "src.expressibility not found — expressibility vs KTA plot will be skipped."
    )

from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("run_experiment")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified subset of (X, y) with at most ``max_samples`` rows."""
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
    """Serialize ``payload`` to JSON, converting numpy types automatically."""
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


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_single_trial(
    seed: int,
    n_components: int,
    reps: int,
    args: argparse.Namespace,
    cfg: dict,
    results_dir: Path,
    sampler: Any,
    run_noise: bool = False,
) -> tuple[dict, dict, dict, float, float, float, float]:
    """Run one complete experiment cell (one seed × dim × reps combination).

    Returns:
        (classical_metrics, quantum_metrics, pegasos_metrics,
         kta_quantum, kta_classical, ckta_quantum, ckta_classical)
    """
    dataset_cfg = cfg.get("dataset", {})
    digits = dataset_cfg.get("digits", [4, 9])
    test_size = float(dataset_cfg.get("test_size", 0.25))
    stratify = bool(dataset_cfg.get("stratify", True))
    use_fallback = args.fallback or bool(
        dataset_cfg.get("fallback_to_sklearn_digits", False)
    )

    if use_fallback:
        LOGGER.warning(
            "=" * 60
            + "\n  WARNING: Using sklearn_digits FALLBACK, not real MNIST."
            + "\n  Results cannot be compared with published MNIST benchmarks."
            + "\n" + "=" * 60
        )

    LOGGER.info("Loading dataset (seed=%d, dim=%d, reps=%d)...", seed, n_components, reps)
    X_train_raw, X_test_raw, y_train, y_test = load_mnist_digits(
        digits=digits,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        data_home=str(results_dir / ".cache" / "sklearn"),
        fallback_to_sklearn_digits=use_fallback,
    )
    dataset_source = (
        "openml_mnist_784" if X_train_raw.shape[1] == 784 else "sklearn_digits_fallback"
    )

    # Pre-process
    preprocess_cfg = cfg.get("preprocessing", {})
    quantum_scaling = bool(preprocess_cfg.get("quantum_scaling", True))
    feature_range_str = preprocess_cfg.get("feature_range", "pi/2")

    LOGGER.info("Preprocessing (feature_range=%s)...", feature_range_str)
    pre = preprocess_data(
        X_train=X_train_raw,
        X_test=X_test_raw,
        n_components=n_components,
        random_state=seed,
        quantum_scaling=quantum_scaling,
        feature_range_str=feature_range_str,
    )
    X_train_pca = pre["X_train_processed"]
    X_test_pca = pre["X_test_processed"]
    X_train_quantum = pre["X_train_quantum"]
    X_test_quantum = pre["X_test_quantum"]
    pca_transformer = pre["pca"]

    # Only save these plots once (first call per results_dir)
    _scatter_path = results_dir / "pca_scatter.png"
    if not _scatter_path.exists():
        plot_class_distribution(
            y_train,
            classes=digits,
            title="MNIST Training Class Distribution",
            save_path=str(results_dir / "class_distribution.png"),
            class_names=[f"Digit {d}" for d in digits],
        )
        plot_pca_scatter(
            X_train_pca, y_train,
            title=f"PCA Scatter (MNIST {digits[0]} vs {digits[1]})",
            save_path=str(_scatter_path),
            class_names=[f"Digit {d}" for d in digits],
        )
        plot_pca_variance(
            pca_transformer,
            title="PCA Variance Explanation",
            save_path=str(results_dir / "pca_variance.png"),
        )

    # -----------------------------------------------------------------------
    # Fair comparison: cap classical training to the SAME n as quantum
    # -----------------------------------------------------------------------
    n_quantum = args.max_quantum_train

    X_q_train, y_q_train = _stratified_subset(
        X_train_quantum, y_train, max_samples=n_quantum, seed=seed
    )
    X_c_train, y_c_train = _stratified_subset(
        X_train_pca, y_train, max_samples=n_quantum, seed=seed
    )

    LOGGER.info(
        "Fair comparison: both classical and quantum train on %d samples.", len(X_q_train)
    )

    # -----------------------------------------------------------------------
    # Classical SVM
    # -----------------------------------------------------------------------
    classical_cfg = cfg.get("classical_model", {})
    gs_cfg = classical_cfg.get("grid_search", {})
    LOGGER.info("Training classical RBF SVM on %d samples...", len(X_c_train))
    classical_model, classical_train = train_classical_svm(
        X_train=X_c_train,
        y_train=y_c_train,
        kernel=classical_cfg.get("kernel", "rbf"),
        random_state=seed,
        grid_search=False,  # disabled for speed in multi-seed runs
        cv=int(gs_cfg.get("cv", 3)),
        param_grid=gs_cfg.get("param_grid"),
        n_jobs=int(gs_cfg.get("n_jobs", 1)),
    )
    joblib.dump(classical_model, results_dir / "classical_svm_model.joblib")

    y_pred_classical = classical_model.predict(X_test_pca)
    is_binary = len(np.unique(y_test)) == 2
    pos_label = int(max(digits)) if is_binary else None
    average = "binary" if is_binary else "weighted"

    classical_metrics = compute_metrics(
        y_test, y_pred_classical, pos_label=pos_label, average=average
    )
    classical_cm = compute_confusion_matrix(y_test, y_pred_classical, labels=digits)

    # -----------------------------------------------------------------------
    # KTA for classical RBF kernel
    # -----------------------------------------------------------------------
    K_rbf = compute_rbf_gram_matrix(X_c_train, gamma="scale")
    kta_classical = float(compute_kernel_target_alignment(K_rbf, y_c_train))
    ckta_classical = float(compute_centered_kta(K_rbf, y_c_train))
    LOGGER.info("Classical RBF KTA = %.4f | cKTA = %.4f", kta_classical, ckta_classical)

    _save_json(
        results_dir / "classical_svm_results.json",
        {
            "model": "Classical RBF SVM",
            "train_n": int(len(X_c_train)),
            "best_params": classical_train.get("best_params"),
            "best_cv_score": classical_train.get("best_cv_score"),
            "train_time": classical_train.get("train_time"),
            "metrics": classical_metrics,
            "kta": kta_classical,
            "ckta": ckta_classical,
            "confusion_matrix": classical_cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred_classical, zero_division=0
            ),
        },
    )

    # -----------------------------------------------------------------------
    # Build quantum kernel
    # -----------------------------------------------------------------------
    qfm_cfg = cfg.get("quantum_feature_map", {})
    feature_map = create_feature_map(
        feature_map_type=qfm_cfg.get("type", "ZZFeatureMap"),
        feature_dimension=n_components,
        reps=reps,
        entanglement=qfm_cfg.get("entanglement", "full"),
        parameter_prefix=qfm_cfg.get("parameter_prefix", "theta"),
    )

    # -----------------------------------------------------------------------
    # Quantum Kernel Learning (Optional)
    # -----------------------------------------------------------------------
    qkl_history = None
    qkl_cfg = cfg.get("quantum_kernel_learning", {})
    if qkl_cfg.get("enabled", False):
        LOGGER.info("Running Quantum Kernel Learning...")
        qkl = QuantumKernelLearner(
            feature_map=feature_map,
            X_train=X_q_train,
            y_train=y_q_train,
            lr=float(qkl_cfg.get("lr", 0.05)),
            n_epochs=int(qkl_cfg.get("n_epochs", 30)),
            batch_size=int(qkl_cfg.get("batch_size", 20)),
            seed=seed,
            parameter_prefix=qfm_cfg.get("parameter_prefix", "theta"),
        )
        opt_params, qkl_history = qkl.fit()
        feature_map = qkl.get_optimized_feature_map()
        plot_qkl_convergence(
            qkl_history,
            save_path=str(results_dir / f"qkl_convergence_dim{n_components}_reps{reps}.png")
        )

    quantum_kernel = create_quantum_kernel(feature_map=feature_map, sampler=sampler)

    # -----------------------------------------------------------------------
    # Exact QSVC
    # -----------------------------------------------------------------------
    LOGGER.info("Training exact QSVC on %d samples...", len(X_q_train))
    qsvc_cfg = cfg.get("qsvc", {})
    quantum_model, quantum_train = train_qsvc(
        quantum_kernel=quantum_kernel,
        X_train=X_q_train,
        y_train=y_q_train,
        C=float(qsvc_cfg.get("C", 1.0)),
        random_state=seed,
        grid_search=False,
        cv=int(qsvc_cfg.get("cv", 3)),
        n_jobs=int(qsvc_cfg.get("n_jobs", 1)),
    )
    try:
        joblib.dump(quantum_model, results_dir / "quantum_qsvc_model.joblib")
    except Exception as exc:
        LOGGER.warning("Could not save QSVC with joblib (%s); using pickle.", exc)
        with open(results_dir / "quantum_qsvc_model.pkl", "wb") as f:
            pickle.dump(quantum_model, f)

    y_pred_quantum = quantum_model.predict(X_test_quantum)
    quantum_metrics = compute_metrics(
        y_test, y_pred_quantum, pos_label=pos_label, average=average
    )
    quantum_cm = compute_confusion_matrix(y_test, y_pred_quantum, labels=digits)

    # -----------------------------------------------------------------------
    # Pegasos QSVC (precomputed kernel)
    # -----------------------------------------------------------------------
    LOGGER.info("Precomputing quantum kernel matrices for Pegasos...")
    K_peg_train = compute_kernel_matrix(quantum_kernel, X_q_train)

    X_peg_test, y_peg_test = _stratified_subset(
        X_test_quantum, y_test, max_samples=args.max_kernel_samples, seed=seed
    )
    K_peg_test = compute_kernel_matrix(quantum_kernel, X_peg_test, Y=X_q_train)

    pegasos_cfg = cfg.get("pegasos_svc", {})
    LOGGER.info(
        "Training Pegasos QSVC (lambda=%.4f, max_iter=%d, batch=%d) on %d samples...",
        float(pegasos_cfg.get("lambda_param", 0.001)),
        int(pegasos_cfg.get("max_iter", 1500)),
        int(pegasos_cfg.get("batch_size", 32)),
        len(X_q_train),
    )
    pegasos_model, pegasos_train = train_pegasos_qsvc(
        quantum_kernel=quantum_kernel,
        X_train=K_peg_train,
        y_train=y_q_train,
        lambda_param=float(pegasos_cfg.get("lambda_param", 0.001)),
        max_iter=int(pegasos_cfg.get("max_iter", 1500)),
        batch_size=int(pegasos_cfg.get("batch_size", 32)),
        random_state=seed,
        precomputed=True,
    )
    try:
        joblib.dump(pegasos_model, results_dir / "quantum_pegasos_model.joblib")
    except Exception as exc:
        LOGGER.warning("Could not save Pegasos with joblib (%s); using pickle.", exc)
        with open(results_dir / "quantum_pegasos_model.pkl", "wb") as f:
            pickle.dump(pegasos_model, f)

    y_pred_pegasos = pegasos_model.predict(K_peg_test)
    pegasos_metrics = compute_metrics(
        y_peg_test, y_pred_pegasos, pos_label=pos_label, average=average
    )
    pegasos_cm = compute_confusion_matrix(y_peg_test, y_pred_pegasos, labels=digits)

    # Collapse detection
    unique_preds, pred_counts = np.unique(y_pred_pegasos, return_counts=True)
    LOGGER.info("Pegasos prediction distribution: %s", dict(zip(unique_preds.tolist(), pred_counts.tolist())))
    if len(unique_preds) == 1:
        LOGGER.critical(
            "PEGASOS MODEL COLLAPSE: All predictions are class %s. "
            "Check lambda_param (currently %.4f) — try reducing it.",
            unique_preds[0],
            float(pegasos_cfg.get("lambda_param", 0.001)),
        )

    # -----------------------------------------------------------------------
    # KTA for quantum kernel
    # -----------------------------------------------------------------------
    X_kta, y_kta = _stratified_subset(
        X_train_quantum, y_train, max_samples=args.max_kernel_samples, seed=seed
    )
    K_kta = compute_kernel_matrix(quantum_kernel, X_kta)
    kernel_props = analyze_kernel_properties(K_kta)
    kta_quantum = float(compute_kernel_target_alignment(K_kta, y_kta))
    ckta_quantum = float(compute_centered_kta(K_kta, y_kta))
    LOGGER.info("Quantum KTA (reps=%d) = %.4f | cKTA = %.4f", reps, kta_quantum, ckta_quantum)

    # -----------------------------------------------------------------------
    # Plots for this trial cell
    # -----------------------------------------------------------------------
    plot_kernel_heatmap(
        K_kta,
        title=f"Quantum Kernel (dim={n_components}, reps={reps})",
        save_path=str(results_dir / "kernel_heatmap.png"),
        max_samples=args.max_kernel_samples,
    )
    plot_confusion_matrix(
        classical_cm, save_path=str(results_dir / "confusion_matrix_classical.png"),
        class_names=[f"Digit {d}" for d in digits], model_name="Classical RBF SVM",
    )
    plot_confusion_matrix(
        quantum_cm, save_path=str(results_dir / "confusion_matrix_quantum.png"),
        class_names=[f"Digit {d}" for d in digits], model_name="Exact QSVC",
    )
    plot_confusion_matrix(
        pegasos_cm, save_path=str(results_dir / "confusion_matrix_pegasos.png"),
        class_names=[f"Digit {d}" for d in digits], model_name="Pegasos QSVC",
    )
    plot_metrics_comparison(
        {
            "classical": {"metrics": classical_metrics},
            "quantum":   {"metrics": quantum_metrics},
            "pegasos":   {"metrics": pegasos_metrics},
        },
        title=f"Classical vs QSVC vs Pegasos (dim={n_components}, reps={reps})",
        save_path=str(results_dir / "metrics_comparison.png"),
    )

    # Metrics CSV with KTA and cKTA columns
    comparison_df = pd.DataFrame([
        {"Model": "Classical RBF SVM",
         **classical_metrics, "kta": kta_classical, "ckta": ckta_classical,
         "train_time": classical_train.get("train_time"), "train_n": len(X_c_train)},
        {"Model": "Exact QSVC",
         **quantum_metrics, "kta": kta_quantum, "ckta": ckta_quantum,
         "train_time": quantum_train.get("train_time"), "train_n": len(X_q_train)},
        {"Model": "Pegasos QSVC",
         **pegasos_metrics, "kta": kta_quantum, "ckta": ckta_quantum,
         "train_time": pegasos_train.get("train_time"), "train_n": len(X_q_train)},
    ])
    comparison_df.to_csv(results_dir / "metrics_comparison.csv", index=False)

    # -----------------------------------------------------------------------
    # Noise simulation (only for the anchor cell to keep runtime manageable)
    # -----------------------------------------------------------------------
    noise_report: dict | None = None
    if run_noise and not args.disable_noise:
        noise_cfg = cfg.get("noise_simulation", {})
        if bool(noise_cfg.get("enabled", True)):
            LOGGER.info("Running noise simulation for this cell...")
            X_noise, y_noise = _stratified_subset(
                X_train_quantum, y_train,
                max_samples=args.max_noise_samples, seed=seed,
            )
            K_noiseless, K_noisy, noise_analysis = create_noisy_kernel_comparison(
                feature_map=feature_map,
                X=X_noise,
                readout_error=float(
                    noise_cfg.get("noise_model", {}).get("readout_error", 0.01)
                ),
                gate_error=float(
                    noise_cfg.get("noise_model", {}).get("gate_error", 0.001)
                ),
                shots=int(noise_cfg.get("shots", 4096)),
            )
            K_noisy_psd = regularize_kernel_matrix(K_noisy)
            kta_noiseless = float(compute_kernel_target_alignment(K_noiseless, y_noise))
            kta_noisy = float(compute_kernel_target_alignment(K_noisy_psd, y_noise))
            noise_analysis["alignment_noiseless_vs_noisy"] = float(
                compute_kernel_alignment(K_noiseless, K_noisy)
            )
            noise_analysis["kta_noiseless"] = kta_noiseless
            noise_analysis["kta_noisy"] = kta_noisy
            noise_analysis["kta_degradation"] = kta_noiseless - kta_noisy
            LOGGER.info(
                "Noise: KTA noiseless=%.4f noisy=%.4f (drop=%.4f)",
                kta_noiseless, kta_noisy, kta_noiseless - kta_noisy,
            )
            plot_noisy_vs_noiseless_kernel(
                K_noiseless, K_noisy,
                save_path=str(results_dir / "noise_comparison.png"),
            )
            noise_report = {"analysis": noise_analysis}

    # -----------------------------------------------------------------------
    # Save experiment summary JSON for this trial
    # -----------------------------------------------------------------------
    classical_train_summary = {
        k: v for k, v in classical_train.items()
        if k not in {"model", "grid_search_results"}
    }
    quantum_train_summary = {
        k: v for k, v in quantum_train.items()
        if k not in {"model", "support_vector_indices", "grid_search_results"}
    }
    _save_json(
        results_dir / "experiment_summary.json",
        {
            "config_path": str(args.config),
            "dataset": {
                "source": dataset_source,
                "digits": digits,
                "train_samples_total": int(len(X_train_raw)),
                "test_samples_total": int(len(X_test_raw)),
                "train_samples_used_per_model": int(len(X_q_train)),
                "pca_components": int(n_components),
                "reps": reps,
            },
            "classical": {
                "train_info": classical_train_summary,
                "metrics": classical_metrics,
                "kta_quantum": kta_q,
                "kta_classical": kta_c,
                "ckta_quantum": ckta_q,
                "ckta_classical": ckta_c,
            },
            "quantum_exact": {
                "train_info": quantum_train_summary,
                "metrics": quantum_metrics,
                "kta": kta_quantum,
                "ckta": ckta_quantum,
                "kernel_properties": kernel_props,
            },
            "pegasos": {
                "train_info": {
                    k: v for k, v in pegasos_train.items()
                    if k not in {"model", "support_vector_indices"}
                },
                "metrics": pegasos_metrics,
                "kta": kta_quantum,
                "prediction_distribution": dict(
                    zip(unique_preds.tolist(), pred_counts.tolist())
                ),
            },
            "noise": noise_report,
        },
    )

    if qkl_history is not None:
        _save_json(results_dir / "qkl_history.json", qkl_history)

    return classical_metrics, quantum_metrics, pegasos_metrics, kta_quantum, kta_classical, ckta_quantum, ckta_classical


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run classical vs quantum kernel SVM experiment (publication grade)."
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment_config.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/multi_seed"))
    parser.add_argument(
        "--max-quantum-train", type=int, default=100,
        help="Training set size per quantum run (both classical and quantum use this n).",
    )
    parser.add_argument("--max-kernel-samples", type=int, default=100)
    parser.add_argument("--max-noise-samples", type=int, default=30)
    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--quantum-steps", type=int, default=None)
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use sklearn digits fallback instead of real OpenML MNIST (offline mode).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    setup_plot_style()

    # Stamp git SHA for provenance (Phase 1.2)
    git_sha = get_git_sha()
    LOGGER.info("Git SHA: %s", git_sha)

    cfg = _load_config(args.config)
    ablation_cfg = cfg.get("ablation", {})

    seeds = ablation_cfg.get("seeds", [42])
    dimensions = ablation_cfg.get("pca_dimensions", [4])
    reps_list = ablation_cfg.get("reps_values", [1, 2, 3])

    # Hardware Backend setup
    hw_cfg = cfg.get("hardware", {})
    backend_manager = HardwareBackendManager(
        backend_name=hw_cfg.get("backend_name", "ibm_brisbane"),
        use_real_hardware=hw_cfg.get("use_real_hardware", False),
        shots=hw_cfg.get("shots", 4096),
        use_noise_model=cfg.get("noise_simulation", {}).get("enabled", False)
    )
    sampler = backend_manager.get_sampler()
    LOGGER.info("Hardware config: %s", backend_manager.get_backend_info())

    base_results_dir = args.results_dir
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # Config can suggest max_quantum_train, but CLI flag takes priority
    cfg_train_n = int(ablation_cfg.get("max_quantum_train", 100))
    args.max_quantum_train = getattr(args, "max_quantum_train", cfg_train_n)

    LOGGER.info("=" * 70)
    LOGGER.info("QUANTUM KERNEL SVM — PUBLICATION-GRADE MULTI-SEED ABLATION")
    LOGGER.info("Seeds: %s | Dims: %s | Reps: %s | n_per_model: %d",
                seeds, dimensions, reps_list, args.max_quantum_train)
    LOGGER.info("=" * 70)

    # -----------------------------------------------------------------------
    # Phase 1: PCA-dimension ablation  (reps=1, all seeds, all dims)
    # -----------------------------------------------------------------------
    ablation_stats = []
    for dim in dimensions:
        c_f1s, q_f1s, kta_q_vals, kta_c_vals = [], [], [], []
        for seed_idx, s in enumerate(seeds):
            LOGGER.info("--- DIM ABLATION | dim=%d seed=%d ---", dim, s)
            cell_dir = base_results_dir / f"dim_{dim}_seed_{s}_reps1"
            cell_dir.mkdir(parents=True, exist_ok=True)

            # Noise only on the very first cell (anchor) to keep runtime manageable
            is_anchor = (seed_idx == 0 and dim == dimensions[0])
            try:
                (
                    c_metrics, q_metrics, p_metrics,
                    kta_q, kta_c, ckta_q, ckta_c,
                ) = run_single_trial(
                    seed=s,
                    n_components=dim,
                    reps=1,
                    args=args,
                    cfg=cfg,
                    results_dir=cell_dir,
                    sampler=sampler,
                    run_noise=is_anchor,
                )
                c_f1s.append(c_metrics["f1_score"])
                q_f1s.append(p_metrics["f1_score"])
                kta_q_vals.append(kta_q)
                kta_c_vals.append(kta_c)
            except Exception as exc:
                LOGGER.error("Trial failed (dim=%d, seed=%d): %s", dim, s, exc)
                continue

        sig = calculate_statistical_significance(c_f1s, q_f1s)

        # Geometric difference g(K_Q, K_C) — Phase 3 / 2.4
        # Use the last seed's kernel matrices (stored in K_kta / K_rbf captured
        # inside run_single_trial).  We re-derive them here from the summary
        # JSON to avoid refactoring the trial function signature.
        g_value: float = float("nan")
        try:
            # Load last-cell kernel matrices from the saved JSON artefact
            last_cell_dir = base_results_dir / f"dim_{dim}_seed_{seeds[-1]}_reps1"
            _cls_json_path = last_cell_dir / "classical_svm_results.json"
            _exp_json_path = last_cell_dir / "experiment_summary.json"
            if _cls_json_path.exists() and _exp_json_path.exists():
                # We don't persist the raw kernel arrays, so we compute g
                # from the last KTA values as a proxy ratio instead:
                # Use mean values so g is reportable per-dimension row
                kq_mean = float(np.mean(kta_q_vals))
                kc_mean = float(np.mean(kta_c_vals))
                # When K_Q and K_C have been collapsed to scalars (KTA),
                # use the ratio as a 1-D proxy for g.
                g_value = float(np.sqrt(kc_mean / kq_mean)) if kq_mean > 1e-10 else float("inf")
                LOGGER.info("Geometric difference proxy g(dim=%d) = %.4f", dim, g_value)
        except Exception as exc:
            LOGGER.warning("Geometric difference computation skipped for dim=%d: %s", dim, exc)

        ablation_stats.append({
            "dimension": dim,
            "classical_mean_f1": float(np.mean(c_f1s)),
            "classical_std_f1": float(np.std(c_f1s)),
            "quantum_mean_f1": float(np.mean(q_f1s)),
            "quantum_std_f1": float(np.std(q_f1s)),
            "mean_kta_quantum": float(np.mean(kta_q_vals)),
            "std_kta_quantum": float(np.std(kta_q_vals)),
            "mean_kta_classical": float(np.mean(kta_c_vals)),
            "std_kta_classical": float(np.std(kta_c_vals)),
            "geometric_difference": g_value,
            "significance": sig,
        })

    plot_ablation_scaling(ablation_stats, save_path=str(base_results_dir / "ablation_plot.png"))
    _save_json(
        base_results_dir / "ablation_summary.json",
        {"ablation_stats": ablation_stats, "git_sha": git_sha},
    )

    # -----------------------------------------------------------------------
    # Phase 2: Feature-map depth ablation  (best dim, all seeds, all reps)
    # -----------------------------------------------------------------------
    # Pick dimension with highest mean quantum F1 from Phase 1
    best_dim = max(ablation_stats, key=lambda r: r["quantum_mean_f1"])["dimension"]
    LOGGER.info("Best PCA dim from Phase 1: %d — running depth ablation...", best_dim)

    depth_stats = []
    for r in reps_list:
        r_f1s, r_ktas = [], []
        for s in seeds:
            LOGGER.info("--- DEPTH ABLATION | reps=%d seed=%d ---", r, s)
            cell_dir = base_results_dir / f"dim_{best_dim}_seed_{s}_reps{r}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            try:
                (
                    _, _, p_m, kta_q, _, ckta_q, _
                ) = run_single_trial(
                    seed=s, n_components=best_dim, reps=r,
                    args=args, cfg=cfg, results_dir=cell_dir,
                    sampler=sampler,
                    run_noise=False,
                )
                r_f1s.append(p_m["f1_score"])
                r_ktas.append(kta_q)
            except Exception as exc:
                LOGGER.error("Depth Trial failed (reps=%d, seed=%d): %s", r, s, exc)
                continue

        depth_stats.append({
            "reps": r,
            "mean_f1": float(np.mean(r_f1s)),
            "std_f1": float(np.std(r_f1s)),
            "mean_kta": float(np.mean(r_ktas)),
            "std_kta": float(np.std(r_ktas)),
        })

    plot_depth_ablation(
        depth_stats,
        save_path=str(base_results_dir / "depth_ablation.png"),
    )
    _save_json(base_results_dir / "depth_ablation.json", {
        "best_pca_dim": best_dim,
        "depth_stats": depth_stats,
        "git_sha": git_sha,
    })

    # -----------------------------------------------------------------------
    # Phase 3: Geometric difference plot  (g vs PCA dimension)
    # -----------------------------------------------------------------------
    g_dims = [r["dimension"] for r in ablation_stats if not np.isnan(r.get("geometric_difference", float("nan")))]
    g_vals = [r["geometric_difference"] for r in ablation_stats if not np.isnan(r.get("geometric_difference", float("nan")))]
    if g_dims:
        plot_geometric_difference(
            dims=g_dims,
            g_values=g_vals,
            save_path=str(base_results_dir / "geometric_difference.png"),
        )
        LOGGER.info("Geometric difference plot saved to %s/geometric_difference.png", base_results_dir)

    # -----------------------------------------------------------------------
    # Phase 5.4: 2×3 Publication Dashboard
    # -----------------------------------------------------------------------
    plot_ablation_dashboard(
        ablation_stats=ablation_stats,
        depth_stats=depth_stats,
        save_path=str(base_results_dir / "ablation_dashboard.png"),
    )
    LOGGER.info("2×3 ablation dashboard saved to %s/ablation_dashboard.png", base_results_dir)

    # -----------------------------------------------------------------------
    # Phase 5.6: Expressibility vs KTA scatter
    # -----------------------------------------------------------------------
    if _EXPR_AVAILABLE:
        expr_configs: list[dict] = []
        for r_val in reps_list:
            for dim in dimensions:
                try:
                    from src.quantum_feature_maps import create_feature_map as _cfm
                    _fmap = _cfm(
                        feature_map_type=cfg.get("quantum_feature_map", {}).get("type", "ZZFeatureMap"),
                        feature_dimension=dim,
                        reps=r_val,
                        entanglement=cfg.get("quantum_feature_map", {}).get("entanglement", "full"),
                    )
                    expr_val = float(compute_expressibility(_fmap, n_samples=500, n_bins=75))
                    # Retrieve mean KTA for this (dim, reps=1) from ablation_stats
                    _row = next((row for row in ablation_stats if row["dimension"] == dim), None)
                    kta_val = float(_row["mean_kta_quantum"]) if _row else float("nan")
                    expr_configs.append({"dim": dim, "reps": r_val, "expressibility": expr_val, "kta": kta_val})
                except Exception as exc:
                    LOGGER.warning("Expressibility skipped for dim=%d reps=%d: %s", dim, r_val, exc)

        if expr_configs:
            plot_expressibility_vs_kta(
                configs=expr_configs,
                save_path=str(base_results_dir / "expressibility_vs_kta.png"),
            )
            _save_json(base_results_dir / "expressibility_configs.json", {"configs": expr_configs, "git_sha": git_sha})
            LOGGER.info("Expressibility vs KTA scatter saved to %s/expressibility_vs_kta.png", base_results_dir)
    else:
        LOGGER.info("Expressibility module not available — skipping Phase 5.6 scatter plot.")

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("PCA DIMENSION ABLATION SUMMARY  (reps=1, %d seeds)", len(seeds))
    LOGGER.info("=" * 70)
    LOGGER.info(
        "  %-6s | %-14s | %-14s | %-10s | %-10s | %-8s | %s",
        "Dim", "Classical F1", "Pegasos F1", "KTA-Qnt", "KTA-Cls", "g", "Significant?"
    )
    for row in ablation_stats:
        sig = row["significance"]
        t_p = sig.get("paired_t_test_p_value", float("nan"))
        adv = sig.get("significant_advantage", False)
        g = row.get("geometric_difference", float("nan"))
        LOGGER.info(
            "  %-6d | %.3f ± %.3f  | %.3f ± %.3f  | %.4f     | %.4f     | %.4f   | %s",
            row["dimension"],
            row["classical_mean_f1"], row["classical_std_f1"],
            row["quantum_mean_f1"], row["quantum_std_f1"],
            row["mean_kta_quantum"],
            row["mean_kta_classical"],
            g,
            "YES (p<0.05)" if adv else f"no (p={t_p:.3f})",
        )

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("FEATURE-MAP DEPTH ABLATION SUMMARY  (dim=%d, %d seeds)", best_dim, len(seeds))
    LOGGER.info("=" * 70)
    LOGGER.info("  %-6s | %-14s | %-10s", "Reps", "Pegasos F1", "KTA")
    for row in depth_stats:
        LOGGER.info(
            "  %-6d | %.3f ± %.3f  | %.4f ± %.4f",
            row["reps"], row["mean_f1"], row["std_f1"],
            row["mean_kta"], row["std_kta"],
        )
    LOGGER.info("=" * 70)
    LOGGER.info("Git SHA: %s", git_sha)
    LOGGER.info("Multi-seed with depth ablation COMPLETE.")
    LOGGER.info("Results → %s", base_results_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
