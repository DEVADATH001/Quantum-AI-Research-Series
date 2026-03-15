# Upgrade Notes (2026-03-10)

## What Changed

- Added `run_experiment.py` for a reproducible end-to-end run:
  - data loading
  - preprocessing
  - classical baseline training
  - quantum kernel + Pegasos training
  - noise analysis
  - artifact export to `results/`

- Fixed major runtime and correctness issues:
  - `src/__init__.py` now uses lazy imports (no hard crash when optional deps are missing).
  - `src/evaluation_metrics.py` now predicts with `X_test` (not `y_test`).
  - `src/preprocessing.py` now scales test quantum features with train min/max.
  - `src/classical_models.py` kernel-matrix helper now computes a real kernel matrix.
  - `src/quantum_kernel_engine.py` now works with current Qiskit/Qiskit ML APIs.
  - `src/quantum_training.py` now maps legacy Pegasos params to current API.
  - `src/noise_simulation.py` now creates a real noisy sampler path for modern Qiskit.
  - `src/visualization.py` now works with and without `seaborn`.
  - `src/data_loader.py` now uses a local cache path and falls back to sklearn digits if OpenML fails.

- Updated dependency versions in `requirements.txt` for Qiskit 2.x era.

## Recommended Run Command

```bash
python run_experiment.py
```

Faster smoke run:

```bash
python run_experiment.py --max-quantum-train 40 --quantum-steps 120
```

## Outputs

After a successful run, check `results/` for:

- `metrics_comparison.csv`
- `experiment_summary.json`
- `kernel_heatmap.png`
- `confusion_matrix_classical.png`
- `confusion_matrix_quantum.png`
- `metrics_comparison.png`
