# Project 03: Quantum Kernel SVM for MNIST

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qiskit ML](https://img.shields.io/badge/Qiskit%20Machine%20Learning-QSVC%20%7C%20Pegasos-6929C4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RBF%20SVM-F7931E)
![Status](https://img.shields.io/badge/Status-Research%20Benchmark-orange)

Research benchmark for comparing classical RBF-SVM baselines with quantum
kernel SVMs on binary digit classification.

The default target task is MNIST digit `4` versus digit `9`. The experiment
compresses images with PCA, encodes the reduced features into a quantum feature
map, computes quantum-kernel diagnostics, trains QSVC variants, and compares the
results against a tuned classical SVM under a matched sample budget.

## README Audit

The previous README was ambitious and detailed, but it needed a hard editorial
pass before being suitable for researchers and maintainers.

What was missing or weak:

- The file contained severe encoding corruption, making equations, diagrams,
  symbols, and tables hard to read on GitHub.
- The tone overstated maturity with phrases such as "publication-grade" and
  "research grade" without consistently separating implemented features from
  aspirational outputs.
- Some performance ranges were presented as expected results rather than
  artifact-backed observations.
- The statistical section did not sufficiently warn that the
  `significant_advantage` field is not direction-aware; a low p-value can still
  correspond to classical outperforming quantum.
- The README claimed several generated files and plots broadly, but the checked
  results in this repository are split across `results/`, `results/multi_seed/`,
  and `results/final_validation/`, and not every listed artifact exists in each
  run.
- The hardware path was described too confidently. The code has a hardware
  backend manager, but the default reproducible path is simulator-based.
- The fallback dataset warning was present but deserved stronger placement:
  sklearn digits is an 8x8 toy dataset and must not be compared to OpenML MNIST.
- The architecture diagram was visually elaborate but unreadable due to
  mojibake. A simpler ASCII architecture is more maintainable.

This rewrite keeps the real technical strengths while removing overstated or
stale claims.

## Project Overview

This module evaluates quantum kernel classifiers against classical SVM
baselines on a controlled binary image-classification task.

| Component | Implementation |
|---|---|
| Dataset | OpenML `mnist_784` by default; sklearn `load_digits` fallback for offline/CI smoke runs |
| Task | Binary classification, default digits `[4, 9]` |
| Preprocessing | Pixel normalization, standardization, PCA, quantum-angle scaling |
| Classical baseline | RBF-SVM with optional `GridSearchCV` |
| Quantum classifiers | Exact `QSVC` and `PegasosQSVC` |
| Quantum kernel | `FidelityQuantumKernel` with configurable feature map |
| Feature map | Default `ZZFeatureMap`, PCA dimension equals qubit count |
| Diagnostics | KTA, centered KTA, kernel properties, geometric difference, expressibility, optional noise analysis |
| Outputs | JSON, CSV, plots, model files, NumPy arrays, and run directories |

The central question is not "does this prove quantum advantage?" It is:

> Under a matched training budget, does the quantum kernel geometry provide a
> useful classification signal relative to a tuned classical RBF kernel?

## Motivation / Research Context

Quantum kernels map classical data into quantum states and classify using state
overlap. They are a natural way to study whether quantum feature spaces expose
useful decision geometry that a classical kernel misses.

This repository treats that question empirically:

- train classical and quantum models on the same number of examples;
- measure both predictive performance and kernel geometry;
- vary PCA dimension and feature-map depth;
- keep noisy simulation and real-hardware paths separate from exact simulation;
- avoid claiming quantum advantage without stronger statistical and hardware
  evidence.

Digits `4` and `9` are used because they are visually similar enough to make
the binary task nontrivial after dimensionality reduction. The fallback
`sklearn_digits` path is useful for smoke testing only.

## Why This README Structure Matters

- **Overview** tells readers what the project actually runs.
- **Research context** explains why quantum kernels are being tested.
- **Architecture** helps engineers find the relevant modules quickly.
- **Mathematics** makes the kernel and metric definitions auditable.
- **Usage** helps new users run either a full experiment or a smoke test.
- **Results and limitations** keep the interpretation scientifically honest.

## Key Features

- End-to-end CLI runner: [`run_experiment.py`](run_experiment.py).
- OpenML MNIST loader with explicit sklearn-digits fallback mode.
- Matched classical/quantum training sample budget via `--max-quantum-train`.
- PCA dimension ablation over configured dimensions.
- Feature-map depth ablation over configured `reps` values.
- Classical RBF-SVM baseline with grid search.
- Exact QSVC and Pegasos QSVC quantum classifiers.
- Precomputed-kernel path for Pegasos compatibility across Qiskit ML versions.
- Kernel diagnostics: KTA, centered KTA, eigenvalue/PSD checks, and
  regularization.
- Geometric-difference and expressibility utilities.
- Optional QKAT-style quantum kernel learning module.
- Backend manager for statevector simulation, fake-backend/noisy Aer paths, and
  optional IBM Runtime hardware mode.
- Unit tests for preprocessing, kernel regularization, KTA, and statistical
  helper behavior.

## System Architecture

```text
03-Quantum-Kernel-SVM-MNIST
|
|-- run_experiment.py
|   |-- load config and CLI overrides
|   |-- create backend/sampler
|   |-- run PCA-dimension ablation
|   |-- run depth ablation at best dimension
|   `-- save aggregate artifacts
|
|-- src/data_loader.py
|   `-- OpenML MNIST or sklearn_digits fallback
|
|-- src/preprocessing.py
|   `-- normalize -> standardize -> PCA -> angle scaling
|
|-- src/quantum_feature_maps.py / feature_map_registry.py
|   `-- build configured quantum feature map
|
|-- src/quantum_kernel_engine.py
|   `-- kernel construction, KTA, cKTA, PSD checks, geometric difference
|
|-- src/classical_models.py
|   `-- RBF-SVM baseline and classical Gram matrix
|
|-- src/quantum_training.py
|   `-- QSVC and PegasosQSVC training
|
|-- src/noise_simulation.py / hardware_backend.py
|   `-- simulator, fake-backend noise, optional IBM Runtime
|
|-- src/evaluation_metrics.py
|   `-- accuracy, precision, recall, F1, statistical helpers
|
`-- src/visualization.py
    `-- plots for PCA, kernels, confusion matrices, ablations, and diagnostics
```

## Mathematical Foundations

### Quantum Feature Map

Classical features are embedded into a quantum state:

```math
|\phi(x)\rangle = U_\phi(x)|0\rangle^{\otimes d}.
```

Here `d` is the PCA dimension and also the number of qubits. The default
feature map is `ZZFeatureMap`, which encodes single-feature and pairwise
interactions through parameterized Pauli-Z and ZZ terms.

### Fidelity Quantum Kernel

The quantum kernel is the squared state overlap:

```math
K_Q(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2.
```

In Qiskit this is implemented through `FidelityQuantumKernel`. The kernel matrix
is then passed to QSVC-style classifiers.

### Classical RBF Kernel

The classical baseline uses an RBF kernel:

```math
K_{\mathrm{RBF}}(x_i, x_j)
= \exp(-\gamma ||x_i - x_j||^2).
```

This is a strong baseline for low-dimensional PCA features and should not be
treated as a weak comparison.

### Kernel-Target Alignment

Kernel-Target Alignment measures how well a kernel agrees with the label
kernel:

```math
\mathrm{KTA}(K, y) =
\frac{\langle K, yy^T \rangle_F}
{||K||_F ||yy^T||_F}.
```

The code also computes centered KTA for some trial summaries.

### Geometric Difference

The geometric-difference diagnostic compares quantum and classical kernel
geometry:

```math
g(K_Q, K_C) =
\sqrt{||K_C^{1/2} K_Q^{-1} K_C^{1/2}||_2}.
```

This is a diagnostic, not a proof of advantage. A favorable value is at most a
necessary condition under the assumptions of the cited theory.

### Pegasos SVM Objective

Pegasos optimizes a regularized hinge-loss objective:

```math
\min_w \frac{\lambda}{2} ||w||^2
+ \frac{1}{n}\sum_i \max(0, 1 - y_i \langle w, \phi(x_i) \rangle).
```

The configured `lambda_param` is `0.001` to avoid over-regularized collapse on
small training sets.

### Computational Complexity

Quantum-kernel matrix construction is the dominant cost:

```text
Kernel construction: O(n^2 * circuit_cost)
Exact SVM training:  O(n^2) to O(n^3), solver-dependent
Pegasos training:   O(max_iter * batch_size), after kernel evaluation
PCA preprocessing:  usually negligible relative to kernel evaluation
```

For `n=100`, a full training Gram matrix has `10,000` entries before test-kernel
or diagnostic matrices are considered.

## Technologies Used

- Python 3.10+
- NumPy, SciPy, pandas, joblib
- matplotlib, seaborn, plotly
- scikit-learn
- Qiskit
- Qiskit Aer
- Qiskit IBM Runtime
- Qiskit Machine Learning
- PyYAML
- Jupyter / notebook / ipykernel

## Repository Structure

```text
03-Quantum-Kernel-SVM-MNIST/
|-- README.md
|-- UPGRADE_NOTES.md
|-- KNOWLEDGE_BASE.md
|-- requirements.txt
|-- run_experiment.py
|-- refactor_harness.py
|-- config/
|   |-- experiment_config.yaml
|   `-- test_config.yaml
|-- notebooks/
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_classical_baseline.ipynb
|   `-- 03_quantum_kernel_svm.ipynb
|-- src/
|   |-- classical_models.py
|   |-- data_loader.py
|   |-- evaluation_metrics.py
|   |-- expressibility.py
|   |-- feature_map_registry.py
|   |-- hardware_backend.py
|   |-- kernel_learning.py
|   |-- kernel_training.py
|   |-- noise_simulation.py
|   |-- preprocessing.py
|   |-- quantum_feature_maps.py
|   |-- quantum_kernel_engine.py
|   |-- quantum_training.py
|   `-- visualization.py
|-- tests/
|   `-- test_pipeline.py
`-- results/
    |-- experiment_summary.json
    |-- metrics_comparison.csv
    |-- multi_seed/
    |-- final_validation/
    `-- *.png / *.npy / *.joblib artifacts
```

## Installation Guide

From the module directory:

```powershell
cd 03-Quantum-Kernel-SVM-MNIST
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first OpenML MNIST run needs network access and can create a local cache
under `results/.cache/sklearn`. Use `--fallback` for a network-free smoke run.

For optional real IBM hardware execution, configure credentials from the
repository root:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 03-Quantum-Kernel-SVM-MNIST
```

Then set `hardware.use_real_hardware: true` in
[`config/experiment_config.yaml`](config/experiment_config.yaml). The default is
simulation.

## Usage Instructions

### Full Configured Experiment

```powershell
python run_experiment.py
```

This uses OpenML MNIST by default, runs configured PCA-dimension and depth
ablations, and writes outputs to `results/multi_seed`.

### Offline / CI Smoke Run

```powershell
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

This uses sklearn's built-in 8x8 digits dataset. It is useful for validating the
pipeline but is not comparable to MNIST results.

### Smaller Custom Run

```powershell
python run_experiment.py `
  --results-dir results/smoke_local `
  --fallback `
  --max-quantum-train 40 `
  --max-kernel-samples 40 `
  --disable-noise `
  --log-level INFO
```

### CLI Options

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `config/experiment_config.yaml` | YAML configuration file |
| `--results-dir` | `results/multi_seed` | Output directory |
| `--max-quantum-train` | `100` | Matched training sample cap for classical and quantum models |
| `--max-kernel-samples` | `100` | Kernel diagnostic matrix sample cap |
| `--max-noise-samples` | `30` | Sample cap for noise comparison |
| `--disable-noise` | off | Skip noise analysis |
| `--quantum-steps` | `None` | Legacy/compatibility option parsed by CLI |
| `--fallback` | off | Use sklearn digits instead of OpenML MNIST |
| `--log-level` | `INFO` | Python logging level |

### Tests

```powershell
python -m unittest discover -s tests
```

or:

```powershell
python -m pytest tests -q
```

## Example Results / Visualizations

Current checked/generated result locations include:

- `results/experiment_summary.json`
- `results/metrics_comparison.csv`
- `results/multi_seed/ablation_summary.json`
- `results/multi_seed/depth_ablation.json`
- `results/final_validation/ablation_summary.json`
- `results/*.png`
- `results/multi_seed/*.png`

The root `results/experiment_summary.json` currently records a fallback
sklearn-digits run:

| Model | F1 score | Accuracy |
|---|---:|---:|
| Classical RBF-SVM | `1.0000` | `1.0000` |
| Quantum Pegasos SVM | `0.6618` | `0.4945` |

The `results/multi_seed/ablation_summary.json` artifact reports:

| PCA Dim | Classical Mean F1 | Quantum Mean F1 | Mean Quantum KTA | Significance Flag |
|---:|---:|---:|---:|---|
| `4` | `0.5693` | `0.6164` | `0.1281` | `false` |
| `6` | `0.6415` | `0.6984` | `0.1876` | `false` |
| `8` | `0.6840` | `0.8123` | `0.2270` | `false` |

The `results/multi_seed/depth_ablation.json` artifact reports best PCA
dimension `8` and decreasing mean F1 as reps increases:

| Reps | Mean F1 | Mean KTA |
|---:|---:|---:|
| `1` | `0.8630` | `0.2251` |
| `2` | `0.7997` | `0.2285` |
| `3` | `0.6877` | `0.2264` |

Interpretation: these saved artifacts are useful diagnostics, but they are not
evidence of quantum advantage. Confirm dataset source, seed count, sample
budget, and whether fallback mode was used before citing any result.

## Experimental Setup

Default config file: [`config/experiment_config.yaml`](config/experiment_config.yaml)

| Component | Default Setting |
|---|---|
| Dataset | OpenML `mnist_784` |
| Fallback dataset | sklearn `load_digits` |
| Digits | `[4, 9]` |
| Test split | `0.25`, stratified |
| PCA dimensions | `[4, 6, 8]` in ablation |
| Quantum feature map | `ZZFeatureMap` |
| Feature-map reps | `[1, 2, 3]` in depth ablation |
| Entanglement | `linear` |
| Quantum feature range | `[0, pi/2]` |
| Classical baseline | RBF-SVM with grid search |
| Exact QSVC | `QSVC` with configurable `C` grid |
| Pegasos QSVC | `lambda_param=0.001`, `max_iter=1500`, `batch_size=32` |
| Default train cap | `100` samples |
| Noise model | optional FakeBrisbane/Aer path |
| Hardware | disabled by default |

## Performance Metrics

- `accuracy`: fraction of correctly classified test examples.
- `precision`, `recall`, `f1_score`: binary classification metrics using the
  larger digit label as positive class unless otherwise configured.
- `KTA`: alignment between a kernel matrix and label kernel.
- `centered KTA`: KTA after centering kernels.
- `kernel_properties`: symmetry, PSD status, eigenvalues, condition number, and
  diagonal/off-diagonal statistics.
- `geometric_difference`: structural comparison of quantum and classical
  kernels.
- `train_time`: wall-clock model-training time.
- `noise_analysis`: optional comparison between noiseless and noisy kernels.
- `git_sha`: commit stamp where available.

Important: the current `significant_advantage` helper flags low p-values without
encoding whether quantum is better than classical. Always compare means and
effect direction before interpreting the flag.

## Limitations

- No quantum advantage is demonstrated by this module.
- OpenML MNIST and sklearn fallback results are not comparable.
- The default quantum training cap is small (`100` samples), mainly to keep
  kernel computation tractable.
- Kernel construction scales quadratically in sample count.
- Statevector simulation does not represent real hardware noise.
- Fake-backend or Aer noise models approximate device behavior but are not live
  calibration studies.
- Real hardware mode exists, but it is not the default validated path.
- Statistical evidence is sensitive to seed count, sample cap, and fallback
  mode.
- The current significance helper is not direction-aware.
- Some checked artifacts are from fallback or validation runs rather than one
  clean canonical benchmark directory.
- `results/.cache` may contain OpenML cache files and restricted temporary
  paths; generated data should be curated before release.

## Future Improvements

- Fix the statistical helper so significance flags include direction and effect
  size.
- Add a canonical result manifest that states dataset source, fallback status,
  seed list, git SHA, and config hash.
- Separate smoke-test artifacts from research artifacts.
- Add a top-level command for a small deterministic smoke test.
- Add exact kernel-matrix persistence for stronger geometric-difference audits.
- Add real hardware evaluation slices with queue, calibration, shot, and backend
  metadata.
- Add Nystrom or batching approximations for larger training sets.
- Add more feature maps beyond `ZZFeatureMap`.
- Add cross-dataset evaluation such as Fashion-MNIST or binary CIFAR features.

## References

- Havlicek et al. Supervised learning with quantum-enhanced feature spaces.
  *Nature*, 2019.
- Schuld and Killoran. Quantum machine learning in feature Hilbert spaces.
  *Physical Review Letters*, 2019.
- Shalev-Shwartz et al. Pegasos: Primal estimated sub-gradient solver for SVM.
  *Mathematical Programming*, 2011.
- Huang et al. Quantum kernel methods work on classical data?
  *Nature Communications*, 2021.
- Thanasilp et al. Exponential concentration and untrainability in quantum
  kernel methods. arXiv:2208.11060, 2022.
- Sim et al. Expressibility and entangling capability of parameterized quantum
  circuits. *Advanced Quantum Technologies*, 2019.
- LeCun et al. Gradient-based learning applied to document recognition.
  *Proceedings of the IEEE*, 1998.
- Qiskit Machine Learning documentation:
  https://qiskit-community.github.io/qiskit-machine-learning/

## Author Information

**DEVADATH H K**

Project 03 of the
[`Quantum AI Research Series`](../README.md). See the repository-level
[`LICENSE`](../LICENSE), [`CITATION.cff`](../CITATION.cff), and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) files for licensing, citation, and
contribution guidance.
