# Quantum Kernel SVM for MNIST Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qiskit-1.x%20%7C%202.x-6929C4?logo=ibm&logoColor=white" alt="Qiskit">
  <img src="https://img.shields.io/badge/qiskit--machine--learning-≥0.9-8A3FFC" alt="Qiskit ML">
  <img src="https://img.shields.io/badge/scikit--learn-≥1.4-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Research%20Grade-red" alt="Research Grade">
</p>

> **A publication-grade empirical study of quantum kernel methods on MNIST.**  
> Implements fair classical–quantum SVM comparisons, multi-seed statistical ablations,  
> Kernel-Target Alignment (KTA) tracking, feature-map depth ablations, noise robustness  
> analysis, and geometric difference metrics — all through a single reproducible CLI pipeline.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Context & Motivation](#2-research-context--motivation)
3. [Key Contributions](#3-key-contributions)
4. [System Architecture](#4-system-architecture)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [Module Reference](#6-module-reference)
7. [Repository Structure](#7-repository-structure)
8. [Installation](#8-installation)
9. [Usage](#9-usage)
10. [Experimental Setup](#10-experimental-setup)
11. [Output Artefacts](#11-output-artefacts)
12. [Performance Benchmarks](#12-performance-benchmarks)
13. [Visualizations](#13-visualizations)
14. [Computational Complexity](#14-computational-complexity)
15. [Limitations](#15-limitations)
16. [Future Work](#16-future-work)
17. [References](#17-references)
18. [Author](#18-author)

---

## 1. Project Overview

This project is a **research-grade empirical evaluation** of quantum kernel Support Vector Machines (SVMs) applied to binary MNIST digit classification (digit 4 vs. digit 9). It goes beyond a tutorial implementation by enforcing **fair experimental comparisons**, tracking **kernel geometry metrics**, running **multi-seed statistical tests**, and measuring the **effect of circuit depth and hardware noise** on classification performance.

The primary executable is `run_experiment.py`, a single reproducible CLI pipeline that:

- Loads real MNIST from OpenML (70 000 samples, 28×28 pixels) or falls back to sklearn digits offline
- Applies a reproducible PCA → scaling → quantum-encoding preprocessing pipeline
- Trains three models on the **same** *n* samples: Classical RBF SVM, Exact QSVC, and Pegasos QSVC
- Computes Kernel-Target Alignment (KTA) for both quantum and classical kernels
- Runs a multi-seed dimension ablation (PCA dims ∈ {4, 6, 8}) and a feature-map depth ablation (reps ∈ {1, 2, 3})
- Generates publication-quality 300 DPI figures and structured JSON result artefacts

---

## 2. Research Context & Motivation

### Why quantum kernels?

Classical kernel SVMs implicitly operate in infinite-dimensional feature spaces via the kernel trick. Quantum kernels offer a **physically distinct** feature space — the exponentially large Hilbert space of *n* qubits — which cannot be efficiently simulated classically for large *n*.

The central research question is:

> *Does the quantum feature space defined by ZZFeatureMap provide a classically-hard kernel that leads to better binary classification on MNIST compared to a well-tuned RBF kernel under a fixed and equal training budget?*

### The NISQ constraint

Current quantum hardware is in the **Noisy Intermediate-Scale Quantum (NISQ)** era: 50–433 qubits, gate error rates of ~0.1–1%, no fault-tolerant error correction, and strict circuit depth budgets. This project works within these constraints:

- **PCA to ≤ 8 qubits** — tractable on statevector simulators; deployable on real hardware
- **reps = 1 default** — minimises circuit depth while preserving entanglement capacity
- **Noise simulation** — models realistic IBM Brisbane readout and gate errors
- **Kernel regularisation** — restores positive semi-definiteness after noisy kernel computation

### Why 4 vs. 9?

Digits 4 and 9 share significant topological overlap (closed loops, similar stroke patterns), making them the hardest MNIST binary pair. A classifier that separates them well is more informative than one separating 0 vs. 1.

---

## 3. Key Contributions

This project provides several features **beyond a standard tutorial**:

| Contribution | Details |
|---|---|
| **Fair comparison** | Classical and quantum models trained on identical *n* samples per trial |
| **KTA tracking** | Kernel-Target Alignment computed for both RBF and quantum kernels in every trial |
| **Feature-map depth ablation** | reps ∈ {1, 2, 3} sweep with F1 and KTA per depth (novel empirical result) |
| **Multi-seed statistics** | Paired t-test + Bonferroni correction + Bootstrap 95% CI + Cohen's d effect size |
| **Geometric difference g(K\_Q, K\_C)** | Necessary condition for quantum advantage (Huang et al. 2022) computed per PCA dimension |
| **Expressibility ε** | KL divergence from Haar distribution (Sim et al. 2019) per (dim, reps) config |
| **Noise robustness curve** | KTA vs. readout error sweep 0 → 5% |
| **QKAT** | Gradient-based kernel alignment training via `src/kernel_training.py` |
| **Git SHA provenance** | Every result JSON is stamped with the commit hash for reproducibility |
| **300 DPI publication figures** | IEEE/ACM camera-ready quality; serif font; heavier axis spines |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         run_experiment.py                           │
│            (Reproducible CLI Orchestrator — Single Entry Point)     │
└──────────┬─────────────────────┬───────────────────┬───────────────┘
           │                     │                   │
           ▼                     ▼                   ▼
   ┌──────────────┐   ┌─────────────────────┐  ┌────────────────────┐
   │ data_loader  │   │   preprocessing      │  │  feature_map_      │
   │  (OpenML /   │──▶│  PCA · Scale ·       │  │  registry          │
   │  sklearn fb) │   │  Quantum Encode      │  │  (plugin system)   │
   └──────────────┘   └─────────┬───────────┘  └────────┬───────────┘
                                │                        │
                    ┌───────────▼────────────────────────▼──────────┐
                    │            quantum_kernel_engine               │
                    │  FidelityQuantumKernel · KTA · compute_g ·    │
                    │  regularize_kernel · get_git_sha               │
                    └──────┬───────────────────┬────────────────────┘
                           │                   │
              ┌────────────▼──────┐  ┌─────────▼───────────────┐
              │  classical_models │  │   quantum_training        │
              │  RBF SVM          │  │   Exact QSVC             │
              │  GridSearchCV     │  │   Pegasos QSVC (precomp) │
              └────────────┬──────┘  └─────────┬───────────────┘
                           │                   │
              ┌────────────▼───────────────────▼───────────────────┐
              │               evaluation_metrics                    │
              │  accuracy · precision · recall · F1                 │
              │  Bootstrap CI · Cohen's d · Bonferroni · KTA       │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │               noise_simulation                      │
              │  IBM Brisbane noise model · ΔK heatmap             │
              │  KTA degradation curve · readout error sweep       │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │     expressibility · kernel_training (QKAT)        │
              │  KL(P_PQC ‖ P_Haar) · gradient kernel alignment   │
              └────────────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────▼───────────────────────────┐
              │                  visualization                      │
              │  2×3 dashboard · t-SNE kernel geometry             │
              │  g(K_Q,K_C) plot · ε vs KTA scatter               │
              │  noise robustness · scalability curves             │
              └────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Foundations

### 5.1 ZZFeatureMap Encoding

Classical data **x** ∈ ℝᵈ is encoded into a quantum state via a parameterised circuit:

```
|φ(x)⟩ = U_Φ(x) H^⊗n |0⟩^⊗n
```

where the encoding unitary for the ZZ feature map is:

```
U_Φ(x) = exp(i Σ_{i<j} x_i x_j Z_i Z_j) · exp(i Σ_i x_i Z_i)
```

- **H^⊗n**: Hadamard layer creating uniform superposition
- **Z_i**: Single-qubit Pauli-Z rotation at angle x_i
- **Z_i Z_j**: Two-qubit ZZ interaction at angle x_i x_j (entanglement)

### 5.2 Quantum Kernel (Fidelity Kernel)

The quantum kernel measures the fidelity between two encoded quantum states:

```
K(x_i, x_j) = |⟨φ(x_i) | φ(x_j)⟩|²
             = |⟨0|^⊗n U_Φ†(x_i) U_Φ(x_j) |0⟩^⊗n|²
```

This is estimated by executing the **overlap circuit** `U_Φ†(x_i) U_Φ(x_j)` on a quantum backend and measuring the probability of the all-zeros bitstring.

### 5.3 Kernel-Target Alignment (KTA)

KTA measures how well a kernel matrix aligns with the ideal label-based kernel:

```
KTA(K, y) = ⟨K, K_y⟩_F / (‖K‖_F · ‖K_y‖_F)
```

where **K_y = y yᵀ** is the ideal label kernel (K_y[i,j] = +1 if same class, −1 otherwise), and ⟨·,·⟩_F is the Frobenius inner product.

- **KTA = 1**: Perfect alignment (kernel perfectly separates classes)
- **KTA = 0**: No alignment (kernel provides no useful signal)

### 5.4 Geometric Difference g(K_Q, K_C)

The geometric difference quantifies the structural dissimilarity between quantum and classical kernels (Huang et al. 2022):

```
g(K_Q, K_C) = sqrt( ‖ K_C^{1/2} K_Q^{-1} K_C^{1/2} ‖₂ )
```

- **g > 1**: Necessary (not sufficient) condition for quantum kernel advantage over the classical kernel on the same task
- **g ≤ 1**: No quantum advantage can be guaranteed from kernel geometry alone

### 5.5 Circuit Expressibility

Expressibility ε measures how well a parameterised circuit samples the full unitary group (Haar distribution):

```
ε = KL( P_PQC(F; θ) ‖ P_Haar(F) )
```

where F = |⟨φ(θ₁) | φ(θ₂)⟩|² is the fidelity distribution over random parameter pairs.

- **ε → 0**: Circuit approaches Haar-random — highly expressive, prone to barren plateaus
- **ε >> 0**: Circuit visits only a restricted subspace — trainable but less powerful

### 5.6 Pegasos Optimisation

Pegasos (Shalev-Shwartz et al. 2011) solves the kernelised SVM primal via stochastic subgradient descent:

```
w_{t+1} = (1 − η_t λ) w_t + η_t Σ_{i ∈ A_t} y_i K(:, i)
```

where η_t = 1/(λ t) is the decaying step size, λ is the regularisation parameter, and **A_t** is the mini-batch at step t. The key hyperparameter is **λ = 0.001** (equivalent to C = 1000), chosen to prevent model collapse on small (n ≤ 200) datasets.

---

## 6. Module Reference

| Module | Role |
|---|---|
| `src/data_loader.py` | OpenML MNIST fetch with local sklearn cache fallback; `--fallback` flag |
| `src/preprocessing.py` | Normalise → Standardise → PCA → Scale to [0, π/2]; reproducible train/test split |
| `src/feature_map_registry.py` | Plugin registry for extensible feature map construction |
| `src/quantum_feature_maps.py` | ZZFeatureMap builder; validated against Qiskit 1.x / 2.x API |
| `src/quantum_kernel_engine.py` | FidelityQuantumKernel · KTA computation · `compute_geometric_difference` · `get_git_sha` |
| `src/classical_models.py` | RBF SVM with GridSearchCV; computes RBF Gram matrix for KTA |
| `src/quantum_training.py` | Exact QSVC + Pegasos QSVC (precomputed kernel path); collapse detection |
| `src/evaluation_metrics.py` | F1/precision/recall/accuracy; Bootstrap 95% CI; Cohen's d; Bonferroni correction |
| `src/noise_simulation.py` | IBM Brisbane readout + gate error noise model; ΔK analysis |
| `src/expressibility.py` | KL(P_PQC ‖ P_Haar) over sampled fidelities (Sim et al. 2019) |
| `src/kernel_training.py` | QKAT — gradient-based kernel alignment training |
| `src/visualization.py` | 300 DPI publication figures: 2×3 dashboard, t-SNE, g-plot, ε vs KTA, noise curves |

---

## 7. Repository Structure

```
03-Quantum-Kernel-SVM-MNIST/
├── README.md                         # This document
├── UPGRADE_NOTES.md                  # Changelog from prototype to research grade
├── requirements.txt                  # Pinned dependencies
├── run_experiment.py                 # ★ Single reproducible entry point
├── refactor_harness.py               # Legacy refactor helper (not needed for runs)
│
├── config/
│   └── experiment_config.yaml        # All hyperparameters with scientific rationale
│
├── src/
│   ├── __init__.py                   # Lazy imports (no crash on missing optional deps)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_map_registry.py       # ★ New — plugin feature map system
│   ├── quantum_feature_maps.py
│   ├── quantum_kernel_engine.py      # ★ Upgraded — geometric difference + git SHA
│   ├── classical_models.py
│   ├── quantum_training.py
│   ├── evaluation_metrics.py         # ★ Upgraded — Bootstrap CI, Cohen's d, Bonferroni
│   ├── noise_simulation.py
│   ├── expressibility.py             # ★ New — Sim et al. 2019 expressibility
│   ├── kernel_training.py            # ★ New — QKAT gradient kernel training
│   └── visualization.py              # ★ Upgraded — 8 new publication-quality plots
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_classical_baseline.ipynb
│   └── 03_quantum_kernel_svm.ipynb
│
├── tests/                            # Unit test suite
├── figures/                          # Static figures for documentation
├── notes/                            # Research notes and scratch analysis
└── results/                          # Generated at runtime — not committed
    └── multi_seed/
        ├── ablation_summary.json
        ├── ablation_plot.png
        ├── ablation_dashboard.png    # ★ New — 2×3 publication dashboard
        ├── depth_ablation.json
        ├── depth_ablation.png
        ├── geometric_difference.png  # ★ New — g(K_Q, K_C) vs PCA dim
        ├── expressibility_vs_kta.png # ★ New — ε vs KTA scatter
        ├── expressibility_configs.json
        └── dim_{d}_seed_{s}_reps{r}/
            ├── experiment_summary.json   # Includes git SHA
            ├── metrics_comparison.csv    # Includes KTA column
            ├── kernel_heatmap.png
            ├── confusion_matrix_*.png
            ├── noise_comparison.png
            └── pca_scatter.png
```

> **Note**: `results/` is generated at runtime and is not tracked by git. Each run is fully reproducible from the config and a fixed seed.

---

## 8. Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| Qiskit | ≥ 1.1 |
| Qiskit Aer | ≥ 0.17 |
| Qiskit Machine Learning | ≥ 0.9 |
| scikit-learn | ≥ 1.4 |
| numpy | ≥ 1.26 |

### Step-by-step setup

```bash
# 1. Clone the repository
git clone https://github.com/DEVADATH-HK/Quantum-AI-Research-Series.git
cd Quantum-AI-Research-Series/03-Quantum-Kernel-SVM-MNIST

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Tip — Offline / CI environments**: Use `--fallback` mode (described below) to skip the OpenML download. The sklearn digits fallback uses 8×8 toy images and **cannot** be compared with standard MNIST benchmarks.

---

## 9. Usage

### 9.1 Full experiment run (recommended)

```bash
python run_experiment.py
```

Downloads real MNIST from OpenML, runs the full multi-seed dimension ablation + depth ablation, and writes all artefacts to `results/multi_seed/`.

### 9.2 Offline / CI smoke test

```bash
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

Uses the sklearn digits fallback and skips noise simulation — completes in 2–5 minutes.

### 9.3 Custom run

```bash
python run_experiment.py \
  --results-dir results/experiment_v2 \
  --max-quantum-train 150 \
  --max-kernel-samples 120 \
  --log-level DEBUG
```

### 9.4 CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/experiment_config.yaml` | Path to YAML config file |
| `--results-dir` | `results/multi_seed` | Root output directory |
| `--max-quantum-train` | `100` | Training samples (same for both models — ensures fair comparison) |
| `--max-kernel-samples` | `100` | Kernel matrix size for KTA/property analysis |
| `--max-noise-samples` | `30` | Samples for noise simulation (keep small to stay tractable) |
| `--disable-noise` | off | Skip noise simulation |
| `--fallback` | off | Use sklearn digits instead of OpenML MNIST |
| `--log-level` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`) |

### 9.5 Configuration

All hyperparameters live in `config/experiment_config.yaml`. Every value has an inline scientific rationale. Key parameters:

```yaml
dataset:
  digits: [4, 9]                      # Hardest MNIST binary pair
  fallback_to_sklearn_digits: false   # true = offline mode only

preprocessing:
  pca:
    n_components: 8                   # Number of qubits for feature encoding
  feature_range: "pi/2"              # [0, π/2] prevents Bloch-sphere over-dispersion
                                     # (Thanasilp et al. 2022 — exponential concentration)

quantum_feature_map:
  type: "ZZFeatureMap"
  reps: 1                            # Minimal depth — NISQ-era justified
  entanglement: "full"               # All-to-all pairwise ZZ interactions

pegasos_svc:
  lambda_param: 0.001                # λ = 0.001 ↔ C = 1000; prevents collapse on n ≤ 200

ablation:
  seeds: [42, 43, 44]               # 3 seeds → paired t-test (extend to 10 for p<0.05 power)
  pca_dimensions: [4, 6, 8]
  reps_values: [1, 2, 3]            # Novel feature-map depth ablation
  max_quantum_train: 100

noise_simulation:
  enabled: true
  backend: "ibm_brisbane"
  noise_model:
    readout_error: 0.01             # Realistic IBM Brisbane specs
    gate_error:    0.001
```

---

## 10. Experimental Setup

### 10.1 Dataset

| Property | Value |
|---|---|
| Dataset | MNIST (OpenML ID 554) |
| Classes | Digit 4 vs. Digit 9 |
| Full dataset size | 70 000 samples (60 000 train / 10 000 test) |
| Training budget per model | 100 samples (stratified) |
| Test set | 25% stratified split |
| Source | OpenML via `sklearn.datasets.fetch_openml` |

### 10.2 Preprocessing Pipeline

```
Raw pixels (784-dim)
    │
    ▼  MinMaxScaler → [0, 1]
    │
    ▼  StandardScaler → μ=0, σ=1
    │
    ▼  PCA → d components (d ∈ {4, 6, 8})
    │
    ▼  MinMaxScaler → [0, π/2]  (quantum angle encoding range)
    │
    ▼  ZZFeatureMap(d qubits, reps, entanglement="full")
```

The PCA variance threshold is set to 0.95 as a diagnostic; the actual number of components is fixed per ablation cell.

### 10.3 Ablation Design

| Ablation | Variable | Values | Fixed |
|---|---|---|---|
| **Phase 1** — PCA dim | n_components | {4, 6, 8} | reps = 1, 3 seeds |
| **Phase 2** — Circuit depth | reps | {1, 2, 3} | best_dim from Phase 1, 3 seeds |

Each (dim, reps, seed) combination is an independent trial with its own result directory.

### 10.4 Statistical Testing

After collecting F1 scores across seeds:

- **Paired t-test**: Tests whether quantum F1 > classical F1 at the p < 0.05 level
- **Bonferroni correction**: α_adjusted = 0.05 / n_dims (corrects for multiple comparisons)
- **Bootstrap 95% CI**: 10 000 bootstrap resamples on F1 score distributions
- **Cohen's d**: Effect size between classical and quantum F1 across seeds

---

## 11. Output Artefacts

After running `python run_experiment.py`, the `results/multi_seed/` directory contains:

### Aggregated files

| File | Contents |
|---|---|
| `ablation_summary.json` | Mean ± std F1, KTA (Q+C), geometric difference g per PCA dim; git SHA |
| `ablation_plot.png` | Error-bar: Classical vs Pegasos F1 vs PCA dim |
| `ablation_dashboard.png` | **★ 2×3 figure**: F1/KTA/scalability vs dim + F1/KTA/noise vs reps |
| `depth_ablation.json` | F1 and KTA per reps value; git SHA |
| `depth_ablation.png` | Dual-axis: F1 and KTA vs circuit depth (reps) |
| `geometric_difference.png` | **★** g(K\_Q, K\_C) vs PCA dim with g=1 threshold line |
| `expressibility_vs_kta.png` | **★** ε vs KTA scatter with Pearson r annotation |
| `expressibility_configs.json` | Expressibility values per (dim, reps) config; git SHA |

### Per-trial files (one directory per dim/seed/reps cell)

| File | Contents |
|---|---|
| `experiment_summary.json` | Full trial summary including KTA, noise analysis, git SHA |
| `metrics_comparison.csv` | Model × metric table with KTA column |
| `kernel_heatmap.png` | Quantum kernel matrix visualisation |
| `confusion_matrix_classical.png` | Classical RBF SVM confusion matrix |
| `confusion_matrix_quantum.png` | Exact QSVC confusion matrix |
| `confusion_matrix_pegasos.png` | Pegasos QSVC confusion matrix |
| `pca_scatter.png` | 2D PCA scatter coloured by class |
| `pca_variance.png` | Explained variance ratio by component |
| `noise_comparison.png` | Noiseless vs noisy kernel heatmap (anchor cell only) |
| `class_distribution.png` | Training class balance |

---

## 12. Performance Benchmarks

Expected performance ranges on **real MNIST** (OpenML), n = 100 training samples, statevector simulation:

| Model | F1 Score | KTA | Notes |
|---|---|---|---|
| Classical RBF SVM | ≥ 0.97 | > 0.80 | GridSearchCV-tuned; strong baseline |
| Exact QSVC (reps=1) | 0.80 – 0.95 | 0.40 – 0.70 | Statevector; 8 qubits |
| Pegasos QSVC (reps=1) | 0.75 – 0.92 | 0.40 – 0.70 | Precomputed kernel; λ=0.001 |

> **Important**: These are indicative ranges. Run `python run_experiment.py` to generate the exact values for your hardware and Qiskit version. Every JSON result includes the git SHA for traceability.

### On statistical significance

With 3 seeds, the paired t-test has limited statistical power (~30–40% for small effect sizes). For a publishable claim of quantum advantage, extend to **10+ seeds** in `config/experiment_config.yaml`:

```yaml
ablation:
  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
```

---

## 13. Visualizations

The visualization module (`src/visualization.py`) produces **300 DPI, serif-font, publication-quality** figures. Key plots:

| Figure | Function | Description |
|---|---|---|
| 2×3 Ablation Dashboard | `plot_ablation_dashboard` | Summary of all ablation results in one figure |
| F1 vs PCA Dim | `plot_ablation_scaling` | Classical vs Quantum F1 error-bar plot |
| KTA + F1 vs Circuit Depth | `plot_depth_ablation` | Effect of reps on performance and kernel quality |
| Geometric Difference | `plot_geometric_difference` | g vs dim with advantage threshold at g=1 |
| Expressibility vs KTA | `plot_expressibility_vs_kta` | Barren-plateau evidence; Pearson correlation |
| Kernel Geometry (t-SNE) | `plot_kernel_geometry_tsne` | 2D embedding of quantum kernel space |
| ΔK Noise Heatmap | `plot_noise_difference_heatmap` | Element-wise difference (RdBu diverging colormap) |
| Noise Robustness Curve | `plot_noise_robustness_curve` | KTA vs readout error sweep |
| Scalability Curve | `plot_scalability_curve` | F1 and training time vs dataset size |

---

## 14. Computational Complexity

| Operation | Complexity | Notes |
|---|---|---|
| Kernel matrix computation | O(n² · circuit_depth) | Dominant cost; each entry = 1 quantum circuit |
| SVM training (exact QSVC) | O(n² → n³) | Standard QP solver |
| Pegasos training | O(n · max\_iter · batch\_size) | Sub-linear in n for fixed iterations |
| PCA preprocessing | O(n · d²) | One-time; negligible vs kernel eval |
| Expressibility (ε) | O(n\_samples · circuit\_depth) | Embarrassingly parallel |

Where:
- **n** = number of training samples
- **d** = PCA components (= number of qubits)
- **circuit\_depth** ∝ reps × entanglement\_gates

For n = 100, d = 8, the kernel matrix requires **10 000 quantum circuit evaluations**. On a statevector simulator, this takes approximately 30–120 seconds on a modern laptop. On real hardware, execution time depends on queue depth.

---

## 15. Limitations

1. **Scale**: n = 100 training samples is orders of magnitude below typical deep learning scale. Results may not generalise.
2. **Simulation fidelity**: Statevector simulation is exact but ignores real hardware noise. The noise model approximates but does not replicate real IBM Brisbane error characteristics.
3. **Statistical power**: 3 seeds provide insufficient power for robust significance claims. A minimum of 10 seeds is recommended for publication.
4. **Quantum advantage**: No quantum advantage over classical RBF SVM is demonstrated or claimed. The geometric difference g is a **necessary, not sufficient** condition.
5. **Feature map scope**: Only ZZFeatureMap is systematically ablated. PauliFeatureMap and hardware-efficient ansätze are not evaluated.
6. **OpenML dependency**: The full MNIST dataset requires an internet connection at first run.

---

## 16. Future Work

Ordered by priority for the next submission cycle:

1. **Hardware execution**: Point `config/experiment_config.yaml` to `QiskitRuntimeService` with a real IBM Quantum device (e.g. `ibm_brisbane`) to validate noise simulation accuracy.
2. **Larger training sets**: Scale n to 500–1000 using `--max-quantum-train` as hardware improves.
3. **Statistical power**: Extend to 10 seeds; publish paired t-test p-values with Bonferroni correction.
4. **PauliFeatureMap ablation**: Compare ZZFeatureMap against PauliFeatureMap(paulis=["Z","ZZ","ZZZ"]).
5. **Cross-dataset validation**: Apply the same pipeline to Fashion-MNIST or binary CIFAR-10.
6. **QKAT integration**: Connect the `kernel_training.py` QKAT optimiser into the main ablation loop to produce trained vs un-trained kernel comparisons.
7. **Scalability Phase**: Activate Phase 4.3 (F1 + wall-clock vs n) to empirically confirm the O(n²) scaling of kernel matrix computation.

---

## 17. References

1. **Havlíček et al.** (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212. https://doi.org/10.1038/s41586-019-0980-2

2. **Schuld & Killoran** (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters*, 122(4), 040504. https://doi.org/10.1103/PhysRevLett.122.040504

3. **Shalev-Shwartz et al.** (2011). Pegasos: Primal estimated sub-GrAdient SOlver for SVM. *Mathematical Programming*, 127(1), 3–30. https://doi.org/10.1007/s10107-010-0420-4

4. **Thanasilp et al.** (2022). Exponential concentration and untrainability in quantum kernel methods. *arXiv:2208.11060*. https://arxiv.org/abs/2208.11060

5. **Huang et al.** (2022). Quantum kernel methods work on classical data? *Nature Communications*, 13, 4468. https://doi.org/10.1038/s41467-021-22539-9

6. **Sim et al.** (2019). Expressibility and entangling capability of parameterized quantum circuits. *Advanced Quantum Technologies*, 2(12), 1900070. https://doi.org/10.1002/qute.201900070

7. **Glick et al.** (2024). Covariant quantum kernels for data with group structure. *PRX Quantum*, 5, 020337. https://doi.org/10.1103/PRXQuantum.5.020337

8. **Qiskit Machine Learning** (2024). https://qiskit-community.github.io/qiskit-machine-learning/

9. **LeCun et al.** (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

---

## 18. Author

**DEVADATH H K**  
Quantum AI Research Series — Project 03

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **IBM Quantum / Qiskit** — quantum simulation and hardware access
- **scikit-learn** — classical ML baseline and preprocessing utilities
- **OpenML** — standardised MNIST dataset access
- **Qiskit Machine Learning** — FidelityQuantumKernel and PegasosQSVC implementations
