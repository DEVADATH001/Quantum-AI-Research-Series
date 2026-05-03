# Project 01: Classical vs Quantum Visualization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-QSVC%20%7C%20Aer%20%7C%20Runtime-6929C4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classical%20Baselines-F7931E)
![Status](https://img.shields.io/badge/Status-Reproducible%20Research%20Demo-orange)

This module contains two deliberately separate experiments:

1. **Iris classification** using PCA, classical baselines, and a quantum support
   vector classifier.
2. **GHZ-127 noise benchmarking** comparing ideal simulation, noisy simulation
   from an IBM backend model, and optional real IBM hardware execution.

The two workflows answer different questions. Iris classification is a small
quantum machine-learning comparison. GHZ-127 is a large-circuit noise stress
test. A good or bad result in one workflow does not validate the other.

## README Audit

The previous README was useful as a short runbook, but it was not yet
research-grade documentation.

What was missing or weak:

- No research motivation explaining why Iris/QSVC and GHZ-127 appear in the same
  module.
- No mathematical definitions for the quantum kernel, GHZ state, or GHZ
  subspace probability.
- No computational-complexity discussion, even though QSVC decision-boundary
  plotting is dominated by kernel evaluations.
- Installation instructions did not distinguish the Iris path from the IBM
  Runtime-dependent GHZ path.
- The GHZ `--skip-real` behavior was underspecified: it skips real hardware, but
  the script still needs IBM Runtime credentials to choose and model a backend.
- Result interpretation was brief and could be mistaken for a quantum-advantage
  claim.
- No technologies, design decisions, limitations, future work, or citation-style
  references.

This rewrite turns the README into a technical module-level guide while keeping
claims constrained to the code and artifacts that exist in this repository.

## Project Overview

This project introduces the repository's core benchmarking attitude: compare
quantum methods against classical references, visualize the outcome, and state
the limitations plainly.

| Workflow | Entry Point | Output | Purpose |
|---|---|---|---|
| Iris classification | [`Quantum_ML_-_Iris_Classification.py`](Quantum_ML_-_Iris_Classification.py) | PCA decision-boundary plot and JSON metrics | Compare logistic regression, RBF-SVM, and QSVC on a small tabular dataset |
| GHZ-127 benchmark | [`compare_ghz_three_way.py`](compare_ghz_three_way.py) | JSON report and bar-chart summary | Measure how a 127-qubit GHZ circuit behaves under ideal, noisy, and optional real execution |
| Legacy GHZ entry point | [`Hardware_Noise_&_Decoherence_Benchmark.py`](Hardware_Noise_&_Decoherence_Benchmark.py) | Same as GHZ benchmark | Compatibility wrapper that calls the maintained GHZ script |

Use the Iris workflow first if you want a quick local QML run. Use the GHZ
workflow only after IBM Runtime credentials are configured.

## Motivation / Research Context

Small quantum ML demos can be misleading when they are not compared to simple
classical baselines. Large entangled circuits can also look impressive on paper
while collapsing under hardware noise. This module puts both ideas side by
side, but keeps them conceptually separate:

- **Iris QSVC** asks whether a quantum fidelity kernel helps on a small
  three-class classification problem after PCA compression to two features.
- **GHZ-127** asks whether a deep, linearly entangled GHZ circuit retains
  probability mass in the ideal GHZ subspace after transpilation and noise.

The current saved artifacts show the intended lesson: the classical Iris
baselines outperform the QSVC in this configuration, and the large GHZ circuit
is highly noise-sensitive under the selected backend model.

## Key Features

- Classical baselines: logistic regression and RBF-SVM.
- Quantum classifier: Qiskit Machine Learning `QSVC` with a
  `FidelityQuantumKernel` and 2D ZZ feature map.
- Runtime-safe decision-boundary plotting with a configurable kernel-evaluation
  budget.
- Structured metrics report for the Iris workflow.
- 127-qubit GHZ circuit construction and backend-aware transpilation.
- Three-way GHZ comparison: ideal Aer MPS simulation, noisy Aer simulation
  derived from an IBM backend, and optional live IBM hardware.
- Queue-aware real-hardware policy with `--max-pending-jobs`, timeout handling,
  and non-strict fallback behavior.
- Notebook versions split by workflow, with a legacy mixed notebook retained for
  reference.

## System Architecture

```text
01-Classical-vs-Quantum-Visualization
|
|-- Iris workflow
|   |-- load sklearn Iris dataset
|   |-- stratified train/test split
|   |-- PCA -> 2 components
|   |-- StandardScaler
|   |-- train Logistic Regression, RBF-SVM, QSVC
|   |-- render decision boundaries
|   `-- write qml_iris_report.json
|
`-- GHZ workflow
    |-- create 127-qubit GHZ circuit
    |-- initialize IBM Runtime service
    |-- select backend with >=127 qubits
    |-- transpile to ISA circuit
    |-- run ideal Aer MPS simulation
    |-- run backend-derived noisy Aer simulation
    |-- optionally run real IBM hardware
    `-- write JSON report and comparison chart
```

The workflows share the same `assets/` output directory but do not share
training data, metrics, or scientific claims.

## Mathematical Foundations

### Iris PCA and Classical Baselines

The Iris workflow compresses four original features into two principal
components:

```text
X in R^{n x 4} -> Z in R^{n x 2}
```

Logistic regression models class probabilities from linear scores. The RBF-SVM
uses the kernel

```text
K_RBF(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)
```

These baselines are important because Iris is small and low-dimensional after
PCA; a simple classical decision boundary is already strong.

### Quantum Kernel Classifier

The QSVC uses a quantum feature map to encode a classical vector into a quantum
state:

```text
|phi(x)> = U_phi(x)|0>
```

The fidelity quantum kernel is

```text
K_Q(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
```

In this module, `U_phi` is a 2-qubit ZZ feature map with `reps=2` and linear
entanglement. The trained QSVC receives the precomputed kernel geometry through
Qiskit Machine Learning.

### GHZ State and Subspace Metric

The ideal `n`-qubit GHZ state is

```text
|GHZ_n> = (|0...0> + |1...1>) / sqrt(2)
```

After measurement, an ideal GHZ circuit should concentrate probability on the
two bitstrings `00...0` and `11...1`. The benchmark reports

```text
p_ghz_subspace = P(00...0) + P(11...1)
```

This is not a full state-fidelity estimate. It is a coarse measurement-basis
diagnostic for whether the sampled distribution remains in the expected GHZ
subspace.

### Computational Complexity

Iris QSVC kernel construction is pairwise in the number of samples:

```text
training kernel cost ~= O(n_train^2)
boundary prediction cost ~= O(n_grid^2 * n_train)
```

The script estimates the combined kernel-evaluation load as

```text
n_train^2 + (grid_resolution^2 * n_train)
```

and chooses a bounded quantum grid when `--quantum-grid` is omitted.

The GHZ benchmark uses a matrix-product-state simulator. This is more practical
than dense statevector simulation for a 127-qubit chain-like circuit, but noisy
simulation can still be significantly slower than the ideal local run.

### Design Decisions

- PCA is fixed to two dimensions so decision boundaries can be visualized.
- The QSVC feature map is intentionally small: two qubits, ZZ interactions, and
  a bounded plotting grid.
- The GHZ circuit uses a linear chain of CNOTs to create a large entangled state
  with simple expected outputs.
- Real hardware is optional and queue-aware because live backend availability is
  not reproducible.
- The GHZ report separates ideal, simulated noisy, and real execution instead
  of merging them into one ambiguous metric.

## Technologies Used

- Python 3.10+
- NumPy
- matplotlib
- scikit-learn
- Qiskit
- Qiskit Aer
- Qiskit IBM Runtime
- Qiskit Machine Learning
- Jupyter / nbconvert for notebook execution
- pylatexenc for Qiskit circuit rendering dependencies

See [`requirements.txt`](requirements.txt) for the module dependency list.

## Repository Structure

```text
01-Classical-vs-Quantum-Visualization/
|-- README.md
|-- requirements.txt
|-- Quantum_ML_-_Iris_Classification.py
|-- compare_ghz_three_way.py
|-- Hardware_Noise_&_Decoherence_Benchmark.py
|-- iris_qml_classification.ipynb
|-- ghz_127_noise_benchmark.ipynb
|-- iris_quantum_bridge.ipynb
`-- assets/
    |-- classical_boundaries.png
    |-- classical_vs_quantum_boundaries.png
    |-- ghz_127_circuit.png
    |-- ghz_127_quasiprobability.png
    |-- local_ghz127_run.json
    |-- local_vs_real_ghz127_comparison.json
    |-- qml_iris_report.json
    |-- quantum_feature_map.png
    |-- three_way_ghz127_comparison.json
    `-- three_way_ghz127_comparison.png
```

## Installation Guide

From the repository root:

```powershell
cd 01-Classical-vs-Quantum-Visualization
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the Iris workflow, local dependencies are sufficient.

For the GHZ workflow, configure IBM Runtime credentials from the repository
root. If you are still inside this module directory after the install step, move
back to the repository root first:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 01-Classical-vs-Quantum-Visualization
```

Important: `compare_ghz_three_way.py --skip-real` skips live hardware execution,
but the script still initializes `QiskitRuntimeService` to select a backend and
build the backend-derived noisy simulator. Saved IBM credentials are therefore
still required for the current GHZ benchmark path.

## Usage Instructions

### Run Iris Classification

```powershell
python Quantum_ML_-_Iris_Classification.py --no-show
```

Useful options:

```powershell
python Quantum_ML_-_Iris_Classification.py `
  --random-state 7 `
  --test-size 0.25 `
  --classical-grid 140 `
  --quantum-grid 30 `
  --quantum-max-kernel-evals 120000 `
  --output-plot assets/classical_vs_quantum_boundaries.png `
  --output-report assets/qml_iris_report.json `
  --no-show
```

If `--quantum-grid` is omitted, the script chooses the largest grid resolution
within `--quantum-max-kernel-evals`, bounded by the script defaults.

### Run GHZ-127 Benchmark

Skip real hardware but still run ideal and noisy backend-derived simulation:

```powershell
python compare_ghz_three_way.py --skip-real
```

Run with explicit shot counts:

```powershell
python compare_ghz_three_way.py `
  --local-shots 1024 `
  --sim-shots 512 `
  --real-shots 256 `
  --skip-real
```

Allow a live real-backend attempt only when the selected backend queue is small:

```powershell
python compare_ghz_three_way.py --max-pending-jobs 2 --real-timeout-seconds 900
```

Fail instead of skipping on real-execution errors:

```powershell
python compare_ghz_three_way.py --strict-real --real-timeout-seconds 900
```

Use a specific backend:

```powershell
python compare_ghz_three_way.py --backend ibm_fez --skip-real
```

Compatibility entry point:

```powershell
python "Hardware_Noise_&_Decoherence_Benchmark.py" --skip-real
```

### Notebook Workflows

Execute notebooks from this module directory:

```powershell
python -m nbconvert --to notebook --execute --inplace iris_qml_classification.ipynb
python -m nbconvert --to notebook --execute --inplace ghz_127_noise_benchmark.ipynb
```

Notebook notes:

- `iris_qml_classification.ipynb` covers the Iris/QSVC workflow.
- `ghz_127_noise_benchmark.ipynb` covers the GHZ benchmark workflow.
- `ghz_127_noise_benchmark.ipynb` has `RUN_BENCHMARK = False` by default and
  reads saved artifacts unless changed.
- `iris_quantum_bridge.ipynb` is a legacy mixed notebook kept for reference.

## Example Results / Visualizations

Current saved Iris report: [`assets/qml_iris_report.json`](assets/qml_iris_report.json)

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | `0.973684` |
| Classical RBF-SVM | `0.973684` |
| Quantum SVC (ZZ-Map) | `0.631579` |

Additional saved Iris metadata:

- random state: `7`
- train/test size: `112 / 38`
- PCA explained variance ratio: `[0.922174, 0.053326]`
- classical/quantum grid: `140 / 30`
- estimated quantum kernel evaluations: `113344`

Current saved GHZ report:
[`assets/three_way_ghz127_comparison.json`](assets/three_way_ghz127_comparison.json)

| Mode | Status | Shots | Unique States | `p_ghz_subspace` |
|---|---|---:|---:|---:|
| Local ideal Aer MPS | completed | `1024` | `2` | `1.0` |
| Simulated noisy `ibm_fez` | completed | `512` | `512` | `0.0` |
| Real IBM `ibm_fez` | skipped | `256` | `0` | `0.0` |

Saved visual artifacts:

- [`assets/classical_vs_quantum_boundaries.png`](assets/classical_vs_quantum_boundaries.png)
- [`assets/three_way_ghz127_comparison.png`](assets/three_way_ghz127_comparison.png)
- [`assets/ghz_127_circuit.png`](assets/ghz_127_circuit.png)
- [`assets/ghz_127_quasiprobability.png`](assets/ghz_127_quasiprobability.png)
- [`assets/quantum_feature_map.png`](assets/quantum_feature_map.png)

Interpretation: in the saved Iris run, classical baselines are stronger than
the QSVC. In the saved GHZ run, the ideal circuit remains in the GHZ subspace,
while the backend-derived noisy simulation spreads probability across many
states. These are benchmark observations, not general claims about all quantum
kernels or all IBM devices.

## Experimental Setup

### Iris Classification

| Component | Setting |
|---|---|
| Dataset | `sklearn.datasets.load_iris()` |
| Classes | 3 Iris species |
| Split | Stratified train/test split |
| Default random seed | `7` |
| Test fraction | `0.25` |
| Dimensionality reduction | PCA to 2 components |
| Scaling | StandardScaler after PCA |
| Classical baselines | Logistic regression, RBF-SVM |
| Quantum model | QSVC with fidelity quantum kernel |
| Feature map | 2D ZZ feature map, `reps=2`, linear entanglement |
| Outputs | Decision-boundary PNG and JSON metrics |

### GHZ-127 Benchmark

| Component | Setting |
|---|---|
| Circuit | 127-qubit GHZ preparation plus measurement |
| Backend selection | Operational IBM backend with at least 127 qubits |
| Transpilation | Qiskit preset pass manager, optimization level 3 |
| Ideal simulation | `AerSimulator(method="matrix_product_state")` |
| Noisy simulation | `AerSimulator.from_backend(real_backend)` when available |
| Real execution | Optional IBM Runtime `SamplerV2` |
| Real skip policy | Explicit `--skip-real`, queue threshold, timeout, or non-strict exception fallback |
| Outputs | JSON report and comparison chart |

## Performance Metrics

Iris workflow:

- `test_accuracy`: fraction of correct predictions on the held-out test split.
- `pca_explained_variance_ratio`: variance retained by the two plotted PCA
  components.
- `quantum_kernel_eval_estimate`: estimated QSVC kernel calls from training and
  boundary plotting.

GHZ workflow:

- `p_all_zero`: measured probability of `00...0`.
- `p_all_one`: measured probability of `11...1`.
- `p_ghz_subspace`: `p_all_zero + p_all_one`.
- `unique_states`: number of distinct measured bitstrings.
- `elapsed_seconds`: wall-clock runtime for each execution mode.
- `top_states`: most frequent sampled states, with shortened and full
  bitstrings.

## Limitations

- Iris is a small, low-dimensional dataset. It is useful for visualization, not
  for quantum-advantage claims.
- The QSVC uses only two PCA features because the goal is decision-boundary
  visualization.
- The RBF-SVM is not extensively tuned beyond the script's fixed defaults.
- Test accuracy is reported for one default split unless the user reruns with
  different seeds.
- GHZ subspace probability is a measurement-basis diagnostic, not full quantum
  state tomography.
- The GHZ benchmark currently requires IBM Runtime credentials even when live
  hardware execution is skipped.
- Backend-derived noisy simulation depends on the selected backend model and
  the installed Qiskit/Aer versions.
- Real hardware results are queue-, calibration-, shot-, and date-dependent.
- The legacy notebook is retained for reference and should not be treated as the
  canonical execution path.

## Future Improvements

- Add a fully offline GHZ mode using a named fake backend or local synthetic
  noise model.
- Add multi-seed Iris evaluation with mean, standard deviation, and confidence
  intervals.
- Add tuned classical SVM baselines or cross-validation for the Iris workflow.
- Report F1 score and confusion matrices, not only test accuracy.
- Add kernel-target alignment for the Iris quantum kernel.
- Persist transpiled GHZ circuit statistics such as two-qubit gate count and
  coupling-map layout.
- Add automated smoke tests for both script entry points.
- Move legacy artifacts into a clearly named `legacy/` or `archive/` area if the
  module grows.

## References

- Fisher, R. A. The use of multiple measurements in taxonomic problems.
  *Annals of Eugenics*, 1936.
- Cortes, C. and Vapnik, V. Support-vector networks. *Machine Learning*, 1995.
- Havlicek et al. Supervised learning with quantum-enhanced feature spaces.
  *Nature*, 2019.
- Greenberger, Horne, and Zeilinger. Going beyond Bell's theorem. In *Bell's
  Theorem, Quantum Theory and Conceptions of the Universe*, 1989.
- Qiskit Machine Learning documentation:
  https://qiskit-community.github.io/qiskit-machine-learning/
- IBM Quantum / Qiskit documentation: https://docs.quantum.ibm.com/

## Author Information

**DEVADATH H K**

Project 01 of the
[`Quantum AI Research Series`](../README.md). See the repository-level
[`LICENSE`](../LICENSE), [`CITATION.cff`](../CITATION.cff), and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) files for licensing, citation, and
contribution guidance.
