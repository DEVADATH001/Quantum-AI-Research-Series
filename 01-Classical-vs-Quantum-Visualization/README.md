# Project 01: Classical vs Quantum Visualization

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-QSVC%20|%20Aer%20|%20Runtime-6929C4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20baselines-F7931E)
![Status](https://img.shields.io/badge/status-reproducible%20demo-orange)

Two experiments, one question: *what actually happens when you put a quantum method next to something classical and compare fairly?*

**Experiment 1** — Iris QSVC: Logistic regression, RBF-SVM, and a 2-qubit quantum support vector classifier on the Iris dataset. Visualized with PCA decision boundaries so you can see where each model draws its lines.

**Experiment 2** — GHZ-127: A 127-qubit GHZ circuit run under ideal simulation, noise-model simulation (from a real IBM backend), and optionally on live hardware. Measures how much of the ideal entanglement structure survives noise.

These are independent experiments. A win on one says nothing about the other.

---

## Results

**Iris (saved run):**

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 97.4% |
| Classical RBF-SVM | 97.4% |
| Quantum SVC (2-qubit ZZ) | 63.2% |

Classical wins by a wide margin. This isn't a bug — Iris after PCA is a 2D problem that classical methods handle easily. The QSVC uses a `ZZFeatureMap` with `reps=2` and linear entanglement, and its kernel geometry just doesn't add value here.

**GHZ-127 (saved run):**

| Execution mode | Shots | Unique states | `p_ghz_subspace` |
|---|---:|---:|---:|
| Ideal Aer MPS | 1,024 | 2 | 1.0 |
| Noisy (ibm_fez model) | 512 | 512 | 0.0 |
| Real hardware | 256 | — | skipped |

Ideal simulation: perfect — all probability in `|00...0⟩` and `|11...1⟩`. Noisy simulation: probability spread uniformly across all 512 sampled states. That's 127-qubit noise in action.

Saved plots: [`classical_vs_quantum_boundaries.png`](assets/classical_vs_quantum_boundaries.png), [`three_way_ghz127_comparison.png`](assets/three_way_ghz127_comparison.png).

---

## Why these experiments

### Iris QSVC

I wanted to see how a small quantum kernel compares against proper classical baselines on a well-understood dataset. After PCA to 2 features, Iris is basically linearly separable — logistic regression already nails it. The QSVC encodes those 2 features into a 2-qubit ZZ feature map and computes a fidelity kernel:

$$K_Q(x_i, x_j) = |\langle \phi(x_i)|\phi(x_j)\rangle|^2$$

The point isn't to show quantum advantage (it doesn't exist here). It's to see exactly *where* the quantum decision boundary differs and *why* it scores lower.

### GHZ-127

A 127-qubit GHZ state should be maximally entangled — half the probability on `|00...0⟩`, half on `|11...1⟩`:

$$|GHZ_{127}\rangle = \frac{|0\rangle^{\otimes 127} + |1\rangle^{\otimes 127}}{\sqrt{2}}$$

In practice, on real NISQ hardware, noise destroys this completely. I measure `p_ghz_subspace = P(00...0) + P(11...1)` as a coarse diagnostic — it's not full state tomography, just a quick check of how much ideal structure remains.

---

## How to run

### Iris classification

```powershell
python Quantum_ML_-_Iris_Classification.py --no-show
```

With custom settings:

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

If you skip `--quantum-grid`, the script auto-picks the largest grid that fits within `--quantum-max-kernel-evals`. The kernel is pairwise, so it gets expensive fast — $O(n_{train}^2)$ for training, $O(grid^2 \times n_{train})$ for plotting.

### GHZ-127

Ideal + noisy only (no live hardware):

```powershell
python compare_ghz_three_way.py --skip-real
```

**Heads up:** `--skip-real` skips hardware execution, but the script still initializes IBM Runtime to pick a backend and build the noise model from it. You need saved IBM credentials even for this path.

Custom shots:

```powershell
python compare_ghz_three_way.py `
  --local-shots 1024 `
  --sim-shots 512 `
  --real-shots 256 `
  --skip-real
```

With real hardware (short queue only):

```powershell
python compare_ghz_three_way.py --max-pending-jobs 2 --real-timeout-seconds 900
```

Specific backend:

```powershell
python compare_ghz_three_way.py --backend ibm_fez --skip-real
```

### Notebooks

```powershell
python -m nbconvert --to notebook --execute --inplace iris_qml_classification.ipynb
python -m nbconvert --to notebook --execute --inplace ghz_127_noise_benchmark.ipynb
```

The GHZ notebook reads saved artifacts by default (`RUN_BENCHMARK = False`). `iris_quantum_bridge.ipynb` is a legacy notebook — kept for reference.

---

## Installation

```powershell
cd 01-Classical-vs-Quantum-Visualization
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For IBM Runtime (needed by the GHZ benchmark even with `--skip-real`):

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 01-Classical-vs-Quantum-Visualization
```

---

## Experimental setup

**Iris:**

| Setting | Value |
|---|---|
| Dataset | `sklearn.datasets.load_iris()` |
| Seed | 7 |
| Test split | 25% |
| PCA | 2 components |
| Feature map | 2-qubit ZZ, `reps=2`, linear entanglement |
| Classifier | QSVC with fidelity quantum kernel |
| Classical baselines | Logistic Regression, RBF-SVM |

**GHZ-127:**

| Setting | Value |
|---|---|
| Circuit | 127-qubit GHZ (linear CNOT chain) |
| Ideal backend | `AerSimulator(method="matrix_product_state")` |
| Noisy backend | `AerSimulator.from_backend(ibm_fez)` |
| Real execution | Optional via `SamplerV2` |
| Transpilation | Preset pass manager, optimization level 3 |

---

## Limitations

- **Iris is a toy dataset.** Two PCA features, 150 samples, nearly linearly separable. Don't draw quantum-advantage conclusions from it.
- The QSVC uses only 2 features by design — the goal is visualization, not maximum performance.
- Test accuracy is from a single train/test split unless you rerun with different seeds.
- `p_ghz_subspace` is coarse — it's a measurement-basis diagnostic, not fidelity estimation.
- The GHZ workflow needs IBM credentials even when skipping hardware. That's annoying, and it's first on the fix list.
- Hardware results depend on calibration, queue, and shot count. They're not reproducible in the usual sense.

---

## What I'd fix next

- An offline GHZ mode with a local synthetic noise model (no IBM credentials needed)
- Multi-seed Iris runs with confidence intervals
- F1 scores and confusion matrices alongside accuracy
- Kernel-target alignment for the QSVC
- Automated smoke tests for both entry points

---

## References

- Fisher, R. A. The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 1936.
- Cortes, C. and Vapnik, V. Support-vector networks. *Machine Learning*, 1995.
- Havlicek et al. Supervised learning with quantum-enhanced feature spaces. *Nature*, 2019.
- Greenberger, Horne, and Zeilinger. Going beyond Bell's theorem. 1989.
- Qiskit Machine Learning: https://qiskit-community.github.io/qiskit-machine-learning/
- IBM Quantum: https://docs.quantum.ibm.com/

---

## Author

**DEVADATH H K** — Part of the [Quantum AI Research Series](../README.md).

See [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), and [CONTRIBUTING.md](../CONTRIBUTING.md).
