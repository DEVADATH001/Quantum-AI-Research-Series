# Project 03: Quantum Kernel SVM for MNIST

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qiskit ML](https://img.shields.io/badge/Qiskit%20ML-QSVC%20|%20Pegasos-6929C4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RBF%20SVM-F7931E)
![Status](https://img.shields.io/badge/status-research%20benchmark-orange)

This benchmarks quantum kernel SVMs against a tuned classical RBF-SVM on binary digit classification. Default task: MNIST digits 4 vs 9 — visually similar enough after PCA to make it non-trivial.

The question I'm asking: *under a matched training budget, does the quantum kernel find structure that a classical RBF kernel misses?*

From the saved results: mostly no. But measuring it precisely is the point.

---

## Results

### Checked-in artifact (fallback sklearn-digits, not MNIST)

| Model | F1 Score | Accuracy | Training time |
|---|---:|---:|---:|
| Classical RBF-SVM | 1.000 | 1.000 | 0.19s |
| Quantum Pegasos SVM | 0.662 | 0.495 | 291s |

That's a 4-qubit kernel (`ZZFeatureMap`, `reps=2`) trained on 80 samples from sklearn's 8×8 digit images, not full 28×28 MNIST. The classical SVM used `GridSearchCV` and found `C=10, gamma=scale`.

### PCA dimension ablation (3 seeds per dim, fallback dataset)

| PCA Dim (= qubits) | Classical Mean F1 | Quantum Mean F1 | Mean KTA | Significant? |
|---:|---:|---:|---:|---|
| 4 | 0.569 | 0.616 | 0.128 | no |
| 6 | 0.642 | 0.698 | 0.188 | no |
| 8 | 0.684 | 0.812 | 0.227 | no |

### Depth ablation (PCA dim = 8, 3 seeds)

| Feature map reps | Mean F1 | Mean KTA |
|---:|---:|---:|
| 1 | 0.863 | 0.225 |
| 2 | 0.800 | 0.229 |
| 3 | 0.688 | 0.226 |

More feature-map depth hurts — deeper circuits don't translate to better classification here. KTA stays flat while F1 drops.

**Important caveats:**
- These are from the **fallback** sklearn digits dataset (8×8 images), not full MNIST (28×28). Don't mix them.
- The `significant_advantage` flag in the code isn't direction-aware — a low p-value could mean classical *or* quantum wins. Always compare means.
- 3 seeds is low statistical power. Take these as preliminary trends, not definitive.

---

## How the pipeline works

| Stage | What happens |
|---|---|
| Data | Fetches MNIST from OpenML (default) or falls back to sklearn `load_digits` |
| Task | Binary classification: digits `[4, 9]` |
| Preprocessing | Normalize → standardize → PCA → angle scaling to $[0, \pi]$ |
| Classical baseline | RBF-SVM with `GridSearchCV` over C and gamma |
| Quantum QSVC | `FidelityQuantumKernel` with `ZZFeatureMap`, exact `QSVC` |
| Quantum Pegasos | Same kernel, `PegasosQSVC` with `lambda=0.001` |
| Diagnostics | KTA, centered KTA, eigenvalue spectrum, geometric difference, expressibility |
| Output | JSON summary, CSV metrics, confusion matrices, kernel heatmaps, model files |

PCA dimension = qubit count. A 4-dim PCA means a 4-qubit circuit — this is what keeps experiments tractable.

---

## The math

**Quantum feature map** encodes PCA features into qubit amplitudes:

$$|\phi(x)\rangle = U_\phi(x)|0\rangle^{\otimes d}$$

Default: `ZZFeatureMap` with single-qubit rotations and pairwise ZZ entangling interactions.

**Fidelity kernel** — the quantum version of a kernel function:

$$K_Q(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$

**Classical RBF kernel** — the baseline to beat:

$$K_{\text{RBF}}(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$

RBF is a strong baseline. Don't treat it as weak just because it's classical.

**Kernel-Target Alignment (KTA)** — measures how well the kernel geometry fits the labels:

$$\text{KTA}(K, y) = \frac{\langle K, yy^T \rangle_F}{||K||_F \cdot ||yy^T||_F}$$

**Geometric difference** — diagnostic for potential quantum advantage (necessary condition, not sufficient):

$$g(K_Q, K_C) = \sqrt{||K_C^{1/2} K_Q^{-1} K_C^{1/2}||_2}$$

A favorable geometric difference doesn't prove advantage. It's at best a prerequisite under strong assumptions (Huang et al., 2021).

**Pegasos SVM** — stochastic sub-gradient descent on the hinge loss:

$$\min_w \frac{\lambda}{2} ||w||^2 + \frac{1}{n}\sum_i \max(0, 1 - y_i \langle w, \phi(x_i) \rangle)$$

**Kernel cost:** $O(n^2 \cdot \text{circuit\_evaluations})$. For $n=100$, that's 10,000 kernel evaluations for the training Gram matrix alone — before test predictions or diagnostics.

---

## How to run

### Full experiment (needs internet for OpenML MNIST)

```powershell
python run_experiment.py
```

Runs PCA-dimension and depth ablations, writes to `results/multi_seed/`.

### Offline smoke run

```powershell
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

Uses sklearn digits instead of downloading MNIST. Good for pipeline validation.

### Custom run

```powershell
python run_experiment.py `
  --results-dir results/smoke_local `
  --fallback `
  --max-quantum-train 40 `
  --max-kernel-samples 40 `
  --disable-noise `
  --log-level INFO
```

### CLI options

| Flag | Default | What it does |
|---|---|---|
| `--config` | `config/experiment_config.yaml` | Config file |
| `--results-dir` | `results/multi_seed` | Where to write outputs |
| `--max-quantum-train` | 100 | Training sample cap (matched for classical and quantum) |
| `--max-kernel-samples` | 100 | Kernel diagnostic matrix sample cap |
| `--disable-noise` | off | Skip noise analysis |
| `--fallback` | off | Use sklearn digits instead of MNIST |
| `--log-level` | INFO | Logging verbosity |

### Tests

```powershell
python -m pytest tests -q
```

---

## Installation

```powershell
cd 03-Quantum-Kernel-SVM-MNIST
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First OpenML run downloads and caches MNIST in `results/.cache/sklearn`. Use `--fallback` if you're offline.

For IBM Runtime (optional):

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 03-Quantum-Kernel-SVM-MNIST
```

Then set `hardware.use_real_hardware: true` in `config/experiment_config.yaml`.

---

## Default configuration

From `config/experiment_config.yaml`:

| Setting | Value |
|---|---|
| Dataset | OpenML `mnist_784` |
| Digits | `[4, 9]` |
| PCA dimensions | `[4, 6, 8]` (ablation) |
| Feature map | `ZZFeatureMap` |
| Feature-map reps | `[1, 2, 3]` (ablation) |
| Entanglement | Linear |
| Training cap | 100 samples |
| Classical | RBF-SVM with grid search |
| Pegasos | `lambda=0.001`, `max_iter=1500`, `batch_size=32` |
| Noise | Optional FakeBrisbane/Aer |
| Hardware | Disabled |

---

## Limitations

- **No quantum advantage demonstrated.** Classical RBF-SVM wins or ties on every configuration tested.
- OpenML MNIST and sklearn fallback are **not comparable datasets.** Don't mix their results.
- Default training cap is 100 samples — mainly to keep kernel computation feasible.
- Statevector simulation doesn't capture hardware noise effects.
- Fake-backend noise models approximate device behavior, not live calibration.
- The `significant_advantage` flag isn't direction-aware. Always compare means manually.
- Some checked-in artifacts are from fallback or validation runs.

---

## What I'd fix next

- Fix the significance helper to report direction and effect size
- A canonical result manifest (dataset source, fallback status, seed list, git SHA, config hash)
- Clean separation between smoke artifacts and research artifacts
- Larger training sets via Nyström or batching approximations
- More feature maps beyond ZZFeatureMap
- Cross-dataset evaluation (Fashion-MNIST, binary CIFAR features)
- Real hardware evaluation with proper metadata

---

## References

- Havlicek et al. Supervised learning with quantum-enhanced feature spaces. *Nature*, 2019.
- Schuld and Killoran. Quantum machine learning in feature Hilbert spaces. *PRL*, 2019.
- Shalev-Shwartz et al. Pegasos: Primal estimated sub-gradient solver for SVM. *Mathematical Programming*, 2011.
- Huang et al. Quantum kernel methods work on classical data? *Nature Communications*, 2021.
- Thanasilp et al. Exponential concentration and untrainability in quantum kernel methods. arXiv:2208.11060, 2022.
- Sim et al. Expressibility and entangling capability of parameterized quantum circuits. *Advanced Quantum Technologies*, 2019.
- LeCun et al. Gradient-based learning applied to document recognition. *Proc. IEEE*, 1998.
- Qiskit Machine Learning: https://qiskit-community.github.io/qiskit-machine-learning/

---

## Author

**DEVADATH H K** — Part of the [Quantum AI Research Series](../README.md).

See [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), and [CONTRIBUTING.md](../CONTRIBUTING.md).
