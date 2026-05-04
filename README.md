# Quantum AI Research Series

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.1+-6929C4)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20portfolio-orange)
![Claim](https://img.shields.io/badge/quantum%20advantage-not%20claimed-red)
![CI](https://github.com/DEVADATH001/Quantum-AI-Research-Series/actions/workflows/ci.yml/badge.svg)

Five independent research experiments studying near-term quantum computing — with rigorous classical baselines, realistic noise models, and transparent reporting of results.

This series was motivated by a desire to contribute to the reproducibility of quantum ML research. Each module is designed to give an honest comparison between quantum and classical approaches, including cases where the classical method clearly wins. Project 04, for example, is an intentional negative-result benchmark: QAOA is evaluated against strong classical solvers on every graph family, and the classical methods consistently outperform it. Reporting that outcome faithfully is part of the goal.

---

## The Five Modules

| # | Project | What it studies | Honest result |
|---|---|---|---|
| 01 | [Classical vs Quantum Visualization](01-Classical-vs-Quantum-Visualization/README.md) | Iris QSVC (2-qubit) vs classical baselines + 127-qubit GHZ noise benchmark | Classical wins on Iris (97% vs 63%). GHZ collapses under noise. |
| 02 | [Quantum Chemistry VQE](02-Quantum-Chemistry-VQE/README.md) | Molecular energy surfaces for H₂, LiH, BeH₂ — VQE vs exact diagonalization | UCCSD reaches chemical accuracy on H₂. Larger molecules are harder. |
| 03 | [Quantum Kernel SVM MNIST](03-Quantum-Kernel-SVM-MNIST/README.md) | 4-qubit quantum kernel SVM vs tuned RBF-SVM on digit classification | Classical F1=1.0, Quantum F1=0.66 on fallback dataset. |
| 04 | [QAOA Max-Cut](04-Optimization-QAOA-MaxCut/README.md) | QAOA on weighted Max-Cut vs GW rounding, greedy, hill climb | **Negative result.** Classical baselines achieve ratio 1.0; QAOA ~0.78–0.86. |
| 05 | [RL Noise Mitigation](05-Reinforcement-Learning-Noise-Mitigation/README.md) | PQC reinforcement learning under noise, with ZNE/PEC mitigation | Smoke results only. Ideal quantum leads; tabular wins on efficiency. |

Each module is self-contained — its own dependencies, entry points, and conclusions. Please do not combine results across projects, as the experimental assumptions and datasets differ.

---

## Research Questions

- Does a quantum model behave measurably differently from a classical one on the same data?
- How much of that difference survives realistic noise?
- Are observed improvements statistically meaningful, or within measurement variance?
- What does using a quantum method actually cost in runtime, shots, or kernel evaluations?

Each module is designed to give concrete, traceable answers to these questions.

---

## Repository Structure

```
Quantum-AI-Research-Series/
├── 01-Classical-vs-Quantum-Visualization/   # Iris QSVC + GHZ-127 noise study
├── 02-Quantum-Chemistry-VQE/                # VQE potential energy surfaces
├── 03-Quantum-Kernel-SVM-MNIST/             # Quantum kernel SVM benchmark
├── 04-Optimization-QAOA-MaxCut/             # QAOA negative-result study
├── 05-Reinforcement-Learning-Noise-Mitigation/  # Quantum RL + error mitigation
├── docs/
│   └── results_guide.md                     # How to interpret and cite each artifact
├── scripts/
│   ├── smoke_all.py                         # Run all five modules' smoke tests
│   ├── check_authorship.py                  # Authorship compliance checks
│   └── setup_ibm_runtime.py                 # IBM Quantum credential setup
├── results/                                 # Root-level result artifacts
├── requirements.txt                         # Broad dependencies for exploration
├── CITATION.cff                             # How to cite this work
└── LICENSE                                  # MIT
```

Each module also has its own `requirements.txt`, `src/`, `config/`, `results/`, and `tests/`.

---

## Quick Start

Python 3.11 is recommended. Project 05 requires it; the others work on 3.10+.

```powershell
git clone https://github.com/DEVADATH001/Quantum-AI-Research-Series.git
cd Quantum-AI-Research-Series
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root `requirements.txt` covers broad exploration. For reproducing specific experiments, please install dependencies from within the relevant module directory.

Fastest sanity check — all five modules at once:

```powershell
python -m scripts.smoke_all
```

Or run a single module:

```powershell
cd 01-Classical-vs-Quantum-Visualization
python Quantum_ML_-_Iris_Classification.py --no-show
```

---

## Running Each Module

### 01 — Iris Classification + GHZ Noise

```powershell
cd 01-Classical-vs-Quantum-Visualization
python Quantum_ML_-_Iris_Classification.py --no-show
python compare_ghz_three_way.py --skip-real
```

`--skip-real` skips live hardware but still uses the IBM noise model for simulation. See the [module README](01-Classical-vs-Quantum-Visualization/README.md) for credential setup.

### 02 — VQE Chemistry

```powershell
cd 02-Quantum-Chemistry-VQE
python scripts/run_verification.py
```

Please check `source_info` in the output JSON — if it reads `synthetic`, those results use placeholder Hamiltonians rather than real chemistry from PySCF.

### 03 — Quantum Kernel SVM

```powershell
cd 03-Quantum-Kernel-SVM-MNIST
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

Note: the checked-in summary uses scikit-learn's 8×8 digits dataset, not full MNIST (28×28). This distinction affects the interpretation of the numbers.

### 04 — QAOA Max-Cut

```powershell
cd 04-Optimization-QAOA-MaxCut
python generate_artifacts.py
```

The verdict file at `results/results_verdict.md` summarises the comparison. Every classical baseline outperforms QAOA on the tested families.

### 05 — Quantum RL

```powershell
cd 05-Reinforcement-Learning-Noise-Mitigation
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

Full benchmark (multi-hour run):

```powershell
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

---

## IBM Runtime Setup (Optional)

For running experiments on real IBM Quantum hardware:

```powershell
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
```

Please do not commit your token. See [`.env.example`](.env.example) for guidance. Every module defaults to local simulators or fake backends — hardware access is never required to reproduce the core results.

---

## Results from Saved Artifacts

These are the numbers from checked-in artifacts, not projected or aspirational values:

| Module | What the artifact shows |
|---|---|
| 01 | Logistic Regression 97.4%, RBF SVM 97.4%, **Quantum SVC 63.2%** — classical leads by a wide margin |
| 02 | H₂ warm-start (10 seeds): UCCSD **100% chemical-accuracy rate**, worst-case mean error 1.57 µHa. EfficientSU2 also 100%, worst error 1.40 µHa. |
| 03 | Fallback digits: Classical F1 1.000, Quantum Pegasos F1 **0.662** — classical wins again |
| 04 | Verdict: `weak`. QAOA approx ratio ~0.69 on benchmark graph. GW rounding hits 1.0 on all held-out families. |
| 05 | Smoke only: Ideal quantum actor-critic leads on raw success rate. Tabular wins on runtime efficiency. |

Please treat these as module-specific results rather than a cross-project leaderboard — experimental assumptions, datasets, and qubit counts differ significantly between modules.

For detailed guidance on interpreting every artifact field, see [docs/results_guide.md](docs/results_guide.md).

---

## Mathematics at a Glance

Each module README contains the full derivations. Here is a brief overview:

**VQE** (Projects 02, 04) — parameterized circuit minimizes energy expectation:

$$E(\theta) = \langle \psi(\theta)|H|\psi(\theta)\rangle$$

**Quantum kernels** (Projects 01, 03) — state fidelity used as a kernel function:

$$K_Q(x_i, x_j) = |\langle \phi(x_i)|\phi(x_j)\rangle|^2$$

**QAOA** (Project 04) — alternating cost and mixer unitaries:

$$|\psi(\gamma,\beta)\rangle = \prod_{\ell=1}^{p} e^{-i\beta_\ell H_B} e^{-i\gamma_\ell H_C} |+\rangle^{\otimes n}$$

**Quantum RL policy** (Project 05) — action probabilities derived from measurement outcomes:

$$\pi_{\theta}(a|s) = \Pr[M_{\text{action}} = a]$$

---

## Tech Stack

- **Core:** Python 3.11+, NumPy, SciPy, pandas, matplotlib, scikit-learn
- **Quantum:** Qiskit, Qiskit Aer, Qiskit Machine Learning, Qiskit Nature, Qiskit Optimization, IBM Runtime
- **Domain-specific:** PySCF (chemistry, Project 02), NetworkX (graphs, Project 04), PyTorch + Gymnasium (RL, Project 05)
- **Testing:** pytest, unittest

---

## Known Limitations

- **No quantum advantage is demonstrated in any module.** This is intentional — the purpose is honest, reproducible evaluation rather than performance optimisation.
- Some workflows depend on PySCF, OpenML, or IBM Runtime. These are optional, but are required for full reproduction of the relevant experiments.
- Hardware results are sensitive to device calibration and queue state and are therefore not fully reproducible benchmarks.
- Some checked-in artifacts reflect smoke tests or fallback runs. Please refer to the individual module READMEs for clarification.
- Statistical power varies across modules. Multi-seed studies exist for some experiments but are not yet complete everywhere.
- Notebooks may lag behind the script-based workflows. The canonical path for reproducing results is through scripts and saved JSON/CSV artifacts.

---

## Planned Improvements

- Per-module dependency lockfiles (conda environments or `uv` pins)
- Active CI pipeline with README link checks, linting, and smoke tests — badge placeholders are already in place
- Full Project 05 benchmark (currently smoke only)
- Noise-scaling analysis for Project 05 (performance vs error-rate curves)

### Recently Completed

- ✅ `python -m scripts.smoke_all` — runs all five modules with bounded runtime
- ✅ `docs/results_guide.md` — artifact interpretation and citation guide
- ✅ Project 02 chemical-accuracy summary from `multiseed_stats_H2_warm.json`

---

## References

- Havlicek et al. "Supervised learning with quantum-enhanced feature spaces." *Nature*, 2019.
- Schuld and Killoran. "Quantum machine learning in feature Hilbert spaces." *PRL*, 2019.
- Peruzzo et al. "A variational eigenvalue solver on a photonic quantum processor." *Nature Communications*, 2014.
- Farhi, Goldstone, and Gutmann. "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028, 2014.
- Goemans and Williamson. "Improved approximation algorithms for maximum cut." *Journal of the ACM*, 1995.
- Shalev-Shwartz et al. "Pegasos: Primal estimated sub-gradient solver for SVM." *Mathematical Programming*, 2011.
- Williams. "Simple statistical gradient-following algorithms for connectionist RL." *Machine Learning*, 1992.
- Sutton and Barto. *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Qiskit: https://docs.quantum.ibm.com/
- Qiskit ML: https://qiskit-community.github.io/qiskit-machine-learning/

---

## Author

**DEVADATH H K** — MIT License

See [LICENSE](LICENSE), [CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [AUTHORSHIP_POLICY.md](AUTHORSHIP_POLICY.md).

```
DEVADATH H K. Quantum AI Research Series. 2026.
```
