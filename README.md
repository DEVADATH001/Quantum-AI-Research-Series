# Quantum AI Research Series

![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.1+-6929C4)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20portfolio-orange)
![Claim](https://img.shields.io/badge/quantum%20advantage-not%20claimed-red)
![CI](https://github.com/DEVADATH001/Quantum-AI-Research-Series/actions/workflows/ci.yml/badge.svg)

Five independent research experiments studying near-term quantum computing — with real baselines, real noise, and no inflated claims.

I built this because too many quantum ML papers test against weak baselines, skip noise modeling, or cherry-pick favorable results. Every module here includes strong classical comparisons, and the results are what they are. Project 04 is literally a negative-result benchmark — QAOA loses to classical solvers on every tested graph. That's the point.

---

## The five modules

| # | Project | What it studies | Honest result |
|---|---|---|---|
| 01 | [Classical vs Quantum Visualization](01-Classical-vs-Quantum-Visualization/README.md) | Iris QSVC (2-qubit) vs classical baselines + 127-qubit GHZ noise benchmark | Classical wins on Iris (97% vs 63%). GHZ collapses under noise. |
| 02 | [Quantum Chemistry VQE](02-Quantum-Chemistry-VQE/README.md) | Molecular energy surfaces for H₂, LiH, BeH₂ — VQE vs exact diagonalization | UCCSD reaches chemical accuracy on H₂. Larger molecules are harder. |
| 03 | [Quantum Kernel SVM MNIST](03-Quantum-Kernel-SVM-MNIST/README.md) | 4-qubit quantum kernel SVM vs tuned RBF-SVM on digit classification | Classical F1=1.0, Quantum F1=0.66 on fallback dataset. |
| 04 | [QAOA Max-Cut](04-Optimization-QAOA-MaxCut/README.md) | QAOA on weighted Max-Cut vs GW rounding, greedy, hill climb | **Negative result.** Classical baselines achieve ratio 1.0; QAOA ~0.78-0.86. |
| 05 | [RL Noise Mitigation](05-Reinforcement-Learning-Noise-Mitigation/README.md) | PQC reinforcement learning under noise, with ZNE/PEC mitigation | Smoke results only. Ideal quantum leads; tabular wins on efficiency. |

Each module is self-contained — its own dependencies, entry points, and conclusions. Don't combine results across projects.

---

## What I was trying to figure out

- Does a quantum model actually behave differently from a classical one on the same data?
- How much of that difference survives noise?
- Are improvements statistically meaningful, or just noise in the measurement?
- What does quantum actually cost in runtime, shots, or kernel evaluations?

These aren't rhetorical questions. Each module is designed to give concrete, traceable answers.

---

## Repository structure

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

## Quick start

Python 3.11 recommended. Project 05 requires it; the others work on 3.10+.

```powershell
git clone https://github.com/DEVADATH001/Quantum-AI-Research-Series.git
cd Quantum-AI-Research-Series
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root `requirements.txt` covers broad exploration. For reproducing specific experiments, install from inside the module directory instead.

Fastest sanity check — all five modules:

```powershell
python -m scripts.smoke_all
```

Or just one module:

```powershell
cd 01-Classical-vs-Quantum-Visualization
python Quantum_ML_-_Iris_Classification.py --no-show
```

---

## Running each module

### 01 — Iris classification + GHZ noise

```powershell
cd 01-Classical-vs-Quantum-Visualization
python Quantum_ML_-_Iris_Classification.py --no-show
python compare_ghz_three_way.py --skip-real
```

`--skip-real` skips live hardware but still needs IBM credentials for the noise model. See the [module README](01-Classical-vs-Quantum-Visualization/README.md).

### 02 — VQE chemistry

```powershell
cd 02-Quantum-Chemistry-VQE
python scripts/run_verification.py
```

Check `source_info` in the output — if it says `synthetic`, those are placeholder Hamiltonians, not real chemistry.

### 03 — Quantum kernel SVM

```powershell
cd 03-Quantum-Kernel-SVM-MNIST
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

The checked-in summary used sklearn's 8×8 digits, not full MNIST. That changes what the numbers mean.

### 04 — QAOA Max-Cut

```powershell
cd 04-Optimization-QAOA-MaxCut
python generate_artifacts.py
```

Read `results/results_verdict.md` first. The verdict is `weak` — every classical baseline beats QAOA here.

### 05 — Quantum RL

```powershell
cd 05-Reinforcement-Learning-Noise-Mitigation
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

Full benchmark (expensive, multi-hour):

```powershell
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

---

## IBM Runtime setup (optional)

For running on real IBM Quantum hardware:

```powershell
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
```

Don't commit your token. See [`.env.example`](.env.example) for guidance. Every module defaults to local simulators or fake backends — hardware is never required.

---

## Actual results from saved artifacts

These are checked-in numbers, not aspirational claims:

| Module | What the artifact shows |
|---|---|
| 01 | Logistic Regression 97.4%, RBF SVM 97.4%, **Quantum SVC 63.2%** — classical wins easily |
| 02 | H₂ warm-start (10 seeds): UCCSD **100% chemical-accuracy rate**, worst-case mean error 1.57 µHa. EfficientSU2 also 100%, worst error 1.40 µHa. |
| 03 | Fallback digits: Classical F1 1.000, Quantum Pegasos F1 **0.662** — classical wins again |
| 04 | Verdict: `weak`. QAOA approx ratio ~0.69 on benchmark graph. GW rounding hits 1.0 on all held-out families. |
| 05 | Smoke only: Ideal quantum actor-critic leads on raw success. Tabular wins on runtime efficiency. |

Don't treat these as a cross-project leaderboard. Each experiment has different assumptions, datasets, and qubit counts.

For detailed guidance on interpreting every artifact field, see [docs/results_guide.md](docs/results_guide.md).

---

## Math at a glance

Each module README has the full treatment. Here's the quick version:

**VQE** (Projects 02, 04) — parameterized circuit minimizes energy:

$$E(\theta) = \langle \psi(\theta)|H|\psi(\theta)\rangle$$

**Quantum kernels** (Projects 01, 03) — state fidelity as a kernel function:

$$K_Q(x_i, x_j) = |\langle \phi(x_i)|\phi(x_j)\rangle|^2$$

**QAOA** (Project 04) — alternating cost/mixer unitaries:

$$|\psi(\gamma,\beta)\rangle = \prod_{\ell=1}^{p} e^{-i\beta_\ell H_B} e^{-i\gamma_\ell H_C} |+\rangle^{\otimes n}$$

**Quantum RL policy** (Project 05) — action probabilities from measurement:

$$\pi_{\theta}(a|s) = \Pr[M_{\text{action}} = a]$$

---

## Tech stack

- **Core:** Python 3.11+, NumPy, SciPy, pandas, matplotlib, scikit-learn
- **Quantum:** Qiskit, Qiskit Aer, Qiskit Machine Learning, Qiskit Nature, Qiskit Optimization, IBM Runtime
- **Domain-specific:** PySCF (chemistry, Project 02), NetworkX (graphs, Project 04), PyTorch + Gymnasium (RL, Project 05)
- **Testing:** pytest, unittest

---

## Limitations — and I mean it

- **No quantum advantage is demonstrated in any module.** That's by design — the point is honest evaluation.
- Some workflows need PySCF, OpenML, or IBM Runtime. They're optional, but you need them for full reproduction.
- Hardware results depend on device calibration and queue state. They aren't reproducible benchmarks.
- Some checked-in artifacts are smoke tests or fallback runs. Read the individual READMEs to know which.
- Statistical power varies. Multi-seed studies exist for some modules but aren't complete everywhere.
- Notebooks may lag behind the script workflows. Scripts + saved JSON/CSV are the canonical path.

---

## What I'd work on next

- Per-module lockfiles (conda envs or `uv` pins)
- CI pipeline with README link checks, linting, and smoke tests — badge placeholders are ready at the top of this file
- Full Project 05 benchmark (not just smoke)
- Noise-scaling analysis for Project 05 (performance vs error-rate curves)

### Recently completed

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
