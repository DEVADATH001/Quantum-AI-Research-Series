# Quantum AI Research Series — Full Project Showcase

**Author:** DEVADATH H K  
**Repo:** https://github.com/DEVADATH001/Quantum-AI-Research-Series  
**License:** MIT  

Five independent quantum computing research experiments — with real baselines, real noise models, and no inflated claims. Every module includes strong classical comparisons. The results are what they are.

---

## Quick Summary

| # | Project | Core Question | Honest Result |
|---|---|---|---|
| 01 | Classical vs Quantum Visualization | Does a 2-qubit quantum kernel beat classical on Iris? | Classical wins (97.4% vs 63.2%). GHZ-127 collapses under noise. |
| 02 | Quantum Chemistry VQE | Can VQE reproduce molecular energy surfaces accurately? | UCCSD hits 100% chemical-accuracy on H₂ across all 10 seeds. |
| 03 | Quantum Kernel SVM MNIST | Does a quantum kernel find structure classical RBF misses on digits? | No. Classical F1=1.000, Quantum F1=0.662. |
| 04 | QAOA Max-Cut | Does QAOA outperform classical solvers on Max-Cut? | **Negative result.** GW rounding hits 1.0 on every graph family; QAOA ~0.78–0.86. |
| 05 | RL Noise Mitigation | How do PQC RL policies degrade under noise and does mitigation help? | Smoke results only. Ideal quantum leads on success; tabular wins on efficiency. |

---

## Project 01 — Classical vs Quantum Visualization

**Directory:** `01-Classical-vs-Quantum-Visualization/`  
**Stack:** Qiskit QSVC, Qiskit Aer, scikit-learn, PySCF (optional)

### What it studies
Two experiments in one module:

- **Iris QSVC** — Logistic Regression, RBF-SVM, and a 2-qubit `ZZFeatureMap` quantum SVC on the Iris dataset. Decision boundaries visualized with PCA so you can see exactly where each model draws its lines.
- **GHZ-127** — 127-qubit GHZ circuit run under ideal simulation, noise-model simulation (from real IBM backend calibration), and optionally on live hardware.

### Results

**Iris:**
| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 97.4% |
| Classical RBF-SVM | 97.4% |
| Quantum SVC (2-qubit ZZ) | 63.2% |

**GHZ-127:**
| Execution mode | Shots | Unique states | `p_ghz_subspace` |
|---|---:|---:|---:|
| Ideal Aer MPS | 1,024 | 2 | 1.0 |
| Noisy (ibm_fez model) | 512 | 512 | 0.0 |

Ideal: perfect — all probability on `|00…0⟩` and `|11…1⟩`. Noisy: probability spread uniformly across all 512 sampled states. That's 127-qubit noise in action.

### Key math

Fidelity kernel:
$$K_Q(x_i, x_j) = |\langle \phi(x_i)|\phi(x_j)\rangle|^2$$

GHZ ideal state:
$$|GHZ_{127}\rangle = \frac{|0\rangle^{\otimes 127} + |1\rangle^{\otimes 127}}{\sqrt{2}}$$

### Entry points
```powershell
python Quantum_ML_-_Iris_Classification.py --no-show
python compare_ghz_three_way.py --skip-real
```

### Files
- `Quantum_ML_-_Iris_Classification.py` — main Iris QSVC script
- `compare_ghz_three_way.py` — 3-way GHZ benchmark (ideal / noisy / hardware)
- `iris_qml_classification.ipynb` — notebook with full run
- `ghz_127_noise_benchmark.ipynb` — GHZ notebook (reads saved artifacts by default)
- `Hardware_Noise_&_Decoherence_Benchmark.py` — standalone noise benchmark
- `assets/` — saved plots

---

## Project 02 — Quantum Chemistry VQE

**Directory:** `02-Quantum-Chemistry-VQE/`  
**Stack:** Qiskit Nature, PySCF, SLSQP optimizer

### What it studies
Sweeps bond lengths for H₂, LiH, and BeH₂. Computes exact ground-state energies by diagonalizing the qubit Hamiltonian, then runs VQE with UCCSD and EfficientSU2 ansatze. Includes warm-start studies (parameter transfer between adjacent bond lengths) and architecture ablations.

### Results (from `multiseed_stats_H2_warm.json`)

10 seeds × 21 bond lengths × 2 ansatze, PySCF real chemistry (not synthetic):

| Ansatz | Chemical-accuracy rate | Worst-case mean error |
|---|---:|---:|
| **UCCSD** | **100%** (all 21 distances, all 10 seeds) | 0.887 µHa |
| **EfficientSU2** | **100%** (all 21 distances, all 10 seeds) | 1.395 µHa |

Chemical accuracy threshold: **1.6 mHartree**. Both ansatze stay well under it across the full H₂ bond-length curve. Errors increase at stretched geometries (R > 2.0 Å) as expected — warm-starting helps by transferring parameters from adjacent geometries.

### Key math

Electronic Hamiltonian:
$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s + E_{\text{nuc}}$$

VQE objective (variational upper bound):
$$E(\theta) = \langle \psi(\theta) | H_q | \psi(\theta) \rangle, \quad \theta^* = \arg\min_\theta E(\theta)$$

Chemical accuracy:
$$|E_{\text{VQE}} - E_{\text{exact}}| \leq 0.0016 \text{ Hartree}$$

### Entry points
```powershell
python scripts/run_verification.py          # smoke check
python -m src.pes_generator --molecule H2   # PES scan
python scripts/run_experiment.py --molecule H2 --seeds 10  # multi-seed
```

> **Important:** Check `source_info` in output JSON. If it says `synthetic`, results are placeholder Hamiltonians only — not real chemistry.

### Modules
- `src/` — PES generator, VQE solver, ansatz builder, mapper
- `scripts/` — verification, multi-seed runs, ablation, warm-start, hardware
- `config/simulation_config.yaml` — molecules, mappings, bond-length grids
- `results/` — PES JSON, multi-seed stats, ablation data, figures

---

## Project 03 — Quantum Kernel SVM for MNIST

**Directory:** `03-Quantum-Kernel-SVM-MNIST/`  
**Stack:** Qiskit ML (QSVC, PegasosQSVC), scikit-learn RBF-SVM, OpenML

### What it studies
Benchmarks 4-qubit `ZZFeatureMap` quantum kernel SVM against a tuned classical RBF-SVM on binary digit classification (digits 4 vs 9). Includes PCA dimension ablation, feature-map depth ablation, Kernel-Target Alignment (KTA) diagnostics, and geometric difference analysis.

### Results

**Primary run (fallback sklearn digits, 4 qubits):**
| Model | F1 Score | Accuracy | Training time |
|---|---:|---:|---:|
| Classical RBF-SVM | 1.000 | 1.000 | 0.19s |
| Quantum Pegasos SVM | 0.662 | 0.495 | 291s |

**PCA dimension ablation (3 seeds, fallback dataset):**
| PCA Dim | Classical F1 | Quantum F1 | KTA | Significant? |
|---:|---:|---:|---:|---|
| 4 | 0.569 | 0.616 | 0.128 | no |
| 6 | 0.642 | 0.698 | 0.188 | no |
| 8 | 0.684 | 0.812 | 0.227 | no |

**Depth ablation (PCA dim=8, 3 seeds):**
| Feature map reps | Mean F1 | Mean KTA |
|---:|---:|---:|
| 1 | 0.863 | 0.225 |
| 2 | 0.800 | 0.229 |
| 3 | 0.688 | 0.226 |

Deeper circuits hurt — KTA stays flat while F1 drops.

### Key math

Quantum fidelity kernel:
$$K_Q(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$

Kernel-Target Alignment:
$$\text{KTA}(K, y) = \frac{\langle K, yy^T \rangle_F}{||K||_F \cdot ||yy^T||_F}$$

Geometric difference (quantum vs classical):
$$g(K_Q, K_C) = \sqrt{||K_C^{1/2} K_Q^{-1} K_C^{1/2}||_2}$$

Kernel cost: $O(n^2 \cdot \text{circuit\_evals})$ — for n=100, that's 10,000 kernel evaluations for training alone.

### Entry points
```powershell
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
python run_experiment.py   # full run, needs internet for OpenML MNIST
```

### Caveats
- Checked-in artifact uses sklearn `load_digits` (8×8), **not** full MNIST (28×28). Don't mix the two.
- The `significant_advantage` flag isn't direction-aware — always compare means manually.
- 3 seeds is low statistical power. Treat ablation as preliminary trends.

---

## Project 04 — QAOA Max-Cut (Negative Result)

**Directory:** `04-Optimization-QAOA-MaxCut/`  
**Stack:** Qiskit, NetworkX, Goemans-Williamson SDP, SPSA optimizer

### What it studies
QAOA on weighted Max-Cut vs proper classical baselines: exact solver, greedy, local search, Goemans-Williamson SDP rounding, and budget-matched hill climb.

Three common QAOA benchmark inflations that this project avoids:
1. Reporting best sampled bitstring instead of expected objective
2. Using weak baselines (random cut, simple greedy)
3. Ignoring computational budget parity

### Results

**Primary benchmark (6-node communication mesh):**
| Method | Expected Cut | Best Sampled Cut | Approx. Ratio |
|---|---:|---:|---:|
| Exact | 4.472 | 4.472 | 1.000 |
| QAOA (p=1) | 2.993 | 4.472 | 0.669 |

The best sampled bitstring happens to match exact — but the **expected** cut is only 0.669 of optimal. QAOA optimizes the expected objective, not the luckiest sample.

**Held-out study (8-node graphs, tuned QAOA vs classical):**
| Graph Family | QAOA (tuned) | Goemans-Williamson | Hill Climb |
|---|---:|---:|---:|
| Communication mesh | 0.783 | 1.000 | 0.996 |
| D-regular | 0.863 | 1.000 | 1.000 |
| Erdos-Renyi | 0.834 | 1.000 | 1.000 |

**Verdict:** `weak` — misleading-risk `medium`. Classical baselines win on every tested family.  
Depth-1 QAOA mean ratio: **0.686**, 95% CI [0.653, 0.706], across 3 runs.

### Key math

Weighted Max-Cut cost:
$$C(z) = \sum_{(i,j)\in E} w_{ij}\frac{1 - z_i z_j}{2}, \quad z_i \in \{+1, -1\}$$

QAOA circuit:
$$|\psi(\gamma,\beta)\rangle = \prod_{\ell=1}^{p} e^{-i\beta_\ell H_B} e^{-i\gamma_\ell H_C} |+\rangle^{\otimes n}$$

Approximation ratio:
$$\rho = C_{\text{method}} / C_{\text{exact}}$$

### Entry points
```powershell
python generate_artifacts.py    # full benchmark + artifacts
python integration_test.py      # quick sanity check
Get-Content results\results_verdict.md
```

### Methods implemented
- QAOA (p=1 and p=2)
- RQAOA (recursive variable elimination)
- CVaR objective variant
- GW SDP rounding (proper Goemans-Williamson)
- Budget-matched hill climb and random cut

---

## Project 05 — Quantum RL with Noise Mitigation

**Directory:** `05-Reinforcement-Learning-Noise-Mitigation/`  
**Stack:** Qiskit Aer (PQC policies), PyTorch (MLP baselines), Gymnasium

> **Status:** Smoke benchmark only. Full benchmark config exists; the complete run hasn't been committed yet.

### What it studies
PQC actor-critic and REINFORCE policies for RL — how they degrade under noise, and whether ZNE/PEC mitigation actually recovers useful performance. Compared against: tabular Q-learning, classical MLP actor-critic, and a random baseline.

### Smoke results (`results_benchmark_smoke/benchmark_report.md`)

| Method | Avg Eval Success | Avg Rank | Success/Runtime |
|---|---:|---:|---:|
| Quantum Actor-Critic (ideal) | 1.000 | 1.0 | 0.893 |
| Quantum REINFORCE (ideal) | 0.875 | 2.0 | 0.584 |
| Quantum Actor-Critic (mitigated) | 0.750 | 3.0 | 0.133 |
| Quantum Actor-Critic (noisy) | 0.750 | 4.0 | 0.406 |
| Tabular REINFORCE | 0.375 | 7.0 | **18.466** |
| MLP Actor-Critic | 0.125 | 9.0 | 4.925 |
| Random baseline | 0.185 | 8.0 | — |

Noise impact on REINFORCE: ~0.250 drop from ideal to noisy. Mitigation recovers only ~0.062 of that in smoke runs.

Note: mitigated underperforming noisy in some metrics is a training-length artifact from the short smoke config — not a real finding.

### Key math

PQC state encoding:
$$|\psi(s)\rangle = \prod_i R_Y(s_i \cdot \pi) H|0\rangle^{\otimes n}$$

Action probabilities (from Born's rule, not softmax):
$$\pi_\theta(a|s) = \Pr[M_{\text{action}} = a] = |\langle a | U_{\text{var}}(\theta) U_{\text{enc}}(s) |+\rangle|^2$$

Parameter-shift gradient rule:
$$\frac{\partial \langle O \rangle}{\partial \theta_i} = \frac{\langle O \rangle_{\theta_i + \pi/2} - \langle O \rangle_{\theta_i - \pi/2}}{2}$$

ZNE 3-point Richardson extrapolation:
$$\langle O \rangle_0 \approx 3\langle O \rangle_\lambda - 3\langle O \rangle_{2\lambda} + \langle O \rangle_{3\lambda}$$

Efficiency metric:
$$\text{QV ratio} = \frac{\text{eff}_\text{quantum}}{\text{eff}_\text{classical}}, \quad \text{eff} = \frac{\text{reward} \times \text{convergence speed}}{\text{wall-clock time}}$$

### Entry points
```powershell
# Smoke (minutes)
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml

# Full (multi-hour)
python -m src.benchmark_suite --suite config/benchmark_suite.yaml

# Single agent
python -m src.train --agent quantum --episodes 200 --noise-level 0.01
```

### Architecture
```
src/
  agents/
    tabular_agent.py       — Q-learning, epsilon-greedy
    classical_agent.py     — MLP actor-critic (PyTorch)
    quantum_agent.py       — PQC actor-critic
  circuits/
    pqc_policy.py          — Feature map + variational ansatz
    noise_models.py        — Synthetic noise channels
  mitigation/
    zne_wrapper.py         — Zero-noise extrapolation
    pec_wrapper.py         — Probabilistic error cancellation
    measurement_filter.py  — Readout calibration
  benchmark_suite.py       — Orchestration
  metrics_engine.py        — Reward/convergence/action consistency
```

---

## Repository Infrastructure

```
Quantum-AI-Research-Series/
├── 01-Classical-vs-Quantum-Visualization/
├── 02-Quantum-Chemistry-VQE/
├── 03-Quantum-Kernel-SVM-MNIST/
├── 04-Optimization-QAOA-MaxCut/
├── 05-Reinforcement-Learning-Noise-Mitigation/
├── docs/results_guide.md        — artifact interpretation guide
├── scripts/
│   ├── smoke_all.py             — runs all 5 modules' smoke tests
│   ├── check_authorship.py      — authorship compliance
│   └── setup_ibm_runtime.py     — IBM credential setup
├── requirements.txt
├── CITATION.cff
└── LICENSE (MIT)
```

**Run all five smoke tests:**
```powershell
python -m scripts.smoke_all
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Core | Python 3.11+, NumPy, SciPy, pandas, matplotlib, scikit-learn |
| Quantum | Qiskit, Qiskit Aer, Qiskit ML, Qiskit Nature, Qiskit Optimization, IBM Runtime |
| Chemistry | PySCF (Project 02) |
| Graphs | NetworkX (Project 04) |
| RL | PyTorch, Gymnasium (Project 05) |
| Testing | pytest, unittest |

---

## Design Philosophy

> No quantum advantage is claimed in any module. That's by design — the point is honest evaluation.

- Every module uses proper classical baselines (not straw-men)
- Noise is modeled explicitly, not ignored
- Negative results are reported without spin (Project 04 is entirely a negative result)
- Synthetic fallbacks are clearly labeled and excluded from research conclusions
- Multi-seed statistical aggregation where it matters (Project 02: 10 seeds × 21 bond lengths)

---

## References (across all modules)

- Havlicek et al. Supervised learning with quantum-enhanced feature spaces. *Nature*, 2019.
- Schuld and Killoran. Quantum machine learning in feature Hilbert spaces. *PRL*, 2019.
- Peruzzo et al. A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, 2014.
- Farhi, Goldstone, and Gutmann. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028, 2014.
- Goemans and Williamson. Improved approximation algorithms for maximum cut. *JACM*, 1995.
- Shalev-Shwartz et al. Pegasos: Primal estimated sub-gradient solver for SVM. *Mathematical Programming*, 2011.
- Sutton and Barto. *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Cerezo et al. Variational quantum algorithms. *Nature Reviews Physics*, 2021.
- Huang et al. Power of data in quantum machine learning. *Nature Communications*, 2021.
- Temme, Bravyi, and Gambetta. Error mitigation for short-depth quantum circuits. *PRL*, 2017.

---

**DEVADATH H K — Quantum AI Research Series, 2026**  
MIT License | https://github.com/DEVADATH001/Quantum-AI-Research-Series
