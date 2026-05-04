# Project 04: QAOA Max-Cut — A Negative-Result Benchmark

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-QAOA%20|%20Runtime-6929C4)
![NetworkX](https://img.shields.io/badge/NetworkX-graph%20benchmarks-green)
![Status](https://img.shields.io/badge/result-negative-red)

This is a negative-result benchmark, and I'm upfront about it. QAOA underperforms classical baselines on every tested graph family. That's the conclusion — it's not a failure of the project.

I built this because QAOA benchmarks often inflate results in three specific ways:
1. Reporting the **best sampled bitstring** instead of the **expected objective** (these are very different things)
2. Using weak classical baselines (random cut, simple greedy) instead of proper solvers
3. Not accounting for computational budget — how many function evaluations does the classical solver get vs QAOA?

This project fixes all three.

---

## Results

### Primary benchmark (6-node communication mesh)

From `results/metrics.csv`:

| Method | Expected Cut | Best Sampled Cut | Approx. Ratio | Runtime |
|---|---:|---:|---:|---:|
| Exact | 4.472 | 4.472 | 1.000 | 0.001s |
| QAOA (p=1) | 2.993 | 4.472 | 0.669 | 24.8s |

The best sampled bitstring happens to match the exact solution — but the expected cut is only 0.669 of optimal. QAOA optimizes the expected objective, not the luckiest sample. Reporting only the best sample would make QAOA look much better than it actually is.

### Held-out study (8-node graphs, tuned QAOA vs classical baselines)

From `results/study_method_summary.csv`:

| Graph Family | QAOA (tuned) | Goemans-Williamson | Budget-Matched Hill Climb |
|---|---:|---:|---:|
| Communication mesh (8, 3) | 0.783 | 1.000 | 0.996 |
| D-regular (8, 3) | 0.863 | 1.000 | 1.000 |
| Erdos-Renyi (8, 0.5) | 0.834 | 1.000 | 1.000 |

Classical baselines hit perfect or near-perfect approximation ratios on all families. QAOA doesn't come close.

### Robustness

Depth-1 QAOA mean approximation ratio: **0.686**, 95% CI [0.653, 0.706], across 3 runs.

**Verdict** (from `results/results_verdict.md`): `weak` — misleading-risk level `medium`. All classical baselines outperform QAOA on every tested family.

---

## The research question

> Under explicit objective-evaluation budgets and proper sampling conventions, does QAOA produce better Max-Cut solutions than strong classical baselines?

**No.** Not on these graphs.

---

## How it works

**Weighted Max-Cut:** partition nodes into two sets to maximize cut weight:

$$C(z) = \sum_{(i,j)\in E} w_{ij}\frac{1 - z_i z_j}{2}, \quad z_i \in \{+1, -1\}$$

Approximation ratio: $\rho = C_{\text{method}} / C_{\text{exact}}$.

**Ising mapping:** the Max-Cut cost maps to a ZZ Hamiltonian:

$$H_C = \text{offset} + H_{ZZ}, \quad H_{ZZ} = -\sum_{(i,j)\in E}\frac{w_{ij}}{2} Z_i Z_j$$

Expected cut: $J(\theta) = \text{offset} + \langle H_{ZZ}\rangle_\theta$. The optimizer minimizes $-J(\theta)$.

**QAOA circuit:** alternating cost and mixer unitaries:

$$|\psi(\gamma,\beta)\rangle = \prod_{\ell=1}^{p} e^{-i\beta_\ell H_B} e^{-i\gamma_\ell H_C} |+\rangle^{\otimes n}$$

Weighted edges use $RZZ(-\gamma w_{ij})$ gates, matching the Max-Cut interaction up to global phase.

**Communication-mesh weights** are synthetic — not from real network data:

$$w_{ij} = 0.45\,\text{latency}_{ij} + 0.35\,\text{interference}_{ij} + 0.20\,(1 - \text{reliability}_{ij})$$

This is a weighted-graph proxy for testing, not a networking optimization model.

**CVaR objective:** optimizes the best-$\alpha$ tail of the cost distribution. At $\alpha=1$, it reduces to the expected value.

**RQAOA:** iteratively estimates $\langle Z_i Z_j \rangle$ correlations and eliminates strongly-correlated variable pairs. This is a pragmatic benchmark implementation, not a novel algorithm.

**Complexity:** exact Max-Cut is $O(2^n)$ — feasible only for small graphs. QAOA cost per run:
```
O(depth × optimizer_evaluations × shots × |E|)
```

---

## What it covers

| Area | Implementation |
|---|---|
| Problem | Weighted Max-Cut |
| Quantum | QAOA, RQAOA |
| Graphs | Communication mesh, D-regular, Erdos-Renyi, Barabasi-Albert |
| Primary benchmark | 6-node weighted communication mesh |
| Execution | Local exact/sampling, noisy Aer, optional IBM Runtime |
| Classical baselines | Exact, greedy, local search, random cut, Goemans-Williamson SDP, budget-matched random + hill climb |
| Artifacts | CSV, JSON, plots, run manifests, verdict report, publication positioning |

---

## How to run

### Full benchmark

```powershell
python generate_artifacts.py
```

Runs benchmark, held-out study, robustness analysis, visualization, and provenance capture.

### Quick sanity check

```powershell
python integration_test.py
```

Small D-regular graph, exact Max-Cut, local QAOA. Prints expected/representative/best sampled cut.

### Read the verdict

```powershell
Get-Content results\results_verdict.md
Get-Content results\publication_positioning.md
Get-Content results\metrics.csv
```

### Tests

```powershell
python -m pytest tests -q
```

---

## Installation

```powershell
cd 04-Optimization-QAOA-MaxCut
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For IBM Runtime (optional):

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 04-Optimization-QAOA-MaxCut
```

Default uses local noisy/fake-backend simulation. Hardware is not the default path.

---

## Default configuration

From `config/experiment_config.yaml`:

| Setting | Value |
|---|---|
| Benchmark graph | Communication mesh, 6 nodes |
| QAOA depths | `[1]` |
| Optimizer | SPSA, `maxiter=12` |
| Quantum mode | Noisy simulator |
| Backend proxy | `ibm_brisbane` / fake backend |
| Shots | 512 |
| Objective | Expected value |
| Held-out tuning seeds | 101, 202, 303 |
| Held-out eval seeds | 404, 505, 606, 707, 808, 909, 1001 |
| Candidate depths | 1, 2 |
| Restarts | 1, 2 |

Tuning sweep selects p=2, 2 restarts, 40 max iterations for the held-out study.

---

## Limitations

- **QAOA loses on every tested graph.** This is a negative result, not a work-in-progress.
- Benchmark graph is 6 nodes; held-out graphs are 8 nodes. These are small — exact baselines are only feasible because of that.
- Communication-mesh weights are synthetic, not real network data.
- RQAOA is implemented as a benchmark method, not a novel algorithm.
- Default optimizer is derivative-free (SPSA). No parameter-shift or analytic-gradient path.
- No live hardware results with calibration and queue provenance in the default artifacts.

---

## What I'd work on next

- Robustness analysis across the full multi-family held-out study
- Larger weighted graphs where exact solution becomes expensive
- Analytic-gradient or parameter-shift optimization
- Live IBM Runtime evaluation with transpilation and calibration metadata
- A proper domain-specific constrained objective beyond the communication-mesh proxy
- Artifact comparison tooling to detect regressions between runs

---

## References

- Farhi, Goldstone, and Gutmann. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028, 2014.
- Farhi, Goldstone, and Gutmann. QAOA applied to a bounded occurrence constraint problem. arXiv:1412.6062, 2014.
- Bravyi et al. Obstacles to variational quantum optimization from symmetry protection. *PRL*, 2020.
- Goemans and Williamson. Improved approximation algorithms for maximum cut. *Journal of the ACM*, 1995.
- Qiskit: https://docs.quantum.ibm.com/
- Qiskit Optimization: https://qiskit-community.github.io/qiskit-optimization/

---

## Author

**DEVADATH H K** — Part of the [Quantum AI Research Series](../README.md).

See [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), and [CONTRIBUTING.md](../CONTRIBUTING.md).
