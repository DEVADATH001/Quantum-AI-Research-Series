# Project 04: QAOA and RQAOA Max-Cut Benchmark

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-QAOA%20%7C%20Runtime-6929C4)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Benchmarks-green)
![Status](https://img.shields.io/badge/Status-Negative%20Results%20Benchmark-orange)

Benchmark-oriented research toolkit for studying QAOA and RQAOA on weighted
Max-Cut instances.

This project is intentionally framed as a reproducible benchmark and
negative-results artifact. It does not claim a QAOA advantage. The current
artifacts show that tuned QAOA underperforms strong classical baselines on the
included held-out study.

## README Audit

The previous README contained valuable technical details, but it read more like
a changelog and results memo than a polished research repository entry point.

What was missing or weak:

- The "What Changed" section dominated the document and obscured the core
  research question.
- Mathematical explanations were present but brief and not organized into a
  complete foundations section.
- Installation and usage were concise, but did not distinguish artifact
  generation, sanity checks, tests, and hardware/runtime assumptions clearly.
- The README contained strong result summaries, but they were mixed into the
  flow instead of being presented as explicit benchmark artifacts.
- The system architecture was described through module bullets, not as a
  coherent pipeline.
- Some important credibility points, such as expected value versus sampled
  bitstrings, budget-matched baselines, provenance, and negative-result framing,
  deserved earlier and clearer placement.
- The current communication-mesh graph is a useful weighted proxy, but it is not
  a full robotics networking model; that limitation needed to be prominent.

This rewrite keeps the accurate negative-results framing while making the README
easier for researchers, engineers, and recruiters to evaluate.

## Project Overview

This module builds weighted graph instances, maps Max-Cut to an Ising
Hamiltonian, optimizes QAOA parameters, optionally applies RQAOA-style variable
reduction, and compares against exact and heuristic classical baselines.

| Area | Implementation |
|---|---|
| Problem | Weighted Max-Cut |
| Quantum methods | QAOA, RQAOA support |
| Graph families | Communication mesh, D-regular, Erdos-Renyi, Barabasi-Albert |
| Default benchmark | 6-node weighted communication mesh |
| Execution modes | Local exact/sampling, noisy Aer/fake-backend proxy, optional IBM Runtime |
| Classical baselines | Exact, greedy, local search, random cut, Goemans-Williamson, budget-matched random and hill climb |
| Artifacts | CSV, JSON, plots, run manifests, verdict and publication-positioning reports |

The central question is:

> Under explicit objective-evaluation and sampling assumptions, does QAOA produce
> better Max-Cut solutions than strong classical baselines on the configured
> small graph families?

Current answer from the generated artifacts: **no**.

## Motivation / Research Context

QAOA is often introduced as a near-term quantum algorithm for combinatorial
optimization. Max-Cut is its canonical benchmark problem. However, small
instances are also easy for classical solvers and heuristics. A credible QAOA
study must therefore:

- compare expected objective values, not only the best sampled bitstring;
- use exact references where possible;
- include strong classical baselines;
- account for objective-evaluation budgets;
- separate local simulation, noisy proxy simulation, and real hardware;
- preserve run provenance and statistical summaries.

This project is useful because it demonstrates that honest benchmarking can
produce negative results that are still scientifically valuable.

## Why This README Structure Matters

- **Overview** identifies the benchmark problem and supported methods.
- **Research context** explains why QAOA/Max-Cut needs careful baselines.
- **Architecture** shows how graphs become artifacts.
- **Mathematics** makes the cost Hamiltonian and objective conventions auditable.
- **Usage** helps users regenerate or validate results.
- **Metrics and limitations** prevent misinterpretation of sampled outputs.

## Key Features

- Weighted Max-Cut Hamiltonian construction with offset tracked separately.
- Correct weighted QAOA circuit convention using `RZZ(-gamma * w_ij)`.
- Expected cut value reported as the primary QAOA objective.
- Representative sampled cut and best sampled cut stored separately.
- CVaR objective option for tail-focused optimization.
- Warm starts across QAOA depth sweeps.
- RQAOA engine with sampled-count correlation estimation and constant tracking.
- Noisy simulator path with fake IBM backend fallback.
- Hardware-feasibility reporting: logical depth, transpiled depth, two-qubit
  gates, and estimated shot pressure.
- Held-out study with tuning/evaluation seed split.
- Strong classical baselines, including Goemans-Williamson SDP rounding and
  budget-matched random/hill-climb baselines.
- Run-scoped artifact generation with stable mirror files under `results/`.
- Explicit `results_verdict` and `publication_positioning` artifacts.
- Unit tests for graph generation, Hamiltonians, QAOA circuits, runtime
  execution, metrics, RQAOA, artifacts, and review logic.

## System Architecture

```text
04-Optimization-QAOA-MaxCut
|
|-- config/experiment_config.yaml
|   `-- graph, qaoa, optimizer, quantum, study, review, hardware settings
|
|-- src/graph_generator.py
|   `-- generate weighted graph instances
|
|-- src/hamiltonian_builder.py
|   `-- graph -> Max-Cut interaction operator + offset
|
|-- src/qaoa_circuit.py
|   `-- build QAOA circuits for weighted Max-Cut
|
|-- src/runtime_executor.py
|   `-- local, noisy simulator, or Runtime execution
|
|-- src/qaoa_optimizer.py
|   `-- optimize expected/CVaR objective and decode sampled outputs
|
|-- src/classical_solver.py
|   `-- exact and heuristic classical baselines
|
|-- src/experimental_study.py
|   `-- held-out tuning and evaluation across graph families
|
|-- src/results_review.py
|   `-- robustness analysis, verdict, misleading-risk review
|
|-- src/artifact_pipeline.py
|   `-- orchestrates benchmark, study, plots, manifests, stable artifacts
|
`-- generate_artifacts.py
    `-- CLI entry point for full artifact generation
```

## Mathematical Foundations

### Weighted Max-Cut

For an undirected weighted graph `G = (V, E)`, assign each node a spin
`z_i in {+1, -1}`. An edge is cut when its endpoints have opposite spins:

```math
C(z) = \sum_{(i,j)\in E} w_{ij}\frac{1 - z_i z_j}{2}.
```

The approximation ratio is:

```math
\rho = \frac{C_{\mathrm{method}}}{C_{\mathrm{exact}}}.
```

### Communication-Mesh Weight Proxy

The communication-mesh generator assigns edge weights from normalized latency,
interference, and reliability loss:

```math
w_{ij}
= 0.45\,\mathrm{latency}_{ij}
+ 0.35\,\mathrm{interference}_{ij}
+ 0.20\,(1 - \mathrm{reliability}_{ij}).
```

This is a graph-weight proxy, not a full robotics or networking optimization
model.

### Ising Hamiltonian

Replacing spins with Pauli-Z operators gives the Max-Cut Hamiltonian:

```math
H_C = \sum_{(i,j)\in E} w_{ij}\frac{I - Z_i Z_j}{2}.
```

The code stores this as:

```math
H_C = \mathrm{offset} + H_{ZZ},
```

where

```math
\mathrm{offset} = \sum_{(i,j)\in E}\frac{w_{ij}}{2},
\qquad
H_{ZZ} = -\sum_{(i,j)\in E}\frac{w_{ij}}{2} Z_i Z_j.
```

To recover expected cut value:

```math
J(\theta) = \mathrm{offset} + \langle H_{ZZ}\rangle_\theta.
```

Because the optimizer minimizes, the loss is:

```math
L(\theta) = -J(\theta).
```

### QAOA State

For depth `p`, QAOA prepares:

```math
|\psi(\gamma,\beta)\rangle =
\prod_{\ell=1}^{p}
e^{-i\beta_\ell H_B}
e^{-i\gamma_\ell H_C}
|+\rangle^{\otimes n}.
```

The mixer is:

```math
H_B = \sum_i X_i.
```

The implementation convention for each weighted edge is:

```math
RZZ(-\gamma w_{ij})
= \exp\left(+i\frac{\gamma w_{ij}}{2}Z_iZ_j\right),
```

which matches the interaction part of the Max-Cut Hamiltonian up to a global
phase.

### CVaR Objective

For bitstrings sorted by descending cut value, CVaR optimizes the best
probability tail of mass `alpha`:

```math
J_{\mathrm{CVaR}}(\theta;\alpha)
= \frac{1}{\alpha}\sum_x q_\theta(x)C(x).
```

When `alpha = 1`, CVaR reduces to the expected-value objective.

### RQAOA Intuition

RQAOA repeatedly estimates pair correlations such as:

```math
\langle Z_i Z_j \rangle
```

and eliminates variables by imposing strong same-spin or opposite-spin
relations. This implementation is a benchmark-oriented pragmatic version, not a
new RQAOA theory contribution.

### Computational Complexity

- Exact Max-Cut by enumeration scales as `O(2^n)` and is only used for small
  graphs.
- One QAOA objective evaluation requires circuit execution and cut expectation
  estimation.
- A QAOA run scales roughly as:

```text
O(depth * objective_evaluations * shots * graph_edges)
```

- Held-out studies multiply this by graph families, tuning seeds, evaluation
  seeds, candidate depths, and restart counts.
- Goemans-Williamson depends on solving an SDP and is heavier than simple local
  heuristics, but it is a strong classical reference for small instances.

## Technologies Used

- Python 3.11+
- NumPy, SciPy, pandas
- NetworkX
- matplotlib, seaborn
- Qiskit, Qiskit Aer, Qiskit IBM Runtime
- Qiskit Optimization, Qiskit Algorithms
- scikit-learn
- PyYAML
- cvxpy for SDP-style baseline support
- pytest

## Repository Structure

```text
04-Optimization-QAOA-MaxCut/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- pytest.ini
|-- generate_artifacts.py
|-- integration_test.py
|-- project_04_qaoa_maxcut.ipynb
|-- PLAN.md
|-- config/
|   `-- experiment_config.yaml
|-- data/
|   `-- robot_network.adjlist
|-- docs/
|   `-- mathematical_formulation.md
|-- notebooks/
|   |-- 01_problem_formulation.ipynb
|   |-- 02_qaoa_execution.ipynb
|   `-- 03_results_analysis.ipynb
|-- src/
|   |-- artifact_manager.py
|   |-- artifact_pipeline.py
|   |-- artifact_schema.py
|   |-- classical_solver.py
|   |-- evaluation_metrics.py
|   |-- experimental_study.py
|   |-- graph_generator.py
|   |-- hamiltonian_builder.py
|   |-- hardware_analysis.py
|   |-- provenance.py
|   |-- qaoa_circuit.py
|   |-- qaoa_optimizer.py
|   |-- results_review.py
|   |-- rqaoa_engine.py
|   |-- runtime_executor.py
|   `-- visualization.py
|-- tests/
`-- results/
    |-- metrics.csv
    |-- benchmark_robustness_summary.csv
    |-- study_method_summary.csv
    |-- study_significance.csv
    |-- hardware_feasibility.csv
    |-- results_verdict.md
    |-- publication_positioning.md
    |-- run_manifest.json
    `-- runs/
```

## Installation Guide

From the module directory:

```powershell
cd 04-Optimization-QAOA-MaxCut
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional development dependencies:

```powershell
python -m pip install .[dev]
```

For live IBM Runtime experiments, configure credentials from the repository root
and then edit `config/experiment_config.yaml` to use the desired Runtime mode:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 04-Optimization-QAOA-MaxCut
```

The default benchmark uses a local noisy/fake-backend proxy, not live hardware.

## Usage Instructions

### Generate All Benchmark Artifacts

```powershell
python generate_artifacts.py
```

This runs the configured benchmark, held-out study, robustness review,
visualization generation, provenance capture, and stable artifact mirroring.

### Run a Lightweight End-to-End Sanity Check

```powershell
python integration_test.py
```

This generates a small D-regular graph, solves exact Max-Cut, runs local QAOA,
and prints expected, representative sampled, and best sampled cut values.

### Run Tests

```powershell
python -m pytest tests -q
```

### Inspect Existing Artifacts

```powershell
Get-Content results\metrics.csv
Get-Content results\results_verdict.md
Get-Content results\publication_positioning.md
```

## Example Results / Visualizations

Current generated metrics in [`results/metrics.csv`](results/metrics.csv):

| Method | Depth | Expected Cut | Representative Sampled Cut | Best Sampled Cut | Approx. Ratio | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Exact | `0` | `4.4724` | `4.4724` | `4.4724` | `1.0000` | `0.0005` |
| QAOA | `1` | `2.9925` | `2.7901` | `4.4724` | `0.6691` | `24.8027` |

The best sampled bitstring reaches the exact value in this run, but the
optimized expected value is much lower. This distinction is central: QAOA
optimizes the expected objective, not the luckiest sample.

Robustness summary from
[`results/benchmark_robustness_summary.csv`](results/benchmark_robustness_summary.csv):

| Depth | Mean Ratio | 95% CI | Mean Sample Gap | Exact Sample Hit Rate | Runs |
|---:|---:|---|---:|---:|---:|
| `1` | `0.6856` | `[0.6526, 0.7064]` | `-0.2278` | `0.0000` | `3` |

Held-out study from
[`results/study_method_summary.csv`](results/study_method_summary.csv):

| Family | QAOA Tuned | Goemans-Williamson | Budget-Matched Hill Climb |
|---|---:|---:|---:|
| `communication_mesh_8_3` | `0.7828` | `1.0000` | `0.9962` |
| `d_regular_8_3` | `0.8632` | `1.0000` | `1.0000` |
| `erdos_renyi_8_0.5` | `0.8341` | `1.0000` | `1.0000` |

Verdict artifact: [`results/results_verdict.md`](results/results_verdict.md)

- Overall label: `weak`
- Misleading-risk level: `medium`
- Main conclusion: classical baselines outperform tuned QAOA on the included
  held-out study families.

Key plots:

- `results/approximation_ratio.png`
- `results/method_comparison.png`
- `results/optimization_convergence.png`
- `results/study_significance_heatmap.png`
- `results/study_budget_fairness.png`
- `results/sample_gap_analysis.png`
- `results/graph_cut_visualization.png`

## Experimental Setup

Default config file: [`config/experiment_config.yaml`](config/experiment_config.yaml)

| Component | Default |
|---|---|
| Benchmark graph | `communication_mesh` |
| Benchmark nodes | `6` |
| Benchmark degree | `3` |
| QAOA depths | `[1]` |
| Optimizer | `SPSA`, `maxiter=12` |
| Quantum mode | `noisy_simulator` |
| Backend proxy | `ibm_brisbane` / fake backend fallback |
| Shots | `512` |
| Objective mode | `expected` |
| CVaR alpha | `0.25` when enabled |
| RQAOA default | disabled |
| Held-out study executor | local |
| Held-out graph families | communication mesh, D-regular, Erdos-Renyi |
| Tuning seeds | `101, 202, 303` |
| Evaluation seeds | `404, 505, 606, 707, 808, 909, 1001` |
| Candidate QAOA depths | `1, 2` |
| Candidate restarts | `1, 2` |

The current tuning sweep selects `p=2`, `2` random restarts, and `40` maximum
iterations for the held-out QAOA study.

## Performance Metrics

- `expected_cut_value`: expected Max-Cut value of the optimized QAOA state.
- `sampled_cut_value`: cut value of the representative sampled bitstring,
  usually the most likely sample under the configured analysis mode.
- `best_sampled_cut_value`: best cut observed among samples; tracked separately
  because it can overstate the optimized distribution.
- `approximation_ratio`: method cut divided by exact cut.
- `minimization_objective`: negative objective minimized by the classical
  optimizer.
- `n_evaluations`: number of QAOA objective evaluations.
- `objective_std` and `objective_stderr`: variation across repeated objective
  evaluations.
- `mean_sample_gap`: sampled representative cut minus expected cut.
- `exact_sample_hit_rate`: fraction of robustness runs whose representative
  sampled bitstring equals the exact optimum.
- `transpiled_depth` and `transpiled_two_qubit_gates`: hardware-feasibility
  diagnostics.
- `config_hash`, `git_commit`, and package versions: provenance captured in
  `results/run_manifest.json`.

## Limitations

- No QAOA advantage is demonstrated.
- The default benchmark graph is small and uses a fake-backend/noisy proxy.
- The communication-mesh objective is a weighted graph proxy, not a full
  robotics networking model.
- The held-out study is stronger than a single benchmark, but still covers only
  small 8-node graph families.
- Exact baselines are feasible only because the graphs are small.
- RQAOA is implemented as a pragmatic benchmark method, not a novel reduction
  algorithm.
- The default optimizer path is derivative-free; there is no project-owned
  analytic-gradient or parameter-shift optimizer pipeline.
- Live hardware calibration, queue effects, and shot-accounted Runtime studies
  are not part of the default generated artifact set.
- Checked-in artifacts are useful for review, but users should regenerate them
  after changing config or dependencies.

## Future Improvements

- Extend robustness analysis to the full multi-family held-out study.
- Add larger weighted graph families where exact solution becomes expensive.
- Add stronger classical baselines for medium-sized instances.
- Add live IBM Runtime evaluation slices with calibration, queue, shot, and
  transpilation provenance.
- Add analytic-gradient or parameter-shift optimization paths.
- Expand the communication-mesh model into a domain-specific constrained
  networking objective.
- Add artifact comparison tooling to detect regressions between generated runs.
- Add clearer separation between stable curated artifacts and run-scoped
  generated outputs.

## References

- Farhi, Goldstone, and Gutmann. A Quantum Approximate Optimization Algorithm.
  arXiv:1411.4028, 2014.
- Farhi, Goldstone, and Gutmann. A Quantum Approximate Optimization Algorithm
  Applied to a Bounded Occurrence Constraint Problem. arXiv:1412.6062, 2014.
- Bravyi et al. Obstacles to Variational Quantum Optimization from Symmetry
  Protection. *Physical Review Letters*, 2020.
- Goemans and Williamson. Improved approximation algorithms for maximum cut and
  satisfiability problems using semidefinite programming. *Journal of the ACM*,
  1995.
- Qiskit documentation: https://docs.quantum.ibm.com/
- Qiskit Optimization documentation:
  https://qiskit-community.github.io/qiskit-optimization/

## Author Information

**DEVADATH H K**

Project 04 of the
[`Quantum AI Research Series`](../README.md). See the repository-level
[`LICENSE`](../LICENSE), [`CITATION.cff`](../CITATION.cff), and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) files for licensing, citation, and
contribution guidance.
