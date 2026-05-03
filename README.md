# Quantum AI Research Series

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Recommended](https://img.shields.io/badge/recommended-Python%203.11-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-research%20workflows-6929C4)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20portfolio-orange)
![Claim](https://img.shields.io/badge/quantum%20advantage-not%20claimed-red)

A five-project research software portfolio for studying near-term quantum computing methods across quantum machine learning, variational chemistry, quantum kernels, combinatorial optimization, and reinforcement learning under noise.

This repository is intentionally benchmark-oriented. It does **not** claim quantum advantage. The goal is to show how quantum AI workflows should be tested: with explicit baselines, reproducible artifacts, hardware/noise constraints, and honest limitations.

---

## README Audit

The previous top-level README was a solid overview, but it was not yet as rigorous as the upgraded project READMEs.

### Missing Sections

- No explicit audit section, even though the repository is intended to be evaluated by researchers and recruiters.
- No clear "repository-level versus module-level" installation guidance.
- No strong warning that some checked-in artifacts are smoke, fallback, or negative-result artifacts.
- No project maturity table summarizing which modules are demonstration, benchmark, or publication-adjacent.
- No section explaining why each README section exists.
- No repository-level quality score.

### Weak Explanations

- The README described all modules evenly, but the evidence level differs substantially across modules.
- Project 03 was described as MNIST-focused without enough emphasis that the checked-in summary uses `sklearn_digits_fallback`.
- Project 04 was described as a QAOA benchmark, but the top-level README did not foreground its negative-result verdict.
- Project 05 mentioned benchmark reports, but did not clearly state that only a two-scenario smoke benchmark report is checked in.
- The installation section did not sufficiently warn that root dependencies are broad and module-local setup is safer for serious work.

### Technical Inaccuracies or Risky Ambiguities

- Python support was stated as 3.10+, while Project 05 explicitly declares Python 3.11+ and saved artifacts record newer local runtimes. The safer recommendation is Python 3.11.
- "Quantum kernel SVM for MNIST" can mislead readers if they inspect the current artifact first; the saved artifact is fallback digits 4-vs-9.
- The top-level README could be read as a unified package, but the repository is a research collection with independent module workflows.
- IBM Runtime setup was present, but not enough distinction was made between optional hardware execution and default local/fake-backend workflows.

### Unclear Instructions

- There was no quick triage path for users who only want to verify the repository.
- Commands were listed, but not grouped by expected cost and reliability.
- The README did not say which outputs are canonical for each module.
- It did not clearly tell users to prefer script entry points over notebook state.

### Missing Research Context

- The previous README did not fully articulate the common research thesis: honest evaluation of near-term quantum methods rather than performance marketing.
- It did not compare evidence strength across modules.
- It did not explain that negative results are part of the contribution.

### Poor Structure

- The README had most sections, but not enough critical framing.
- It mixed overview, usage, and evidence without a strong reviewer-facing narrative.
- The final quality assessment requested by the user was absent.

---

## Project Overview

The Quantum AI Research Series contains five independent research modules:

| Project | Topic | Current Evidence Level | README |
|---|---|---|---|
| Project 01 | Classical ML vs quantum SVC visualization, plus GHZ-127 noise comparison | Educational benchmark with saved Iris and GHZ artifacts | [01 README](01-Classical-vs-Quantum-Visualization/README.md) |
| Project 02 | VQE potential energy surfaces for molecular systems | Research-style chemistry workflow with exact references and fallback caveats | [02 README](02-Quantum-Chemistry-VQE/README.md) |
| Project 03 | Quantum kernel SVM for binary digit classification | Benchmark pipeline; checked-in summary uses fallback digits, not full MNIST | [03 README](03-Quantum-Kernel-SVM-MNIST/README.md) |
| Project 04 | QAOA/RQAOA Max-Cut benchmarking | Negative-result benchmark against strong classical baselines | [04 README](04-Optimization-QAOA-MaxCut/README.md) |
| Project 05 | Quantum RL policy learning under noise and mitigation | Benchmark framework; checked-in benchmark report is smoke-level | [05 README](05-Reinforcement-Learning-Noise-Mitigation/README.md) |

The repository should be read as a research portfolio, not a single production package. Each module has its own assumptions, commands, artifacts, and limitations.

This section is important because it gives new users the map before they enter module-specific details.

---

## Motivation / Research Context

Near-term quantum AI projects are often evaluated with weak baselines, hand-picked examples, or unclear hardware assumptions. This repository is built around a stricter principle:

> A quantum experiment is only useful if the baseline, noise model, resource cost, and failure modes are documented.

The common research questions are:

- When does a quantum model behave differently from a classical baseline?
- How much of that behavior survives noise, finite shots, and hardware constraints?
- Are reported improvements statistically credible or just artifacts of small samples?
- What does the method cost in runtime, shots, kernel evaluations, or circuit executions?
- When are negative results the honest conclusion?

This section is important because it explains why the repository emphasizes benchmarking and caveats over marketing claims.

---

## Key Features

- Five focused modules covering QML visualization, VQE, quantum kernels, QAOA/RQAOA, and quantum RL.
- Script-first workflows with saved artifacts in JSON, CSV, Markdown, PNG, SVG, and notebook formats.
- Stronger classical baselines where implemented, including RBF SVMs, exact diagonalization, exact Max-Cut, Goemans-Williamson-style baselines, local search, tabular RL, and MLP RL.
- Noise-aware experiments using local simulators, fake-backend proxies, compact noise models, and optional IBM Runtime paths.
- Mathematical documentation for variational circuits, quantum kernels, VQE, QAOA, Max-Cut Ising mappings, and policy-gradient RL.
- Explicit negative-result framing in Projects 04 and 05.
- Module-level READMEs upgraded for research users, engineers, and recruiters.
- Citation, license, contribution, authorship, and code-of-conduct files at the repository root.

This section is important because it lets readers quickly determine whether the repository contains serious research-engineering work.

---

## System Architecture

```text
Quantum-AI-Research-Series/
|
|-- 01-Classical-vs-Quantum-Visualization/
|   |-- Iris classification: logistic regression, RBF SVM, QSVC
|   `-- GHZ-127 comparison: ideal, noisy, optional real hardware
|
|-- 02-Quantum-Chemistry-VQE/
|   |-- molecule definitions and electronic-structure drivers
|   |-- qubit Hamiltonians and exact diagonalization references
|   `-- VQE ansatz/optimizer sweeps and PES artifacts
|
|-- 03-Quantum-Kernel-SVM-MNIST/
|   |-- data loading with MNIST/OpenML or fallback digits
|   |-- PCA preprocessing and quantum feature maps
|   `-- classical/quantum SVM metrics, ablations, and plots
|
|-- 04-Optimization-QAOA-MaxCut/
|   |-- weighted graph instances and Ising mappings
|   |-- QAOA/RQAOA experiments
|   `-- exact, heuristic, and budget-matched baseline comparisons
|
|-- 05-Reinforcement-Learning-Noise-Mitigation/
|   |-- key-and-door RL scenarios
|   |-- measurement-defined quantum policies
|   `-- ideal, noisy, mitigated, and report-generation workflows
|
|-- scripts/
|   |-- setup_ibm_runtime.py
|   `-- check_authorship.py
|
|-- PROJECT_KNOWLEDGE_BASE.md
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md
|-- AUTHORSHIP_POLICY.md
|-- LICENSE
|-- requirements.txt
|-- pyproject.toml
|-- setup.py
`-- README.md
```

Repository-level design:

```text
module config or script
   |
   v
local simulator / exact solver / optional runtime backend
   |
   v
classical baseline + quantum method
   |
   v
JSON/CSV metrics + plots + module README interpretation
```

This section is important because researchers and engineers need to see how code, artifacts, and documentation relate.

---

## Mathematical Foundations

### Variational Quantum Algorithms

Projects 02 and 04 use parameterized quantum circuits optimized by a classical outer loop:

$$
|\psi(\theta)\rangle = U(\theta)|0\rangle
$$

For VQE, the objective is the Hamiltonian expectation:

$$
E(\theta) = \langle \psi(\theta)|H|\psi(\theta)\rangle
$$

$$
\theta^\* = \arg\min_{\theta} E(\theta)
$$

Project 02 compares VQE energy against exact diagonalization and uses chemical accuracy as a reference threshold:

$$
|E_{\mathrm{VQE}} - E_{\mathrm{exact}}| \leq 0.0016 \ \mathrm{Hartree}
$$

### Quantum Kernels

Projects 01 and 03 encode classical vectors into quantum states:

$$
|\phi(x)\rangle = U_{\phi}(x)|0\rangle
$$

The quantum kernel is state fidelity:

$$
K_Q(x_i, x_j) = |\langle \phi(x_i)|\phi(x_j)\rangle|^2
$$

Project 03 also tracks Kernel-Target Alignment:

$$
\mathrm{KTA}(K, y) =
\frac{\langle K, yy^T\rangle_F}{\|K\|_F \|yy^T\|_F}
$$

Kernel construction scales as:

$$
O(n^2 C_{\mathrm{kernel}})
$$

where `n` is the number of examples and `C_kernel` is the cost of one kernel estimate.

### Max-Cut and QAOA

Project 04 maps weighted Max-Cut to an Ising objective:

$$
C(z) = \sum_{(i,j)\in E} w_{ij}\frac{1 - z_i z_j}{2}
$$

The corresponding cost Hamiltonian is:

$$
H_C = \sum_{(i,j)\in E} w_{ij}\frac{I - Z_iZ_j}{2}
$$

QAOA prepares:

$$
|\psi(\gamma,\beta)\rangle =
\prod_{\ell=1}^{p}
e^{-i\beta_\ell H_B}
e^{-i\gamma_\ell H_C}
|+\rangle^{\otimes n}
$$

The project reports expected objective values separately from sampled bitstrings because high expected value does not guarantee a strong representative sample.

### Measurement-Defined Quantum Policies

Project 05 defines the policy directly through action-register measurement:

$$
\pi_{\theta}(a|s) = \Pr[M_{\mathrm{action}} = a]
$$

For parameter-shift-compatible gates:

$$
\frac{\partial f(\theta)}{\partial \theta_i}
=
\frac{
f(\theta_i + \pi/2) - f(\theta_i - \pi/2)
}{2}
$$

Under finite shots, noise, and mitigation, this estimator becomes stochastic and can be biased.

This section is important because the repository contains algorithms whose assumptions should be readable without opening every source file.

---

## Technologies Used

Core stack:

- Python 3.10+ at the repository level; Python 3.11 is recommended for current module compatibility.
- Qiskit, Qiskit Aer, Qiskit Machine Learning, Qiskit Nature, Qiskit Optimization, and IBM Runtime interfaces.
- NumPy, SciPy, pandas, matplotlib, scikit-learn, NetworkX, PyYAML, Jupyter.
- PySCF for chemistry workflows when available.
- pytest and unittest for module-level checks.

Important dependency note: root `requirements.txt` is broad. For serious reproduction, install from the target module and follow its README.

This section is important because quantum software stacks change quickly and reproducibility depends on version-aware setup.

---

## Repository Structure

```text
.
|-- 01-Classical-vs-Quantum-Visualization/
|-- 02-Quantum-Chemistry-VQE/
|-- 03-Quantum-Kernel-SVM-MNIST/
|-- 04-Optimization-QAOA-MaxCut/
|-- 05-Reinforcement-Learning-Noise-Mitigation/
|-- scripts/
|-- results/
|-- results_smoke/
|-- PROJECT_KNOWLEDGE_BASE.md
|-- report.md
|-- requirements.txt
|-- pyproject.toml
|-- setup.py
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md
|-- AUTHORSHIP_POLICY.md
|-- AUTHORSHIP_COMPLIANCE_REPORT.md
|-- LICENSE
`-- README.md
```

Top-level files:

| File | Purpose |
|---|---|
| `PROJECT_KNOWLEDGE_BASE.md` | Repository knowledge base and project notes |
| `requirements.txt` | Broad dependency list for the whole research collection |
| `pyproject.toml` | Repository metadata package definition |
| `setup.py` | Minimal package metadata shim |
| `CITATION.cff` | Citation metadata |
| `CONTRIBUTING.md` | Contribution guidance |
| `AUTHORSHIP_POLICY.md` | Authorship policy |
| `LICENSE` | MIT license |

This section is important because the repository mixes source code, experiments, generated artifacts, and governance files.

---

## Installation Guide

### Recommended Root Setup

Use Python 3.11 unless a module README says otherwise:

```powershell
git clone <repository-url>
cd Quantum-AI-Research-Series
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root install is useful for exploration. For research reproduction, install inside the module you want to run.

### Module-Specific Setup

Example:

```powershell
cd 05-Reinforcement-Learning-Noise-Mitigation
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Optional IBM Runtime Setup

Do not commit credentials. Use environment variables or local config:

```powershell
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
```

See [`.env.example`](.env.example) and [`scripts/setup_ibm_runtime.py`](scripts/setup_ibm_runtime.py).

This section is important because new users need a safe install path and a clear distinction between local simulation and optional cloud hardware.

---

## Usage Instructions

### Fast Repository Tour

Start with Project 01:

```powershell
cd 01-Classical-vs-Quantum-Visualization
python Quantum_ML_-_Iris_Classification.py --no-show
```

Run GHZ comparison without live hardware:

```powershell
python compare_ghz_three_way.py --skip-real
```

Note: this script may still initialize IBM Runtime paths depending on configuration. See the Project 01 README for details.

### Chemistry Verification

```powershell
cd ..\02-Quantum-Chemistry-VQE
python scripts/run_verification.py
```

Check `source_info` in outputs. Synthetic fallback results are useful for pipeline testing, not for chemistry claims.

### Quantum Kernel Benchmark

```powershell
cd ..\03-Quantum-Kernel-SVM-MNIST
python run_experiment.py --fallback --max-quantum-train 40 --disable-noise
```

The checked-in summary uses fallback digits, not full MNIST/OpenML.

### QAOA Max-Cut Artifacts

```powershell
cd ..\04-Optimization-QAOA-MaxCut
python generate_artifacts.py
```

Read `results/results_verdict.md` before interpreting QAOA performance.

### Quantum RL Smoke Benchmark

```powershell
cd ..\05-Reinforcement-Learning-Noise-Mitigation
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

For the full configured benchmark:

```powershell
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

This section is important because users should have a shortest path to reproducible execution without accidentally launching expensive hardware or long benchmark jobs.

---

## Example Results / Visualizations

Representative checked-in artifacts:

| Project | Artifact | What it shows |
|---|---|---|
| 01 | `assets/classical_vs_quantum_boundaries.png` | Iris decision boundaries for classical and quantum classifiers |
| 01 | `assets/three_way_ghz127_comparison.png` | Local/noisy/real-style GHZ-127 comparison artifact |
| 02 | `results/figures/pes_curve_*.png` | Potential energy surface curves |
| 02 | `results/figures/error_*.png` | VQE error against reference energies |
| 03 | `results/kernel_heatmap.png` | Quantum-kernel structure |
| 03 | `results/confusion_matrix_classical.png` | Classical classifier confusion matrix |
| 03 | `results/confusion_matrix_quantum.png` | Quantum classifier confusion matrix |
| 04 | `results/method_comparison.png` | QAOA and classical baseline comparison |
| 04 | `results/study_significance_heatmap.png` | Statistical comparison heatmap |
| 05 | `results_benchmark_smoke/paper_report/figures/figure_1_leaderboard.png` | Smoke benchmark leaderboard |
| 05 | `results_benchmark_smoke/paper_report/figures/figure_3_noise_forest.png` | Noise and mitigation summary |

These outputs are module-specific. They should not be combined into a single cross-project leaderboard.

This section is important because visual artifacts help recruiters and researchers quickly inspect the project, while the caveat prevents overinterpretation.

---

## Experimental Setup

| Project | Problem | Quantum Method | Baselines | Current Caveat |
|---|---|---|---|---|
| 01 | Iris and GHZ-127 | QSVC, GHZ circuit simulation | Logistic regression, RBF SVM, ideal simulator | Iris QSVC underperforms classical baselines in saved report |
| 02 | Molecular PES scans | VQE ansatze and optimizers | Exact diagonalization | Synthetic fallback is not valid chemistry evidence |
| 03 | Binary digit classification | Quantum kernel SVM / Pegasos QSVC | Tuned RBF SVM | Saved summary uses fallback digits; quantum underperforms classical there |
| 04 | Weighted Max-Cut | QAOA and RQAOA | Exact, GW-style, greedy, local search, budget-matched heuristics | Current verdict: weak evidence, medium misleading risk |
| 05 | Key-and-door RL | Measurement-defined quantum policy | Random, tabular, MLP, actor-critic | Full benchmark configured, smoke report checked in |

This section is important because each module has a different experimental contract. Comparing them without context would be misleading.

---

## Performance Metrics

Metric families by module:

- Project 01: accuracy, classification reports, decision boundaries, GHZ fidelity/probability concentration, backend comparison.
- Project 02: exact energy, VQE energy, absolute error, chemical-accuracy rate, convergence traces.
- Project 03: accuracy, precision, recall, F1 score, confusion matrix, Kernel-Target Alignment, kernel conditioning, runtime.
- Project 04: expected cut value, approximation ratio, sampled cut values, sample gap, classical baseline rank, runtime, hardware feasibility.
- Project 05: held-out success, held-out reward, reward AUC, success AUC, runtime, estimated shots, noise drop, mitigation recovery.

Selected checked-in results:

| Project | Result Snapshot | Interpretation |
|---|---|---|
| 01 | Iris saved report: Logistic Regression 0.973684, RBF SVM 0.973684, Quantum SVC 0.631579 | Quantum classifier is not competitive on this saved Iris run |
| 03 | Fallback digits summary: classical F1 1.000, quantum F1 0.6618 | Current checked-in fallback result favors classical SVM |
| 04 | `results_verdict.md`: overall label `weak`; classical baselines outperform tuned QAOA | Negative-result benchmark, not a QAOA advantage claim |
| 05 | Smoke benchmark: ideal quantum actor-critic leads raw smoke success, tabular wins runtime efficiency | Smoke validation only; full benchmark evidence not checked in |

This section is important because performance claims should be attached to actual artifacts and their scope.

---

## Limitations

- This is a research portfolio, not a unified production library with one stable public API.
- No module currently proves practical quantum advantage.
- Some workflows rely on optional heavy dependencies or external services such as PySCF, OpenML, and IBM Runtime.
- Hardware execution is optional, queue-dependent, and sensitive to calibration state.
- Generated artifacts have different maturity levels: some are smoke tests, some are fallback runs, and some are negative-result studies.
- Notebook state may lag behind script workflows; scripts and saved JSON/CSV artifacts are the preferred reproducibility path.
- Statistical power varies by module and should be improved before publication-level claims.
- Dependency versions differ across modules because the projects were built as research experiments rather than one monolithic package.

This section is important because professional open-source research should say what it does not establish.

---

## Future Improvements

- Add a top-level smoke-test command that verifies all five modules with bounded runtime.
- Add per-module lock files or `uv`/conda environment files.
- Standardize artifact schemas across modules.
- Add CI for README link checks, linting, and module smoke tests.
- Publish full benchmark bundles for Project 05 and expanded multi-seed artifacts for underpowered studies.
- Add `docs/` at the repository root with a cross-project results guide.
- Separate generated artifacts from source code more consistently.
- Add a model card or experiment card for each project.
- Add hardware-run provenance templates for IBM Runtime experiments.

This section is important because it tells maintainers and reviewers how the portfolio can become more reproducible and publication-ready.

---

## References

- Havlicek et al. "Supervised learning with quantum-enhanced feature spaces." *Nature*, 2019.
- Schuld and Killoran. "Quantum machine learning in feature Hilbert spaces." *Physical Review Letters*, 2019.
- Peruzzo et al. "A variational eigenvalue solver on a photonic quantum processor." *Nature Communications*, 2014.
- Farhi, Goldstone, and Gutmann. "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028, 2014.
- Goemans and Williamson. "Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming." *Journal of the ACM*, 1995.
- Shalev-Shwartz et al. "Pegasos: Primal estimated sub-gradient solver for SVM." *Mathematical Programming*, 2011.
- Williams. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 1992.
- Sutton and Barto. *Reinforcement Learning: An Introduction*, 2nd edition, 2018.
- Qiskit documentation: https://docs.quantum.ibm.com/
- Qiskit Machine Learning documentation: https://qiskit-community.github.io/qiskit-machine-learning/

This section is important because research software should connect implementation choices to established literature.

---

## Author Information

**Author:** DEVADATH H K

This repository is released under the MIT License. See:

- [LICENSE](LICENSE)
- [CITATION.cff](CITATION.cff)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [AUTHORSHIP_POLICY.md](AUTHORSHIP_POLICY.md)

Recommended citation:

```text
DEVADATH H K. Quantum AI Research Series. 2026.
```

This section is important because open-source research repositories should expose ownership, licensing, citation, and contribution expectations.

---

## Brutal Quality Check

Current top-level README quality after this rewrite:

| Category | Score | Rationale |
|---|---:|---|
| Clarity | 9.2/10 | The README now separates portfolio scope, module scope, evidence level, commands, and caveats. |
| Technical depth | 9.0/10 | Includes core equations and explains VQE, kernels, QAOA, and quantum RL without overwhelming the top level. |
| Documentation quality | 9.2/10 | Structure matches research repository expectations and links to module READMEs for detail. |
| Research credibility | 9.1/10 | Explicitly states negative results, fallback artifacts, smoke artifacts, and no quantum-advantage claim. |
| Open-source usability | 8.8/10 | Install and run paths are clear; a top-level smoke command and CI would improve usability. |

Overall score: **9.1/10**.

Final improvements needed:

- add a single `python -m scripts.smoke_all` or equivalent root command;
- add CI badges only after CI exists;
- publish a full Project 05 benchmark report, not only smoke artifacts;
- add per-module locked environments;
- add a root `docs/results_guide.md` explaining how to cite and interpret each artifact.
