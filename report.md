# Quantum AI Research Series: Repository Status Report

This report summarizes the current verified state of the repository as of **March 16, 2026**. It is intended as a technical status document, not a marketing summary. The emphasis is on what is implemented, what has been verified locally, what committed artifacts actually show, and where the repository still has reproducibility or maturity gaps.

## Executive Summary

The repository is a five-project quantum-AI research portfolio covering quantum machine learning, variational quantum chemistry, combinatorial optimization, and quantum reinforcement learning under noise. The strongest parts of the portfolio are:

- Project 02, which has the cleanest modular architecture and the strongest local test coverage.
- Project 01, which clearly separates an educational QML visualization from a hardware-style GHZ benchmark.
- Project 05, which has an end-to-end training pipeline with committed logs and figures across ideal, noisy, and mitigated modes.

The main documentation and reproducibility caveats are:

- Project 03 has a maintained script pipeline, but the committed results currently reflect a smoke-run fallback configuration rather than a fresh benchmark from the current default setup.
- Project 04 has solid implementation building blocks and a working integration test, but it does not yet package a full experiment runner and committed result bundle at the same level as Projects 02, 03, and 05.
- Project requirements vary slightly by module, so per-project environments remain the safer path for reproducibility than a single root install.

## Repository Verification Snapshot

The following checks were run locally during this update:

| Check | Result | Notes |
| --- | --- | --- |
| `python scripts/check_authorship.py` | Passed | `Scanned files: 149`, `Violations detected: 0` |
| `python -m unittest 02-Quantum-Chemistry-VQE.tests.test_unit -v` | Passed | 12 tests passed |
| `python -m unittest tests.test_pipeline -v` from `03-Quantum-Kernel-SVM-MNIST` | Passed | 4 tests passed after restoring KTA logic |
| `python 04-Optimization-QAOA-MaxCut/integration_test.py` | Passed | Verified approximation ratio `0.7143` |
| `python -m pytest 04-Optimization-QAOA-MaxCut/tests -q -p no:cacheprovider` | Passed | 6 tests passed, 1 skipped |

## Project 01: Classical vs Quantum Visualization

### Scope

Project 01 contains two intentionally separate studies:

1. Iris classification with logistic regression, classical RBF-SVM, and QSVC.
2. A GHZ-127 benchmark comparing local ideal simulation, noisy simulation, and optional IBM execution.

This separation is correct and important: the GHZ benchmark is a hardware stress test, not evidence for classification performance.

### Verified Artifacts

- [`qml_iris_report.json`](01-Classical-vs-Quantum-Visualization/assets/qml_iris_report.json)
- [`three_way_ghz127_comparison.json`](01-Classical-vs-Quantum-Visualization/assets/three_way_ghz127_comparison.json)
- [`classical_vs_quantum_boundaries.png`](01-Classical-vs-Quantum-Visualization/assets/classical_vs_quantum_boundaries.png)
- [`three_way_ghz127_comparison.png`](01-Classical-vs-Quantum-Visualization/assets/three_way_ghz127_comparison.png)

### Current Findings

- Iris metrics in the committed report:
  - Logistic Regression accuracy: `0.973684`
  - Classical RBF-SVM accuracy: `0.973684`
  - Quantum SVC accuracy: `0.631579`
- GHZ benchmark in the committed artifact:
  - selected backend target: `ibm_fez`
  - ISA circuit depth: `382`
  - local `p_ghz_subspace`: `1.0`
  - noisy simulation `p_ghz_subspace`: `0.0`
  - real hardware status in the committed artifact: `skipped`

### Interpretation

This project currently demonstrates two useful truths:

- small QML demos are easy to visualize but do not imply quantum advantage;
- deep GHZ circuits are highly sensitive to realistic noise, even when the ideal simulator remains perfectly in the target subspace.

## Project 02: Quantum Chemistry VQE

### Scope

Project 02 implements a modular VQE stack for molecular potential-energy-surface scans, with exact reference energies, configurable ansatz selection, config validation, plotting, runtime abstraction, and test coverage.

### Verified Artifacts

- [`pes_LiH_table.csv`](02-Quantum-Chemistry-VQE/results/raw_data/pes_LiH_table.csv)
- [`pes_H2_20260316_040559_table.csv`](02-Quantum-Chemistry-VQE/results/raw_data/pes_H2_20260316_040559_table.csv)
- [`simulation_config.yaml`](02-Quantum-Chemistry-VQE/config/simulation_config.yaml)

### Current Findings

- Local unit test suite passed: 12/12 tests.
- The committed LiH table contains 26 bond-length points with no recorded failures.
- Against the repository chemical-accuracy target of `0.0016 Ha`:
  - UCCSD is within chemical accuracy at `24/26` LiH points.
  - EfficientSU2 is within chemical accuracy at `19/26` LiH points.
- The chemistry artifacts currently committed were generated in `local_statevector` mode.

### Interpretation

This is the most mature research-software module in the repository. It has the clearest separation of concerns and the best testing story. The main caveat is environmental: PySCF-backed integrals are constrained by Python version, and the synthetic fallback should be treated as a smoke-test path, not a publication-grade substitute for real electronic-structure integrals.

## Project 03: Quantum Kernel SVM for MNIST

### Scope

Project 03 compares a classical SVM baseline with a quantum-kernel pipeline on reduced digit data, with preprocessing, kernel analysis, evaluation metrics, noise utilities, and plotting.

### Verified Artifacts

- [`experiment_summary.json`](03-Quantum-Kernel-SVM-MNIST/results/experiment_summary.json)
- [`metrics_comparison.csv`](03-Quantum-Kernel-SVM-MNIST/results/metrics_comparison.csv)
- [`kernel_heatmap.png`](03-Quantum-Kernel-SVM-MNIST/results/kernel_heatmap.png)

### Current Findings

- Project 03 unit tests passed: 4/4 tests.
- During this update, the kernel-target-alignment helper was repaired after a partial edit left it returning `None`.
- The committed summary artifact shows:
  - dataset source: `sklearn_digits_fallback`
  - digits: `4` vs `9`
  - PCA components in the stored artifact: `4`
  - classical accuracy: `1.0`
  - quantum accuracy: `0.4945054945`
  - kernel PSD check: `true`
  - kernel condition number: `47.2789284618`

### Interpretation

The codebase is in better shape than the stored artifacts suggest. The maintained pipeline exists and is runnable, but the committed results are still diagnostic artifacts from a fallback dataset and an earlier training configuration. They are useful as evidence that kernel analysis and plotting work, but they should not be presented as the final MNIST benchmark for the current script defaults.

## Project 04: Optimization with QAOA and RQAOA

### Scope

Project 04 implements graph generation, Max-Cut Hamiltonian construction, QAOA circuit generation, classical optimization, recursive QAOA, runtime utilities, transpilation helpers, and tests.

### Verified Artifacts

- [`integration_test.py`](04-Optimization-QAOA-MaxCut/integration_test.py)
- [`config/experiment_config.yaml`](04-Optimization-QAOA-MaxCut/config/experiment_config.yaml)
- notebook suite under [`notebooks/`](04-Optimization-QAOA-MaxCut/notebooks)

### Current Findings

- The integration path ran successfully on March 16, 2026.
- Verified result from `integration_test.py`:
  - graph: 6-node, 3-regular instance
  - exact optimal cut: `7`
  - QAOA cut value: `5`
  - approximation ratio: `0.7143`
- The pytest suite for classical solving, Hamiltonian construction, QAOA circuit logic, and RQAOA logic passed after adding a repository-root test path shim.

### Interpretation

Project 04 has a credible algorithmic core and a working local integration path. Its weakness is packaging and evidence retention: unlike Projects 02, 03, and 05, it does not yet leave behind a committed experiment-results bundle that a third party can inspect without rerunning code.

## Project 05: Reinforcement Learning and Noise Mitigation

### Scope

Project 05 implements a small quantum policy-gradient environment with ideal, noisy, and mitigated execution modes. It includes a parameterized quantum policy network, parameter-shift gradients, REINFORCE training, an execution abstraction, and a mitigation engine with TREX-style correction and ZNE.

### Verified Artifacts

- [`summary.json`](05-Reinforcement-Learning-Noise-Mitigation/results/summary.json)
- [`learning_curves.png`](05-Reinforcement-Learning-Noise-Mitigation/results/learning_curves.png)
- [`convergence_comparison.png`](05-Reinforcement-Learning-Noise-Mitigation/results/convergence_comparison.png)

### Current Findings

The committed summary reports:

- `ideal`
  - convergence episode: `5`
  - total runtime: `4.9553 s`
  - final success rate: `1.0`
- `noisy`
  - convergence episode: `5`
  - total runtime: `8.7254 s`
  - final success rate: `1.0`
- `mitigated`
  - convergence episode: `5`
  - total runtime: `56.7747 s`
  - final success rate: `1.0`

### Interpretation

On this toy environment, mitigation improves neither final success rate nor convergence episode relative to the easier modes, but it does impose a large runtime penalty. That does not invalidate the mitigation work; it shows that the task is simple enough that runtime overhead dominates the comparison. This is exactly the kind of caveat that should remain explicit in any presentation of the project.

## Cross-Project Conclusions

Several portfolio-level conclusions are already visible:

- Classical baselines remain stronger than current quantum methods on the repository's supervised-learning demos.
- Chemistry and optimization are the two most natural domains here for reference-driven benchmarking because exact classical comparators are available.
- Hardware realism matters: GHZ fidelity collapses in noisy simulation, and mitigation in RL comes with meaningful runtime cost.
- The repository is strongest when it is explicit about its role as a NISQ-era research portfolio rather than as evidence of broad quantum advantage.

## Risks and Gaps

- The root environment is convenient but not sufficient for exact reproduction across all modules.
- Project 03 needs a fresh archived result set from the current maintained pipeline.
- Project 04 needs a standardized experiment runner and committed outputs.
- Project 05 should eventually be tested on harder environments before stronger claims about mitigation utility are made.
- Some committed documentation files in the repository are still less precise than the actual code and artifact state.

## Recommended Next Steps

1. Refresh Project 03 results on OpenML MNIST using the current maintained runner and archive the new JSON/CSV outputs.
2. Add a Project 04 experiment script that writes structured results to `results/`.
3. Add CI smoke checks for Projects 02, 03, and 04.
4. Add per-project environment guidance or lock files to reduce version drift.
5. Keep root-level reporting aligned with committed artifacts rather than inferred or historical numbers.

## Metadata

**Report generated:** March 16, 2026  
**Author:** DEVADATH H K  
