# Project 02: Quantum Chemistry VQE

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-Electronic%20Structure-6929C4)
![VQE](https://img.shields.io/badge/Algorithm-VQE%20PES-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Research software for generating molecular potential energy surfaces with exact
diagonalization and Variational Quantum Eigensolver (VQE) workflows.

The module is designed for small electronic-structure benchmarks where the full
pipeline is inspectable: molecule construction, active-space reduction,
fermion-to-qubit mapping, ansatz selection, VQE optimization, exact reference
comparison, statistical aggregation, and visualization.

## README Audit

The previous README was a useful operational note, but it was not yet a
research-grade project document.

What was missing or weak:

- Limited research context: it described the workflow but did not explain why
  VQE/PES studies matter for near-term quantum chemistry.
- No mathematical foundation for the electronic Hamiltonian, qubit mapping,
  VQE objective, or chemical accuracy threshold.
- No computational complexity discussion, despite exact diagonalization and VQE
  scaling being central to interpretation.
- Weak explanation of ansatz choices, active spaces, warm-starting, and
  multi-restart design.
- Incomplete command coverage: the README mentioned PES and verification, but
  not the dispatcher, multi-seed runner, warm-start study, ablation script, or
  hardware bridge.
- Synthetic fallback was mentioned, but not strongly enough. The config warns
  that synthetic fallback is invalid for research, and some research scripts
  currently force it on; users must inspect `source_info` before trusting
  physical conclusions.
- No current-artifact summary, performance-metric definitions, limitations, or
  future-work roadmap.
- The structure was serviceable for a local run, but not strong enough for
  researchers, engineers, or recruiters reviewing the project.

This rewrite upgrades the README into a professional module-level research
document while avoiding claims that are not supported by the codebase.

## Project Overview

This module computes molecular potential energy surfaces (PES) by scanning bond
lengths and comparing:

1. **Exact diagonalization** of the mapped qubit Hamiltonian.
2. **VQE with chemistry-inspired and hardware-efficient ansatze**.
3. **Classical PySCF baselines** in the multi-seed runner where available
   (`HF` and `CISD`).

Configured molecules:

| Molecule | Geometry Model | Default Basis | Active-Space Handling |
|---|---|---|---|
| `H2` | Linear diatomic H-H scan | `sto-3g` | Full small problem |
| `LiH` | Linear Li-H scan | `sto-3g` | Freeze core plus 2-electron / 2-orbital active space |
| `BeH2` | Linear H-Be-H scan | `sto-3g` | Freeze core plus 4-electron / 3-orbital active space |

Primary research questions:

- How accurately do `UCCSD` and `EfficientSU2` reproduce exact PES curves?
- How often do ansatze reach chemical accuracy across bond lengths?
- Does warm-starting parameters across adjacent bond lengths reduce optimizer
  effort?
- How does ansatz architecture trade off energy error and circuit cost?

## Motivation / Research Context

Quantum chemistry is one of the most natural applications for quantum
computers because molecular electronic structure is itself a quantum many-body
problem. Near-term hardware, however, cannot run large fault-tolerant chemistry
algorithms. VQE is a hybrid alternative: a quantum circuit prepares trial states
and a classical optimizer updates circuit parameters to minimize molecular
energy.

This module uses small molecules because exact diagonalization is still
available. That exact reference is essential: without it, VQE energies are hard
to interpret. The module therefore acts as a controlled research platform for
studying VQE behavior, not as a production chemistry package.

## Why This README Structure Matters

- **Overview** tells readers what problem the module solves.
- **Research context** explains why the experiment is scientifically relevant.
- **Architecture** helps engineers locate responsibilities in the codebase.
- **Mathematical foundations** make the model and metrics auditable.
- **Installation and usage** make the project runnable by new users.
- **Results and metrics** help researchers interpret generated artifacts.
- **Limitations and future work** prevent overclaiming and guide maintenance.

## Key Features

- Config-driven PES scans for `H2`, `LiH`, and `BeH2`.
- PySCF/Qiskit Nature electronic-structure problem construction when PySCF is
  installed.
- Synthetic fallback Hamiltonians for smoke testing only.
- Fermion-to-qubit mappings: parity, Jordan-Wigner, and Bravyi-Kitaev.
- Parity mapping with two-qubit reduction when configured.
- Exact diagonalization via `NumPyMinimumEigensolver`.
- VQE ansatz factory for `UCCSD`, `EfficientSU2`, and `RYRZ` compatibility.
- Hartree-Fock initial states prepended to supported ansatze.
- Warm-starting across bond lengths by transferring optimal parameters.
- Multi-restart VQE retry path when the error remains above threshold.
- Local statevector estimator path and IBM Runtime estimator path.
- Multi-seed aggregation, chemical-accuracy rates, confidence intervals, and
  warm-start speedup analysis.
- Architecture ablation for `EfficientSU2` rotation sets, repetitions, and
  entanglement patterns.
- QASM export for selected optimized circuits.
- Unit tests for config validation, mapping, ansatz construction, exact solver,
  VQE wrapper, PES failure handling, and molecule driver paths.

## System Architecture

```text
02-Quantum-Chemistry-VQE
|
|-- config/simulation_config.yaml
|   `-- molecule grids, mapping, ansatz list, optimizer, runtime, metrics
|
|-- src/molecule_driver.py
|   `-- PySCF/Qiskit Nature molecule construction or synthetic fallback
|
|-- src/problem_builder.py
|   `-- second-quantized operator -> qubit SparsePauliOp
|
|-- src/classical_solver.py
|   `-- exact diagonalization, HF, CISD, FCI helpers
|
|-- src/ansatz_factory.py
|   `-- UCCSD / EfficientSU2 / RYRZ circuit construction
|
|-- src/vqe_engine.py
|   `-- VQE execution, callbacks, warm starts, multi-restart retry
|
|-- src/pes_generator.py
|   `-- bond-length loop, exact/VQE comparison, failure capture, artifacts
|
|-- src/statistical_analysis.py
|   `-- multi-seed aggregation and warm-start speedup metrics
|
`-- scripts/
    |-- run_verification.py
    |-- run_experiment.py
    |-- run_multi_seed.py
    |-- run_warm_start_study.py
    |-- run_ablation_study.py
    `-- run_hardware_experiment.py
```

The canonical batch path is script-driven. The notebook is useful for
inspection and visualization, but scripts are the reproducible source of truth.

## Mathematical Foundations

### Electronic Structure Hamiltonian

In second quantization, the molecular electronic Hamiltonian in a chosen basis
can be written as:

```math
H =
\sum_{pq} h_{pq} a_p^\dagger a_q
+ \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s
+ E_{\mathrm{nuc}} .
```

`PySCFDriver` and Qiskit Nature build this problem from molecular geometry,
charge, spin, and basis-set configuration. For larger molecules in this module,
freeze-core and active-space transformations reduce the problem size before
mapping.

### Fermion-to-Qubit Mapping

Quantum circuits operate on qubits, so fermionic operators must be transformed
into Pauli operators:

```math
H_q = \sum_k c_k P_k,
```

where each `P_k` is a tensor product of Pauli operators. The configured mapping
defaults to parity mapping:

```yaml
vqe:
  mapping: "parity"
```

The code also supports `jordan_wigner` and `bravyi_kitaev`.

### Exact Reference Energy

For small mapped Hamiltonians, the module computes an exact reference by
diagonalizing the qubit operator:

```math
H_q |\psi_i\rangle = E_i |\psi_i\rangle,
\qquad
E_0 = \min_i E_i.
```

The reported total energy adds Hamiltonian constants such as nuclear repulsion
and frozen-core contributions:

```math
E_{\mathrm{total}} = E_{\mathrm{electronic}} + \sum_j C_j.
```

### VQE Objective

VQE prepares a parameterized quantum state

```math
|\psi(\theta)\rangle = U(\theta)|0\rangle
```

and minimizes the energy expectation:

```math
E(\theta) =
\langle \psi(\theta) | H_q | \psi(\theta) \rangle
+ \sum_j C_j.
```

The optimizer solves:

```math
\theta^* = \arg\min_\theta E(\theta).
```

By the variational principle, ideal VQE energies are upper bounds to the exact
ground-state energy within the chosen ansatz family and basis.

### Chemical Accuracy

The default threshold is:

```math
|E_{\mathrm{VQE}} - E_{\mathrm{exact}}| \le 0.0016 \text{ Hartree}.
```

This is approximately `1.6 mHartree`, commonly used as a practical chemistry
accuracy target.

### Potential Energy Surface

For a bond length `R`, the PES point is:

```math
E_0(R) = \min_{\psi} \langle \psi | H(R) | \psi \rangle.
```

The module scans a grid of `R` values and plots:

- exact PES curve;
- VQE PES curve per ansatz;
- absolute VQE error;
- chemical-accuracy success rate;
- optimizer convergence traces.

### Computational Complexity

- Exact diagonalization scales exponentially with qubit count and is practical
  only for small active spaces.
- VQE measurement cost scales with the number of Pauli terms, shot count, and
  optimizer evaluations.
- Multi-seed PES cost scales roughly as:

```text
O(n_seeds * n_distances * n_ansatze * n_restarts * optimizer_evaluations)
```

- Warm-starting can reduce optimizer effort when adjacent bond-length optima are
  close in parameter space, but it is not guaranteed to help every ansatz or
  molecule.

### Design Decisions

- The default `sto-3g` basis keeps experiments small and reproducible.
- `H2` is kept as a full small reference problem; `LiH` and `BeH2` use active
  spaces to remain tractable.
- `UCCSD` provides a chemistry-inspired ansatz; `EfficientSU2` provides a
  hardware-efficient comparison.
- The PES loop records failures rather than aborting the whole scan.
- The callback history stores optimizer traces for convergence plots.
- Local mode uses `StatevectorEstimator`; IBM Runtime mode switches to SPSA for
  noise resilience.
- Synthetic fallback exists only to validate software plumbing when PySCF is
  unavailable.

## Technologies Used

- Python 3.10+
- NumPy and SciPy
- matplotlib
- PyYAML
- Pydantic
- PySCF
- Qiskit
- Qiskit Nature
- Qiskit Algorithms
- Qiskit Aer
- Qiskit IBM Runtime
- unittest / pytest
- Jupyter and ipywidgets for notebook exploration

## Repository Structure

```text
02-Quantum-Chemistry-VQE/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- setup.py
|-- environment.yml
|-- check_uccsd.py
|-- config/
|   `-- simulation_config.yaml
|-- notebooks/
|   `-- H2_Dissociation_Curve.ipynb
|-- scripts/
|   |-- run_ablation_study.py
|   |-- run_experiment.py
|   |-- run_hardware_experiment.py
|   |-- run_multi_seed.py
|   |-- run_verification.py
|   `-- run_warm_start_study.py
|-- src/
|   |-- adapt_vqe_engine.py
|   |-- advanced_plotting.py
|   |-- ansatz_factory.py
|   |-- backend_manager.py
|   |-- classical_solver.py
|   |-- config_schema.py
|   |-- data_processor.py
|   |-- experiment_tracker.py
|   |-- extensions.py
|   |-- hamiltonian_optimizer.py
|   |-- interfaces.py
|   |-- molecule_driver.py
|   |-- noise_model_builder.py
|   |-- optimizer_callbacks.py
|   |-- pes_generator.py
|   |-- plotting.py
|   |-- problem_builder.py
|   |-- runtime_executor.py
|   |-- statistical_analysis.py
|   `-- vqe_engine.py
|-- tests/
|   `-- test_unit.py
`-- results/
    |-- raw_data/
    `-- figures/
```

## Installation Guide

From the module directory:

```powershell
cd 02-Quantum-Chemistry-VQE
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

PySCF is required for physically meaningful molecular integrals. The
`requirements.txt` file installs PySCF for Python versions below 3.14:

```text
pyscf>=2.5; python_version < "3.14"
```

For IBM Runtime execution, configure credentials from the repository root:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 02-Quantum-Chemistry-VQE
```

## Usage Instructions

### 1. Run a Verification Smoke Test

```powershell
python scripts/run_verification.py
```

This runs a small H2 sweep, checks for finite energies, and compares the exact
diagonalization path against PySCF FCI when PySCF is available. If PySCF is not
installed, it warns that synthetic surrogate Hamiltonians are being used.

### 2. Run a Single PES Scan

```powershell
python -m src.pes_generator --config config/simulation_config.yaml --molecule H2
```

Other configured molecules:

```powershell
python -m src.pes_generator --config config/simulation_config.yaml --molecule LiH
python -m src.pes_generator --config config/simulation_config.yaml --molecule BeH2
```

### 3. Run the Research Dispatcher

Default multi-seed mode:

```powershell
python scripts/run_experiment.py --molecule H2 --seeds 10
```

Cold-start multi-seed run:

```powershell
python scripts/run_experiment.py --molecule H2 --seeds 10 --no-warm-start
```

Warm-start study:

```powershell
python scripts/run_experiment.py --molecule H2 --seeds 10 --mode warm_start_study
```

Architecture ablation:

```powershell
python scripts/run_experiment.py --molecule H2 --mode ablation
```

### 4. Run Individual Research Scripts

```powershell
python scripts/run_multi_seed.py --molecule H2 --seeds 10
python scripts/run_warm_start_study.py --molecule H2 --seeds 10
python scripts/run_ablation_study.py
python scripts/run_hardware_experiment.py
```

The hardware script is an experimental bridge. It selects an IBM backend,
applies additional Hamiltonian optimization/tapering logic, and uses a
sessionless Runtime estimator path. Treat it as a hardware exploration script,
not the default reproducible benchmark.

### 5. Run Tests

```powershell
python -m unittest tests.test_unit -v
```

If `pytest` is installed:

```powershell
python -m pytest tests -q
```

## Example Results / Visualizations

Current generated artifacts include:

- Raw PES JSON files:
  - `results/raw_data/pes_H2_*.json`
  - `results/raw_data/pes_H2_*_table.csv`
- Multi-seed statistics:
  - `results/raw_data/multiseed_stats_H2_warm.json`
  - `results/raw_data/multiseed_stats_H2.json`
  - `results/raw_data/multiseed_stats_LIH.json`
  - `results/raw_data/multiseed_stats_BEH2.json`
- Summary markdown:
  - `results/raw_data/multiseed_summary_H2_warm.md`
- Ablation artifact:
  - `results/raw_data/ablation_study_H2.json`
- Figures:
  - `results/figures/pes_curve_*.png`
  - `results/figures/error_*.png`
  - `results/figures/chem_acc_rates_*.png`
  - `results/figures/convergence_*.png`
  - `results/figures/warm_start_comparison_*.png`
  - `results/figures/pareto_front_H2_Ablation.png`
- QASM exports:
  - `results/figures/H2_UCCSD_*.qasm`
  - `results/figures/H2_EfficientSU2_*.qasm`

Example saved H2 warm-start multi-seed metadata:

| Field | Value |
|---|---:|
| Molecule | `H2` |
| Seeds | `10` |
| Warm start | `true` |
| Chemical accuracy threshold | `0.0016 Ha` |
| Ansatz set | `UCCSD`, `EfficientSU2` |
| Distance range | `0.5` to `2.5 Angstrom` |

The saved `multiseed_summary_H2_warm.md` reports chemical-accuracy success for
both ansatze across the H2 scan. Before using those numbers in a research claim,
inspect the corresponding JSON `source_info` and config to confirm that PySCF,
not synthetic fallback, generated the Hamiltonians.

## Experimental Setup

Default configuration file: [`config/simulation_config.yaml`](config/simulation_config.yaml)

| Component | Default |
|---|---|
| Random seed | `7` |
| Molecules | `H2`, `LiH`, `BeH2` |
| Basis | `sto-3g` |
| Mapping | `parity` |
| Two-qubit reduction | Used with parity mapping |
| Warm start | Enabled |
| Ansatz set | `UCCSD`, `EfficientSU2` |
| EfficientSU2 reps | `3` |
| EfficientSU2 entanglement | `circular` |
| Optimizer | `SLSQP`, `maxiter=200`, `tol=1e-6` |
| Local backend | `StatevectorEstimator` |
| IBM Runtime shots | `4096` |
| Chemical accuracy | `1.6 mHa` |
| Multi-seed default | `10` seeds in config and dispatcher |

Configured bond grids:

| Molecule | Start | End | Step |
|---|---:|---:|---:|
| `H2` | `0.5` | `2.5` | `0.1` |
| `LiH` | `1.0` | `3.5` | `0.1` |
| `BeH2` | `1.0` | `2.5` | `0.1` |

## Performance Metrics

- `exact_energies`: exact diagonalization reference energy at each bond length.
- `vqe_energies`: optimized VQE total energy per ansatz and bond length.
- `absolute_error`: `|E_vqe - E_exact|`.
- `chemical_accuracy_rate`: fraction of seeds within `0.0016 Ha`.
- `iterations` / `evaluations`: optimizer effort recorded from VQE callback or
  optimizer result metadata.
- `ci95_low` / `ci95_high`: bootstrap or aggregation confidence interval fields
  in multi-seed summaries.
- `mapping_stats`: full and reduced qubit counts, mapper, reduction usage, and
  molecule provenance.
- `source_info`: per-distance source label such as `pyscf` or synthetic
  fallback.
- `failures`: structured records for bond-length or ansatz failures that did
  not abort the full PES scan.
- `cost` in ablation studies: CNOT-count proxy from decomposed ansatz circuits.

## Limitations

- Synthetic fallback Hamiltonians are invalid for research or publication. They
  exist only to keep smoke tests runnable without PySCF.
- The default config currently sets `allow_synthetic_fallback: true`; change it
  to `false` for research runs and confirm that `source_info` reports `pyscf`.
- Some research scripts currently override `allow_synthetic_fallback` to `true`.
  Review and patch this before treating multi-seed outputs as physical chemistry
  evidence.
- The experiments use a minimal `sto-3g` basis and small active spaces.
- Exact diagonalization is only practical because the mapped problems are small.
- VQE success on H2 does not imply scalability to chemically relevant molecules.
- Hardware execution is experimental, backend-dependent, and not the default
  reproducible path.
- The ablation script uses CNOT count as a proxy cost without backend-specific
  transpilation.
- Some generated markdown summaries degrade when `tabulate` is not installed.
- The project stores many generated artifacts in the repository, which helps
  demonstration but can make source/result boundaries less clean.

## Future Improvements

- Set `allow_synthetic_fallback: false` by default and add a separate
  smoke-test config for synthetic mode.
- Ensure every research runner respects the publication-safe fallback setting.
- Add a top-level result-verification script that fails if any research artifact
  was generated from synthetic Hamiltonians.
- Add backend-transpiled circuit metrics for ansatz cost comparisons.
- Add more classical chemistry references, such as MP2 and FCI summaries in
  exported result tables where feasible.
- Add parameter-shift or analytic-gradient experiments for selected ansatze.
- Add a hardware-feasibility table with depth, two-qubit gates, shots, and
  estimated runtime.
- Add richer statistical reports with paired tests for warm-start versus
  cold-start conditions.
- Move generated artifacts into versioned run directories with manifest files.

## References

- Peruzzo et al. A variational eigenvalue solver on a photonic quantum
  processor. *Nature Communications*, 2014.
- McClean et al. The theory of variational hybrid quantum-classical algorithms.
  *New Journal of Physics*, 2016.
- Kandala et al. Hardware-efficient variational quantum eigensolver for small
  molecules and quantum magnets. *Nature*, 2017.
- Bartlett and Musial. Coupled-cluster theory in quantum chemistry. *Reviews of
  Modern Physics*, 2007.
- Seeley, Richard, and Love. The Bravyi-Kitaev transformation for quantum
  computation of electronic structure. *Journal of Chemical Physics*, 2012.
- Qiskit Nature documentation: https://qiskit-community.github.io/qiskit-nature/
- PySCF documentation: https://pyscf.org/
- IBM Quantum documentation: https://docs.quantum.ibm.com/

## Author Information

**DEVADATH H K**

Project 02 of the
[`Quantum AI Research Series`](../README.md). See the repository-level
[`LICENSE`](../LICENSE), [`CITATION.cff`](../CITATION.cff), and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) files for licensing, citation, and
contribution guidance.
