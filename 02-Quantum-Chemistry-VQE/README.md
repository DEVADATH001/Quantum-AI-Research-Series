# Project 02: Quantum Chemistry VQE

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-electronic%20structure-6929C4)
![VQE](https://img.shields.io/badge/algorithm-VQE%20PES-orange)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to study how well VQE reproduces molecular potential energy surfaces — with exact diagonalization as the reference, not just a "lower is better" hand-wave.

The module sweeps bond lengths for H₂, LiH, and BeH₂, computes exact ground-state energies by diagonalizing the qubit Hamiltonian, runs VQE with multiple ansatz choices, and compares. You can also run warm-start studies (does transferring parameters between adjacent bond lengths help convergence?) and architecture ablations (circuit depth, entanglement, rotation gates).

The short version for H₂: UCCSD nails it. EfficientSU2 is cheaper but less reliable. LiH and BeH₂ depend on active-space config and which part of the curve you're looking at.

---

## Why VQE + potential energy surfaces

Molecular energy is inherently a quantum problem, and VQE is the standard near-term approach: a parameterized quantum circuit prepares trial states, a classical optimizer pushes the parameters toward minimum energy. But VQE energies are nearly uninterpretable without a reference.

That's why exact diagonalization is baked into every run. For small molecules with small active spaces, it's still feasible, and it turns VQE from "we got a number" into "we're X mHartree from the ground truth."

Chemical accuracy threshold: **1.6 mHartree** — I track this explicitly.

$$|E_{\text{VQE}} - E_{\text{exact}}| \leq 0.0016 \text{ Hartree}$$

---

## What this computes

**Electronic Hamiltonian** from PySCF molecular integrals:

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s + E_{\text{nuc}}$$

PySCF + Qiskit Nature builds this from geometry, charge, spin, and basis set. Fermion-to-qubit mapping converts it to Pauli operators:

$$H_q = \sum_k c_k P_k$$

Default mapping: parity with two-qubit reduction. Also supports Jordan-Wigner and Bravyi-Kitaev.

**VQE objective** — by the variational principle, this is always an upper bound:

$$E(\theta) = \langle \psi(\theta) | H_q | \psi(\theta) \rangle$$

$$\theta^* = \arg\min_\theta E(\theta)$$

For each bond length $R$, I compute $E_0(R)$ exactly and via VQE, then plot the PES curve and per-point error.

---

## Key features

- **Config-driven PES scans** for H₂, LiH, BeH₂
- **PySCF integration** for real Hamiltonian construction (required for meaningful results)
- **Synthetic fallback** — smoke testing only, not research. Don't trust results with `source_info: synthetic`.
- **Multiple mappings:** parity (default), Jordan-Wigner, Bravyi-Kitaev
- **Two ansatze:** UCCSD (physics-informed, more parameters) and EfficientSU2 (hardware-efficient, fewer CNOTs)
- **Warm-starting:** transfer optimal θ from one bond length to the next
- **Multi-seed aggregation** with chemical-accuracy rates and confidence intervals
- **Architecture ablation:** rotation sets, reps, entanglement patterns, CNOT-count proxy
- **IBM Runtime path** for hardware experiments (experimental, not default)
- **Unit tests** for configs, mappings, ansatze, solvers, and failure handling

---

## How to run

### 1. Start here — smoke verification

```powershell
python scripts/run_verification.py
```

Runs a quick H₂ sweep. **Check `source_info` in the output.** If it says `synthetic`, the Hamiltonians are placeholders — not real chemistry. If PySCF is installed, it also validates against FCI.

### 2. PES scans

```powershell
python -m src.pes_generator --config config/simulation_config.yaml --molecule H2
python -m src.pes_generator --config config/simulation_config.yaml --molecule LiH
python -m src.pes_generator --config config/simulation_config.yaml --molecule BeH2
```

### 3. Research runs (multi-seed)

```powershell
# Warm-start, 10 seeds
python scripts/run_experiment.py --molecule H2 --seeds 10

# Cold start for comparison
python scripts/run_experiment.py --molecule H2 --seeds 10 --no-warm-start

# Warm-start study mode
python scripts/run_experiment.py --molecule H2 --seeds 10 --mode warm_start_study

# Architecture ablation
python scripts/run_experiment.py --molecule H2 --mode ablation
```

### 4. Individual scripts

```powershell
python scripts/run_multi_seed.py --molecule H2 --seeds 10
python scripts/run_warm_start_study.py --molecule H2 --seeds 10
python scripts/run_ablation_study.py
python scripts/run_hardware_experiment.py   # experimental
```

### 5. Tests

```powershell
python -m pytest tests -q
```

---

## Installation

```powershell
cd 02-Quantum-Chemistry-VQE
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

**PySCF is required for real chemistry.** Without it, the module falls back to synthetic Hamiltonians that are only useful for testing the pipeline.

```
pyscf>=2.5; python_version < "3.14"
```

For IBM Runtime:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 02-Quantum-Chemistry-VQE
```

---

## Default configuration

From `config/simulation_config.yaml`:

| Setting | Value |
|---|---|
| Molecules | H₂, LiH, BeH₂ |
| Basis set | `sto-3g` |
| Mapping | Parity, two-qubit reduction |
| Ansatze | UCCSD, EfficientSU2 |
| Optimizer | SLSQP, `maxiter=200`, `tol=1e-6` |
| Seeds | 10 |
| Chemical accuracy | 1.6 mHartree |
| Warm start | Enabled |

Bond-length grids:

| Molecule | Range (Å) | Step | Points |
|---|---|---|---:|
| H₂ | 0.5 – 2.5 | 0.1 | 21 |
| LiH | 1.0 – 3.5 | 0.1 | 26 |
| BeH₂ | 1.0 – 2.5 | 0.1 | 16 |

**Computational cost** scales as:
```
O(n_seeds × n_distances × n_ansatze × n_restarts × optimizer_evaluations)
```

---

## Saved artifacts

The `results/` directory contains PES JSON, multi-seed statistics, ablation data, convergence traces, warm-start comparisons, and QASM circuit exports. Key files:

- `results/raw_data/multiseed_stats_H2_warm.json` — warm vs cold-start comparison
- `results/raw_data/ablation_study_H2.json` — ansatz depth/entanglement trade-offs
- `results/figures/pes_curve_*.png` — visual PES curves
- `results/figures/error_*.png` — VQE error vs exact reference
- `results/figures/warm_start_comparison_*.png` — warm-start convergence plots

**Before citing any result:** inspect `source_info` in the JSON. If any distance point used synthetic Hamiltonians, those numbers don't represent real chemistry.

### Checked-in results from `multiseed_stats_H2_warm.json`

10 seeds, warm-start enabled, 21 bond lengths (0.5–2.5 Å, 0.1 Å step), PySCF + parity mapping:

| Ansatz | Chemical-accuracy rate | Worst-case mean error | Worst-case std | Distance range |
|---|---:|---:|---:|---|
| **UCCSD** | **100%** (all 21 distances, all 10 seeds) | 0.887 µHa (at 2.5 Å) | 0.262 µHa | 0.5–2.5 Å |
| **EfficientSU2** | **100%** (all 21 distances, all 10 seeds) | 1.395 µHa (at 2.3 Å) | 0.442 µHa | 0.5–2.5 Å |

Both ansatze achieve **100% chemical-accuracy rate** — every seed at every bond length produces an error below the 1.6 mHartree threshold. The worst-case errors are still well under the threshold (UCCSD: ~0.9 µHa, EfficientSU2: ~1.4 µHa), indicating robust convergence.

**Error trend:** errors increase at stretched bond lengths (R > 2.0 Å), which is expected — the wavefunction becomes more multi-reference as the bond stretches, making classical optimizer convergence harder. Warm-starting helps by transferring parameters from the adjacent (easier) geometry.

**Confidence intervals** (95%) are extremely tight — on the order of 10⁻⁷ Hartree — confirming that the 10-seed average is statistically stable.

### What each metric means

| Metric | Description |
|---|---|
| `exact_energies` | Ground-state energy from exact diagonalization |
| `vqe_energies` | VQE total energy per ansatz |
| `absolute_error` | `|E_vqe - E_exact|` |
| `chemical_accuracy_rate` | Fraction of seeds within 0.0016 Ha |
| `iterations` / `evaluations` | Optimizer effort |
| `source_info` | `pyscf` or `synthetic` — this tells you whether to trust the result |
| `failures` | Per-point failure records (partial failures don't kill the scan) |

---

## Limitations

- **Synthetic fallback is not research data.** The config defaults to `allow_synthetic_fallback: true` — set it to `false` for serious runs.
- Some scripts override the fallback setting. Check before treating outputs as physical results.
- `sto-3g` minimal basis and small active spaces keep things tractable, but they don't represent chemically relevant accuracy for real applications.
- H₂ is the simplest molecule in quantum chemistry. VQE success on H₂ doesn't tell you much about scalability.
- Hardware execution is experimental and backend-specific. It's not the default path.
- CNOT count in ablation is a proxy — not backend-transpiled gate cost.

---

## What I'd fix next

- Default `allow_synthetic_fallback: false`, with a separate smoke config for synthetic mode
- A verification script that fails if any research artifact came from synthetic Hamiltonians
- Backend-transpiled circuit metrics for honest ansatz cost comparison
- Parameter-shift gradient experiments
- More classical references (MP2, FCI) in exported tables
- Versioned run directories with manifests instead of flat dumps

---

## References

- Peruzzo et al. A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, 2014.
- McClean et al. The theory of variational hybrid quantum-classical algorithms. *New Journal of Physics*, 2016.
- Kandala et al. Hardware-efficient VQE for small molecules and quantum magnets. *Nature*, 2017.
- Bartlett and Musial. Coupled-cluster theory in quantum chemistry. *Reviews of Modern Physics*, 2007.
- Seeley, Richard, and Love. The Bravyi-Kitaev transformation for quantum computation. *J. Chem. Phys.*, 2012.
- Qiskit Nature: https://qiskit-community.github.io/qiskit-nature/
- PySCF: https://pyscf.org/
- IBM Quantum: https://docs.quantum.ibm.com/

---

## Author

**DEVADATH H K** — Part of the [Quantum AI Research Series](../README.md).

See [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), and [CONTRIBUTING.md](../CONTRIBUTING.md).
