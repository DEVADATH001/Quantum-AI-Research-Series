# Quantum AI Research Series

Five-project research portfolio that moves from classical-vs-quantum ML
comparisons to chemistry, optimization, and quantum RL under noise.

## What This Repository Is

This is not a single monolithic app. It is a structured series of focused
experiments:

1. [`01-Classical-vs-Quantum-Visualization`](01-Classical-vs-Quantum-Visualization/README.md)
2. [`02-Quantum-Chemistry-VQE`](02-Quantum-Chemistry-VQE/README.md)
3. [`03-Quantum-Kernel-SVM-MNIST`](03-Quantum-Kernel-SVM-MNIST/README.md)
4. [`04-Optimization-QAOA-MaxCut`](04-Optimization-QAOA-MaxCut/README.md)
5. [`05-Reinforcement-Learning-Noise-Mitigation`](05-Reinforcement-Learning-Noise-Mitigation/README.md)

## Project Map

### 01: Classical vs Quantum Visualization

- Scope: Iris QML classification and a separate GHZ-127 noise benchmark.
- Key reality: GHZ-127 is a hardware stress test, not a QML accuracy booster.
- Maintained entrypoints: Python scripts + split notebooks.

### 02: Quantum Chemistry VQE

- Scope: PES benchmarking (exact diagonalization vs VQE) for H2/LiH workflows.
- Structure: modular `src/` package with config validation and test coverage.
- Includes runtime/fallback handling for PySCF availability.

### 03: Quantum Kernel SVM (MNIST)

- Scope: classical SVM vs quantum-kernel classifier on reduced image features.
- Format: notebook-driven experiment.

### 04: QAOA Max-Cut

- Scope: formulate Max-Cut as QUBO/Ising and solve approximately with QAOA.
- Format: notebook-driven optimization experiment.

### 05: Quantum RL Noise Mitigation

- Scope: toy quantum policy-gradient loop under noiseless/noisy/mitigated settings.
- Format: notebook-driven RL experiment.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then either:

- run project-specific scripts from each module README, or
- open notebooks with `jupyter notebook`.

## IBM Runtime Setup

Use environment variable + setup script (do not hardcode tokens):

```powershell
setx IBM_QUANTUM_TOKEN "YOUR_NEW_TOKEN"
python scripts/setup_ibm_runtime.py
```

Reference files:

- [`scripts/setup_ibm_runtime.py`](scripts/setup_ibm_runtime.py)
- [`.env.example`](.env.example)

## Suggested Execution Order

1. Project 01 to validate environment and Qiskit runtime path.
2. Project 02 for chemistry workflow and module architecture.
3. Projects 03-05 as focused notebook studies (kernel ML, QAOA, RL+noise).

## Reproducibility Notes

- Each module has its own README with module-specific commands and outputs.
- Prefer running from the module directory so relative paths resolve cleanly.
- GHZ real-hardware runs can queue; Project 01 includes queue-aware auto-skip.

## Repository Conventions

- Keep credentials in environment variables only.
- Use small, focused commits (`feat`, `docs`, `chore`).
- Keep generated result artifacts inside each project folder (for example,
  `01-Classical-vs-Quantum-Visualization/assets`).

---

Author: DEVADATH H K
