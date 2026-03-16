# 02-Quantum-Chemistry-VQE

## What This Module Does

This module runs Potential Energy Surface (PES) scans for molecules and compares:

1. Exact diagonalization baseline.
2. VQE with configurable ansatz set (for example `UCCSD`, `EfficientSU2`).

Primary configured workflows are for `H2` and `LiH`.

## Scientific Goal

For a molecule and bond-length grid:

1. Build electronic structure problem.
2. Map to qubit Hamiltonian.
3. Compute exact reference energy.
4. Run VQE per ansatz.
5. Compare `|E_vqe - E_exact|` against chemical-accuracy threshold.

Default chemical accuracy threshold in config is `1.6 mHartree = 0.0016 Hartree`.

## Folder Layout

```text
02-Quantum-Chemistry-VQE/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- setup.py
|-- config/
|   `-- simulation_config.yaml
|-- src/
|   |-- pes_generator.py
|   |-- molecule_driver.py
|   |-- problem_builder.py
|   |-- ansatz_factory.py
|   |-- vqe_engine.py
|   |-- runtime_executor.py
|   |-- classical_solver.py
|   |-- plotting.py
|   |-- data_processor.py
|   |-- config_schema.py
|   |-- optimizer_callbacks.py
|   |-- interfaces.py
|   `-- extensions.py
|-- scripts/
|   `-- run_verification.py
|-- notebooks/
|   `-- H2_Dissociation_Curve.ipynb
|-- results/
|   |-- raw_data/
|   `-- figures/
`-- tests/
    `-- test_unit.py
```

## Configuration Model

Config file: `config/simulation_config.yaml`

Top-level sections:

- `general`: random seed and synthetic-fallback policy.
- `molecules`: per-molecule distance sweep and electronic structure settings.
- `vqe`: ansatz list and optimizer settings.
- `runtime`: backend mode and runtime options.
- `analysis`: error thresholds (chemical accuracy).

Validation is enforced by Pydantic schema in `src/config_schema.py`.

## Runtime Modes

Local mode:

- Set `runtime.backend: "local"`.
- Uses `StatevectorEstimator` (no sampling noise).

IBM Runtime mode:

- Set `runtime.backend` to an IBM backend name (for example `ibm_fez`).
- Uses `EstimatorV2` with configured `resilience_level`, `optimization_level`, and `shots`.
- Requires valid IBM Runtime credentials in environment/account setup.

Synthetic fallback:

- Controlled by `general.allow_synthetic_fallback`.
- If PySCF/integral generation is unavailable, synthetic surrogate problems can be used.
- Output JSON includes `source_info` so synthetic vs PySCF provenance is explicit.

## Installation

```powershell
cd 02-Quantum-Chemistry-VQE
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional dev extras:

```powershell
python -m pip install .[dev]
```

## Run Commands

Run full PES for H2:

```powershell
python -m src.pes_generator --config config/simulation_config.yaml --molecule H2
```

Run full PES for LiH:

```powershell
python -m src.pes_generator --config config/simulation_config.yaml --molecule LiH
```

Quick local verification sweep:

```powershell
python scripts/run_verification.py
```

## Testing

Unit tests:

```powershell
python -m unittest tests.test_unit -v
```

If `pytest` is installed:

```powershell
pytest -q
```

## Outputs Produced

Raw data:

- `results/raw_data/pes_<molecule>.json`
- `results/raw_data/pes_<molecule>_table.csv`

Figures:

- `results/figures/pes_curve_<molecule>.png`
- `results/figures/error_<molecule>.png`
- `results/figures/convergence_<molecule>_<ansatz>_<bond>A.png`

## Reliability and Failure Handling

- Config is validated before execution.
- PES loop is fault-tolerant:
  - per-distance failures are captured,
  - per-ansatz failures are captured,
  - scan continues and logs details under `results["failures"]`.
- Mapping metadata and qubit counts are retained in `mapping_stats`.

## Notebook

`notebooks/H2_Dissociation_Curve.ipynb` is the interactive analysis notebook.

Use the script pipeline for reproducible batch output; use notebook for
inspection and visualization.

## Common Pitfalls

- Run from module directory (`02-Quantum-Chemistry-VQE`) to keep relative paths valid.
- Use module execution (`python -m src.pes_generator`), not
  `python src/pes_generator.py`.
- IBM runtime mode will fail without valid credentials/backend access.

Author: DEVADATH H K
