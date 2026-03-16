# 01-Classical-vs-Quantum-Visualization

## What This Module Does

This module contains two separate experiments:

1. Iris classification with classical baselines and QSVC.
2. GHZ-127 hardware/noise benchmark (local ideal vs noisy simulation vs optional real IBM run).

These are intentionally independent. GHZ benchmark outcomes do not improve Iris
classification accuracy, and Iris metrics do not validate large-scale GHZ
hardware fidelity.

## Folder Layout

```text
01-Classical-vs-Quantum-Visualization/
|-- README.md
|-- Quantum_ML_-_Iris_Classification.py
|-- compare_ghz_three_way.py
|-- Hardware_Noise_&_Decoherence_Benchmark.py
|-- iris_qml_classification.ipynb
|-- ghz_127_noise_benchmark.ipynb
|-- iris_quantum_bridge.ipynb
|-- requirements.txt
`-- assets/
    |-- classical_vs_quantum_boundaries.png
    |-- qml_iris_report.json
    |-- three_way_ghz127_comparison.json
    `-- three_way_ghz127_comparison.png
```

## Environment Setup

```powershell
cd 01-Classical-vs-Quantum-Visualization
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Iris Classification

```powershell
python Quantum_ML_-_Iris_Classification.py --no-show
```

Useful CLI options:

- `--random-state 7`
- `--test-size 0.25`
- `--classical-grid 140`
- `--quantum-grid 30`
- `--quantum-max-kernel-evals 120000`
- `--output-plot assets/classical_vs_quantum_boundaries.png`
- `--output-report assets/qml_iris_report.json`

Runtime behavior:

- If `--quantum-grid` is omitted, the script automatically chooses a safe grid
  resolution under the kernel-evaluation budget.
- Output paths are resolved relative to this module directory.

## Run GHZ-127 Benchmark

Default command:

```powershell
python compare_ghz_three_way.py --local-shots 1024 --sim-shots 512 --real-shots 256
```

Real execution policy:

1. If `--skip-real` is passed, real IBM execution is skipped.
2. Otherwise, if backend queue is above `--max-pending-jobs` (default `0`),
   real execution is skipped.
3. Otherwise, script attempts real execution with
   `--real-timeout-seconds` (default `900`).
4. If real execution fails or times out, script skips real unless
   `--strict-real` is enabled.

Examples:

```powershell
# Explicitly skip real run
python compare_ghz_three_way.py --skip-real

# Auto-skip when queue is above 2 jobs
python compare_ghz_three_way.py --max-pending-jobs 2

# Fail hard on real-run failures
python compare_ghz_three_way.py --strict-real --real-timeout-seconds 900
```

Compatibility entrypoint:

```powershell
python "Hardware_Noise_&_Decoherence_Benchmark.py" --local-shots 1024 --sim-shots 512 --real-shots 256
```

## Notebook Workflows

Execute split notebooks:

```powershell
python -m nbconvert --to notebook --execute --inplace iris_qml_classification.ipynb
python -m nbconvert --to notebook --execute --inplace ghz_127_noise_benchmark.ipynb
```

Notebook notes:

- `iris_qml_classification.ipynb` covers only Iris/QSVC workflow.
- `ghz_127_noise_benchmark.ipynb` covers only GHZ benchmark workflow.
- `ghz_127_noise_benchmark.ipynb` has `RUN_BENCHMARK = False` by default and
  reads saved benchmark artifacts.
- `iris_quantum_bridge.ipynb` is a legacy mixed notebook kept for reference.

## Generated Outputs

Iris:

- `assets/classical_vs_quantum_boundaries.png`
- `assets/qml_iris_report.json`

GHZ:

- `assets/three_way_ghz127_comparison.json`
- `assets/three_way_ghz127_comparison.png`

## Current Baseline Snapshot

From `assets/qml_iris_report.json`:

- Logistic Regression test accuracy: `0.973684`
- Classical RBF-SVM test accuracy: `0.973684`
- Quantum SVC (ZZ-Map) test accuracy: `0.631579`
- Quantum grid: `30`

From `assets/three_way_ghz127_comparison.json`:

- `generated_utc`: `2026-03-08T23:49:54.568282+00:00`
- ISA circuit depth: `382`
- `p_ghz_subspace` local/simulated/real: `1.0 / 0.0 / 0.0`

Interpretation:

- Iris is used as a controlled QML workflow demo, not quantum-advantage proof.
- GHZ-127 result demonstrates NISQ noise/decoherence limitations at this depth.

Author: DEVADATH H K
