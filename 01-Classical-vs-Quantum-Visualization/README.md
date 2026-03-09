# 01-Classical-vs-Quantum-Visualization

## Research Abstract

This project builds a practical bridge between classical and quantum representations of the Iris dataset.
It compares classical decision boundaries (Logistic Regression and RBF-SVM) against quantum state encoding intuition using `ZZFeatureMap`, Bell-state entanglement, and a 127-qubit GHZ construction.

The central question is not whether Iris needs quantum computing. It is how feature mapping changes the geometry of data representations as dimensionality grows.

## Project Structure

```text
01-Classical-vs-Quantum-Visualization/
|-- README.md
|-- compare_ghz_three_way.py
|-- requirements.txt
|-- iris_quantum_bridge.ipynb
`-- assets/
    |-- classical_boundaries.png
    |-- ghz_127_circuit.png
    |-- quantum_feature_map.png
    |-- three_way_ghz127_comparison.json
    `-- three_way_ghz127_comparison.png
```

## Technical Scope

- Dataset: Scikit-Learn Iris (3 classes, 4 features)
- Classical baselines:
  - Logistic Regression
  - SVM (RBF kernel)
- Classical visualization:
  - PCA projection to 2D
  - Side-by-side decision boundaries
- Quantum visualization:
  - Bell state circuit
  - 127-qubit GHZ state (sparse visual view + full circuit construction)
  - `ZZFeatureMap(feature_dimension=4)` for Iris encoding
- Runtime stack:
  - Qiskit 1.x
  - Qiskit Runtime Primitives V2 (`SamplerV2`)
  - Aer local backend for reproducible SamplerV2 execution
  - ISA-aware transpilation via
    `generate_preset_pass_manager(optimization_level=3, backend=backend)`

## Why Quantum For Iris?

Iris is classically solvable and is not a quantum advantage benchmark.

This notebook uses Iris as a controlled, explainable substrate to demonstrate Hilbert space mapping and the conceptual transition needed for high-dimensional datasets where fixed classical kernels can struggle under the curse of dimensionality.

## Running The Notebook

```powershell
cd 01-Classical-vs-Quantum-Visualization
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook iris_quantum_bridge.ipynb
```

## 3-Way GHZ Comparison (Local vs Simulated vs IBM Quantum)

Run this from the project folder:

```powershell
python compare_ghz_three_way.py --local-shots 1024 --sim-shots 512 --real-shots 256
```

Outputs:

- `assets/three_way_ghz127_comparison.json`
- `assets/three_way_ghz127_comparison.png`

## IBM Runtime Note

The notebook transpiles to a 127-qubit backend target (ISA workflow) and executes SamplerV2 locally on Aer for reproducibility.
If you configure `IBM_QUANTUM_TOKEN`, you can switch to a live IBM backend directly inside the notebook.
If your account requires an explicit instance, also set `IBM_QUANTUM_INSTANCE`.
