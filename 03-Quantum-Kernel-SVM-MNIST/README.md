# Quantum Kernel SVM for MNIST Classification

<p align="center">
  <img src="https://img.shields.io/badge/Quantum-Kernel-SVM-blue" alt="Quantum Kernel SVM">
  <img src="https://img.shields.io/badge/Qiskit-2.x-purple" alt="Qiskit 2.x">
  <img src="https://img.shields.io/badge/MNIST-4--vs--9-green" alt="MNIST 4 vs 9">
</p>

## Overview

This research project demonstrates how classical data can be mapped into **high-dimensional Hilbert space** using quantum feature maps and classified using **quantum kernel methods**. The project compares classical RBF kernel SVM against Quantum Kernel SVM with Pegasos optimization on MNIST digit classification (4 vs 9).

## Research Objective

Classify handwritten digits **4 vs 9** from the MNIST dataset using:

1. **Classical RBF SVM** - Standard sklearn SVM with RBF kernel
2. **Quantum Kernel SVM** - Using ZZFeatureMap and FidelityQuantumKernel with PegasosQSVC

Compare the models using:
- Accuracy
- Precision  
- Recall
- F1 Score

Analyze the **effect of quantum noise on kernel matrices**.

---

## Project Motivation

### Why Quantum Kernels?

Classical kernel methods rely on explicitly computing (or approximating) kernel functions in lower-dimensional spaces. Quantum kernels offer a fundamentally different approach:

1. **Exponential Hilbert Space**: For n qubits, the Hilbert space has 2^n dimensions - potentially capturing complex feature interactions
2. **Native Entanglement**: Quantum circuits can naturally represent entangled feature correlations
3. **Quantum Advantage Potential**: For certain data distributions, quantum feature spaces may provide better separation

### Hilbert Space Mapping

The ZZFeatureMap encodes classical vectors |x⟩ into quantum states |φ(x)⟩ through parameterized rotations:

```
|φ(x)⟩ = U(x)U^Φ(x)|0⟩^⊗n
```

Where:
- U(x) = ⊗ᵢ Rₓ(xᵢ) applies rotation gates
- U^Φ(x) applies entangling phases based on feature interactions

### Kernel Overlap Formula

The quantum kernel measures similarity through quantum state overlap:

```
K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
```

This is the quantum analog of classical kernel functions, measuring the "quantum fidelity" between encoded data points.

---

## Repository Structure

```
03-Quantum-Kernel-SVM-MNIST/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   └── experiment_config.yaml        # Experiment configuration
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_loader.py                # MNIST data loading (4 vs 9)
│   ├── preprocessing.py             # Normalization, Standardization, PCA
│   ├── classical_models.py            # Classical RBF SVM with GridSearchCV
│   ├── quantum_feature_maps.py       # ZZFeatureMap implementation
│   ├── quantum_kernel_engine.py      # FidelityQuantumKernel computation
│   ├── quantum_training.py           # PegasosQSVC implementation
│   ├── evaluation_metrics.py         # Model evaluation metrics
│   ├── visualization.py              # Plotting utilities
│   └── noise_simulation.py           # IBM noise model simulation
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data loading and preprocessing
│   ├── 02_classical_baseline.ipynb   # Classical SVM baseline
│   └── 03_quantum_kernel_svm.ipynb    # Quantum kernel SVM experiments
│
├── results/
│   ├── kernel_heatmap.png            # Quantum kernel visualization
│   ├── metrics_comparison.csv        # Model comparison metrics
│   └── confusion_matrix.png          # Confusion matrices
│
└── utils/
    ├── __init__.py
    └── q_utils.py                   # Utility functions
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- Qiskit >= 2.0
- Qiskit Machine Learning >= 0.9

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install from the root project:

```bash
cd Quantum-AI-Research-Series
pip install -r requirements.txt
```

---

## Running Experiments

### Quick Start

Run the reproducible CLI pipeline:

```bash
python run_experiment.py
```

Fast smoke run:

```bash
python run_experiment.py --max-quantum-train 40 --quantum-steps 120
```

Notebook files are kept for exploration, but `run_experiment.py` is the maintained execution path.

### Configuration

Edit `config/experiment_config.yaml` to customize:

```yaml
dataset:
  digits: [4, 9]      # Binary classification
  test_size: 0.25

preprocessing:
  pca:
    n_components: 4    # PCA components (qubits)

classical_model:
  kernel: "rbf"
  grid_search:
    param_grid:
      C: [0.1, 1, 10, 100]
      gamma: ["scale", "auto", 0.1, 0.01]

quantum_feature_map:
  type: "ZZFeatureMap"
  reps: 3
  entanglement: "full"

pegasos_svc:
  lambda_param: 1.0
  max_iter: 1000

noise_simulation:
  enabled: true
  backend: "ibm_brisbane"
```

---

## Data Preprocessing

### MNIST Dataset

We use MNIST digits **4 vs 9** because they have similar shapes, making the classification more challenging.

### Preprocessing Pipeline

1. **Normalization**: Scale pixel values to [0, 1]
2. **Standardization**: Zero mean, unit variance
3. **PCA**: Reduce to 8 principal components

> **Note**: PCA was used to reduce the feature set to 8 dimensions (capturing a more realistic chunk of the variance while remaining simulatable) to accommodate the qubit constraints of NISQ-era hardware.

---

## Quantum Feature Map

### ZZFeatureMap

The ZZFeatureMap encodes classical data into quantum states:

```python
from qiskit.circuit.library import ZZFeatureMap

feature_map = ZZFeatureMap(
    feature_dimension=8,
    reps=2,
    entanglement="linear"
)
```

### Configuration

- **feature_dimension**: Number of qubits (= 8 for PCA components)
- **reps**: Number of circuit repetitions (2)
- **entanglement**: "linear" for adjacent feature correlation (less gate depth)

### Hilbert Space Dimensionality

| Qubits | Hilbert Space Dimension |
|--------|------------------------|
| 4      | 16 (2⁴)                |
| 8      | 256 (2⁸)               |
| 12     | 4096 (2¹²)             |

---

## Quantum Kernel

### FidelityQuantumKernel

The FidelityQuantumKernel computes the kernel matrix as:

```python
from src.quantum_kernel_engine import create_quantum_kernel

quantum_kernel = create_quantum_kernel(feature_map=feature_map)

# Compute kernel matrix
K = quantum_kernel.evaluate(x_vec=X_train)
```

### Kernel Matrix

The kernel matrix K has elements:
- **K[i,i] ≈ 1**: Self-overlap (normalized states)
- **K[i,j] ≈ 1**: Similar quantum embeddings
- **K[i,j] ≈ 0**: Orthogonal quantum embeddings (dissimilar)

---

## Exact QSVC

### Algorithm

To ensure a fair mathematical comparison with classical RBF SVM, we use the exact `QSVC` (Quantum Support Vector Classifier) rather than a stochastic optimizer.

```python
from qiskit_machine_learning.algorithms import QSVC

qsvc = QSVC(
    quantum_kernel=quantum_kernel,
    C=1.0
)

qsvc.fit(X_train, y_train)
```

---

## Noise Simulation

### NISQ Noise Effects

Current quantum computers are **Noisy Intermediate-Scale Quantum (NISQ)** devices with:
- Limited qubits (50-100+)
- Error rates ~0.1-1%
- No error correction

### Noise Sources

1. **Decoherence**: Qubits lose quantum information over time
2. **Gate Errors**: Imperfect quantum gates accumulate
3. **Measurement Errors**: Readout noise
4. **Cross-talk**: Unwanted qubit interactions

### Noise Impact on Kernels

- Diagonal elements deviate from 1
- Off-diagonal elements lose correlation
- Positive semi-definiteness may be violated
- Classification performance degrades

### Mitigation Strategies

1. **Error Mitigation**: ZNE, PEC
2. **Circuit Optimization**: Reduce depth
3. **Feature Map Design**: Fewer reps
4. **Post-processing**: Kernel regularization

---

## Computational Complexity

### Comparison: Quantum Kernel vs RBF Kernel

| Aspect | RBF Kernel | Quantum Kernel |
|--------|-----------|----------------|
| Feature Space | Implicit (∞-dim) | Hilbert Space (2ⁿ-dim) |
| Computation | O(n²d) | O(n² × circuit_depth) |
| Memory | O(n²) | O(n²) |
| Training | O(n² to n³) | O(n² × iterations) |
| Advantage | Mature, fast | Potential for complex patterns |

Where:
- n = number of samples
- d = feature dimension
- circuit_depth = feature_map.reps × entanglement_pattern

---

## Results

### Expected Output

The project generates:
- `kernel_heatmap.png`: Visualization of quantum kernel matrix
- `metrics_comparison.csv`: CSV with accuracy, precision, recall, F1
- `confusion_matrix.png`: Confusion matrices for both models

### Sample Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Classical RBF SVM | ~95% | ~0.95 | ~0.95 | ~0.95 |
| Quantum Kernel SVM | Variable | Variable | Variable | Variable |

*Note: Quantum kernel results vary based on noise, random seed, and hardware characteristics.*

---

## Notebooks

### 1. Data Preprocessing (01_data_preprocessing.ipynb)

- Load MNIST dataset (4 vs 9)
- Apply PCA dimensionality reduction
- Visualize PCA scatter plot
- Show variance explained

### 2. Classical Baseline (02_classical_baseline.ipynb)

- Train RBF SVM with GridSearchCV
- Evaluate metrics
- Generate confusion matrix

### 3. Quantum Kernel SVM (03_quantum_kernel_svm.ipynb)

- Implement ZZFeatureMap
- Compute FidelityQuantumKernel
- Train PegasosQSVC
- Compare with classical baseline
- Analyze noise effects

---

## Documentation

### Key Concepts Explained

#### What is a Quantum Kernel?

A quantum kernel is a function that measures similarity between quantum states encoding classical data. Unlike classical kernels, quantum kernels operate in the exponentially large Hilbert space.

#### Why Pegasos?

Pegasos provides memory-efficient training for large kernel matrices through stochastic gradient descent, making it practical for quantum kernel SVMs.

#### Hilbert Space Mapping

Classical data x is mapped to quantum state |φ(x)⟩ through parameterized quantum circuits. The kernel then measures overlap between these quantum states.

---

## Future Work

### Advanced Extensions

1. **Kernel Alignment Optimization**: Train feature map parameters to maximize kernel alignment
2. **Parameterized Feature Maps**: Use variational feature maps (e.g., QAOA-based)
3. **Noise Mitigation**: Apply ZNE, PEC for better results
4. **Quantum Circuit Depth Analysis**: Study effect of circuit depth
5. **Hybrid Classical-Quantum Kernels**: Combine RBF and quantum kernels

### Research Directions

- Explore different feature maps (ZZFeatureMap, PauliFeatureMap)
- Investigate quantum advantage for specific data distributions
- Develop better noise mitigation techniques
- Scale to larger datasets with efficient training

---

## References

1. Havlicek et al., "Supervised learning with quantum enhanced feature spaces" (Nature 2019)
2. Schuld et al., "Quantum machine learning in feature Hilbert spaces" (Phys. Rev. Lett. 2019)
3. Zhou et., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM" (ICML 2007)
4. Qiskit Machine Learning Documentation

---

## License

MIT License

---

Author: DEVADATH H K

---

## Acknowledgments

- IBM Quantum via Qiskit
- scikit-learn for classical ML baseline
- OpenML for MNIST dataset
