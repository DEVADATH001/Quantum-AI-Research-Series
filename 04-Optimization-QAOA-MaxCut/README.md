# 04-Optimization-QAOA-MaxCut

<p align="center">
  <img src="https://qiskit.org/images/qiskit-logo.png" alt="Qiskit" width="200"/>
</p>

A production-grade quantum optimization research platform for solving the Max-Cut problem using **Quantum Approximate Optimization Algorithm (QAOA)** and **Recursive QAOA (RQAOA)**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quantum Optimization Background](#quantum-optimization-background)
- [Max-Cut Problem](#max-cut-problem)
- [QAOA Algorithm](#qaoa-algorithm)
- [Recursive QAOA](#recursive-qaoa-rqaoa)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hardware-Aware Optimization](#hardware-aware-optimization)
- [API Reference](#api-reference)
- [Industrial Use Cases](#industrial-use-cases)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository demonstrates how **quantum-classical hybrid optimization loops** can solve combinatorial optimization problems, specifically the **Max-Cut problem**, which has applications in robot communication networks, VLSI design, and clustering.

### Primary Research Objective

Solve the **Max-Cut problem** for a graph that simulates a **robot communication mesh network** and benchmark performance using:

1. **Classical brute-force solution** (exact)
2. **Standard QAOA** (approximate)
3. **Recursive QAOA** (scalable approximate)

### Target Performance

$$\text{Approximation Ratio} = r = \frac{\text{QAOA\_Value}}{\text{Optimal\_Value}}$$

A ratio **r > 0.8** is considered strong for NISQ algorithms.

---

## Quantum Optimization Background

### Hybrid Quantum-Classical Computing

Quantum computers cannot yet run complete optimization algorithms independently. Instead, we use **hybrid approaches** where:

1. **Quantum Processing Unit (QPU)**: Prepares quantum states and computes expectation values
2. **Classical Processor**: Optimizes parameters and manages the control loop

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID LOOP                              │
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌────────────────┐   │
│   │ Classical │───▶│  Quantum     │───▶│   Classical    │   │
│   │ Optimizer │    │  Circuit     │    │   Evaluator    │   │
│   └──────────┘    └──────────────┘    └────────────────┘   │
│        ▲                                         │          │
│        └─────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Why Quantum for Optimization?

- **Quantum states** can represent superpositions of many solutions
- **Entanglement** can encode correlations between variables
- **Quantum tunneling** may help escape local minima
- **Exponential speedups** for certain problem classes (in theory)

---

## Max-Cut Problem

### Problem Definition

Given an undirected graph $G = (V, E)$ with $n$ vertices, partition the vertices into two disjoint sets $S$ and $V \setminus S$ to **maximize the number of edges crossing between the sets**.

### Mathematical Formulation

The Max-Cut cost function is:

$$C = \frac{1}{2} \sum_{(i,j) \in E} (1 - z_i z_j)$$

where:
- $z_i \in \{-1, +1\}$ represents the partition assignment
- $z_i = +1$ → vertex $i$ in set A
- $z_i = -1$ → vertex $i$ in set B

### Ising Hamiltonian

Converting to a quantum Hamiltonian:

$$H = \sum_{(i,j) \in E} w_{ij} \frac{(1 - Z_i Z_j)}{2}$$

Where:
- $Z_i$ is the Pauli-Z operator on qubit $i$
- The ground state corresponds to the optimal cut

### Example

For a 3-node graph:

```
    0 --- 1
    |     |
    |     |
    2 --- 3

Optimal Cut: {0, 3} vs {1, 2}
Edges in cut: (0,1), (2,3), (0,2), (1,3) = 4 edges
```

---

## QAOA Algorithm

### Overview

The **Quantum Approximate Optimization Algorithm (QAOA)** is a hybrid quantum-classical algorithm for combinatorial optimization.

### Circuit Structure

A QAOA circuit with $p$ layers consists of:

```
┌─────────────────────────────────────────────────────────────┐
│ Initial State: |+⟩⊗n = (|0⟩ + |1⟩)⊗n / 2^(n/2)             │
├─────────────────────────────────────────────────────────────┤
│ Layer 1:                                                     │
│   U_C(γ₁) = exp(-iγ₁ H_C)    [Cost Hamiltonian]            │
│   U_M(β₁) = exp(-iβ₁ H_M)    [Mixer Hamiltonian]          │
├─────────────────────────────────────────────────────────────┤
│ Layer 2:                                                     │
│   U_C(γ₂) = exp(-iγ₂ H_C)                                   │
│   U_M(β₂) = exp(-iβ₂ H_M)                                   │
├─────────────────────────────────────────────────────────────┤
│ ...                                                          │
├─────────────────────────────────────────────────────────────┤
│ Layer p:                                                     │
│   U_C(γ_p) = exp(-iγ_p H_C)                                 │
│   U_M(β_p) = exp(-iβ_p H_M)                                 │
└─────────────────────────────────────────────────────────────┘
```

### Parameters

- **γ (gamma)**: Cost Hamiltonian parameters - controls solution quality
- **β (beta)**: Mixer Hamiltonian parameters - controls exploration
- **p**: Number of layers - increases expressibility

### Cost Hamiltonian (Max-Cut)

$$H_C = \sum_{(i,j) \in E} \frac{(1 - Z_i Z_j)}{2}$$

Implemented as RZZ gates:
```python
for (i, j) in edges:
    circuit.rzz(2 * gamma, i, j)
```

### Mixer Hamiltonian

$$H_M = \sum_{i=1}^{n} X_i$$

Implemented as RX rotations:
```python
for i in range(n_qubits):
    circuit.rx(2 * beta, i)
```

---

## Recursive QAOA (RQAOA)

### Concept

RQAOA addresses QAOA's scalability limitations by **recursively reducing problem size** through exploiting correlations.

### Algorithm Steps

1. **Run QAOA** on the full problem
2. **Analyze correlations** between variables: $\langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle$
3. **Identify strong correlations**: $|C_{ij}| > \threshold$
4. **Eliminate correlated variables**: Fix $z_i = s_j \cdot z_j$ (same or opposite)
5. **Reduce problem**: Create smaller graph without eliminated nodes
6. **Recurse**: Solve smaller problem
7. **Reconstruct**: Expand solution back to original size

### Benefits

- **Reduced search space**: Each elimination removes one variable
- **Exploits structure**: Uses problem correlations
- **Better scalability**: Can handle larger graphs
- **Improved quality**: Often better than standard QAOA

### Visual Representation

```
Level 0: 15 qubits ──────▶ Solve QAOA ──────▶ Find correlations
                           │
                           ▼
Level 1: 12 qubits ──────▶ Solve smaller ──▶ Find correlations
                           │
                           ▼
Level 2: 8 qubits  ──────▶ ...
                           │
                           ▼
Level k: 4 qubits  ──────▶ Solve exact ────▶ Reconstruct solution
```

---

## Project Structure

```
04-Optimization-QAOA-MaxCut/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── PLAN.md                  # Implementation plan
│
├── config/
│   └── experiment_config.yaml    # Experiment configuration
│
├── data/
│   └── robot_network.adjlist    # Sample robot mesh graph
│
├── src/
│   ├── __init__.py              # Package initialization
│   ├── graph_generator.py       # Graph generation (D-regular, etc.)
│   ├── hamiltonian_builder.py   # Ising Hamiltonian construction
│   ├── qaoa_circuit.py          # QAOA circuit implementation
│   ├── qaoa_optimizer.py        # Classical optimization loop
│   ├── rqaoa_engine.py          # Recursive QAOA implementation
│   ├── classical_solver.py      # Brute-force exact solver
│   ├── evaluation_metrics.py    # Performance metrics
│   ├── runtime_executor.py      # Qiskit Runtime execution
│   └── visualization.py         # Plotting utilities
│
├── notebooks/
│   ├── 01_problem_formulation.ipynb  # Max-Cut math & Ising
│   ├── 02_qaoa_execution.ipynb        # QAOA execution details
│   └── 03_results_analysis.ipynb     # Results & analysis
│
├── results/
│   ├── energy_landscape.png
│   ├── approximation_ratio.png
│   ├── graph_cut_visualization.png
│   └── metrics.csv
│
└── utils/
    ├── __init__.py
    ├── qiskit_helpers.py      # Qiskit utility functions
    └── circuit_transpiler.py   # Hardware-aware transpilation
```

---

## Installation

### Requirements

- Python ≥ 3.10
- Qiskit ≥ 1.0
- NetworkX ≥ 3.0
- NumPy, SciPy, Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or for development:

```bash
pip install -e .
```

---

## Quick Start

### 1. Generate a Robot Communication Mesh

```python
from src.graph_generator import GraphGenerator

# Create graph generator
generator = GraphGenerator(seed=42)

# Generate D-regular graph (robot mesh)
graph = generator.generate_robot_mesh(
    n_robots=15,
    connectivity=3
)

print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### 2. Build Hamiltonian

```python
from src.hamiltonian_builder import HamiltonianBuilder

builder = HamiltonianBuilder()
hamiltonian, offset = builder.build_maxcut_hamiltonian(graph)

print(f"Hamiltonian: {len(hamiltonian)} terms")
```

### 3. Run QAOA

```python
from src.qaoa_circuit import QAOACircuitBuilder
from src.qaoa_optimizer import QAOAOptimizer

# Create circuit
circuit_builder = QAOACircuitBuilder(n_qubits=15, p=2)
circuit = circuit_builder.build_qaoa_circuit(hamiltonian)

# Optimize
optimizer = QAOAOptimizer(p=2, optimizer_type="COBYLA")
result = optimizer.optimize(objective_function, n_qubits=15, graph=graph)
```

### 4. Evaluate Results

```python
from src.evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()
ratio = metrics.compute_approximation_ratio(qaoa_value, optimal_value)

print(f"Approximation Ratio: {ratio:.4f}")
```

### 5. Visualize

```python
from src.visualization import Visualizer

viz = Visualizer()

# Plot solution
viz.plot_graph_cut(graph, partition, cut_edges)
viz.plot_approximation_ratio(depths, ratios)
viz.plot_energy_landscape(gamma_grid, beta_grid, cost_grid)
```

---

## Hardware-Aware Optimization

### Why Hardware Mapping Matters

Quantum processors only allow entanglement between **physically connected qubits** (coupling map). Mapping problem graphs to hardware topology is crucial for performance.

### Coupling Map Example

```
IBM Brisbane (127 qubits):
    
    0 ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 6
    │   │   │   │   │   │   │
    70  71  72  73  74  75  76
    │   │   │   │   │   │   │
   ... ... ... ... ... ... ...

Only CNOT gates between connected qubits!
```

### Mapping Strategy

```python
from utils.circuit_transpiler import CircuitTranspiler

transpiler = CircuitTranspiler(
    backend=backend,
    optimization_level=3
)

# Map to hardware
mapped_circuit = transpiler.map_to_hardware(qaoa_circuit)
```

### Best Practices

1. **Choose problem size** within hardware capabilities
2. **Match graph structure** to coupling map when possible
3. **Use transpiler** to find optimal qubit assignment
4. **Reduce circuit depth** to minimize noise effects

---

## API Reference

### Core Modules

| Module | Description |
|--------|-------------|
| `graph_generator.py` | Generate D-regular, Erdős-Rényi, Barabási-Albert graphs |
| `hamiltonian_builder.py` | Build Ising Hamiltonians from graphs |
| `qaoa_circuit.py` | Construct parameterized QAOA circuits |
| `qaoa_optimizer.py` | Classical optimization loop (COBYLA, SPSA) |
| `rqaoa_engine.py` | Recursive QAOA implementation |
| `classical_solver.py` | Brute-force exact solver |
| `evaluation_metrics.py` | Approximation ratio & performance metrics |
| `runtime_executor.py` | Qiskit Runtime V2 execution |

### Key Classes

```python
# Graph Generation
GraphGenerator(seed=42)
├── generate_d_regular_graph(n_nodes, degree)
├── generate_robot_mesh(n_robots, connectivity)
└── get_cut_value(graph, partition)

# Hamiltonian Construction  
HamiltonianBuilder()
├── build_maxcut_hamiltonian(graph) -> (SparsePauliOp, offset)
└── create_mixer_hamiltonian(n_qubits)

# QAOA Circuit
QAOACircuitBuilder(n_qubits, p)
├── build_qaoa_circuit(hamiltonian)
└── get_initial_parameters(strategy="random")

# Optimization
QAOAOptimizer(p, optimizer_type="COBYLA")
└── optimize(objective_function, n_qubits, graph)

# Recursive QAOA
RQAOAEngine(p, correlation_threshold=0.8)
└── solve(graph, optimal_value)

# Evaluation
EvaluationMetrics()
├── compute_approximation_ratio(qaoa_value, optimal_value)
└── evaluate_solution(graph, qaoa_bitstring, optimal_bitstring)
```

---

## Industrial Use Cases

### 1. Robot Communication Networks

Optimize data routing in mesh networks of robots:

```
Robot 0 ──▶ Robot 1 ──▶ Robot 2
   │          │          │
   ▼          ▼          ▼
Cut edges = communication paths optimized
```

### 2. VLSI Circuit Design

Partition chip components to minimize wiring:

```
┌─────────────────────┐
│  Component A   │  Component B   │
│   [Logic]      │    [Logic]     │
│       ─────────┼────────        │ ← Minimize crossing wires
│   [Memory]     │   [Memory]     │
└─────────────────────┘
```

### 3. Clustering & Community Detection

Group related entities while maximizing inter-group connections.

### 4. Network Security

Optimize firewall placement and network segmentation.

### 5. Portfolio Optimization

Balance risk and return in investment portfolios.

---

## Performance Metrics

### Approximation Ratio

$$r = \frac{\text{QAOA Cut Value}}{\text{Optimal Cut Value}}$$

| Ratio | Rating | Description |
|-------|--------|-------------|
| r ≥ 0.95 | Excellent | Near-optimal |
| r ≥ 0.90 | Very Good | High quality |
| r ≥ 0.80 | Good | NISQ-acceptable |
| r ≥ 0.70 | Fair | Needs improvement |
| r < 0.70 | Poor | Not recommended |

### Expected Performance

For random 3-regular graphs:
- p=1: r ≈ 0.70-0.80
- p=2: r ≈ 0.80-0.90
- p=3: r ≈ 0.85-0.95

### Benchmarking Results

See `notebooks/03_results_analysis.ipynb` for detailed benchmarks.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure PEP8 compliance
5. Submit a pull request

---

## License

MIT License - see LICENSE file for details.

---

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm.
2. Hadfield, S., et al. (2019). From the Quantum Approximate Optimization Algorithm to the Quantum Alternating Operator Ansatz.
3. Bärtschi, A., & Eidenbenz, S. (2019). Deterministic Preparation of Dicke States.
4. Qiskit Documentation: https://qiskit.org/documentation/

---

<p align="center">
  <b>Built with Qiskit 💜</b>
</p>

