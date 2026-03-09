# Project 04: Solving Max-Cut with QAOA

## Objective

Formulate Max-Cut as a QUBO/Ising problem and solve it approximately using QAOA on small random graphs.

## Stack

- Python
- NumPy / SciPy
- NetworkX
- Matplotlib
- Qiskit
- Qiskit Optimization
- Jupyter Notebook

## Notebook

- `project_04_qaoa_maxcut.ipynb`

## Tasks

1. Generate small random graph (3-5 nodes).
2. Formulate Max-Cut QUBO.
3. Configure QAOA with different depths (`p=1`, `p=2`, ...).
4. Optimize parameters via classical optimizer.
5. Compare with exact classical brute-force result.

## Expected Outputs

- Best bitstring/cut value from QAOA.
- Exact best cut value for comparison.
- Performance trend by QAOA depth.

