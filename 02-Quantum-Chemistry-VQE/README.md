# Project 02: Simulating Molecular Ground States with VQE

## Objective

Estimate molecular ground-state energy (starting with H2) using VQE and compare against a classical/eigensolver reference.

## Stack

- Python
- NumPy / SciPy
- Matplotlib
- Qiskit
- Qiskit Nature
- PySCF (classical reference)
- Jupyter Notebook

## Notebook

- `project_02_vqe_h2.ipynb`

## Tasks

1. Define H2 molecular geometry and basis.
2. Build second-quantized Hamiltonian.
3. Map to qubit operator (Jordan-Wigner).
4. Configure ansatz + optimizer + estimator backend.
5. Run VQE and compare with exact/reference energy.

## Expected Outputs

- VQE convergence plot (energy vs iteration).
- Final estimated ground-state energy.
- Comparison with exact diagonalization/reference.

