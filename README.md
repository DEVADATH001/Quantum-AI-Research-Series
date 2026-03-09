# Quantum AI Research Series

A 5-project, beginner-to-advanced portfolio series that builds from classical ML + quantum basics to quantum optimization and quantum reinforcement learning.

## Repository Structure

1. `01-Classical-vs-Quantum-Visualization`
2. `02-Quantum-Chemistry-VQE`
3. `03-Quantum-Kernel-SVM-MNIST`
4. `04-Optimization-QAOA-MaxCut`
5. `05-Reinforcement-Learning-Noise-Mitigation`

## Quick Start

```powershell
# From repository root
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## IBM Quantum Token Setup (Do Not Hardcode Secrets)

1. Rotate any previously exposed API key in IBM Quantum dashboard.
2. Set token in environment variable:

```powershell
setx IBM_QUANTUM_TOKEN "YOUR_NEW_TOKEN"
```

3. Save account once from Python:

```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token=os.environ["IBM_QUANTUM_TOKEN"],
    overwrite=True
)
```

## Git Workflow

Use one branch per project:

```powershell
git checkout -b feat/project-1
# work, commit, push, PR, merge
```

Recommended commit style:

- `feat: ...` for new notebooks/code
- `docs: ...` for README/report updates
- `chore: ...` for setup/config

## Deliverables Checklist

- Project README with objective, tools, and results
- Jupyter notebook with clear markdown sections
- At least one metric chart/table
- Reproducible environment and run instructions

