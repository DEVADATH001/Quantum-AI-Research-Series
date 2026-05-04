# Project 05: Quantum RL with Noise Mitigation

![Python](https://img.shields.io/badge/Python-3.11%20required-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-PQC%20policies-6929C4)
![PyTorch](https://img.shields.io/badge/PyTorch-neural%20baselines-EE4C2C)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Key--and--Door-4B8BBE)
![Status](https://img.shields.io/badge/status-smoke%20only-yellow)

This module studies how parameterized quantum circuit (PQC) policies perform for reinforcement learning — specifically, how they degrade under noise and whether error mitigation actually helps in practice.

**Fair warning:** only a smoke benchmark report is committed. The full benchmark config exists and the framework is ready, but that complete run hasn't been done yet. Take the smoke numbers as pipeline validation, not definitive science.

---

## Smoke benchmark results

From `results_benchmark_smoke/benchmark_report.md` — a two-scenario smoke suite on a Key-and-Door environment:

| Method | Avg Eval Success | Avg Rank | Success / Runtime | Wins |
|---|---:|---:|---:|---:|
| Quantum Actor-Critic (ideal) | 1.000 | 1.0 | 0.893 | 2 |
| Quantum REINFORCE (ideal) | 0.875 | 2.0 | 0.584 | 0 |
| Quantum Actor-Critic (mitigated) | 0.750 | 3.0 | 0.133 | 0 |
| Quantum Actor-Critic (noisy) | 0.750 | 4.0 | 0.406 | 0 |
| Quantum REINFORCE (mitigated) | 0.688 | 5.0 | 0.123 | 0 |
| Quantum REINFORCE (noisy) | 0.625 | 6.0 | 0.307 | 0 |
| Tabular REINFORCE | 0.375 | 7.0 | 18.466 | 0 |
| Random baseline | 0.185 | 8.0 | — | 0 |
| MLP Actor-Critic | 0.125 | 9.0 | 4.925 | 0 |
| MLP REINFORCE | 0.062 | 10.0 | 1.751 | 0 |

**Noise impact:** quantum REINFORCE drops ~0.250 in success rate from ideal to noisy. Mitigation recovers only ~0.062 of that.

The mitigated variant underperforming noisy in some metrics is a training-duration artifact from the smoke config — not a real finding. Don't generalize from it.

**Do not cite these as final results.** The smoke config uses short training horizons and minimal seeds.

---

## What the framework tests

| Dimension | Coverage |
|---|---|
| Policy types | Tabular Q-learning, classical MLP actor-critic, quantum PQC actor-critic |
| Algorithms | REINFORCE, actor-critic (both classical and quantum) |
| Noise levels | Clean, low (0.1%), medium (1%), high (5%) error rates |
| Mitigation | Zero-noise extrapolation (ZNE), probabilistic error cancellation (PEC), measurement error calibration |
| Metrics | Mean reward, convergence steps, action consistency (Wasserstein), wall-clock efficiency |
| Backend | Local statevector (default), fake-backend noise, optional IBM Runtime |

---

## The questions driving the design

1. How fast does PQC policy performance degrade with increasing noise?
2. Does ZNE recover useful performance, or just add runtime overhead?
3. Where's the efficiency break-even — when does quantum's wall-clock cost become unjustifiable vs a classical network?
4. Can measurement error mitigation be applied efficiently inside the RL update loop?

The smoke results give preliminary signal. The full benchmark would give complete answers.

---

## The math

**PQC state encoding** — angle encoding with Hadamard initialization (4 qubits for 4 state components):

$$|\psi(s)\rangle = \prod_i R_Y(s_i \cdot \pi) H|0\rangle^{\otimes n}$$

**Variational ansatz** — trainable rotation layers with entanglement:

$$U_{\text{var}}(\theta) = \prod_{l=1}^{L} U_{\text{ent}} \cdot \prod_{i=1}^{n} R_Y(\theta_{l,i})$$

**Action probabilities** come from quantum measurement, not softmax:

$$\pi_\theta(a|s) = \Pr[M_{\text{action}} = a] = |\langle a | U_{\text{var}}(\theta) U_{\text{enc}}(s) |+\rangle|^2$$

This is a genuine probability from Born's rule, not an approximation.

**REINFORCE gradient:**

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_{t=0}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

where $G_t = \sum_{k=t}^T \gamma^{k-t} r_k$.

**Parameter-shift rule** for quantum gradients:

$$\frac{\partial \langle O \rangle}{\partial \theta_i} = \frac{\langle O \rangle_{\theta_i + \pi/2} - \langle O \rangle_{\theta_i - \pi/2}}{2}$$

Cost: 2 extra circuit executions per parameter per gradient step.

**ZNE (Richardson, 3-point extrapolation):**

$$\langle O \rangle_0 \approx 3\langle O \rangle_\lambda - 3\langle O \rangle_{2\lambda} + \langle O \rangle_{3\lambda}$$

Requires 3× circuit evaluations at scaled noise levels.

**Action consistency** via Wasserstein distance:

$$W_1(\pi_1, \pi_2) = \int_0^1 |F_{\pi_1}^{-1}(q) - F_{\pi_2}^{-1}(q)| \, dq$$

Measures how much the policy distribution shifts between ideal and noisy/mitigated execution.

**Efficiency metric:**

$$\text{QV ratio} = \frac{\text{eff}_\text{quantum}}{\text{eff}_\text{classical}}, \quad \text{eff} = \frac{\text{reward} \times \text{convergence speed}}{\text{wall-clock time}}$$

QV ratio < 1 means classical is more efficient per unit of compute.

---

## Architecture

```
config/
  benchmark_suite.yaml           ← Full benchmark config
  benchmark_suite_smoke.yaml     ← Smoke/CI config

src/
  agents/
    tabular_agent.py             ← Q-learning, epsilon-greedy
    classical_agent.py           ← MLP actor-critic (PyTorch)
    quantum_agent.py             ← PQC actor-critic
  circuits/
    pqc_policy.py                ← Feature map + variational ansatz
    noise_models.py              ← Synthetic noise channels
  mitigation/
    zne_wrapper.py               ← Zero-noise extrapolation
    pec_wrapper.py               ← Probabilistic error cancellation
    measurement_filter.py        ← Readout calibration
  benchmark_suite.py             ← Orchestration runner
  metrics_engine.py              ← Reward/convergence/consistency
  visualization.py               ← Plotting
```

---

## How to run

**Python 3.11 is required** — uses `tomllib`, generics syntax, and `match` statements.

### Smoke benchmark (minutes)

```powershell
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

### Full benchmark (multi-hour)

```powershell
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

### Single agent training

```powershell
python -m src.train --agent quantum --episodes 200 --noise-level 0.01
python -m src.train --agent classical --episodes 200
python -m src.train --agent tabular --episodes 200
```

### Tests

```powershell
python -m pytest tests -q
```

---

## Installation

```powershell
cd 05-Reinforcement-Learning-Noise-Mitigation
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Key dependencies: Qiskit Aer, qiskit-ibm-runtime, PyTorch (CPU or CUDA), Gymnasium, NumPy, SciPy, matplotlib, PyYAML.

For IBM Runtime:

```powershell
cd ..
setx IBM_QUANTUM_TOKEN "YOUR_TOKEN"
python scripts/setup_ibm_runtime.py
cd 05-Reinforcement-Learning-Noise-Mitigation
```

Then set `hardware.backend_name` in `config/benchmark_suite.yaml` to a real IBM device.

---

## Smoke vs full benchmark

| Parameter | Smoke | Full |
|---|---|---|
| Episodes per run | 50 | 1,000 |
| Seeds per agent | 1 | 10 |
| Noise levels | 3 | 5 |
| Mitigation strategies | ZNE only | ZNE, PEC, measurement |
| Max steps per episode | 200 | 500 |
| Expected runtime | Minutes | Multi-hour |

The smoke config validates the pipeline. The full config is where real conclusions come from.

---

## Limitations

- **Only smoke results are committed.** The full benchmark hasn't been run yet.
- The environment is a Key-and-Door grid — low-complexity control task. Quantum expressibility comparisons here don't transfer directly to harder environments.
- PQC computation is simulation-only by default. Hardware QRL has circuit-depth constraints the simulator doesn't capture.
- Richardson ZNE assumes linear noise scaling — that's an approximation.
- The smoke mitigated result looking worse than noisy is a training-length artifact. Don't generalize it.
- Parameter-shift costs 2 evaluations per parameter per gradient step — wall-clock cost scales with circuit width.
- Agent comparison doesn't control for model-parameter count between quantum and classical networks.

---

## What I'd work on next

- Run and commit the full benchmark
- Noise scaling analysis: performance vs error rate curve
- PEC and measurement mitigation in the full evaluation
- Convergence-step tracking (not just final reward)
- Backend-transpiled circuit depth vs theoretical depth
- A second environment (MountainCar, LunarLander) to test generalization
- PQC hyperparameter sensitivity (qubit count, layers, encoding)
- Matched-parameter classical baseline for fair efficiency comparison

---

## References

- Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed. 2018.
- Williams, R. J. Simple statistical gradient-following algorithms for connectionist RL. *Machine Learning*, 1992.
- Cerezo et al. Variational quantum algorithms. *Nature Reviews Physics*, 2021.
- Mitarai et al. Quantum circuit learning. *Physical Review A*, 2018.
- Li et al. Quantum reinforcement learning in continuous action space. arXiv:2012.10711, 2020.
- Temme, Bravyi, and Gambetta. Error mitigation for short-depth quantum circuits. *PRL*, 2017.
- Kim et al. Evidence for the utility of quantum computing before fault tolerance. *Nature*, 2023.
- Gymnasium: https://gymnasium.farama.org/
- Qiskit: https://docs.quantum.ibm.com/
- PyTorch: https://pytorch.org/

---

## Author

**DEVADATH H K** — Part of the [Quantum AI Research Series](../README.md).

See [LICENSE](../LICENSE), [CITATION.cff](../CITATION.cff), and [CONTRIBUTING.md](../CONTRIBUTING.md).
