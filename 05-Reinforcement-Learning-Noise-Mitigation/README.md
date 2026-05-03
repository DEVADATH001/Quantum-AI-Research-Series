# Project 05: Reinforcement Learning for Quantum Noise Mitigation

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-2.3.x-purple)
![Status](https://img.shields.io/badge/status-research%20benchmark-orange)
![Claim](https://img.shields.io/badge/quantum%20advantage-not%20claimed-red)

Research benchmark for measurement-defined quantum reinforcement learning (QRL) policies under ideal simulation, NISQ-style noise, and lightweight error mitigation.

This repository studies a narrow question:

> How do variational quantum policies behave when their action distribution is defined directly by measurement probabilities, and how much do readout correction and zero-noise extrapolation recover under controlled RL benchmarks?

The project is designed as a reproducible systems-and-evaluation artifact. It is not presented as evidence of quantum advantage.

---

## README Audit

The previous README had a useful honest framing, but it was too compressed for a research-facing GitHub project.

### Missing Sections

- No complete installation guide with Python version, editable install, and dependency options.
- No clear separation between smoke runs, default training runs, and the intended full eight-scenario benchmark.
- No detailed usage section for console scripts exposed in `pyproject.toml`.
- No mathematical foundations beyond one policy equation.
- No computational complexity or hardware-cost discussion despite the project having a dedicated hardware audit.
- No explicit limitations section tied to saved empirical evidence.
- No references or author-maintainer section.

### Weak Explanations

- The environment was named but not explained as a finite-state key-and-door Markov decision process.
- The quantum policy was described as "measurement-native" without enough circuit, action-register, or state-encoding context.
- The mitigation stack was listed without clarifying that it is heuristic, not an exact inverse of hardware noise.
- The benchmark outputs were described structurally but not connected to the actual metrics a reader should inspect.

### Technical Inaccuracies or Risky Ambiguities

- The README implied a mature benchmark artifact, but the checked-in saved benchmark report is a two-scenario smoke suite. The full eight-scenario benchmark is configured but no full result bundle is checked in.
- It did not clearly distinguish `quantum_reinforce` from the upgraded `quantum_actor_critic` method.
- It did not warn that in-loop ZNE multiplies circuit executions and is not realistic for full hardware training.
- The saved `results/run_manifest.json` records five training seeds, while some configs define ten benchmark seeds. This matters for reproducibility and statistical interpretation.

### Unclear Instructions

- Commands were listed, but there was no recommended order for new users.
- Output locations were described, but not which artifacts correspond to training, smoke benchmark, paper report, audit, or mitigation ablation.
- IBM hardware-related workflows were mentioned without explaining the intended simulator-train and hardware-evaluate split.

### Missing Research Context

- The README did not explain why measurement-defined policies are different from expectation-value-logit quantum classifiers.
- It did not connect REINFORCE, actor-critic learning, GAE, parameter-shift gradients, and finite-shot bias.
- It did not summarize the negative-result interpretation: noisy quantum policies degrade, mitigation partially recovers in some cases, and classical baselines remain critical.

### Poor Structure

- The previous structure was useful for maintainers but not enough for researchers, engineers, or recruiters evaluating the project.
- There was no structured "what to run first" path.
- Research credibility depended on local artifacts that were not surfaced in the README.

---

## Project Overview

This project implements a benchmark-first quantum RL system for a discrete key-and-door navigation task. The agent must first move to a key location, then reverse direction to reach a locked goal. The environment is small enough for controlled quantum-policy experiments but nontrivial because the optimal action depends on the latent "has key" phase of the episode.

The main experimental comparison includes:

- classical baselines: random, tabular REINFORCE, MLP REINFORCE, MLP actor-critic
- quantum baselines: quantum REINFORCE under ideal, noisy, and mitigated execution
- upgraded quantum method: quantum actor-critic with a measurement-defined quantum actor and classical MLP critic
- mitigation modes: none/noisy, readout correction, ZNE, and combined mitigation
- resource metrics: runtime, estimated shots, success per runtime second, and success per million shots

The current checked-in evidence supports a conservative conclusion: quantum policies can perform well in idealized small simulations, noise degrades performance, lightweight mitigation can recover part of the loss in some runs, and classical baselines remain essential. No quantum advantage is claimed.

---

## Motivation / Research Context

Near-term quantum RL research often suffers from three documentation and evaluation problems:

1. quantum policies are treated as classical logits instead of measurement-defined probability models;
2. noisy and mitigated results are reported without resource-normalized costs;
3. weak classical baselines make apparent improvements difficult to interpret.

This repository addresses those issues by using a fixed benchmark family, explicit quantum execution modes, stronger classical comparators, and saved provenance artifacts. The goal is not to show superiority of quantum RL, but to make the behavior of quantum policies under noise measurable and reproducible.

This section matters because researchers need to know the scientific question before interpreting results. Engineers and recruiters need to know whether the project is a toy demo or a structured experimental system.

---

## Key Features

- Measurement-defined quantum policies: actions are sampled from Born probabilities on an action register, not from a classical softmax over expectation values.
- Key-and-door benchmark family: eight configured scenarios varying corridor size, reward sparsity, and action-slip stochasticity.
- Multiple learning algorithms: quantum REINFORCE, quantum actor-critic, tabular REINFORCE, MLP REINFORCE, MLP actor-critic, and random baseline.
- Noise-aware execution: ideal, noisy, and mitigated quantum execution modes.
- Lightweight mitigation: qubit-wise readout correction and ZNE-style probability extrapolation.
- Research-grade artifacts: JSON summaries, per-seed logs, benchmark reports, paper-style tables, figures, hardware audit, and experiment-design audit.
- Console entry points: installable commands for training, benchmark runs, sweeps, audits, reports, and hardware evaluation.
- Explicit negative-result framing: the project warns against unsupported quantum-advantage claims.

This section matters because it gives readers a fast but accurate inventory of what the repository actually implements.

---

## System Architecture

```text
05-Reinforcement-Learning-Noise-Mitigation/
|-- agent/             # quantum policy, REINFORCE learner, actor-critic learner, gradients
|-- benchmarks/        # named benchmark scenario registry
|-- config/            # training, smoke, benchmark, tuning, and hardware configs
|-- core/              # seeds, schemas, public runner helpers
|-- docs/              # benchmark specification and technical note
|-- environments/      # key-and-door navigation environment
|-- hardware/          # backend adapter helpers
|-- methods/           # method-level wrappers
|-- reporting/         # paper-style report generation
|-- results/           # saved default training, audits, sweeps, and ablations
|-- results_benchmark_smoke/
|   |-- benchmark_report.json
|   |-- benchmark_report.md
|   `-- paper_report/
|-- src/               # CLI modules and experiment orchestration
|-- tests/             # smoke and regression tests
`-- utils/             # Qiskit and plotting helpers
```

High-level execution flow:

```text
YAML config
   |
   v
benchmark/training CLI
   |
   +--> KeyDoorNavigationEnv
   |
   +--> classical baselines
   |
   +--> QuantumPolicyNetwork
          |
          +--> ideal/noisy/mitigated runtime executor
          |
          +--> parameter-shift gradient estimator
          |
          +--> REINFORCE or actor-critic optimizer
   |
   v
summary.json + per-seed logs + reports + figures
```

This section matters because research users need to understand the pipeline before modifying experiments, and engineers need to see module boundaries.

---

## Mathematical Foundations

### Environment

The benchmark is a finite-horizon Markov decision process:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, H)
$$

where:

- `S` encodes position and whether the key has been collected;
- `A = {left, right}`;
- `P` includes optional action-slip probability;
- `r` includes step cost, wall penalty, locked-goal penalty, optional shaping reward, key reward, and goal reward;
- `H` is the maximum episode length.

The checked-in default task uses 4 positions, 8 observable states, 2 actions, horizon 8, slip probability 0.05, and 64 held-out evaluation episodes.

### Measurement-Defined Quantum Policy

For state `s` and trainable parameters `theta`, the quantum actor prepares:

$$
|\psi(\theta, s)\rangle = U(\theta, s)|0\rangle
$$

The policy is defined directly by measuring the action register:

$$
\pi_{\theta}(a \mid s) = \Pr[M_{\text{action}} = a]
$$

For a two-action environment, the action register uses one qubit. Additional qubits are latent/state-processing qubits. The current policy uses hybrid angle embedding followed by a variational ansatz such as `RealAmplitudes`.

### REINFORCE Objective

The policy-gradient objective uses advantage-weighted log probabilities and entropy regularization:

$$
\mathcal{L}_{\text{pg}}(\theta)
= -\sum_t A_t \log \pi_{\theta}(a_t \mid s_t)
- \beta \sum_t H(\pi_{\theta}(\cdot \mid s_t))
$$

For the legacy quantum REINFORCE baseline, advantages are reward-to-go values minus an action-independent baseline:

$$
A_t = G_t - b_t
$$

### Parameter-Shift Gradient

Under ideal parameter-shift-compatible gates:

$$
\frac{\partial \pi_{\theta}(a \mid s)}{\partial \theta_i}
=
\frac{
\pi_{\theta_i + \pi/2}(a \mid s)
-
\pi_{\theta_i - \pi/2}(a \mid s)
}{2}
$$

This identity is exact for ideal Born probabilities. Under finite shots, noise, and mitigation, the estimator is stochastic and may be biased.

### Quantum Actor-Critic and GAE

The upgraded method uses:

- actor: measurement-defined quantum policy;
- critic: classical MLP value function over one-hot state features;
- advantage estimator: generalized advantage estimation.

Temporal-difference residual:

$$
\delta_t = r_t + \gamma V_{\phi}(s_{t+1}) - V_{\phi}(s_t)
$$

GAE advantage:

$$
A_t^{\text{GAE}}
=
\sum_{\ell=0}^{H-t-1}(\gamma \lambda)^\ell \delta_{t+\ell}
$$

The critic is trained using squared value error against GAE-induced returns.

### Mitigation Model

The mitigation stack is intentionally lightweight:

- readout correction: applies action-register confusion-matrix correction;
- ZNE: evaluates folded circuits at scale factors such as 1, 2, and 3, then extrapolates probabilities back toward zero noise.

These methods are approximations for trend analysis. They should not be interpreted as exact hardware-noise inverses.

### Computational Complexity

Let:

- `T` be maximum episode length;
- `E` be training episodes per seed;
- `P` be quantum parameter count;
- `S` be shots per circuit;
- `K` be the number of ZNE scale factors;
- `N` be the number of seeds;
- `C` be the number of scenarios.

For parameter-shift training, the rough number of quantum probability queries per seed is:

$$
O(E \cdot T \cdot 2P)
$$

With ZNE, the circuit-execution multiplier becomes:

$$
O(K \cdot E \cdot T \cdot 2P)
$$

The full benchmark scales approximately as:

$$
O(C \cdot N \cdot E \cdot T \cdot P \cdot K \cdot S)
$$

This cost model explains why the project recommends simulator training plus small hardware-evaluation slices rather than full in-loop hardware training.

This section matters because research-level documentation should expose the actual algorithmic assumptions and costs.

---

## Technologies Used

- Python 3.11+
- Qiskit 2.3.x
- Qiskit Aer 0.17.x
- Qiskit IBM Runtime 0.45.x
- Qiskit Machine Learning 0.9.x
- NumPy
- pandas
- Matplotlib
- PyYAML
- unittest
- Jupyter

This section matters because reproducibility depends on compatible scientific-computing and quantum SDK versions.

---

## Repository Structure

Important files:

| Path | Purpose |
|---|---|
| `agent/quantum_policy.py` | Measurement-defined variational quantum policy |
| `agent/reinforce_learner.py` | Legacy quantum REINFORCE learner |
| `agent/actor_critic_learner.py` | Quantum actor-critic learner |
| `agent/gradient_estimator.py` | Parameter-shift gradient estimator |
| `environments/simple_nav_env.py` | Key-and-door navigation environment |
| `benchmarks/scenario_registry.py` | Eight-scenario benchmark registry |
| `src/training_pipeline.py` | Main single-config training runner |
| `src/benchmark_suite.py` | Multi-scenario benchmark runner |
| `src/mitigation_ablation_suite.py` | Mitigation ablation runner |
| `src/hardware_audit.py` | Hardware feasibility analysis |
| `src/experiment_design_audit.py` | Scientific audit of experiment design |
| `reporting/paper_report.py` | Paper-style table and figure generation |
| `docs/technical_note.md` | Mathematical and algorithmic notes |
| `docs/benchmark_spec.md` | Benchmark definition |
| `results/summary.json` | Saved default training summary |
| `results_benchmark_smoke/benchmark_report.md` | Saved two-scenario smoke benchmark report |

This section matters because new users need to know where to inspect implementation, configuration, and evidence.

---

## Installation Guide

From this project directory:

```bash
cd 05-Reinforcement-Learning-Noise-Mitigation
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and the local package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

After editable installation, console scripts become available:

```bash
qrl-train --help
qrl-benchmark --help
qrl-report --help
qrl-audit --help
```

This section matters because reproducible experiments require a predictable environment and an install path that exposes the project CLIs.

---

## Usage Instructions

### 1. Run a Fast Smoke Training Pass

```bash
python -m src.training_pipeline --config config/smoke_test.yaml
```

Use this first to verify dependencies and local execution.

### 2. Run the Default Training Configuration

```bash
python -m src.training_pipeline --config config/training_config.yaml
```

This writes logs, plots, and `summary.json` under the configured results directory.

### 3. Run the Two-Scenario Smoke Benchmark

```bash
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

Saved smoke benchmark outputs are already present in `results_benchmark_smoke/`.

### 4. Run the Full Eight-Scenario Benchmark

```bash
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

This is the intended research benchmark. It is more expensive than the smoke suite because it runs eight scenarios and the configured main seed list.

### 5. Build a Paper-Style Report Bundle

```bash
python -m src.paper_report --results-root results_benchmark_smoke
```

The report builder creates Markdown/LaTeX tables and PNG/SVG figures.

### 6. Run the Experiment Design Audit

```bash
python -m src.experiment_design_audit --config config/training_config.yaml
```

This evaluates the evidence strength of the current experimental design.

### 7. Run the Hardware Audit

```bash
python -m src.hardware_audit --config config/training_config.yaml
```

Use this before attempting IBM Runtime execution.

### 8. Run Tests

```bash
python -m unittest discover -s tests
```

This section matters because a README should make the shortest path from clone to useful output obvious.

---

## Example Results / Visualizations

Saved result artifacts include:

- `results/learning_curves.png`
- `results/final_policy_plot.png`
- `results/convergence_comparison.png`
- `results/baseline_comparison.png`
- `results_benchmark_smoke/paper_report/figures/figure_1_leaderboard.png`
- `results_benchmark_smoke/paper_report/figures/figure_3_noise_forest.png`
- `results_benchmark_smoke/paper_report/figures/figure_4_shot_frontier.png`
- `results_benchmark_smoke/paper_report/figures/figure_5_runtime_frontier.png`

Current checked-in smoke benchmark leaderboard:

| Method | Avg eval success | Avg rank | Wins | Success/runtime sec | Success/million shots |
|---|---:|---:|---:|---:|---:|
| Quantum Actor-Critic ideal | 1.000 | 1.000 | 2 | 0.893 | 22.247 |
| Quantum REINFORCE ideal | 0.875 | 2.000 | 0 | 0.584 | 19.466 |
| Quantum Actor-Critic mitigated | 0.750 | 3.000 | 0 | 0.133 | 8.343 |
| Quantum Actor-Critic noisy | 0.750 | 4.000 | 0 | 0.406 | 16.685 |
| Quantum REINFORCE mitigated | 0.688 | 5.000 | 0 | 0.123 | 7.647 |
| Quantum REINFORCE noisy | 0.625 | 6.000 | 0 | 0.307 | 13.905 |
| Tabular REINFORCE | 0.375 | 7.000 | 0 | 18.466 | NA |
| Random baseline | 0.185 | 8.000 | 0 | NA | NA |
| MLP Actor-Critic | 0.125 | 9.000 | 0 | 4.925 | NA |
| MLP REINFORCE | 0.062 | 10.000 | 0 | 1.751 | NA |

Important interpretation: this table is from the two-scenario smoke benchmark, not the full eight-scenario research benchmark. It validates the benchmark runner and report generator, but it is not sufficient for a research claim.

This section matters because readers should be able to inspect actual artifacts and understand what evidence level each result supports.

---

## Experimental Setup

### Environment

Default training environment:

- positions: 4
- observable states: 8
- actions: 2
- key position: 0
- goal position: 3
- max episode steps: 8
- slip probability: 0.05
- key reward: 0.15
- goal reward: 1.0
- progress shaping: enabled in default scenario

### Quantum Policy

Default quantum policy:

- qubits: 3 for 4-position scenarios
- action-register qubits: 1
- latent qubits: 2
- ansatz: `RealAmplitudes`
- ansatz repetitions: 1
- entanglement: linear
- state encoding: hybrid angle embedding
- shots: 128

The hardware audit reports 6 trainable policy parameters for the default 3-qubit policy.

### Benchmark Family

The full configured research suite contains:

1. `default_4pos`
2. `sparse_4pos`
3. `high_slip_4pos`
4. `sparse_high_slip_4pos`
5. `default_5pos`
6. `sparse_5pos`
7. `high_slip_5pos`
8. `sparse_high_slip_5pos`

These scenarios vary corridor size, reward sparsity, and action-slip probability.

### Seeds

The full benchmark config defines:

```text
[7, 21, 33, 47, 63, 77, 91, 105, 119, 133]
```

The saved `results/summary.json` and `results/run_manifest.json` currently record five seeds:

```text
[7, 21, 33, 47, 63]
```

Smoke benchmark runs use seed `7`.

This section matters because differences between configured experiments and checked-in result artifacts affect reproducibility and statistical strength.

---

## Performance Metrics

The project reports:

- held-out evaluation success
- held-out evaluation reward
- reward AUC
- success AUC
- final average reward
- final success rate
- convergence episode
- runtime per seed
- estimated shots per seed
- success per runtime second
- success per million shots
- ideal-to-noisy performance drop
- mitigated-to-noisy recovery
- paired seed statistics and sign-flip p-values

Saved default training summary highlights:

| Method / mode | Eval success | Eval reward | Reward AUC | Mean runtime sec |
|---|---:|---:|---:|---:|
| Quantum REINFORCE ideal | 0.850 | 1.047 | 0.989 | 6.341 |
| Quantum REINFORCE noisy | 0.666 | 0.792 | 0.783 | 11.107 |
| Quantum REINFORCE mitigated | 0.750 | 0.930 | 0.905 | 55.842 |
| Tabular REINFORCE | 0.794 | 0.947 | 0.549 | 0.064 |
| MLP REINFORCE | 0.991 | 1.248 | 0.705 | 0.055 |

Key paired comparisons from the saved default training summary:

| Comparison | Metric | Mean difference | Sign-flip p-value | Interpretation |
|---|---:|---:|---:|---|
| ideal - noisy | eval success | 0.184 | 0.0625 | noise degrades quantum policy performance |
| mitigated - noisy | eval success | 0.084 | 0.1250 | mitigation partially recovers success, not strong enough for a firm claim |
| ideal - tabular | eval success | 0.056 | 0.1250 | small ideal quantum edge over tabular in this run, not significant |
| noisy - tabular | eval success | -0.128 | 0.0625 | noisy quantum underperforms tabular |
| ideal - MLP | eval success | -0.141 | 0.0625 | MLP baseline outperforms ideal quantum on held-out success |

Mitigation ablation from `results/mitigation_ablation/ablation_report.json`:

| Mitigation condition | Mitigated eval success | Eval-success gain over noisy | Reward-AUC gain over noisy |
|---|---:|---:|---:|
| none | 0.438 | 0.000 | 0.000 |
| readout only | 0.438 | 0.000 | 0.008 |
| ZNE only | 0.469 | 0.031 | 0.017 |
| both | 0.469 | 0.031 | 0.039 |

This section matters because a research README should summarize results with enough detail to prevent cherry-picked interpretation.

---

## Hardware Feasibility

The hardware audit reports that a single forward pass is plausible on IBM-style hardware, but full in-loop policy-gradient training is not realistic as a default workflow.

Default policy footprint:

- logical qubits: 3
- logical depth: 6
- logical operations: 9 `ry`, 3 `rz`, 2 `cx`
- transpiled depth on the audited backend model: 20
- transpiled two-qubit gates: 2

Worst-case workload estimates per seed:

| Mode | Circuit executions per seed | Shots per seed | Two-qubit gate executions per seed |
|---|---:|---:|---:|
| ideal/noisy | 5,312 | 679,936 | 1,359,872 |
| mitigated | 15,936 | 2,039,808 | 8,159,232 |

Recommended hardware workflow:

1. train in ideal simulation or fake-backend simulation;
2. freeze checkpoints;
3. run small held-out hardware evaluation slices;
4. use readout-focused mitigation by default;
5. reserve ZNE for small final calibration studies.

This section matters because quantum experiments must report resource feasibility, not only task scores.

---

## Limitations

- No quantum advantage is demonstrated.
- The checked-in benchmark report is a two-scenario smoke suite, not the full eight-scenario benchmark.
- The default saved training summary has five seeds, while the benchmark config defines ten seeds.
- The task family is handcrafted and small; it is useful for controlled analysis but not broad RL generalization.
- The MLP baseline can outperform the quantum policy on held-out success in saved default results.
- Parameter-shift training is shot-expensive.
- In-loop ZNE is not hardware-realistic for full training.
- The noise model can use compact approximations that do not capture all device-level effects.
- Sign-flip p-values are limited by small paired seed counts.
- Results are sensitive to configuration, seed set, and evaluation budget.

This section matters because professional research documentation should state what the project does not prove.

---

## Future Improvements

- Run and publish the full eight-scenario benchmark under `results/benchmark_suite/`.
- Increase paired seed count for stronger statistical power.
- Add budget-matched classical baselines across wall-clock and sample budgets.
- Add additional environment families beyond key-and-door navigation.
- Separate training-time and evaluation-time mitigation costs more explicitly in reports.
- Add checkpoint-based hardware evaluation for frozen policies.
- Add CI that runs smoke benchmarks and validates report schemas.
- Add richer statistical reporting, including effect sizes and multiple-comparison corrections.
- Add plots comparing final success, AUC, runtime efficiency, and shot efficiency across all eight scenarios.

This section matters because maintainers and researchers need a clear roadmap for strengthening the artifact.

---

## References

- Sutton, R. S., and Barto, A. G. *Reinforcement Learning: An Introduction*. MIT Press, 2018.
- Williams, R. J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*, 1992.
- Schulman, J. et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." ICLR, 2016.
- Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. "Evaluating analytic gradients on quantum hardware." *Physical Review A*, 2019.
- Temme, K., Bravyi, S., and Gambetta, J. M. "Error mitigation for short-depth quantum circuits." *Physical Review Letters*, 2017.
- Kandala, A. et al. "Error mitigation extends the computational reach of a noisy quantum processor." *Nature*, 2019.
- Qiskit documentation: https://docs.quantum.ibm.com/

This section matters because research documentation should connect implementation choices to established methods.

---

## Author Information

This project is part of the **Quantum AI Research Series**.

Intended audience:

- quantum machine learning researchers;
- reinforcement learning engineers;
- research software engineers;
- recruiters evaluating applied quantum-AI engineering work.

Recommended citation style for informal use:

```text
Quantum AI Research Series, Project 05:
Reinforcement Learning for Quantum Noise Mitigation.
Measurement-defined quantum RL benchmark under ideal, noisy, and mitigated execution.
```

This section matters because professional open-source repositories should identify project ownership, scope, and intended audience.

---

## Brutal Quality Check

Current README quality after this rewrite:

| Category | Score | Rationale |
|---|---:|---|
| Clarity | 9.0/10 | The project goal, workflow, commands, and result interpretation are now explicit. |
| Technical depth | 9.0/10 | Includes RL objective, quantum policy definition, GAE, parameter-shift gradients, mitigation assumptions, and complexity. |
| Documentation quality | 9.1/10 | Provides structure, commands, artifacts, architecture, limitations, and reproducibility notes. |
| Research credibility | 8.8/10 | Strong honesty around negative results and hardware cost, but full benchmark results are not checked in. |
| Open-source usability | 8.7/10 | Install and run paths are clear; usability would improve with CI badges and published full benchmark artifacts. |

Overall score: **8.9/10**.

Final improvements needed before calling this a polished publication artifact:

- run the full eight-scenario benchmark and commit the report bundle;
- resolve the five-seed versus ten-seed artifact mismatch;
- add CI for smoke tests and schema validation;
- add a concise `CITATION.cff`;
- include final paper-style plots from the full benchmark rather than only smoke outputs.
