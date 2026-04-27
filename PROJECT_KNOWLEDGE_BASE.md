# Quantum AI Research Series: Structured Knowledge Base

Author of repo: DEVADATH H K

Purpose of this document: turn the repository into study material for mastering the concepts, algorithms, math, and code architecture behind the five-project Quantum AI Research Series.

---

## Step 1: Project Understanding

### What Problem This Repository Solves

This repository is not one application. It is a research-learning portfolio that studies how near-term quantum computing methods interact with classical AI, optimization, chemistry, and reinforcement learning.

At the highest level, it solves this learning and research problem:

> How do we build, evaluate, and honestly benchmark quantum AI algorithms under realistic constraints instead of treating them as toy demos or exaggerated quantum-advantage claims?

The five modules cover different parts of the quantum AI landscape:

1. `01-Classical-vs-Quantum-Visualization`
   - Compares classical ML and quantum kernel classification on Iris.
   - Separately stress-tests a 127-qubit GHZ circuit under noise/hardware constraints.

2. `02-Quantum-Chemistry-VQE`
   - Computes molecular potential energy surfaces using exact diagonalization and VQE.
   - Studies how ansatz choice, mapping, optimizers, and chemical accuracy affect quantum chemistry workflows.

3. `03-Quantum-Kernel-SVM-MNIST`
   - Evaluates quantum kernel SVMs against classical RBF SVMs on binary MNIST.
   - Tracks kernel-target alignment, depth ablations, geometric difference proxies, expressibility, and noise.

4. `04-Optimization-QAOA-MaxCut`
   - Solves weighted Max-Cut using QAOA/RQAOA and compares against exact and strong classical baselines.
   - Frames results as a benchmark and negative-results artifact rather than a quantum-advantage claim.

5. `05-Reinforcement-Learning-Noise-Mitigation`
   - Studies measurement-defined quantum policies in reinforcement learning under ideal, noisy, and mitigated execution.
   - Compares quantum policy-gradient methods against classical RL baselines.

### Why This Problem Is Important

Quantum AI is often taught with clean textbook examples, but real research needs harder questions:

- Does a quantum model beat a strong classical baseline under the same data budget?
- Is the quantum feature space actually useful, or just different?
- What happens when hardware noise, shot noise, transpilation depth, and queue constraints enter the loop?
- Are reported improvements statistically meaningful?
- Can generated artifacts be reproduced from config, seed, and provenance?

This repo is important because it teaches research discipline:

- fair baseline comparison,
- multi-seed evaluation,
- hardware-aware design,
- explicit negative-result framing,
- statistical summaries,
- provenance tracking,
- resource-aware metrics.

### Domains Covered

- Quantum Machine Learning
- Quantum Chemistry
- Variational Quantum Algorithms
- Quantum Optimization
- Kernel Methods
- Support Vector Machines
- Reinforcement Learning
- Noise Mitigation
- Benchmark Engineering
- Experimental Design and Scientific Reporting

### Real-World Applications

The repository's examples are research-scale proxies, but they connect to real applications:

- Quantum chemistry:
  - molecular ground-state energy estimation,
  - reaction pathways,
  - materials discovery,
  - drug and catalyst design.

- Quantum kernels:
  - classification with nonclassical feature spaces,
  - small-data scientific ML,
  - kernel geometry diagnostics.

- QAOA and Max-Cut:
  - scheduling,
  - network partitioning,
  - communication channel assignment,
  - routing and clustering,
  - weighted conflict optimization.

- Quantum RL:
  - decision-making with quantum policies,
  - hardware-aware policy evaluation,
  - mitigation-aware learning systems.

- GHZ noise benchmarking:
  - entanglement stress testing,
  - decoherence visualization,
  - hardware-readiness assessment.

---

## Step 2: Concept Extraction

### Concept 1: Variational Quantum Algorithms

Simple explanation:
Variational quantum algorithms use a quantum circuit with adjustable parameters. A classical optimizer changes those parameters to minimize or maximize an objective.

Advanced explanation:
A parameterized quantum circuit prepares a state

```text
|psi(theta)> = U(theta) |0>
```

and a classical optimizer searches for parameters that optimize an expectation value:

```text
min_theta <psi(theta)| H |psi(theta)>
```

Why used here:

- VQE minimizes molecular Hamiltonian energy in Project 02.
- QAOA maximizes expected Max-Cut value in Project 04.
- Quantum RL optimizes policy parameters in Project 05.

Connections:

- Requires ansatz design.
- Requires measurement/estimation.
- Depends heavily on optimizer choice.
- Degrades under noise and finite shots.

### Concept 2: Ansatz

Simple explanation:
An ansatz is the structure of the quantum circuit used as the model.

Advanced explanation:
It defines the family of states reachable by the algorithm. A good ansatz is expressive enough to contain useful solutions but shallow enough to run on noisy hardware.

Why used here:

- Project 02 uses UCCSD and hardware-efficient ansatze.
- Project 03 uses feature maps such as ZZFeatureMap.
- Project 04 uses alternating cost/mixer QAOA layers.
- Project 05 uses RealAmplitudes or EfficientSU2 policy circuits.

Connections:

- Expressibility affects trainability.
- Depth affects noise sensitivity.
- Entanglement affects representational power.

### Concept 3: Quantum Feature Map

Simple explanation:
A feature map turns classical data into a quantum state.

Advanced explanation:
For input vector x, the circuit U_phi(x) prepares

```text
|phi(x)> = U_phi(x) |0...0>
```

Then a quantum kernel compares two inputs by state overlap:

```text
K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
```

Why used here:

- Project 01 uses quantum feature maps for Iris classification.
- Project 03 uses quantum kernels for MNIST 4-vs-9 classification.

Connections:

- Kernel SVM uses K directly.
- Kernel-target alignment measures whether K matches labels.
- Expressibility and concentration affect whether the kernel is useful.

### Concept 4: Kernel Methods

Simple explanation:
A kernel is a similarity function. SVMs can classify data using only pairwise similarities.

Advanced explanation:
Instead of learning in original input space, kernel methods implicitly use a feature map phi:

```text
K(x_i, x_j) = <phi(x_i), phi(x_j)>
```

Quantum kernels replace phi with a quantum state embedding.

Why used here:

- Project 03 asks whether a quantum kernel provides better decision geometry than a classical RBF kernel.

Connections:

- PCA reduces input dimension to number of qubits.
- KTA measures alignment with labels.
- Pegasos trains an SVM using the precomputed Gram matrix.

### Concept 5: Potential Energy Surface

Simple explanation:
A potential energy surface shows molecular energy as bond length changes.

Advanced explanation:
For each geometry R, the electronic Hamiltonian H(R) changes. The goal is to estimate

```text
E_0(R) = min_psi <psi|H(R)|psi>
```

Plotting E_0(R) gives the dissociation curve and equilibrium bond distance.

Why used here:

- Project 02 scans H2 and LiH bond lengths and compares exact energy to VQE energy.

Connections:

- Requires molecule driver.
- Requires fermion-to-qubit mapping.
- Uses chemical accuracy threshold.

### Concept 6: Fermion-to-Qubit Mapping

Simple explanation:
Molecular electrons are fermions, but quantum computers operate on qubits. A mapping converts fermionic operators into qubit Pauli operators.

Advanced explanation:
Electronic Hamiltonians contain creation/annihilation operators. Mappings such as parity, Jordan-Wigner, or Bravyi-Kitaev transform them into sums of Pauli strings:

```text
H = sum_k c_k P_k
```

where P_k is a tensor product of I, X, Y, Z.

Why used here:

- Project 02 maps molecular Hamiltonians into `SparsePauliOp` objects for exact diagonalization and VQE.

Connections:

- Active space controls problem size.
- Two-qubit reduction can lower qubit count.
- Ansatz must match the mapped problem.

### Concept 7: Max-Cut as Ising Hamiltonian

Simple explanation:
Max-Cut divides graph nodes into two groups so that high-weight edges cross the partition.

Advanced explanation:
Represent each node by spin z_i in {+1, -1}. Edge (i,j) is cut when z_i != z_j:

```text
C(z) = sum_(i,j in E) w_ij (1 - z_i z_j) / 2
```

Replacing z_i with Pauli Z_i gives the cost Hamiltonian.

Why used here:

- Project 04 uses Max-Cut as the optimization target for QAOA.

Connections:

- QAOA alternates cost and mixer unitaries.
- Approximation ratio compares QAOA to exact optimum.
- Classical baselines test whether QAOA is actually useful.

### Concept 8: Measurement-Defined Quantum Policy

Simple explanation:
A quantum policy chooses actions from measured bit probabilities.

Advanced explanation:
For state s and parameters theta:

```text
|psi(theta, s)> = U(theta, s)|0>
pi_theta(a|s) = Pr[M_action = a]
```

The action probabilities come directly from Born-rule measurement outcomes, not from classical softmax logits.

Why used here:

- Project 05 uses this as the core quantum RL policy.

Connections:

- Policy-gradient updates require gradients of probabilities.
- Parameter shift estimates gradients.
- Noise mitigation changes measured probabilities.

### Concept 9: Noise and Mitigation

Simple explanation:
Quantum hardware is noisy. Mitigation tries to reduce measurement and gate-error effects without full error correction.

Advanced explanation:
Noise changes the observed probability distribution:

```text
p_observed ~= Noise(p_ideal)
```

Mitigation approximates an inverse or extrapolated correction, often imperfectly.

Why used here:

- Project 01 shows GHZ collapse under large-circuit noise.
- Project 03 simulates noisy kernels.
- Project 04 uses noisy simulator/fake backend proxies.
- Project 05 compares ideal, noisy, and mitigated quantum policies.

Connections:

- Finite shots introduce estimator variance.
- ZNE and readout correction may reduce bias but increase cost.
- Hardware feasibility depends on depth, two-qubit gate count, and shot budget.

### Concept 10: Scientific Benchmarking

Simple explanation:
A benchmark is not just a result. It is a controlled way to compare methods.

Advanced explanation:
Good benchmarks define:

- fixed scenarios,
- fixed seeds,
- baselines,
- metrics,
- statistical tests,
- reproducible outputs,
- limitations.

Why used here:

- Projects 03, 04, and 05 are explicitly benchmark-focused.

Connections:

- Avoids overstated quantum advantage claims.
- Connects code to paper-style reporting.
- Makes negative results useful.

---

## Step 3: Algorithm Breakdown

### Algorithm 1: Classical Logistic Regression and RBF-SVM

Appears in:

- `01-Classical-vs-Quantum-Visualization/Quantum_ML_-_Iris_Classification.py`
- `03-Quantum-Kernel-SVM-MNIST/src/classical_models.py`

Intuition:

- Logistic regression learns a linear boundary.
- RBF-SVM learns a nonlinear boundary using distance-based similarity.

Step-by-step:

1. Load data.
2. Normalize/scale features.
3. Train classifier.
4. Predict labels.
5. Compare accuracy/F1 against quantum models.

Mathematical formulation:

Logistic regression:

```text
p(y=1|x) = sigmoid(w^T x + b)
```

RBF kernel:

```text
K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)
```

SVM decision function:

```text
f(x) = sign(sum_i alpha_i y_i K(x_i, x) + b)
```

Strengths:

- Strong baseline.
- Cheap compared with quantum kernels.
- RBF kernel is hard to beat on small tabular/image-feature tasks.

Limitations:

- Classical kernel may miss structure that a useful quantum feature map could expose.
- Grid search can overfit small validation sets.

### Algorithm 2: Quantum Support Vector Classifier

Appears in:

- `01-Classical-vs-Quantum-Visualization/Quantum_ML_-_Iris_Classification.py`
- `03-Quantum-Kernel-SVM-MNIST/src/quantum_training.py`
- `03-Quantum-Kernel-SVM-MNIST/src/quantum_kernel_engine.py`
- `03-Quantum-Kernel-SVM-MNIST/run_experiment.py`

Intuition:

Map data into quantum states. Classify using state similarity instead of ordinary Euclidean similarity.

Step-by-step:

1. Reduce data dimension using PCA.
2. Scale features to angle range, usually `[0, pi/2]`.
3. Build a feature map circuit such as ZZFeatureMap.
4. Compute Gram matrix:

```text
K_ij = |<phi(x_i)|phi(x_j)>|^2
```

5. Train QSVC or Pegasos QSVC.
6. Evaluate accuracy, F1, KTA, confusion matrix.

Mathematical formulation:

```text
|phi(x)> = U_phi(x)|0>
K_Q(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
```

Strengths:

- Accesses quantum Hilbert-space geometry.
- Easy to compare against classical kernels.
- Useful for studying kernel alignment and expressibility.

Limitations:

- Kernel computation scales as O(n^2).
- Small training sets limit conclusions.
- Noise can make Gram matrices non-PSD.
- Quantum advantage is not guaranteed just because the kernel is quantum.

### Algorithm 3: Kernel-Target Alignment

Appears in:

- `03-Quantum-Kernel-SVM-MNIST/src/quantum_kernel_engine.py`
- `03-Quantum-Kernel-SVM-MNIST/src/kernel_learning.py`
- `03-Quantum-Kernel-SVM-MNIST/run_experiment.py`

Intuition:

A good kernel says same-class points are similar and different-class points are dissimilar.

Step-by-step:

1. Build kernel matrix K.
2. Build label kernel yy^T.
3. Compute normalized Frobenius alignment.

Mathematical formulation:

```text
KTA(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F)
```

Strengths:

- Gives diagnostic signal beyond accuracy.
- Useful before full classifier training.
- Lets quantum and classical kernels be compared fairly.

Limitations:

- High KTA does not always guarantee best generalization.
- Sensitive to labels, scaling, centering, and class balance.

### Algorithm 4: Pegasos QSVC

Appears in:

- `03-Quantum-Kernel-SVM-MNIST/src/quantum_training.py`
- `03-Quantum-Kernel-SVM-MNIST/run_experiment.py`

Intuition:

Pegasos trains a support vector classifier with stochastic gradient steps instead of solving a full quadratic program.

Step-by-step:

1. Precompute kernel matrix.
2. Initialize model.
3. Repeatedly sample mini-batches.
4. Update the primal/dual representation using subgradients.
5. Predict using kernel similarities.

Mathematical formulation:

```text
min_w (lambda / 2)||w||^2 + (1/n) sum_i max(0, 1 - y_i <w, phi(x_i)>)
eta_t = 1 / (lambda t)
```

Strengths:

- More scalable than exact SVM for larger n.
- Works with precomputed quantum kernels.

Limitations:

- Sensitive to `lambda_param`.
- Can collapse to one-class predictions if regularization/training is poor.
- Still depends on expensive kernel matrix construction.

### Algorithm 5: VQE

Appears in:

- `02-Quantum-Chemistry-VQE/src/vqe_engine.py`
- `02-Quantum-Chemistry-VQE/src/pes_generator.py`

Intuition:

Guess a quantum state, measure its energy, and adjust the circuit until energy is minimized.

Step-by-step:

1. Build molecular electronic structure problem.
2. Convert fermionic Hamiltonian to qubit Hamiltonian.
3. Choose ansatz.
4. Pick initial parameters.
5. Estimate energy:

```text
E(theta) = <psi(theta)|H|psi(theta)>
```

6. Classical optimizer updates theta.
7. Repeat until convergence.
8. Compare VQE energy to exact energy.

Mathematical formulation:

```text
E_0 <= E(theta) = <0|U^dagger(theta) H U(theta)|0>
theta* = argmin_theta E(theta)
```

Strengths:

- Natural fit for quantum chemistry.
- Hybrid design works on near-term devices.
- Variational principle gives an upper bound in ideal conditions.

Limitations:

- Optimizer can get stuck.
- Ansatz may not express the ground state.
- Noise biases energy estimates.
- Barren plateaus can make gradients vanish.

### Algorithm 6: Exact Diagonalization

Appears in:

- `02-Quantum-Chemistry-VQE/src/classical_solver.py`
- `04-Optimization-QAOA-MaxCut/src/classical_solver.py`

Intuition:

For small enough systems, solve the problem exactly and use it as ground truth.

Step-by-step for chemistry:

1. Build Hamiltonian matrix.
2. Diagonalize it.
3. Take smallest eigenvalue.

Step-by-step for Max-Cut:

1. Enumerate all bitstrings for small graph.
2. Compute cut value for each.
3. Return best.

Mathematical formulation:

```text
H |psi_k> = E_k |psi_k>
E_0 = min_k E_k
```

Strengths:

- Gives reliable reference.
- Essential for measuring approximation error.

Limitations:

- Exponential scaling.
- Only practical for small molecules/graphs.

### Algorithm 7: QAOA

Appears in:

- `04-Optimization-QAOA-MaxCut/src/qaoa_circuit.py`
- `04-Optimization-QAOA-MaxCut/src/qaoa_optimizer.py`
- `04-Optimization-QAOA-MaxCut/src/runtime_executor.py`

Intuition:

QAOA alternates between applying the problem objective and mixing between bitstrings. A classical optimizer tunes the angles.

Step-by-step:

1. Start in equal superposition:

```text
|+>^n
```

2. Apply cost unitary for angle gamma:

```text
U_C(gamma) = exp(-i gamma H_C)
```

3. Apply mixer unitary for angle beta:

```text
U_B(beta) = exp(-i beta sum_i X_i)
```

4. Repeat for p layers.
5. Estimate expected cut value.
6. Optimize gamma and beta.
7. Decode measurement distribution.

Mathematical formulation:

```text
|psi(gamma, beta)> =
prod_l exp(-i beta_l H_B) exp(-i gamma_l H_C) |+>^n
J(theta) = E[C] = <psi(theta)|H_C|psi(theta)>
```

This repo minimizes:

```text
L(theta) = -J(theta)
```

Strengths:

- Problem-aware circuit.
- Low depth for small p.
- Natural benchmark for NISQ optimization.

Limitations:

- Hard classical optimization landscape.
- No guarantee of outperforming classical heuristics.
- Small p can underperform.
- Noise hurts depth scaling.

### Algorithm 8: CVaR-QAOA Objective

Appears in:

- `04-Optimization-QAOA-MaxCut/src/qaoa_optimizer.py`
- `04-Optimization-QAOA-MaxCut/docs/mathematical_formulation.md`

Intuition:

Instead of optimizing average cut value, focus on the best tail of measured bitstrings.

Step-by-step:

1. Get probability distribution over bitstrings.
2. Sort bitstrings by cut value.
3. Keep top alpha probability mass.
4. Average cut value over that tail.
5. Maximize this risk-sensitive objective.

Mathematical formulation:

```text
J_CVaR(theta; alpha) = (1/alpha) sum_x q_theta(x) C(x)
```

where q keeps only the best alpha mass.

Strengths:

- Encourages high-quality samples.
- Useful when best samples matter more than expectation.

Limitations:

- Requires measurement distribution.
- Not always supported by estimator-only hardware paths.
- Can increase variance.

### Algorithm 9: RQAOA

Appears in:

- `04-Optimization-QAOA-MaxCut/src/rqaoa_engine.py`

Intuition:

Run QAOA, infer strong variable correlations, fix or merge variables, and reduce the graph.

Step-by-step:

1. Run QAOA on current graph.
2. Estimate pair correlations.
3. Select strongest correlation.
4. Impose relation between two variables.
5. Reduce graph and update weights/constants.
6. Repeat until small enough.
7. Solve remaining problem exactly or heuristically.

Mathematical formulation:

Correlations often use:

```text
<Z_i Z_j>
```

Strong positive or negative correlations suggest same-side or opposite-side assignments.

Strengths:

- Can reduce problem size.
- Connects variational outputs to classical preprocessing.

Limitations:

- Wrong reductions can lock in bad decisions.
- Correlation estimates are noisy under finite shots.
- Implementation is benchmark-oriented, not a new RQAOA theory claim.

### Algorithm 10: REINFORCE

Appears in:

- `05-Reinforcement-Learning-Noise-Mitigation/agent/reinforce_learner.py`
- `05-Reinforcement-Learning-Noise-Mitigation/src/baselines.py`

Intuition:

Actions that led to better returns become more likely.

Step-by-step:

1. Roll out policy in environment.
2. Collect states, actions, rewards.
3. Compute discounted returns.
4. Estimate policy gradient:

```text
grad J(theta) ~= sum_t G_t grad log pi_theta(a_t|s_t)
```

5. Update policy parameters.

Mathematical formulation:

```text
G_t = sum_{k=t}^T gamma^(k-t) r_k
L(theta) = -sum_t A_t log pi_theta(a_t|s_t) - beta H(pi)
```

Strengths:

- Simple and general.
- Works with stochastic policies.

Limitations:

- High variance.
- Sensitive to sparse rewards.
- Quantum gradients add shot/noise cost.

### Algorithm 11: Quantum Actor-Critic with GAE

Appears in:

- `05-Reinforcement-Learning-Noise-Mitigation/agent/actor_critic_learner.py`
- `05-Reinforcement-Learning-Noise-Mitigation/src/rl_utils.py`

Intuition:

The actor chooses actions. The critic estimates how good states are. The critic reduces gradient variance.

Step-by-step:

1. Roll out quantum actor in environment.
2. Critic predicts V(s_t).
3. Compute temporal difference residual:

```text
delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
```

4. Compute GAE:

```text
A_t = sum_l (gamma lambda)^l delta_{t+l}
```

5. Use parameter shift to estimate actor gradient.
6. Train critic on return targets.
7. Update both with Adam.

Strengths:

- Lower variance than REINFORCE.
- Better benchmark method than simple policy gradient.
- Separates quantum actor from classical value estimation.

Limitations:

- More moving parts.
- Critic errors bias actor updates.
- Quantum action probability estimates are costly.

### Algorithm 12: Parameter-Shift Gradient Estimation

Appears in:

- `05-Reinforcement-Learning-Noise-Mitigation/agent/gradient_estimator.py`
- `05-Reinforcement-Learning-Noise-Mitigation/docs/technical_note.md`

Intuition:

For certain gates, the derivative can be computed by evaluating the circuit at shifted parameter values.

Mathematical formulation:

```text
d pi_theta(a|s) / d theta_i =
  [pi_{theta_i + pi/2}(a|s) - pi_{theta_i - pi/2}(a|s)] / 2
```

Strengths:

- Exact for ideal compatible gates.
- Hardware-friendly because it uses circuit evaluations.

Limitations:

- Requires two evaluations per parameter.
- Under finite shots/noise/mitigation, estimator becomes stochastic and may be biased.

### Algorithm 13: Zero-Noise Extrapolation and Readout Mitigation

Appears in:

- `05-Reinforcement-Learning-Noise-Mitigation/src/mitigation_engine.py`
- `05-Reinforcement-Learning-Noise-Mitigation/src/noise_models.py`

Intuition:

Run noisier versions of a circuit, observe a trend, and extrapolate back toward zero noise. Also correct measured bit flips using a confusion matrix.

Step-by-step:

1. Fold circuit to increase effective noise scale.
2. Measure probabilities at multiple scales.
3. Fit/extrapolate probability trend to zero-noise intercept.
4. Apply readout correction to action-register probabilities.

Strengths:

- Lightweight and usable on NISQ devices.
- Helps study mitigation trends.

Limitations:

- Not true error correction.
- Adds shot cost.
- Can amplify variance or produce biased probabilities.

---

## Step 4: Code-to-Concept Mapping

### Root Files

| File | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `README.md` | Maps the five projects and execution order | Project architecture | Treats repo as a research series, not a monolith |
| `requirements.txt` | Shared dependency list | Reproducibility | Gives common environment starting point |
| `scripts/setup_ibm_runtime.py` | IBM Runtime account setup | Hardware execution | Keeps credentials out of source code |
| `report.md` | Top-level generated/report-style material | Scientific reporting | Collects outcomes in reusable prose |

### Project 01: Classical vs Quantum Visualization

| File | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `Quantum_ML_-_Iris_Classification.py` | Runs Iris classification with classical and quantum methods | QSVC, classical baselines | Small dataset makes decision boundaries visible |
| `compare_ghz_three_way.py` | Compares ideal local, noisy simulation, and optional real GHZ-127 execution | GHZ state, noise, hardware queue policy | Separates hardware stress test from ML performance |
| `Hardware_Noise_&_Decoherence_Benchmark.py` | Compatibility entrypoint for GHZ benchmark | Decoherence benchmark | Preserves legacy workflow |
| `assets/qml_iris_report.json` | Stores Iris metrics | Reproducible metrics | Makes claims inspectable |
| `assets/three_way_ghz127_comparison.json` | Stores GHZ benchmark result | Hardware/noise evidence | Captures circuit depth and subspace probability |

### Project 02: Quantum Chemistry VQE

| File/Function | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `src/pes_generator.py::PESGenerator` | Orchestrates bond-length sweeps | Potential energy surface | Separates workflow from chemistry primitives |
| `src/molecule_driver.py::get_molecule_problem` | Builds H2/LiH electronic problem, with fallback | Electronic structure setup | Keeps PySCF dependency and fallback provenance explicit |
| `src/problem_builder.py::build_mapped_hamiltonian` | Maps molecular Hamiltonian to qubit operator | Fermion-to-qubit mapping | Centralizes mapper choice and qubit-reduction metadata |
| `src/classical_solver.py::get_exact_energy_from_qubit_operator` | Computes exact reference energy | Exact diagonalization | Needed to measure VQE error |
| `src/ansatz_factory.py::get_ansatz` | Builds UCCSD/EfficientSU2 ansatze | Ansatz design | Lets experiments compare physically motivated and hardware-efficient circuits |
| `src/vqe_engine.py::VQEEngine` | Runs VQE and captures optimizer history | Variational optimization | Wraps Qiskit VQE into serializable research records |
| `src/runtime_executor.py::get_estimator` | Chooses local estimator or IBM Runtime estimator | Backend abstraction | Same pipeline can target local or hardware |
| `src/config_schema.py` | Validates YAML config | Reproducibility and safety | Bad configs fail early |
| `src/plotting.py` | Plots PES, error, convergence | Scientific visualization | Turns raw energy lists into interpretable evidence |

Important code-level note:

In the current `src/pes_generator.py`, the call to `run_vqe_qubit_with_retry` appears to use keyword names that do not match `src/vqe_engine.py`:

```text
called with: qubit_op=..., max_restarts=...
signature:   qubit_operator=..., n_restarts=...
```

This is likely a runtime bug unless there is another local change not reflected in the inspected file. It is a good example of why unit tests around orchestration code matter.

### Project 03: Quantum Kernel SVM MNIST

| File/Function | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `run_experiment.py::run_single_trial` | Executes one seed/dim/reps experiment | Controlled ablation cell | Makes fair classical-vs-quantum comparison repeatable |
| `src/data_loader.py::load_mnist_digits` | Loads OpenML MNIST or sklearn fallback | Dataset control | Allows real runs and offline smoke tests |
| `src/preprocessing.py::preprocess_data` | Normalizes, standardizes, PCA-reduces, angle-scales data | Quantum-compatible preprocessing | Number of PCA components equals qubits |
| `src/feature_map_registry.py` | Builds named feature maps | Feature-map experimentation | Supports ZZ, Z, Pauli, IQP, HEA variants |
| `src/quantum_feature_maps.py::create_feature_map` | Constructs Qiskit feature map circuits | Quantum encoding | Hides Qiskit API details |
| `src/quantum_kernel_engine.py::create_quantum_kernel` | Creates FidelityQuantumKernel | Quantum kernel | Main bridge from circuit to SVM |
| `src/quantum_kernel_engine.py::compute_kernel_target_alignment` | Computes KTA | Kernel diagnostics | Measures whether a kernel matches labels |
| `src/classical_models.py::train_classical_svm` | Trains RBF SVM | Classical baseline | Strong baseline prevents weak quantum claims |
| `src/quantum_training.py::train_qsvc` | Trains exact QSVC | Quantum SVM | Direct quantum-kernel classifier |
| `src/quantum_training.py::train_pegasos_qsvc` | Trains Pegasos QSVC | Stochastic SVM optimization | More scalable alternative |
| `src/noise_simulation.py` | Builds noisy kernel comparison | NISQ robustness | Tests kernel degradation under readout/gate error |
| `src/expressibility.py` | Estimates feature-map expressibility | Circuit diagnostics | Connects circuit capacity to trainability |
| `src/kernel_learning.py` | Trains kernel alignment | QKAT | Tries to improve KTA by tuning kernel parameters |
| `src/visualization.py` | Generates dashboards and plots | Research communication | Turns ablations into publication-style figures |

### Project 04: QAOA Max-Cut

| File/Function | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `src/graph_generator.py::GraphGenerator` | Builds graph families including communication mesh | Problem generation | Enables controlled synthetic and domain-proxy benchmarks |
| `src/hamiltonian_builder.py::HamiltonianBuilder` | Converts graph to Max-Cut Hamiltonian plus offset | Ising/QUBO mapping | Separates constant offset from ZZ interaction |
| `src/qaoa_circuit.py::QAOACircuitBuilder` | Builds QAOA circuits | Alternating operator ansatz | Keeps circuit construction separate from optimization |
| `src/qaoa_optimizer.py::MaxCutQAOAProblem` | Owns graph, Hamiltonian, circuit, executor, objective | End-to-end QAOA problem | Keeps objective and decoding aligned |
| `src/qaoa_optimizer.py::QAOAOptimizer` | Runs COBYLA/SPSA/Nelder-Mead/L-BFGS-B | Classical outer-loop optimization | Supports optimizer comparisons and diagnostics |
| `src/runtime_executor.py::RuntimeExecutor` | Runs local/noisy/hardware-like execution | Backend execution | Separates objective estimation from algorithm logic |
| `src/classical_solver.py::ClassicalSolver` | Exact and heuristic baselines | Baseline comparison | Tests QAOA against nontrivial classical methods |
| `src/rqaoa_engine.py::RQAOAEngine` | Reduces graph using correlations | Recursive QAOA | Studies reduction-based hybrid strategy |
| `src/experimental_study.py` | Runs held-out tuning/evaluation | Experimental design | Prevents overfitting hyperparameters to test instances |
| `src/evaluation_metrics.py` | Computes approximation ratios, CIs, corrections | Benchmark statistics | Makes results interpretable and honest |
| `src/results_review.py` | Produces scientific verdict | Research quality control | Flags weak/misleading evidence |
| `src/artifact_pipeline.py` | Generates all artifacts | Reproducibility | One entrypoint for checked-in results |
| `src/provenance.py` | Captures config hash, git state, packages | Provenance | Makes runs traceable |

### Project 05: Reinforcement Learning Noise Mitigation

| File/Function | What It Does | Concept Implemented | Why Designed This Way |
|---|---|---|---|
| `environments/simple_nav_env.py::KeyDoorNavigationEnv` | Key-and-door navigation task | RL environment | Small enough for quantum policy experiments |
| `agent/quantum_policy.py::QuantumPolicyNetwork` | Builds measurement-defined quantum actor | Quantum policy | Action probabilities come from measurements |
| `agent/gradient_estimator.py::ParameterShiftGradientEstimator` | Estimates policy gradients | Parameter shift | Hardware-compatible gradient method |
| `agent/reinforce_learner.py::ReinforceLearner` | Trains quantum REINFORCE | Policy gradient | Simple baseline quantum learner |
| `agent/actor_critic_learner.py::QuantumActorCriticLearner` | Trains quantum actor plus classical critic | Actor-critic and GAE | Lower-variance upgraded quantum method |
| `src/baselines.py` | Tabular and MLP RL baselines | Classical RL comparison | Prevents weak quantum-only claims |
| `src/mitigation_engine.py` | Readout correction and ZNE-style mitigation | Noise mitigation | Studies ideal/noisy/mitigated gaps |
| `src/noise_models.py` | Loads or builds noise models | NISQ simulation | Supports fake-backend and custom noise experiments |
| `src/training_pipeline.py` | Runs training and saves summaries | Experiment orchestration | Centralizes mode comparisons and metrics |
| `src/benchmark_suite.py` | Runs fixed scenario suite | Benchmarking | Eight scenarios and fixed seeds |
| `src/research_stats.py` | Bootstrap CIs, Holm correction, paired tests | Statistical analysis | Makes claims more rigorous |
| `reporting/paper_report.py` | Builds paper-style report bundle | Scientific reporting | Converts results into reusable research artifacts |

---

## Step 5: Mathematical Understanding

### Quantum State Preparation

Most quantum models in the repo follow:

```text
|psi(theta, x)> = U(theta, x)|0...0>
```

where:

- x may be molecular geometry, graph parameters, image features, or RL state.
- theta are trainable parameters.
- U is the circuit.

### Born Rule

Quantum measurement probabilities are:

```text
p(bitstring b) = |<b|psi>|^2
```

In Project 05:

```text
pi_theta(a|s) = Pr[M_action = a]
```

This is the policy itself.

### VQE Energy

Chemistry Hamiltonian:

```text
H = sum_k c_k P_k
```

where P_k are Pauli strings.

Energy objective:

```text
E(theta) = <psi(theta)|H|psi(theta)>
```

Optimization:

```text
theta* = argmin_theta E(theta)
```

Chemical accuracy:

```text
|E_VQE - E_exact| <= 0.0016 Hartree
```

Intuition:
VQE searches for the lowest-energy state allowed by the ansatz. If the ansatz cannot represent the ground state, optimization cannot fix that.

### Potential Energy Surface

For bond length R:

```text
E_0(R) = min_psi <psi|H(R)|psi>
```

The PES curve is:

```text
R -> E_0(R)
```

Intuition:
The minimum of the curve approximates equilibrium bond length.

### Quantum Kernel

Feature map:

```text
|phi(x)> = U_phi(x)|0>
```

Kernel:

```text
K_Q(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
```

Intuition:
If two inputs produce similar quantum states, they have high kernel similarity.

### ZZ Feature Map

Approximate structure:

```text
U_phi(x) = exp(i sum_i x_i Z_i) exp(i sum_{i<j} x_i x_j Z_i Z_j)
```

Intuition:
Single-feature terms encode each component. Pairwise ZZ terms encode feature interactions through entanglement.

### Kernel-Target Alignment

Label kernel:

```text
Y = y y^T
```

Alignment:

```text
KTA(K, y) = <K, Y>_F / (||K||_F ||Y||_F)
```

Intuition:
KTA is high when the kernel makes same-class examples similar and opposite-class examples dissimilar.

### Centered KTA

Kernel centering:

```text
H = I - (1/n) 11^T
K_c = H K H
Y_c = H Y H
```

Then:

```text
cKTA = <K_c, Y_c>_F / (||K_c||_F ||Y_c||_F)
```

Intuition:
Centering removes global similarity bias.

### Expressibility

The repo estimates how close a circuit's fidelity distribution is to the Haar-random distribution:

```text
epsilon = KL(P_PQC(F) || P_Haar(F))
```

where:

```text
F = |<psi(theta_1)|psi(theta_2)>|^2
```

Intuition:
Very expressive circuits can behave like random unitaries, which may hurt trainability and create concentration.

### Max-Cut Objective

Binary spin form:

```text
C(z) = sum_(i,j in E) w_ij (1 - z_i z_j) / 2
```

Quantum Hamiltonian:

```text
H_C = sum_(i,j in E) w_ij (I - Z_i Z_j) / 2
```

Offset decomposition used in Project 04:

```text
H_C = offset + H_ZZ
offset = sum_(i,j) w_ij / 2
H_ZZ = -sum_(i,j) w_ij Z_i Z_j / 2
```

Expected cut:

```text
J(theta) = offset + <H_ZZ>_theta
```

Loss minimized by optimizer:

```text
L(theta) = -J(theta)
```

### QAOA State

For p layers:

```text
|psi(gamma, beta)> =
prod_{l=1}^p exp(-i beta_l H_B) exp(-i gamma_l H_C) |+>^n
```

Mixer:

```text
H_B = sum_i X_i
```

Intuition:
Cost unitary adds phase based on solution quality. Mixer moves probability mass between candidate bitstrings.

### Approximation Ratio

```text
ratio = method_cut / exact_cut
```

Intuition:
Ratio 1.0 means optimal. A ratio below strong classical heuristics is evidence against practical advantage.

### RL Returns

Discounted return:

```text
G_t = sum_{k=t}^T gamma^(k-t) r_k
```

Advantage with baseline:

```text
A_t = G_t - b_t
```

Policy-gradient loss:

```text
L_pg(theta) = -sum_t A_t log pi_theta(a_t|s_t) - beta sum_t H(pi_theta(.|s_t))
```

### GAE

Temporal-difference residual:

```text
delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
```

Generalized advantage:

```text
A_t = sum_l (gamma lambda)^l delta_{t+l}
```

Intuition:
GAE trades bias and variance using lambda.

### Parameter Shift

For compatible gates:

```text
d f(theta) / d theta_i =
  [f(theta_i + pi/2) - f(theta_i - pi/2)] / 2
```

In policy terms:

```text
d pi_theta(a|s) / d theta_i =
  [pi_{theta_i + pi/2}(a|s) - pi_{theta_i - pi/2}(a|s)] / 2
```

Intuition:
Gradients are measured by running shifted circuits.

---

## Step 6: Study Notes

### Quick Revision Notes

- The repo is a five-part quantum AI research portfolio, not a single app.
- Project 01 teaches visual comparison and hardware noise reality.
- Project 02 teaches VQE for molecular PES generation.
- Project 03 teaches quantum kernels, fair SVM baselines, KTA, and ablations.
- Project 04 teaches Max-Cut, QAOA, RQAOA, approximation ratios, and negative-result benchmarking.
- Project 05 teaches measurement-defined quantum policies, policy gradients, actor-critic, and mitigation.
- The recurring pattern is: define problem -> encode into quantum object -> optimize/evaluate -> compare against classical baselines -> report limitations.
- Quantum advantage is not claimed. The repo repeatedly emphasizes honest benchmarking.
- Noise, finite shots, circuit depth, and small sample sizes are central constraints.
- Strong baselines are essential. A weak classical baseline makes quantum results meaningless.

### Detailed Notes

#### 1. Why Quantum AI Is Hard

Quantum systems live in Hilbert spaces whose dimension grows exponentially with qubits. This is the source of potential power, but also the source of difficulty:

- State simulation becomes expensive.
- Measurement gives samples, not direct full state access.
- Noise accumulates with circuit depth.
- Optimization landscapes can be flat, noisy, or nonconvex.

This repo teaches that quantum AI must be evaluated as a complete system, not just as a circuit diagram.

#### 2. The Classical Baseline Principle

Every quantum claim should ask:

```text
Did the quantum method beat a strong classical method under the same budget?
```

Examples:

- Project 03 uses classical RBF SVM on the same n samples as QSVC.
- Project 04 compares QAOA against exact, greedy, local search, Goemans-Williamson, random, and budget-matched hill climbing.
- Project 05 compares quantum RL against tabular and MLP baselines.

#### 3. Why PCA Appears in Quantum ML

Quantum feature maps usually require one qubit per feature. Raw MNIST has 784 pixels, which is not feasible. PCA reduces dimensionality:

```text
784 pixels -> d principal components -> d qubits
```

But PCA also changes the task. The quantum model is not operating on raw MNIST; it is operating on compressed features.

#### 4. Why KTA Matters

Accuracy can be noisy and sample-dependent. KTA asks a deeper question:

```text
Does the kernel geometry match the label structure?
```

If quantum KTA is much lower than classical RBF KTA, quantum SVM performance is likely weak.

#### 5. Why VQE Needs Exact Energy

Without exact or high-quality reference energy, a VQE result is hard to interpret. Project 02 compares:

```text
delta = |E_VQE - E_exact|
```

and checks whether delta is below chemical accuracy.

#### 6. Why QAOA Expected Value and Sampled Bitstrings Are Separate

Project 04 correctly distinguishes:

- expected cut value: what QAOA optimizes,
- most likely sampled bitstring: representative output,
- best sampled bitstring: optimistic artifact.

This prevents misleading reporting where a lucky sample looks better than the optimized distribution.

#### 7. Why Quantum RL Uses Measurement-Native Policies

Some quantum RL demos measure expectation values and feed them into a classical softmax. This repo instead defines:

```text
policy = measured action probability
```

That makes the policy genuinely quantum-measurement-native, but it also makes gradients and mitigation more delicate.

### Key Formulas

```text
VQE energy:
E(theta) = <psi(theta)|H|psi(theta)>
```

```text
Chemical accuracy:
|E_VQE - E_exact| <= 0.0016 Hartree
```

```text
Quantum kernel:
K_Q(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
```

```text
RBF kernel:
K_RBF(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)
```

```text
KTA:
KTA(K, y) = <K, yy^T>_F / (||K||_F ||yy^T||_F)
```

```text
Max-Cut:
C(z) = sum_(i,j) w_ij (1 - z_i z_j) / 2
```

```text
QAOA objective:
J(theta) = offset + <H_ZZ>_theta
L(theta) = -J(theta)
```

```text
Approximation ratio:
ratio = method_cut / exact_cut
```

```text
Policy:
pi_theta(a|s) = Pr[M_action = a]
```

```text
Discounted return:
G_t = sum_{k=t}^T gamma^(k-t) r_k
```

```text
GAE:
A_t = sum_l (gamma lambda)^l delta_{t+l}
```

```text
Parameter shift:
d f / d theta_i = [f(theta_i + pi/2) - f(theta_i - pi/2)] / 2
```

### Important Concepts

- NISQ: noisy intermediate-scale quantum devices.
- Ansatz: parameterized circuit family.
- Feature map: data-to-quantum-state encoding.
- Kernel matrix: pairwise similarity matrix.
- PSD regularization: fixing invalid noisy kernels.
- Expressibility: how broadly a circuit explores Hilbert space.
- Barren plateau: vanishing-gradient problem.
- Chemical accuracy: practical quantum chemistry error threshold.
- Ising model: spin representation of binary optimization.
- CVaR: tail-focused risk-sensitive objective.
- GAE: advantage estimator balancing bias and variance.
- ZNE: zero-noise extrapolation.
- Readout mitigation: correction for measurement bit-flip errors.
- Provenance: metadata needed to reproduce a run.

### Common Mistakes

- Treating a quantum demo as evidence of quantum advantage.
- Comparing quantum models against weak classical baselines.
- Forgetting that PCA changes the input problem.
- Reporting best sampled QAOA bitstring as if it were the optimized expected value.
- Ignoring finite-shot uncertainty.
- Using too many circuit layers and then blaming the optimizer rather than noise/depth.
- Assuming mitigation is equivalent to error correction.
- Interpreting 3-seed statistics as publishable proof.
- Forgetting to include Hamiltonian constants in chemistry energy.
- Confusing molecular electronic energy with total molecular energy.
- Treating KTA as a guarantee of accuracy rather than a diagnostic.
- Ignoring whether a kernel matrix is positive semidefinite after noise.
- Using real hardware without queue, timeout, and fallback policy.

---

## Step 7: Learning Path

### Beginner Stage

Goal: understand the vocabulary and run small examples.

Learn first:

1. Linear algebra:
   - vectors,
   - matrices,
   - eigenvalues,
   - inner products,
   - tensor products.

2. Classical ML basics:
   - logistic regression,
   - SVM,
   - kernels,
   - train/test split,
   - accuracy and F1.

3. Quantum basics:
   - qubits,
   - superposition,
   - measurement,
   - Pauli gates,
   - parameterized circuits.

Practice:

- Run Project 01 Iris classification.
- Inspect `assets/qml_iris_report.json`.
- Change quantum feature-map grid size and compare runtime.
- Run GHZ benchmark with `--skip-real`.

Build next:

- Add one more classical baseline to Project 01.
- Plot how QSVC accuracy changes with feature-map reps.

### Intermediate Stage

Goal: understand hybrid quantum algorithms and fair experiments.

Learn:

1. VQE:
   - Hamiltonians,
   - ansatz,
   - variational principle,
   - classical optimizers.

2. Quantum kernels:
   - fidelity,
   - Gram matrices,
   - KTA,
   - PSD regularization.

3. QAOA:
   - Ising mapping,
   - cost Hamiltonian,
   - mixer Hamiltonian,
   - approximation ratio.

Practice:

- Run Project 02 for H2 with local mode.
- Compare UCCSD vs EfficientSU2.
- Run Project 03 in fallback mode.
- Run Project 04 integration test and inspect `results/metrics.csv`.

Build next:

- Add a new feature map to Project 03 through `feature_map_registry.py`.
- Add a new graph family to Project 04.
- Add a test ensuring QAOA's expected cut is not confused with best sampled cut.

### Advanced Stage

Goal: reason like a researcher.

Learn:

1. Statistical experiment design:
   - multi-seed evaluation,
   - paired tests,
   - bootstrap CIs,
   - Holm/Bonferroni correction.

2. Noise-aware quantum computing:
   - readout error,
   - depolarizing error,
   - transpilation,
   - shot budgets,
   - mitigation limits.

3. Quantum RL:
   - policy gradients,
   - actor-critic,
   - GAE,
   - parameter-shift estimators.

Practice:

- Extend Project 03 to 10 seeds and compare significance.
- Extend Project 04 held-out study to larger graphs where exact solution becomes expensive.
- Run Project 05 smoke benchmark and inspect ideal/noisy/mitigated gaps.

Build next:

- Add live IBM Runtime evaluation slices with strict provenance.
- Add analytic gradient or parameter-shift QAOA optimizer.
- Add a stronger mitigation ablation in Project 05.
- Write a paper-style negative-results report using repo artifacts.

---

## Step 8: Questions

### Interview Questions

1. What is the difference between a variational quantum algorithm and a classical neural network?
2. Why does VQE use the variational principle?
3. What is chemical accuracy and why is it important?
4. Why do quantum kernel methods need a feature map?
5. What does kernel-target alignment measure?
6. Why is an RBF SVM a strong baseline for quantum kernel experiments?
7. What is the Max-Cut objective in Ising form?
8. Why does QAOA alternate cost and mixer unitaries?
9. What is the difference between expected cut value and best sampled cut?
10. Why can QAOA underperform budget-matched hill climbing?
11. What is a measurement-defined quantum policy?
12. How does parameter shift estimate quantum gradients?
13. Why is noise mitigation not the same as error correction?
14. What makes a quantum AI benchmark scientifically credible?
15. Why are multi-seed experiments important?

### Conceptual Questions

1. If a quantum kernel has low KTA but high accuracy on one split, how would you interpret it?
2. Why might increasing feature-map depth reduce performance?
3. What happens if a noisy quantum kernel is not positive semidefinite?
4. Why does Project 03 cap classical and quantum training samples to the same n?
5. Why is exact diagonalization useful even though it does not scale?
6. How does active-space selection change a chemistry problem?
7. How do Hamiltonian constants affect reported VQE energy?
8. Why does Project 04 use held-out tuning and evaluation seeds?
9. Why can the most likely QAOA bitstring be worse than the best sampled bitstring?
10. In Project 05, why is the critic classical instead of quantum?
11. How can readout mitigation improve probabilities but still bias gradients?
12. Why should quantum advantage claims require strong classical baselines?

### Deep Research Questions

1. Under what data distributions can a quantum kernel outperform a classical RBF kernel?
2. How should geometric difference be computed and interpreted when kernel matrices are noisy?
3. Can kernel alignment training improve generalization, or does it overfit small datasets?
4. How does feature-map expressibility trade off with concentration and trainability?
5. Which ansatz families give chemistry accuracy at minimum circuit depth?
6. Can adaptive ansatz construction outperform fixed UCCSD or EfficientSU2 in this repo?
7. What QAOA parameter initialization strategy improves held-out performance?
8. When does CVaR-QAOA outperform expectation-based QAOA?
9. Can RQAOA reductions remain reliable under realistic shot noise?
10. How should quantum RL account for mitigation-induced bias in policy gradients?
11. What is the fairest hardware budget metric for comparing quantum and classical RL?
12. Can frozen quantum policies trained in simulation transfer to hardware evaluation slices?

---

## Step 9: Weaknesses and Missing Knowledge Areas

### Conceptual Gaps

1. Quantum advantage is mostly diagnostic, not demonstrated.
   - The repo is honest about this, especially in Projects 03, 04, and 05.
   - Study next: quantum advantage criteria, classical simulability, kernel lower bounds.

2. Hardware execution is mostly proxy-based.
   - Fake backends and noise models are useful, but live hardware behavior can differ.
   - Study next: IBM Runtime primitives, calibration data, transpilation, queue-aware experiments.

3. Some domain mappings are proxies.
   - The Project 04 communication mesh is a weighted graph proxy, not a full robotics communication model.
   - Study next: domain-specific optimization modeling.

4. Quantum RL benchmark is small.
   - The key-and-door task is useful for controlled experiments but not a large RL benchmark.
   - Study next: policy-gradient benchmarks, POMDPs, sample efficiency.

### Inefficient or Risky Design Choices

1. Kernel computation scales as O(n^2).
   - Project 03 becomes expensive as sample count grows.
   - Possible improvement: approximate kernels, Nyström methods, batching/caching.

2. Project 03 geometric difference appears to use a KTA-ratio proxy in `run_experiment.py`.
   - The true geometric difference requires kernel matrices, not scalar KTA means.
   - Possible improvement: persist K_Q and K_C or recompute them for exact g(K_Q, K_C).

3. Project 02 has a likely keyword mismatch in VQE retry orchestration.
   - `PESGenerator.run` appears to call `run_vqe_qubit_with_retry` with `qubit_op` and `max_restarts`.
   - `VQEEngine.run_vqe_qubit_with_retry` expects `qubit_operator` and `n_restarts`.
   - Possible improvement: add unit test around PES VQE retry path and fix keywords.

4. Several projects rely on optional heavy dependencies.
   - PySCF, Qiskit Aer, Qiskit Machine Learning, OpenML, and Runtime availability affect reproducibility.
   - Possible improvement: clearer environment lockfiles per module.

5. Notebook workflows are secondary to scripts.
   - This is good for reproducibility but can confuse learners who expect notebooks to be source of truth.
   - Possible improvement: add "script is canonical" note inside each notebook.

### Missing Knowledge Areas to Study Next

- Quantum information:
  - density matrices,
  - decoherence channels,
  - trace distance,
  - fidelity,
  - entanglement entropy.

- Quantum chemistry:
  - second quantization,
  - Hartree-Fock,
  - configuration interaction,
  - coupled cluster,
  - active-space methods.

- Quantum ML:
  - quantum kernels,
  - barren plateaus,
  - trainability,
  - data re-uploading,
  - expressibility.

- Optimization:
  - QUBO,
  - Ising models,
  - semidefinite programming,
  - Goemans-Williamson,
  - stochastic local search.

- Reinforcement learning:
  - REINFORCE,
  - actor-critic,
  - GAE,
  - off-policy evaluation,
  - exploration under sparse rewards.

- Statistics:
  - paired tests,
  - permutation tests,
  - bootstrap confidence intervals,
  - multiple-testing correction,
  - effect sizes.

---

## Step 10: Personal Knowledge Base Format

### Mental Model of the Whole Repo

```text
Quantum AI Research Series
|
|-- Project 01: See the contrast
|   |-- classical vs quantum Iris classification
|   |-- GHZ hardware/noise stress test
|
|-- Project 02: Estimate physics
|   |-- molecule -> qubit Hamiltonian -> VQE -> PES
|
|-- Project 03: Compare feature spaces
|   |-- MNIST -> PCA -> quantum feature map -> kernel -> SVM
|
|-- Project 04: Optimize combinatorial structure
|   |-- graph -> Ising Hamiltonian -> QAOA/RQAOA -> benchmark
|
|-- Project 05: Learn policies under noise
    |-- environment -> measurement-defined policy -> policy gradient -> mitigation
```

### The Repeated Research Pattern

```text
1. Define problem
2. Encode into quantum representation
3. Choose ansatz or feature map
4. Select backend/executor
5. Optimize or train
6. Compare against exact/classical baselines
7. Quantify uncertainty and resource cost
8. Save artifacts with provenance
9. State limitations honestly
```

### What To Be Able To Explain From Memory

- Why quantum kernels compare state overlaps.
- Why VQE energy minimization works.
- Why QAOA maps graph cuts into Pauli-Z Hamiltonians.
- Why measurement-defined policies are different from expectation-logit policies.
- Why finite shots and noise change every algorithmic claim.
- Why strong classical baselines are non-negotiable.
- Why negative results are still valuable research outputs.

### Final Mastery Checklist

- [ ] I can derive the Max-Cut Hamiltonian from the graph objective.
- [ ] I can explain VQE's variational principle.
- [ ] I can explain why PCA dimension equals qubit count in Project 03.
- [ ] I can compute KTA by hand for a small kernel matrix.
- [ ] I can distinguish expected QAOA objective from sampled bitstring quality.
- [ ] I can explain parameter-shift gradients.
- [ ] I can describe why mitigation may introduce bias.
- [ ] I can inspect a result artifact and judge whether the claim is supported.
- [ ] I can add a new baseline or ablation without breaking experiment fairness.
- [ ] I can write a limitation section that is scientifically honest.

