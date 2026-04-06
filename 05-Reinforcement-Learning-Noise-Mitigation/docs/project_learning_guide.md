# Quantum RL Noise Mitigation - Deep Study Guide

This document turns the project into structured learning material.
It is written as a mentor-style knowledge base, not as a marketing summary.
The goal is to help you understand:

- what the project is trying to do
- how the main algorithms work
- how the math maps to code
- what is scientifically strong, weak, and incomplete
- what to study next to go from "I ran the repo" to "I deeply understand the system"

---

## How To Use This Guide

Read this in layers:

1. Read **Project Understanding** to build a mental model.
2. Read **Core Concepts** to learn the vocabulary.
3. Read **Algorithm Breakdown** to understand the mechanics.
4. Read **Code-to-Concept Mapping** while opening the actual files.
5. Read **Mathematical Understanding** with a notebook beside you and derive the formulas yourself.
6. Use **Study Notes**, **Learning Path**, and **Questions** for revision and mastery.

If you want the shortest path through the repo, the most important files are:

- `environments/simple_nav_env.py`
- `agent/quantum_policy.py`
- `agent/gradient_estimator.py`
- `agent/reinforce_learner.py`
- `agent/actor_critic_learner.py`
- `src/runtime_executor.py`
- `src/mitigation_engine.py`
- `src/training_pipeline.py`
- `src/benchmark_suite.py`
- `reporting/paper_report.py`
- `docs/technical_note.md`
- `docs/benchmark_spec.md`

---

# STEP 1 - Project Understanding

## 1.1 What Problem This Project Solves

This project studies how a **measurement-defined quantum policy** behaves when used for **reinforcement learning** on a small sequential task, especially under **realistic NISQ-style noise** and **error mitigation**.

More concretely, it asks:

- Can a variational quantum circuit act as a policy for a sequential decision problem?
- How much does hardware noise degrade that policy?
- Can readout correction and zero-noise extrapolation recover some of the lost performance?
- How does the quantum method compare against simple and stronger classical baselines?

This is not mainly a "build the best agent" project.
It is a **benchmark and characterization project**:

- benchmark the behavior of a quantum policy
- compare it against classical policies
- quantify noise sensitivity
- quantify mitigation cost versus benefit

## 1.2 Why This Problem Is Important

This problem matters for three reasons.

### A. Quantum ML often overclaims

Many quantum machine learning projects jump too quickly from "it runs" to "it is better."
This repo tries to be more honest:

- noise matters
- mitigation is costly
- classical baselines are strong
- small tasks can be misleading

That makes the project useful as a **research methodology artifact**.

### B. Reinforcement learning is harder than static supervised learning

RL combines:

- sequential decisions
- stochastic transitions
- delayed rewards
- exploration
- unstable gradient estimates

Adding quantum circuits on top of that creates a much harder system than a static classifier.

### C. NISQ hardware is noisy and limited

Even if a quantum policy is mathematically elegant, real devices introduce:

- gate error
- decoherence
- readout error
- shot noise
- queue and runtime cost

So any realistic quantum RL claim must account for hardware limitations.

## 1.3 What Domain This Project Belongs To

This project sits at the intersection of several domains:

- **Reinforcement Learning**
- **Variational Quantum Algorithms**
- **Quantum Machine Learning**
- **Hybrid Quantum-Classical Optimization**
- **NISQ Error Mitigation**
- **Research Benchmarking and Experimental Design**

The repo is best understood as a **hybrid quantum-classical RL benchmark system**.

## 1.4 Real-World Applications

The direct environment is a toy task, but the ideas matter in broader settings:

- **Hardware-aware decision systems**: studying how noisy quantum policies behave before using them in more serious control problems
- **Adaptive experiment design**: choosing actions based on uncertainty and noisy measurements
- **Benchmark design for QML**: building more rigorous comparisons between quantum and classical learning systems
- **Hybrid control pipelines**: quantum feature generation with classical optimization and evaluation
- **Error-mitigation-aware algorithm design**: deciding when mitigation is worth its cost

Important caveat:
this repo is not a direct production system for robotics, finance, or operations.
Its value is primarily as a **research and learning platform**.

---

# STEP 2 - Concept Extraction

This section breaks the project into its core concepts.
Each concept is explained from simple to advanced.

## Concept 1 - Markov Decision Process (MDP)

**Simple explanation**

An MDP is a formal way to describe decision making over time.
An agent observes a state, takes an action, gets a reward, and moves to a new state.

**Advanced explanation**

An MDP is usually written as `(S, A, P, R, gamma)`:

- `S`: state space
- `A`: action space
- `P(s' | s, a)`: transition dynamics
- `R(s, a, s')`: reward
- `gamma`: discount factor

RL methods try to learn a policy `pi(a|s)` that maximizes expected discounted return.

**Why it is used here**

The environment is sequential.
The right action changes depending on whether the key has already been collected.
That makes the problem a real control problem, not just one-step classification.

**How it connects**

It is the outer shell that all other concepts live inside:

- the quantum circuit defines a policy inside the MDP
- REINFORCE and actor-critic optimize that policy
- noise affects policy evaluation
- mitigation changes how probabilities are estimated

## Concept 2 - Key-and-Door Sequential Benchmark

**Simple explanation**

The agent lives in a 1D corridor.
It must first go to the key, then reverse direction and go to the goal.

**Advanced explanation**

The benchmark introduces:

- delayed reward
- phase-dependent optimal behavior
- stochastic action slip
- sparse versus shaped rewards
- multiple difficulty settings across corridor size and slip levels

This is designed to be more meaningful than a trivial one-step task.

**Why it is used here**

A quantum policy should be tested on something sequential but still computationally manageable.
This task is small enough to run many experiments, yet nontrivial enough to expose instability.

**How it connects**

It drives:

- state encoding
- reward-to-go computation
- advantage estimation
- evaluation metrics like success rate and reward AUC

## Concept 3 - Reward Shaping vs Sparse Reward

**Simple explanation**

Sparse reward means the agent only gets useful feedback rarely.
Reward shaping gives it small hints along the way.

**Advanced explanation**

In this project:

- `goal_reward` rewards finishing
- `key_reward` rewards collecting the key
- `progress_reward_scale` rewards movement toward the current target
- `step_penalty`, `wall_penalty`, and `locked_goal_penalty` discourage bad behavior

Sparse settings remove shaping and key bonus, making credit assignment harder.

**Why it is used here**

To create multiple difficulty levels and test whether methods are robust across task designs.

**How it connects**

Sparse rewards increase the variance of policy-gradient methods, which makes:

- baselines more important
- actor-critic more useful
- statistical evaluation more important

## Concept 4 - Stochastic Dynamics / Slip Probability

**Simple explanation**

Sometimes the environment flips the chosen action.

**Advanced explanation**

If the agent selects action `a`, the environment may execute `1-a` with probability `p_slip`.
This introduces transition noise in the MDP itself, separate from quantum hardware noise.

**Why it is used here**

It prevents the task from being too deterministic and makes policy robustness matter.

**How it connects**

It interacts with:

- exploration
- variance of returns
- benchmark difficulty
- the distinction between environment stochasticity and hardware stochasticity

## Concept 5 - Policy

**Simple explanation**

A policy tells the agent how likely each action is in each state.

**Advanced explanation**

The policy is a conditional probability distribution:

`pi_theta(a|s)`

In this project, the policy is defined by a quantum circuit:

- state is encoded into circuit rotations
- trainable parameters shape the final quantum state
- measuring action qubits produces action probabilities

**Why it is used here**

The whole point of the project is to test a quantum policy under noise.

**How it connects**

The policy is the object optimized by REINFORCE and actor-critic.
Its probabilities drive:

- action sampling
- log-probability gradients
- entropy regularization

## Concept 6 - Variational Quantum Circuit (VQC)

**Simple explanation**

A VQC is a quantum circuit with trainable parameters.

**Advanced explanation**

The project builds a circuit with:

- state-dependent angle embedding
- one or more data reupload blocks
- a trainable ansatz such as `RealAmplitudes` or `EfficientSU2`
- an action register whose measurement defines the policy

The trainable parameters are classical numbers updated by a classical optimizer.

**Why it is used here**

VQCs are one of the main ways to build trainable quantum models on near-term devices.

**How it connects**

It is the quantum core of the actor.
It interacts with:

- state encoding
- parameter-shift differentiation
- noise models
- circuit depth and hardware feasibility

## Concept 7 - Measurement-Defined Policy

**Simple explanation**

The action distribution comes directly from measuring designated qubits.

**Advanced explanation**

This matters because it is more faithful to a quantum probabilistic model than taking expectation values and applying a classical softmax.

In the current design:

- the action register size is `log2(number_of_actions)`
- the final Born probabilities over that register define the action distribution

This makes the policy genuinely probability-native at the circuit output.

**Why it is used here**

It fixes a common conceptual weakness in quantum policy implementations:
using a classical post-processing head that hides what the quantum circuit is actually producing.

**How it connects**

It affects:

- how probabilities are estimated
- how gradients are computed
- how readout mitigation is applied

## Concept 8 - Parameter Shift Rule

**Simple explanation**

Instead of backpropagating through a quantum circuit directly, the project estimates derivatives by running slightly shifted versions of the circuit.

**Advanced explanation**

For suitable gates, the derivative of a probability with respect to parameter `theta_i` can be written as:

`d p / d theta_i = [p(theta_i + pi/2) - p(theta_i - pi/2)] / 2`

This is exact in the ideal setting for Pauli-generated rotation families.

**Why it is used here**

It is the main way the project computes policy gradients for the quantum actor.

**How it connects**

It creates the main computational cost:

- for `P` parameters, each logical gradient estimate needs roughly `2P` shifted evaluations
- this is one reason hardware training is expensive

## Concept 9 - REINFORCE

**Simple explanation**

REINFORCE increases the probability of actions that led to high return and decreases the probability of actions that led to low return.

**Advanced explanation**

The gradient identity is:

`nabla J(theta) = E[ sum_t G_t * nabla log pi_theta(a_t|s_t) ]`

In practice the project uses:

- reward-to-go returns
- an action-independent baseline
- entropy regularization

This reduces variance and improves stability.

**Why it is used here**

It is the simplest policy-gradient algorithm and serves as the legacy baseline quantum learner.

**How it connects**

It is the foundation for understanding the more advanced actor-critic method.

## Concept 10 - Actor-Critic and GAE

**Simple explanation**

The actor chooses actions.
The critic estimates how good states are.
The critic helps the actor learn with lower variance.

**Advanced explanation**

The actor remains quantum.
The critic is classical.

The project uses:

- a quantum actor for the policy
- a classical MLP critic for `V(s)`
- generalized advantage estimation (GAE) to compute lower-variance advantages

GAE mixes short-term and long-term credit assignment using `gamma` and `lambda`.

**Why it is used here**

Pure REINFORCE has high variance.
Actor-critic is the main algorithmic upgrade because it stabilizes training without making the entire system quantum.

**How it connects**

It connects:

- value estimation
- advantage estimation
- lower-variance policy gradients
- stronger comparison against classical actor-critic baselines

## Concept 11 - Error Mitigation

**Simple explanation**

Error mitigation tries to partially undo noise without full quantum error correction.

**Advanced explanation**

The project uses two mitigation ideas:

- **Readout correction**: correct measurement bias using a confusion matrix model
- **Zero-noise extrapolation (ZNE)**: intentionally scale noise via circuit folding, then extrapolate back toward zero noise

These are not perfect corrections.
They are approximate and can themselves add variance or bias.

**Why it is used here**

Because real NISQ hardware is noisy, and mitigation is often the only practical way to improve results.

**How it connects**

Mitigation affects:

- probability estimation
- gradient quality
- runtime cost
- shot cost
- hardware feasibility

## Concept 12 - Statistical Rigor in ML Experiments

**Simple explanation**

You should not trust a method just because one run looked good.

**Advanced explanation**

This project includes:

- fixed seed lists
- multi-scenario benchmarking
- confidence intervals
- paired comparisons
- sign-flip tests
- Hodges-Lehmann effect estimates
- Holm correction

That turns the repo from a demo into a more serious research artifact.

**Why it is used here**

Because quantum ML is especially vulnerable to overinterpretation from small, noisy experiments.

**How it connects**

It turns raw run logs into research claims.

---

# STEP 3 - Algorithm Breakdown

This section identifies the main algorithms and explains how each one works.

## Algorithm 1 - Key-and-Door Environment Dynamics

### Intuition

The agent must solve a two-phase navigation task:

1. Go to the key.
2. After collecting it, go to the goal.

This forces context-sensitive behavior.

### Step-by-step

1. Reset places the agent at one of the allowed start positions without a key.
2. The agent chooses left or right.
3. With probability `slip_probability`, the chosen action is flipped.
4. The environment applies penalties or rewards:
   - step penalty every move
   - wall penalty if stuck against boundary
   - locked-goal penalty if goal is reached before key
   - progress reward for moving closer to current target
   - key reward when key is collected
   - goal reward when goal is reached with key
5. Episode ends if the goal is reached or the time limit is hit.

### Mathematical view

State can be written as:

`s = (position, has_key)`

The encoded integer state is:

`state_id = position + n_positions * 1[has_key]`

The transition kernel is stochastic because of slip.

### Where it appears

- `environments/simple_nav_env.py`

### Strengths

- sequential, not one-step
- delayed reward
- interpretable dynamics
- easy to scale in difficulty

### Limitations

- still a small toy environment
- only two actions
- limited representational challenge

## Algorithm 2 - Quantum State Encoding

### Intuition

The discrete environment state must be converted into quantum rotations.

### Step-by-step

1. Represent the state index as a structured numeric object.
2. Build qubit angles with `build_state_angles(...)`.
3. For each reupload block and qubit:
   - apply `RY(state_angle)`
   - apply `RZ(0.5 * state_angle)`
4. Compose trainable ansatz blocks after the data encoding.

### Mathematical view

For qubit `q` and state-derived angle `x_q`, a block contains:

`RY(x_q) RZ(0.5 x_q)`

The `"hybrid"` encoding combines:

- binary structure from the state identity
- smooth phase ramp from normalized state index

### Where it appears

- `utils/qiskit_helpers.py`
- `agent/quantum_policy.py`

### Strengths

- simple
- deterministic
- richer than pure binary encoding

### Limitations

- hand-designed
- not learned
- may be weak for larger state spaces

## Algorithm 3 - Measurement-Native Quantum Policy

### Intuition

The quantum circuit itself defines the action probabilities through measurement.

### Step-by-step

1. Build a parameterized circuit with:
   - state embedding
   - trainable ansatz blocks
2. Reserve `log2(n_actions)` qubits as the action register.
3. Bind state-dependent and trainable parameters.
4. Execute the circuit.
5. Measure only the action qubits.
6. Interpret the resulting bitstring distribution as the policy distribution.

### Mathematical view

Let `U_theta(s)` be the state-conditioned circuit.
Then the action probability is:

`pi_theta(a|s) = Prob(measuring action register in bitstring a | U_theta(s) |0...0>)`

That is a Born-rule probability, not a classical softmax over manually chosen logits.

### Where it appears

- `agent/quantum_policy.py`
- `src/runtime_executor.py`

### Strengths

- conceptually honest quantum policy
- measurement interpretation is direct
- easier to reason about under readout noise

### Limitations

- currently requires number of actions to be a power of two
- scaling action space increases required action-register qubits

## Algorithm 4 - Parameter-Shift Policy Gradient

### Intuition

To update the quantum actor, we need derivatives of action probabilities with respect to circuit parameters.

### Step-by-step

1. For each parameter `theta_i`, create:
   - `theta_i + pi/2`
   - `theta_i - pi/2`
2. Evaluate the action probabilities at both shifted parameter settings.
3. Estimate the derivative of each action probability using the difference formula.
4. Convert that derivative into a log-probability gradient.
5. Multiply by return or advantage weight.

### Mathematical view

For a policy probability:

`d pi_theta(a|s) / d theta_i = [pi_{theta_i + pi/2}(a|s) - pi_{theta_i - pi/2}(a|s)] / 2`

Then:

`d log pi_theta(a|s) / d theta_i = (1 / pi_theta(a|s)) * d pi_theta(a|s) / d theta_i`

The code stabilizes small probabilities with a floor before dividing.

### Where it appears

- `agent/gradient_estimator.py`

### Strengths

- analytic in the ideal gate family setting
- straightforward to implement

### Limitations

- expensive: `2P` shifted circuits for `P` parameters
- noisy under finite shots
- may become unstable when probabilities are very small

## Algorithm 5 - REINFORCE with Entropy Regularization

### Intuition

If an action helped produce high return, increase its probability.
If it led to low return, decrease it.
Entropy keeps the policy from collapsing too quickly.

### Step-by-step

1. Roll out an episode with the current policy.
2. Compute reward-to-go returns.
3. Subtract a timestep baseline to reduce variance.
4. Compute `nabla log pi(a_t|s_t)` for each step.
5. Weight each by the advantage-like signal.
6. Add entropy regularization.
7. Update parameters with Adam.

### Mathematical view

The surrogate loss is:

`L_actor(theta) = - sum_t A_t log pi_theta(a_t|s_t) - beta sum_t H(pi_theta(.|s_t))`

where:

- `A_t` is return or baseline-adjusted return
- `H(p) = - sum_a p(a) log p(a)`

### Where it appears

- `agent/reinforce_learner.py`
- `agent/gradient_estimator.py`

### Strengths

- conceptually simple
- good baseline algorithm

### Limitations

- high variance
- sample inefficient
- sensitive to shot noise and hyperparameters

## Algorithm 6 - Quantum Actor + Classical Critic

### Intuition

Let the quantum circuit model the policy, but let a classical network estimate state value.
This reduces the variance of the policy update.

### Step-by-step

1. Roll out a trajectory using the quantum policy.
2. At each state, also predict `V(s)` with the classical critic.
3. Use rewards and value estimates to compute GAE advantages.
4. Update the actor using policy gradients weighted by those advantages.
5. Update the critic by regressing predicted values toward target returns.

### Mathematical view

Temporal-difference residual:

`delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`

GAE:

`A_t = delta_t + gamma lambda delta_{t+1} + gamma^2 lambda^2 delta_{t+2} + ...`

Return target:

`R_t = A_t + V(s_t)`

Critic loss:

`L_value(phi) = 0.5 sum_t (V_phi(s_t) - R_t)^2`

### Where it appears

- `agent/actor_critic_learner.py`
- `src/baselines.py` for the classical actor-critic counterpart
- `src/rl_utils.py`

### Strengths

- lower variance than REINFORCE
- stronger benchmark method
- more realistic RL training loop

### Limitations

- more moving parts
- critic quality affects actor update quality
- still expensive when the actor is quantum

## Algorithm 7 - Adam Optimization

### Intuition

Adam adapts the step size per parameter using running estimates of first and second gradient moments.

### Step-by-step

1. Track moving average of gradients.
2. Track moving average of squared gradients.
3. Bias-correct both estimates.
4. Scale update by `m_hat / (sqrt(v_hat) + eps)`.
5. Optionally clip gradients.

### Mathematical view

Standard Adam:

- `m_t = beta1 m_{t-1} + (1-beta1) g_t`
- `v_t = beta2 v_{t-1} + (1-beta2) g_t^2`
- `theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)`

### Where it appears

- `src/optim.py`

### Strengths

- stable
- widely used
- good default optimizer for noisy gradients

### Limitations

- not magical
- can still fail with very high-variance gradients

## Algorithm 8 - Readout Error Mitigation

### Intuition

Measured bitstrings are biased because the hardware can misread `0` as `1` or vice versa.
Readout mitigation tries to correct that.

### Step-by-step

1. Estimate per-qubit readout error rates.
2. Build a confusion matrix for measurement outcomes.
3. Invert or pseudo-invert the measurement distortion.
4. Apply the correction to measured probabilities.
5. Reproject onto the probability simplex if needed.

### Mathematical view

If `p_true` is the true distribution and `M` is the readout matrix:

`p_meas = M p_true`

Then mitigation approximates:

`p_true ~= M^{-1} p_meas`

### Where it appears

- `src/runtime_executor.py`
- `src/noise_models.py`
- `src/mitigation_engine.py`

### Strengths

- cheap compared with full error correction
- directly relevant for measurement-defined policies

### Limitations

- correction can amplify noise
- model mismatch causes residual bias

## Algorithm 9 - Zero-Noise Extrapolation (ZNE)

### Intuition

If you can run versions of a circuit at higher effective noise, you can extrapolate backward toward what the zero-noise answer might have been.

### Step-by-step

1. Start with the original circuit.
2. Produce folded circuits that are functionally equivalent but noisier.
3. Measure probabilities at several achieved noise scales.
4. Fit an extrapolation model.
5. Estimate the zero-noise distribution from that fit.

### Mathematical view

For achieved noise scales `lambda_1, lambda_2, ..., lambda_k`, fit:

`p(lambda)`

and estimate:

`p(0)`

The project uses achieved fold scales rather than merely requested ones, which is mathematically important.

### Where it appears

- `src/mitigation_engine.py`
- `src/runtime_executor.py`

### Strengths

- standard NISQ-era mitigation technique
- often easy to integrate with simulators and runtime pipelines

### Limitations

- increases runtime and shot cost
- extrapolation can be unstable
- deeper folded circuits may be much less hardware realistic

## Algorithm 10 - Statistical Comparison Pipeline

### Intuition

A method should be judged across seeds and scenarios, not by a single lucky run.

### Step-by-step

1. Run each method across fixed seeds.
2. Aggregate metrics like success and reward.
3. Build confidence intervals.
4. Compare methods using paired differences on shared seeds.
5. Report effect sizes and corrected p-values.

### Mathematical view

The code uses:

- bootstrap confidence intervals for means
- sign-flip tests on paired differences
- Hodges-Lehmann estimates
- Holm correction across comparison families

### Where it appears

- `src/research_stats.py`
- `src/experiment_design_audit.py`
- `src/benchmark_suite.py`

### Strengths

- more scientifically valid than reporting means only
- controls overclaiming

### Limitations

- still depends on sample size
- statistical significance is not the same as scientific importance

---

# STEP 4 - Code-to-Concept Mapping

This section maps important files to the theory they implement.

## Architecture Map

### `environments/simple_nav_env.py`

**What it does**

Defines the key-and-door environment, its reward logic, state encoding, and Gym-style wrapper.

**Concepts**

- MDPs
- delayed reward
- stochastic transitions
- reward shaping

**Why designed this way**

It keeps the benchmark self-contained, interpretable, and easy to parameterize for scenario sweeps.

### `agent/quantum_policy.py`

**What it does**

Builds the measurement-defined variational quantum policy.

**Concepts**

- state encoding
- variational quantum circuits
- Born-rule action probabilities
- action register

**Why designed this way**

It makes the policy genuinely quantum at the measurement layer instead of using a classical readout head on expectation values.

### `agent/gradient_estimator.py`

**What it does**

Implements parameter-shift-based log-probability and trajectory-gradient estimation for the quantum actor.

**Concepts**

- parameter-shift differentiation
- policy gradients
- entropy regularization
- probability stabilization

**Why designed this way**

Quantum circuits do not use ordinary backpropagation through gates the same way classical neural nets do.
Parameter shift is the natural analytic derivative tool for this gate family.

### `agent/reinforce_learner.py`

**What it does**

Implements batched REINFORCE training for the quantum actor.

**Concepts**

- Monte Carlo policy gradient
- reward-to-go
- variance reduction with baseline
- optimizer stability tools

**Why designed this way**

REINFORCE is the simplest faithful baseline for quantum policy optimization.

### `agent/actor_critic_learner.py`

**What it does**

Implements the upgraded quantum actor-critic learner with a classical value network.

**Concepts**

- actor-critic
- GAE
- value-function learning
- hybrid quantum-classical optimization

**Why designed this way**

It upgrades the legacy quantum REINFORCE learner into a more stable RL method without adding the cost of a quantum critic.

### `src/baselines.py`

**What it does**

Defines classical baselines:

- random
- tabular REINFORCE
- MLP REINFORCE
- MLP actor-critic

**Concepts**

- fair benchmarking
- classical reference methods
- matched objective families

**Why designed this way**

Without strong classical baselines, a quantum benchmark is not scientifically persuasive.

### `src/rl_utils.py`

**What it does**

Provides shared RL computations:

- discounted returns
- baseline adjustment
- baseline updates
- generalized advantage estimation

**Concepts**

- credit assignment
- variance reduction
- actor-critic targets

**Why designed this way**

These are reusable mathematical primitives shared across methods.

### `src/optim.py`

**What it does**

Provides a lightweight Adam implementation with clipping and per-step stats.

**Concepts**

- adaptive optimization
- gradient clipping
- training diagnostics

**Why designed this way**

Research code often needs transparent, inspectable optimizers rather than opaque framework defaults.

### `src/runtime_executor.py`

**What it does**

Executes circuits in ideal, noisy, mitigated, and hardware modes and returns action probabilities.

**Concepts**

- circuit execution
- shot-based sampling
- noise-aware inference
- layout-aware measurement handling

**Why designed this way**

It centralizes the difference between ideal simulation, noisy simulation, mitigation, and hardware evaluation.

### `src/noise_models.py`

**What it does**

Creates compact or full fake-backend-based noise models and extracts readout information.

**Concepts**

- NISQ noise
- backend realism
- readout error modeling

**Why designed this way**

Noise assumptions need to be explicit, configurable, and reusable across modes.

### `src/mitigation_engine.py`

**What it does**

Applies readout mitigation and ZNE extrapolation.

**Concepts**

- readout correction
- circuit folding
- zero-noise extrapolation

**Why designed this way**

Mitigation logic should be separate from circuit execution so it can be tested and audited more easily.

### `src/evaluation.py`

**What it does**

Evaluates trained policies, aggregates histories, and creates plots and summary metrics.

**Concepts**

- held-out evaluation
- AUC metrics
- training diagnostics
- visualization

**Why designed this way**

Training performance and evaluation performance are not the same thing.
This module makes that distinction explicit.

### `src/research_stats.py`

**What it does**

Provides bootstrap CIs, paired effect summaries, sign-flip tests, Hodges-Lehmann estimates, and Holm correction.

**Concepts**

- uncertainty quantification
- effect size estimation
- multiple-testing correction

**Why designed this way**

The project tries to support research claims with more than just averages.

### `benchmarks/scenario_registry.py`

**What it does**

Defines the eight named benchmark scenarios and the smoke suite.

**Concepts**

- benchmark design
- controlled task difficulty variation

**Why designed this way**

It avoids "one default config" syndrome and makes the benchmark reproducible.

### `src/benchmark_suite.py`

**What it does**

Runs full benchmark suites across scenarios and summarizes winners, ranks, efficiency, and noise profiles.

**Concepts**

- benchmark orchestration
- scenario-level aggregation
- cross-method comparison

**Why designed this way**

A research benchmark needs a top-level runner, not only per-config scripts.

### `core/schemas.py`

**What it does**

Defines versioned artifact schemas such as `ScenarioSpec`, `MethodSpec`, `RunResult`, `BenchmarkSuiteResult`, and `PaperReportBundle`.

**Concepts**

- research reproducibility
- machine-readable experiment outputs

**Why designed this way**

Versioned schemas make results easier to validate, compare, and archive.

### `core/runner.py`

**What it does**

Provides stable APIs for training, evaluation, benchmark execution, and report building.

**Concepts**

- modular architecture
- public programmatic interface

**Why designed this way**

This turns the repo from "a collection of scripts" into a system with reusable entry points.

### `reporting/paper_report.py`

**What it does**

Builds paper-style figures, tables, and a report bundle from saved benchmark results.

**Concepts**

- research communication
- reproducible reporting

**Why designed this way**

The report should be reproducible from saved artifacts, not rebuilt manually from notebooks.

### `src/training_pipeline.py`

**What it does**

Orchestrates end-to-end runs:

- config loading
- per-seed training
- evaluation
- artifact saving
- aggregation

**Concepts**

- experiment orchestration
- reproducible pipelines
- per-method execution modes

**Why designed this way**

It is the backbone that ties together environment, methods, noise, mitigation, and reporting.

## Mental Execution Flow

When you run the project, the conceptual flow is:

1. Load config.
2. Choose benchmark scenario.
3. Construct environment.
4. Construct quantum policy and executor.
5. Train a method:
   - collect trajectories
   - estimate returns or advantages
   - compute gradients
   - update parameters
6. Evaluate the final checkpoint.
7. Save structured run results.
8. Aggregate across seeds and scenarios.
9. Build benchmark report and paper figures.

---

# STEP 5 - Mathematical Understanding

This section extracts the most important math used in the project and explains it intuitively.

## 5.1 MDP Objective

The learning objective is to maximize expected discounted return:

`J(theta) = E_{tau ~ pi_theta} [ sum_{t=0}^{T-1} gamma^t r_t ]`

where:

- `tau` is a trajectory
- `theta` are policy parameters
- `gamma` is the discount factor

**Intuition**

The agent should prefer policies that lead to more future reward, not only immediate reward.

## 5.2 Policy Definition

For a state `s`, the quantum circuit produces action probabilities:

`pi_theta(a|s) = Prob(a from action-register measurement after U_theta(s))`

**Intuition**

The circuit prepares a quantum state whose measurement statistics become the policy.

This is important because:

- the action distribution is not manually imposed afterward
- readout error directly affects the policy probabilities

## 5.3 Return / Reward-to-Go

For a trajectory with rewards `r_t`, the reward-to-go is:

`G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k`

**Intuition**

`G_t` tells you how good the future turned out from time `t` onward.

**Why this version is used**

Reward-to-go has lower variance than assigning the full episode return to every timestep.

## 5.4 Baseline-Adjusted Policy Gradient

The standard unbiased policy-gradient identity with an action-independent baseline is:

`nabla J(theta) = E[ sum_t (G_t - b_t) nabla log pi_theta(a_t|s_t) ]`

In the code, the REINFORCE learner uses a timestep-indexed moving baseline.

**Intuition**

The baseline does not change which action was taken.
It only subtracts a reference level, reducing variance.

## 5.5 Entropy Regularization

Entropy of a policy at a state is:

`H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)`

The actor objective includes:

`- beta H(pi(.|s_t))`

inside the loss, which means maximizing entropy in the optimization view.

**Intuition**

Entropy prevents early collapse to an overly deterministic policy.

## 5.6 Parameter-Shift Derivative

For a suitable gate family:

`d pi_theta(a|s) / d theta_i = [pi_{theta_i + pi/2}(a|s) - pi_{theta_i - pi/2}(a|s)] / 2`

Then:

`d log pi_theta(a|s) / d theta_i = [1 / pi_theta(a|s)] * d pi_theta(a|s) / d theta_i`

**Intuition**

The project estimates how the action probability changes when a single parameter is nudged in both directions.

**Important caution**

This identity is exact in the ideal gate model, but under:

- finite shots
- noise
- mitigation

the estimator becomes stochastic and may become biased after post-processing.

## 5.7 REINFORCE Surrogate Used In Practice

For one trajectory:

`L_actor(theta) = - sum_t A_t log pi_theta(a_t|s_t) - beta sum_t H(pi_theta(.|s_t))`

where `A_t` is a return-like weight.

In REINFORCE:

- `A_t` is baseline-adjusted reward-to-go

In actor-critic:

- `A_t` is the GAE advantage

**Intuition**

This loss is a training surrogate whose gradient produces the desired policy-gradient direction.

## 5.8 Generalized Advantage Estimation (GAE)

First compute temporal-difference residuals:

`delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`

Then compute:

`A_t = delta_t + gamma lambda delta_{t+1} + gamma^2 lambda^2 delta_{t+2} + ...`

The target return becomes:

`R_t = A_t + V(s_t)`

**Intuition**

GAE smoothly trades off:

- low bias and high variance
- higher bias and lower variance

The parameter `lambda` controls this tradeoff.

## 5.9 Value Loss

The critic learns by minimizing squared prediction error:

`L_value(phi) = 0.5 sum_t (V_phi(s_t) - R_t)^2`

**Intuition**

The critic tries to predict future return from state.
Better value estimates give better actor advantages.

## 5.10 Readout Correction

Let `M` be a measurement confusion matrix and `p_true` the true distribution.
Measured distribution satisfies:

`p_meas = M p_true`

Then mitigation estimates:

`p_true ~= M^{-1} p_meas`

In practice the code uses qubit-wise readout information and then projects the result back onto the simplex if needed.

**Intuition**

You are correcting the detector, not the quantum state itself.

## 5.11 Zero-Noise Extrapolation

Let `p(lambda)` be the measured probability distribution at effective noise scale `lambda`.
You evaluate the circuit at several achieved scales and fit a curve.
Then estimate:

`p(0)`

**Intuition**

You deliberately make the circuit noisier in a controlled way, then infer what the answer would be with less noise.

**Important subtlety**

The code now uses the **achieved** noise scaling from folding, not just the requested scale factor.
That is mathematically more correct.

## 5.12 Resource Metrics

This project also treats compute cost as part of the math of the experiment.

Examples include:

- estimated total shots per seed
- parameter bindings per timestep
- success per runtime second
- success per million shots

**Intuition**

A method that is slightly better but vastly more expensive may not be practically better.

---

# STEP 6 - Study Notes

## 6.1 Quick Revision Notes

### One-paragraph memory aid

This repo is a benchmark-first hybrid quantum RL system.
It trains a measurement-defined variational quantum policy on a key-and-door sequential environment under ideal, noisy, and mitigated execution modes.
The legacy quantum method is REINFORCE.
The upgraded quantum method is a quantum actor with a classical value critic using GAE.
The project compares these against random, tabular, MLP REINFORCE, and MLP actor-critic baselines, then summarizes performance, efficiency, and statistical evidence across a fixed benchmark suite.

### Core memory bullets

- Environment is a sequential key-then-goal corridor.
- State is discrete, encoded into quantum rotations.
- Policy is defined by Born probabilities on an action register.
- Quantum gradients use parameter shift.
- REINFORCE uses return-like weights.
- Actor-critic uses classical value estimates and GAE.
- Noise is simulated using backend-inspired models.
- Mitigation uses readout correction and ZNE.
- Benchmarking uses fixed seeds, multiple scenarios, and statistical comparisons.

## 6.2 Detailed Notes

### Detailed Note 1 - What makes this a "hybrid" system

The actor is quantum, but almost everything else is classical:

- optimizer
- return computation
- baseline tracking
- critic
- statistics
- report generation

This is normal and realistic for near-term quantum ML.

### Detailed Note 2 - Why the measurement-defined policy matters

Earlier styles of quantum policy often used:

- expectation values as logits
- classical softmax afterward

That makes the policy less natively quantum.
This project instead uses direct measurement probabilities on an action register.
That is conceptually cleaner and easier to interpret under readout noise.

### Detailed Note 3 - Why actor-critic is the main algorithmic upgrade

Pure REINFORCE has very noisy updates because it estimates return using full trajectory samples.
Actor-critic reduces variance by learning a value function.
In this project, the critic is classical because that is far more cost-effective than adding a quantum critic.

### Detailed Note 4 - Why mitigation is not automatically a win

Mitigation can improve prediction quality, but it also:

- increases runtime
- increases circuit count
- increases shot cost
- can amplify variance
- can become unstable when extrapolation is poor

So mitigation must be judged on both **accuracy** and **cost**.

### Detailed Note 5 - Why benchmark design is part of the research contribution

The strongest part of this repo is not "we found a quantum advantage."
It is closer to:

- honest benchmark design
- fair baseline comparison
- resource-aware evaluation
- mitigation-aware analysis

That is a valid systems-and-benchmark contribution.

## 6.3 Key Formulas

### RL objective

`J(theta) = E[ sum_t gamma^t r_t ]`

### Reward-to-go

`G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k`

### Policy gradient

`nabla J(theta) = E[ sum_t (G_t - b_t) nabla log pi_theta(a_t|s_t) ]`

### Entropy

`H(pi(.|s)) = - sum_a pi(a|s) log pi(a|s)`

### REINFORCE surrogate

`L_actor(theta) = - sum_t A_t log pi_theta(a_t|s_t) - beta sum_t H(pi_theta(.|s_t))`

### TD residual

`delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`

### GAE

`A_t = sum_{l=0}^\infty (gamma lambda)^l delta_{t+l}`

### Critic return target

`R_t = A_t + V(s_t)`

### Value loss

`L_value(phi) = 0.5 sum_t (V_phi(s_t) - R_t)^2`

### Parameter shift

`d pi_theta(a|s) / d theta_i = [pi(theta_i + pi/2) - pi(theta_i - pi/2)] / 2`

### Log-prob derivative

`d log pi / d theta_i = (1 / pi) * d pi / d theta_i`

### Readout model

`p_meas = M p_true`

### ZNE target

`p(0)` estimated from fitted `p(lambda)`

## 6.4 Important Concepts To Remember

- discrete MDP
- delayed reward
- reward shaping
- stochastic transitions
- variational quantum circuit
- data reuploading
- measurement-native policy
- parameter shift
- REINFORCE
- baseline variance reduction
- actor-critic
- GAE
- readout correction
- zero-noise extrapolation
- resource-aware benchmarking
- confidence intervals and paired tests

## 6.5 Common Mistakes

### Mistake 1 - Thinking the circuit alone trains itself

It does not.
The quantum circuit is only one part of a hybrid learning loop.
Training logic is mostly classical.

### Mistake 2 - Confusing environment noise with hardware noise

They are different:

- slip probability is MDP stochasticity
- gate/readout noise is quantum execution noise

### Mistake 3 - Assuming parameter shift is always exact in practice

It is exact for the ideal gate identity, not for the entire noisy finite-shot post-processed estimation chain.

### Mistake 4 - Treating mitigation as free performance

Mitigation adds overhead and can fail.
Always inspect efficiency metrics too.

### Mistake 5 - Ignoring classical baselines

A quantum result without strong classical comparison is usually not meaningful.

### Mistake 6 - Reading one benchmark scenario as a universal conclusion

Performance can change across:

- reward sparsity
- slip level
- corridor length

That is why the scenario suite matters.

---

# STEP 7 - Learning Path

This learning path is designed to turn this project into a mastery roadmap.

## Beginner Stage

### What to learn first

1. Basic reinforcement learning:
   - MDPs
   - policy
   - reward
   - return
   - exploration
2. Basic quantum computing:
   - qubits
   - gates
   - measurement
   - amplitudes and probabilities
3. Variational quantum circuits:
   - parameterized gates
   - ansatz
   - classical optimization loop

### What to practice

- Draw the key-and-door environment on paper.
- Manually compute state transitions.
- Manually compute discounted returns for one sample episode.
- Build a tiny 1-qubit circuit in Qiskit and measure probabilities.

### What to build next

- A classical random agent
- A tabular policy for the same environment
- A tiny 1-qubit measurement-based classifier

## Intermediate Stage

### What to learn

1. Policy gradients:
   - REINFORCE derivation
   - entropy regularization
   - variance reduction
2. Actor-critic:
   - value functions
   - TD error
   - GAE
3. Quantum differentiation:
   - parameter shift
   - shot-noise effects

### What to practice

- Derive REINFORCE from the log-derivative trick.
- Derive GAE from TD residuals.
- Reproduce the parameter-shift formula for a simple circuit.
- Trace one training step through `reinforce_learner.py`.
- Trace one training step through `actor_critic_learner.py`.

### What to build next

- Add a new classical baseline
- Add a new state encoding
- Add a new benchmark scenario with harder horizons

## Advanced Stage

### What to learn

1. NISQ noise modeling:
   - depolarization
   - thermal relaxation
   - readout noise
2. Error mitigation:
   - confusion matrices
   - ZNE theory and failure modes
3. Experimental design:
   - confidence intervals
   - paired tests
   - effect size
   - fair hyperparameter tuning
4. Hybrid quantum-classical systems design:
   - interface boundaries
   - artifact schemas
   - reproducible pipelines

### What to practice

- Compare compact and full fake-backend noise models.
- Run mitigation ablations and interpret the tradeoffs.
- Inspect shot-efficiency and runtime-efficiency frontiers.
- Try to prove where estimator bias may enter under mitigation.

### What to build next

- A different environment family beyond key-and-door
- A policy with larger action spaces
- A stronger hardware-evaluation slice
- A new mitigation-policy coupling strategy

## Suggested Study Order By File

If you want to study the code in a good order:

1. `environments/simple_nav_env.py`
2. `utils/qiskit_helpers.py`
3. `agent/quantum_policy.py`
4. `src/rl_utils.py`
5. `agent/gradient_estimator.py`
6. `agent/reinforce_learner.py`
7. `agent/actor_critic_learner.py`
8. `src/baselines.py`
9. `src/runtime_executor.py`
10. `src/mitigation_engine.py`
11. `src/evaluation.py`
12. `src/research_stats.py`
13. `src/training_pipeline.py`
14. `src/benchmark_suite.py`
15. `reporting/paper_report.py`

---

# STEP 8 - Generate Questions

## 8.1 Interview Questions

1. What is the difference between REINFORCE and actor-critic?
2. Why does a value baseline reduce policy-gradient variance?
3. What is generalized advantage estimation?
4. How does the parameter-shift rule work?
5. Why is a measurement-defined policy more conceptually faithful than a softmax over expectation values?
6. What is the difference between readout correction and zero-noise extrapolation?
7. Why is a strong classical baseline necessary in quantum ML experiments?
8. What are the main bottlenecks in training variational quantum policies?
9. Why can mitigation improve one metric while hurting efficiency?
10. Why are fixed seeds and paired comparisons important in benchmark studies?

## 8.2 Conceptual Questions

1. Why does the optimal action in the environment depend on whether the key has been collected?
2. Why is this environment more meaningful than a one-step classification task?
3. Why does entropy regularization help early training?
4. Why does dividing by a tiny action probability create instability?
5. Why does actor-critic often scale better than Monte Carlo REINFORCE?
6. Why does measuring the action register define a valid policy distribution?
7. Why can ZNE become unreliable as circuit depth grows?
8. Why should hardware feasibility be considered part of algorithm evaluation?
9. Why is average rank across scenarios sometimes more informative than one best-case score?
10. Why is "statistically significant" not the same as "scientifically important"?

## 8.3 Deep Research Questions

1. Under what task structures do measurement-defined quantum policies offer representational benefits over classical policies?
2. How does the variance of parameter-shift policy gradients scale with action count, shot budget, and ansatz depth?
3. When does the cost of mitigation outweigh its benefit in RL training loops?
4. Can policy architectures be designed to be intrinsically more robust to readout noise?
5. Is there a better hybrid interface than "quantum actor, classical critic" for this benchmark family?
6. How should one fairly budget quantum and classical methods when wall-clock cost, shots, and tuning effort all matter?
7. Can benchmark difficulty be increased without making the task meaningless for near-term quantum circuits?
8. Which components of the pipeline introduce estimator bias, and can that bias be quantified cleanly?
9. How should one separate algorithmic weakness from hardware-induced weakness in quantum RL?
10. What would count as persuasive evidence of genuine quantum value in this setting?

---

# STEP 9 - Identify Weaknesses

This section is intentionally critical.
Deep understanding includes seeing what is missing.

## 9.1 Conceptual Gaps

### Gap 1 - Small environment family

The benchmark is much better than the original trivial task, but it is still a small corridor domain.
You should not overgeneralize from it to all RL or all quantum RL.

### Gap 2 - No clear quantum advantage mechanism

The project is valuable as a benchmark, but it does not yet explain why quantum structure should outperform strong classical baselines on this problem family.

### Gap 3 - Hardware realism is partial

The repo takes hardware seriously, but most training still occurs in simulation or fake-backend noise.
That is scientifically reasonable, but it limits the strength of hardware claims.

## 9.2 Inefficient Design Choices

### Issue 1 - Parameter-shift cost scales badly

For `P` trainable parameters, you need about `2P` shifted evaluations per logical gradient estimate.
That is a major scalability limit.

### Issue 2 - Mitigation multiplies cost

ZNE requires multiple noise-scaled evaluations.
This makes already expensive quantum training even more expensive.

### Issue 3 - Action space restriction

The current policy assumes the number of actions is a power of two.
That is fine for this benchmark, but restrictive in general.

## 9.3 Missing Knowledge Areas

You should study these next if you want research-level mastery:

- policy-gradient theorem derivations
- actor-critic convergence intuition
- variance reduction in RL
- advanced quantum circuit ansatz design
- quantum natural gradients and other VQA optimizers
- readout mitigation theory
- zero-noise extrapolation stability analysis
- statistical testing for ML benchmarks
- fair tuning and compute budgeting
- hardware transpilation and layout effects

## 9.4 Scientifically Missing Next Steps

To push the project further, study and possibly implement:

- stronger environment families
- larger action spaces
- alternative quantum policy heads
- more hardware evaluation
- deeper ablations on ansatz and encoding
- cleaner causal analysis of why mitigation helps or fails

## 9.5 What You Should Study Next

### If your goal is "understand the RL deeply"

Study:

- Sutton and Barto chapters on policy gradients and actor-critic
- GAE derivation and intuition
- variance reduction techniques in Monte Carlo RL

### If your goal is "understand the quantum side deeply"

Study:

- variational quantum algorithms
- parameter-shift derivations
- expressivity and barren plateaus
- hardware-native transpilation and noise

### If your goal is "become a research engineer"

Study:

- experiment tracking
- artifact schemas
- statistical benchmarking
- reproducible reporting pipelines

---

# STEP 10 - Structured Knowledge Base

This final section turns the project into a reusable personal knowledge map.

## 10.1 One-Screen Knowledge Graph

### Problem Layer

- sequential decision making under noise
- hybrid quantum-classical learning
- benchmark-first evaluation

### Method Layer

- quantum measurement-defined actor
- REINFORCE baseline
- actor-critic upgrade
- classical baselines

### Systems Layer

- runtime executor
- noise models
- mitigation engine
- structured experiment runner

### Evaluation Layer

- held-out policy evaluation
- confidence intervals
- paired method comparison
- benchmark report generation

## 10.2 Personal Notes Template

Use this template when studying or extending the repo:

### Topic

What concept am I studying?

### File

Which file best implements it?

### Core equation

What is the governing equation?

### Intuition

Can I explain it without symbols?

### Failure mode

Where can this break?

### Experimental consequence

How would this show up in a benchmark result?

## 10.3 Final Mental Model

The cleanest way to think about this project is:

> A small but increasingly research-grade system for studying whether a measurement-defined quantum policy can learn a sequential control task under NISQ noise, how mitigation changes that picture, and how the result compares to classical baselines under reproducible experimental design.

That sentence captures the real contribution much better than "a quantum RL agent."

---

# Suggested Self-Study Exercises

1. Re-derive the REINFORCE objective used in the repo and explain every term in words.
2. Re-derive GAE and explain why `lambda` controls a bias-variance tradeoff.
3. Trace one episode manually through the environment and compute all rewards.
4. Trace one training update manually through the quantum REINFORCE learner.
5. Explain why actor-critic is the stronger algorithmic choice here.
6. Draw the full dataflow from `state -> circuit -> action probabilities -> environment -> returns -> gradients -> parameter update`.
7. Compare the role of environment stochasticity and hardware noise in one paragraph.
8. Explain when readout correction helps and when it can hurt.
9. Explain why ZNE is expensive and why the achieved scaling factor matters mathematically.
10. Write your own one-page critique of whether this repo demonstrates quantum value or mainly benchmarking discipline.

---

# Closing Perspective

If you learn this project well, you are not only learning a quantum RL codebase.
You are learning how modern research systems are built from:

- mathematical objectives
- model architectures
- optimization loops
- hardware assumptions
- experiment design
- reporting discipline

That broader view is what turns project familiarity into research maturity.
