# Quantum Kernel SVM — Complete Knowledge Base

**Project**: 03-Quantum-Kernel-SVM-MNIST  
**Author of Notes**: Generated from deep code analysis by Antigravity  
**Purpose**: Master-level understanding for research, interviews, and publication

---

## Table of Contents

1. [Step 1 — What This Project Actually Solves](#step-1--what-this-project-actually-solves)
2. [Step 2 — Concept Map](#step-2--concept-map)
3. [Step 3 — All Algorithms, Deep Breakdown](#step-3--all-algorithms-deep-breakdown)
4. [Step 4 — Code-to-Concept Mapping](#step-4--code-to-concept-mapping)
5. [Step 5 — Complete Mathematics](#step-5--complete-mathematics)
6. [Step 6 — Study Notes](#step-6--study-notes)
7. [Step 7 — Learning Path](#step-7--learning-path)
8. [Step 8 — Interview & Research Questions](#step-8--interview--research-questions)
9. [Step 9 — Weaknesses & Gaps](#step-9--weaknesses--gaps)

---

## Step 1 — What This Project Actually Solves

### The Core Problem

Classifying handwritten digits **4 vs. 9** from the MNIST dataset sounds simple —
and classically it is. An RBF SVM achieves ≥97% F1 with 100 training samples.
The real problem this project investigates is:

> **Does a quantum feature space offer any structural advantage over a classical
> kernel on real, non-trivially-separable image data under a fixed, equal
> computational budget?**

That is not the same as "train a quantum classifier." It is an empirical
scientific question about the geometry of quantum vs. classical feature spaces.

### Why This Is Scientifically Hard

1. **Fair comparison is deceptively difficult.** If quantum gets 1000 samples and
   classical gets 100, any difference is meaningless. This project enforces
   equal `n` per trial.
2. **Kernel concentration kills naive implementations.** At high qubit count
   with full angle range [0, π], all off-diagonal kernel values collapse to a
   single constant — the kernel matrix becomes proportional to the identity,
   carrying zero information. Most tutorials miss this (Thanasilp et al. 2022).
3. **Statistical noise from seeds.** One trial can be misleading; multi-seed
   paired tests are required to say anything credible.
4. **Hardware noise changes the kernel geometry.** On real quantum hardware,
   the computed kernel K_noisy ≠ K_ideal. This project models and measures
   that gap.

### Domain

| Layer | Domain |
|---|---|
| Core ML | Supervised classification, kernel methods |
| Quantum computing | Parameterised quantum circuits, state fidelity |
| Quantum ML | Quantum kernel methods, feature map design |
| Statistics | Hypothesis testing, effect sizes, bootstrap CI |
| Research methods | Ablation studies, reproducibility, provenance |

### Real-World Significance

- **Near-term quantum ML path**: Kernel methods are the most
  hardware-ready quantum ML approach because they offload learning to
  classical SVM and only use quantum hardware to evaluate K(x_i, x_j).
- **Quantum advantage research**: Understanding *when* g(K_Q, K_C) > 1 is a
  necessary milestone before claiming any advantage.
- **NISQ hardware benchmarking**: The noise robustness analysis directly
  informs which circuit configurations are viable on real QPUs today.

---

## Step 2 — Concept Map

### Concept 1: The Kernel Trick

**Simple**: Instead of explicitly computing features, measure similarity
between data points. If you can define a meaningful similarity function
K(x_i, x_j), you can train an SVM without ever constructing features manually.

**Advanced**: The kernel trick works because SVMs only need inner products of
feature vectors, not the vectors themselves. If φ(x) is a feature map, then
K(x_i, x_j) = ⟨φ(x_i), φ(x_j)⟩. You never need φ explicitly — you just need K.

**Why used here**: Classical RBF kernel implicitly maps to an infinite-dimensional
Gaussian feature space. Quantum kernel maps to a 2^n-dimensional Hilbert space.
Both are tractable through the kernel trick.

**Connects to**: Feature maps, SVM dual formulation, Hilbert space.

---

### Concept 2: Hilbert Space as a Feature Space

**Simple**: When you encode classical data into a quantum state, the state
lives in a complex vector space of dimension 2^n (for n qubits). That space
is called a Hilbert space.

**Advanced**: For 8 qubits, the Hilbert space has 2^8 = 256 complex dimensions.
Critically, the structure of this space is determined by the quantum circuit —
not freely chosen like PCA. The ZZ gates create entangled features that have
no classical polynomial equivalent.

**Why used here**: The hope is that the Hilbert space contains separating
hyperplanes for the 4-vs-9 task that the RBF Gaussian kernel cannot find.
The geometric difference g quantifies whether this is even possible.

**Connects to**: Quantum feature maps, fidelity kernel, entanglement.

---

### Concept 3: ZZFeatureMap

**Simple**: A quantum circuit that takes a classical data vector x and rotates
qubits based on x_i (single-qubit gates) and x_i × x_j (two-qubit gates).

**Advanced**: The circuit structure is:
```
H^⊗n → [Z rotations by x_i] → [ZZ rotations by x_i × x_j] → (repeat reps times)
```
The ZZ interaction encodes *pairwise feature products*, creating correlations
that no single-qubit gate can capture alone. This is why entanglement matters.

**Why used here**: ZZFeatureMap is the standard Qiskit feature map for
quantum kernel experiments. It has known expressibility and the interaction
structure creates features correlated across PCA components.

**Connects to**: Entanglement, expressibility, circuit depth (reps).

**Critical subtlety**: reps=1 is deliberately chosen. Higher reps:
- Increases circuit depth → more noise
- Increases gate count → longer simulation time
- Does NOT necessarily increase useful expressibility for ≤8 qubits
- Can increase kernel concentration if combined with [0, π] angle range

---

### Concept 4: Fidelity Quantum Kernel

**Simple**: The quantum kernel K(x_i, x_j) tells you how similar two
data points are in quantum feature space. It is computed by running a quantum
circuit and measuring the probability of getting all zeros.

**Advanced**:
```
K(x_i, x_j) = |⟨φ(x_i) | φ(x_j)⟩|²
             = |⟨0|^⊗n U†(x_i) U(x_j) |0⟩^⊗n|²
```
This is called the *fidelity* between two quantum states. It is:
- Always in [0, 1]
- Symmetric: K(x_i, x_j) = K(x_j, x_i)
- Diagonal K(x, x) = 1 (a state has fidelity 1 with itself)
- Positive semi-definite by construction

The Qiskit implementation uses `ComputeUncompute` fidelity: run U(x_j), then
U†(x_i), measure. The all-zeros probability is K.

**Why used here**: It is the natural "inner product" in quantum feature space.
FidelityQuantumKernel in Qiskit handles all circuit construction, transpilation,
and batched evaluation automatically.

**Connects to**: Overlap circuit, sampler primitive, PSD regularisation.

---

### Concept 5: Kernel-Target Alignment (KTA)

**Simple**: KTA measures how well the kernel's notion of similarity matches
the actual class labels. KTA=1 means the kernel perfectly knows which points
are in the same class. KTA=0 means random.

**Advanced**:
```
KTA(K, y) = ⟨K, K_y⟩_F / (‖K‖_F · ‖K_y‖_F)
```
where K_y = y·yᵀ is the ideal target kernel (K_y[i,j] = +1 same class, -1 different).
⟨·,·⟩_F is the Frobenius inner product — flatten both matrices, take dot product.

This is used as a **proxy for expected SVM performance before training**. High
KTA → the kernel geometry already aligns with the decision boundary.

**Why used here**: KTA enables comparing quantum vs. classical kernel quality
*without* training an SVM. It is used in the depth ablation to see how reps
affects kernel quality independently of classifier training noise.

**Connects to**: Frobenius norm, geometric difference, expressibility correlation.

---

### Concept 6: Exponential Concentration

**Simple**: As the number of qubits grows and the angle range [0, π] is used,
all quantum kernel values collapse to the same number — the kernel matrix
becomes useless.

**Advanced**: For a random data distribution, the variance of off-diagonal
kernel entries decays exponentially with qubit count:
```
Var[K(x_i, x_j)] ∝ e^{-n}
```
When all entries are equal, K ≈ c·I (identity times a constant), and the SVM
has no gradient signal — every point looks like every other point.

**Detection in code** (`quantum_kernel_engine.py`, `monitor_exponential_concentration`):
```python
off_diagonal_variance = np.var(kernel_matrix[upper_triangle])
if off_diagonal_variance < 1e-4:
    logger.warning("CRITICAL: Exponential Concentration detected!")
```

**Mitigation**: Scale features to [0, π/2] instead of [0, π]. This halves
the Bloch sphere rotation range, keeping off-diagonal entries meaningfully spread.

**Why critical**: A tutorial that scales to [0, π] and uses 8+ qubits will
silently produce a collapsed kernel and report near-random F1, wrongly
concluding "quantum doesn't work." The preprocessing here is designed to
prevent this.

---

### Concept 7: Geometric Difference g(K_Q, K_C)

**Simple**: g tells you whether the shape of the quantum kernel matrix is
fundamentally different from the classical kernel matrix. If g > 1, they
differ enough that quantum has the *possibility* of being better.

**Advanced** (Huang et al. 2022):
```
g(K_Q, K_C) = sqrt( ‖K_C^{1/2} · K_Q^{-1} · K_C^{1/2}‖₂ )
```
- K_C^{1/2}: square root via eigendecomposition (always exists for PSD matrices)
- K_Q^{-1}: inverse of quantum kernel (requires regularisation: K_Q + εI)
- ‖·‖₂: spectral norm = largest singular value

**Interpretation**:
- g > 1: Necessary (NOT sufficient) for quantum advantage
- g ≤ 1: Quantum kernel is a contraction of classical kernel → no advantage possible
- g = 1: Kernels are identical up to scaling

**Critical distinction**: g > 1 does NOT guarantee quantum is better on a
specific task. It only guarantees the kernel geometries differ enough that
advantage is *possible* in principle.

**Connects to**: KTA (if g > 1 but KTA_Q < KTA_C, the quantum kernel spans
a different space but that space doesn't help for this task).

---

### Concept 8: Circuit Expressibility ε

**Simple**: Expressibility measures how much of the full quantum state space
a circuit can reach. A highly expressive circuit can generate many different
quantum states for different parameter values.

**Advanced** (Sim et al. 2019):
```
ε = KL( P_PQC(F ; θ) ‖ P_Haar(F) )
```
- P_PQC: Distribution of fidelities |⟨φ(θ₁)|φ(θ₂)⟩|² over random parameter pairs
- P_Haar: Fidelity distribution for Haar-random states = Beta(1, 2^n − 1)
- KL: Kullback-Leibler divergence (measures how different two distributions are)

**Interpretation**:
- ε → 0: Circuit is Haar-random → maximally expressive → prone to barren plateaus
- ε → ∞: Circuit only reaches a tiny subspace → easy to train but limited
- For ZZFeatureMap(reps=1), ε is moderate (not random, not trivial)

**Trade-off**: High expressibility + high qubit count → barren plateaus (gradients vanish). This is why ZZFeatureMap at reps=1 is a deliberate balance.

**In code** (`expressibility.py`):
1. Sample 1000 random parameter pairs (θ, φ)
2. Compute |⟨φ(θ)|φ(φ)⟩|² for each pair using `Statevector`
3. Build histogram of these fidelities
4. Compare against Beta(1, 2^n−1) histogram
5. KL divergence = ε

---

### Concept 9: Pegasos Optimisation

**Simple**: Pegasos is a fast SVM solver that uses random mini-batches instead
of solving a full quadratic program. It makes quantum kernel SVM practical.

**Advanced**: Pegasos solves the primal SVM objective with stochastic subgradient descent:
```
min_{w}  λ/2 ‖w‖² + (1/n) Σ max(0, 1 − y_i ⟨w, φ(x_i)⟩)
```
At step t, using mini-batch A_t:
```
η_t = 1/(λ·t)
w_{t+1} = (1 − η_t·λ) w_t + η_t/|A_t| Σ_{i∈A_t: y_i⟨w_t,φ(x_i)⟩<1} y_i φ(x_i)
```
In kernel form (with precomputed K):
```
w_{t+1} corresponds to: α_i updates on support vectors
```

**Critical hyperparameter**: λ = 0.001 (not the default 1.0). Why?
- λ corresponds to 1/(C·n) in standard SVM notation
- λ = 1.0, n = 100 → C = 0.01 (extreme over-regularisation)
- Model predicts majority class for all inputs (collapse)
- λ = 0.001, n = 100 → C = 10 (reasonable)

---

### Concept 10: Bootstrap Confidence Intervals

**Simple**: Run the experiment with multiple different random seeds. A bootstrap
CI tells you the range within which the true mean performance falls, with 95%
confidence — accounting for finite-sample randomness.

**Advanced**:
```
For observed scores [s₁, s₂, ..., sₖ] across k seeds:
  For i = 1 to B=2000 bootstrap rounds:
    Sample k scores with replacement
    Record mean_i
  CI_95 = [percentile(2.5), percentile(97.5)] of {mean_i}
```

**Why it matters**: With 3 seeds, the sample mean is volatile. Bootstrap CI
quantifies how reliable the mean estimate is. Overlapping CIs for quantum and
classical means there is no evidence of difference.

---

### Concept 11: Cohen's d Effect Size

**Simple**: p-values tell you *whether* there's a difference; Cohen's d tells
you *how big* the difference is. A statistically significant but tiny effect
is scientifically unimportant.

**Formula**:
```
d = (mean_Q − mean_C) / pooled_std

pooled_std = sqrt( [(n_Q-1)·σ²_Q + (n_C-1)·σ²_C] / (n_Q + n_C - 2) )
```

**Interpretation**:
- |d| < 0.2: Negligible
- |d| ≈ 0.5: Medium
- |d| ≥ 0.8: Large

---

### Concept 12: Bonferroni Correction

**Simple**: If you run 3 tests (for PCA dims 4, 6, 8), you have a 3× higher
chance of a false positive by luck. Bonferroni correction tightens the
threshold to compensate.

**Formula**: α_corrected = α / m = 0.05 / 3 ≈ 0.0167

A result is only significant if p < 0.0167, not p < 0.05.

---

### Concept 13: PSD Regularisation

**Simple**: Noisy quantum kernels sometimes have tiny negative eigenvalues
(numerically, matrices should have all non-negative eigenvalues). Regularisation
fixes this so the SVM solver doesn't crash.

**Strategies in code**:

**Shift** (fast):
```
K_regularised = K + |min_eigenvalue| * I + ε * I
```
Pros: O(n²) cost. Cons: changes all eigenvalues equally.

**Clip** (exact):
```
K = V · diag(λ) · Vᵀ  (eigendecomposition)
K_regularised = V · diag(max(λ, ε)) · Vᵀ  (clip negatives)
```
Pros: Nearest PSD matrix in Frobenius norm. Cons: O(n³) cost.

After regularisation, diagonal is re-normalised to 1.0 to restore the
fidelity kernel property (K(x,x) = 1).

---

## Step 3 — All Algorithms, Deep Breakdown

### Algorithm 1: Classical RBF SVM with GridSearchCV

**Intuition**: Implicitly maps data to infinite-dimensional Gaussian feature
space. Data points that are geometrically close get kernel value ≈ 1;
distant points get ≈ 0. SVM finds the maximum-margin hyperplane in this space.

**Step-by-step**:
1. Compute RBF kernel: K(x_i, x_j) = exp(−γ ‖x_i − x_j‖²)
2. Solve dual QP: max Σα_i − ½ Σ Σ α_i α_j y_i y_j K(x_i,x_j)
3. Decision boundary: f(x) = sign(Σ α_i y_i K(x_i, x) + b)
4. GridSearchCV: Try C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, auto, 0.1, 0.01}, pick best 3-fold CV

**Location**: `src/classical_models.py`

**Strengths**: Fast, mature, interpretable via support vectors.
**Limitations**: Gaussian kernel may not capture entangled feature interactions.

---

### Algorithm 2: Quantum Fidelity Kernel Computation

**Intuition**: Run a quantum circuit for each pair of data points. Measure the
probability of getting all zeros. That probability IS the kernel value.

**Step-by-step**:
1. For each pair (x_i, x_j):
   a. Build circuit: U†(x_i) · U(x_j) (compute-uncompute)
   b. Execute on simulator (StatevectorSampler for ideal case)
   c. Read probability of measuring |00…0⟩
   d. K[i,j] = that probability
2. Build full n×n matrix
3. Check for PSD violation, monitor exponential concentration
4. Regularise if needed

**Critical implementation**: Qiskit's `StatevectorSampler` computes this
exactly without shot noise. For noisy simulation, `BackendSamplerV2` with
a noise model gives shot-limited estimates.

**Location**: `src/quantum_kernel_engine.py`, `compute_kernel_matrix()`

---

### Algorithm 3: Exact QSVC Training

**Intuition**: Use the precomputed quantum kernel matrix as input to a
standard SVM solver. Exactly the same as classical SVM, just with a quantum kernel.

**Step-by-step**:
1. Precompute K_quantum (n×n)
2. Pass to `QSVC(quantum_kernel=qk, C=1.0)`
3. Qiskit internally uses libsvm to solve the quadratic program
4. Prediction: f(x_test) = sign(K_test_train · α · y + b)

**Location**: `src/quantum_training.py`, `train_qsvc()`

---

### Algorithm 4: Pegasos QSVC Training

**Intuition**: Instead of solving a full QP (expensive for large n), use
stochastic subgradient steps on mini-batches. Each step costs O(batch_size²)
kernel evaluations instead of O(n²).

**Step-by-step**:
1. Precompute K_train (n×n) once
2. Initialise support vector set as empty
3. For t = 1 to max_iter:
   a. Draw random mini-batch A_t ⊂ {1..n}, size batch_size
   b. Compute decayed step size η_t = 1/(λ·t)
   c. Identify violated constraints in A_t (misclassified points)
   d. Update dual coefficients for support vectors
4. Prediction: precomputed kernel row K_test vs. support vectors

**Key convergence guarantee**: Pegasos finds ε-approximate solution in
O(1/(λ·ε)) iterations, independent of n. This is the memory efficiency win.

**Location**: `src/quantum_training.py`, `train_pegasos_qsvc()`

---

### Algorithm 5: Expressibility Computation (Sim et al. 2019)

**Intuition**: Sample many random parameter pairs from the circuit. Build a
histogram of fidelities. Compare it to what you'd get with a truly random
(Haar) circuit. KL divergence between them = expressibility.

**Step-by-step**:
1. For i = 1 to n_samples:
   a. Draw θ ~ Uniform[0, 2π]^k
   b. Draw φ ~ Uniform[0, 2π]^k
   c. Compute |⟨ψ(θ)|ψ(φ)⟩|² via Statevector inner product
2. Build 75-bin histogram of the n_samples fidelities
3. Sample n_samples×10 from Beta(1, 2^n−1) (Haar distribution)
4. Build histogram of Haar fidelities
5. Smooth both histograms: add ε=1e-10, normalise to probabilities
6. KL = Σ p_i · log(p_i / q_i)

**Location**: `src/expressibility.py`, `compute_expressibility()`

---

### Algorithm 6: Geometric Difference Computation (Huang et al. 2022)

**Step-by-step**:
1. Add regularisation: K_Q_reg = K_Q + ε·I (prevents singular matrix)
2. Invert: K_Q_inv = K_Q_reg⁻¹ (via np.linalg.inv)
3. Compute K_C^{1/2} via eigendecomposition: K_C = V·diag(λ)·Vᵀ → K_C^{1/2} = V·diag(√λ)·Vᵀ
4. Form M = K_C^{1/2} · K_Q_inv · K_C^{1/2}
5. Spectral norm: ‖M‖₂ = largest singular value (= largest eigenvalue for PSD M)
6. g = √‖M‖₂

**Location**: `src/quantum_kernel_engine.py`, `compute_geometric_difference()`

---

### Algorithm 7: Bootstrap Confidence Interval

**Step-by-step**:
1. Collect scores = [F1_seed42, F1_seed43, F1_seed44]
2. For i = 1 to 2000:
   a. Resample scores with replacement → boot_sample
   b. Record mean(boot_sample) → boot_means[i]
3. CI_lower = percentile(boot_means, 2.5)
4. CI_upper = percentile(boot_means, 97.5)

**Why 2000 resamples?** Converges to stable CI estimates. 10000 would be
more precise but unnecessarily slow for 3-seed experiments.

**Location**: `src/evaluation_metrics.py`, `bootstrap_confidence_interval()`

---

### Algorithm 8: Meyer-Wallach Entanglement Capability

**Intuition**: Measure how entangled the average output state is across
random parameters. 0 = separable (product state), 1 = maximally entangled.

**Formula**:
```
Q̄ = (4/n) · Σ_{k=1}^{n} E_θ[SA(ρ_k)]  where SA is linear entropy
```
SA(ρ_k) = 1 − Tr(ρ_k²) for the k-th qubit's reduced density matrix.

**Location**: `src/expressibility.py`, `compute_entanglement_capability()`

---

## Step 4 — Code-to-Concept Mapping

### `src/preprocessing.py`

| Function | Concept | Why Designed This Way |
|---|---|---|
| `normalize_pixels` | Data normalisation | Pixel range [0, 255] → [0, 1]; prevents numerical issues |
| `standardize_features` | Zero-mean normaliation | Required before PCA; equal weight per dimension |
| `apply_pca` | Dimensionality reduction | 784 pixels → d components = d qubits; fit on train only |
| `parse_feature_range` | Angle encoding range | Parses "pi/2" string to (0, π/2) float pair |
| `scale_for_quantum` | Quantum angle range | Applies `data_min` from **train** to both train and test |
| `preprocess_data` | Full pipeline | Returns two versions: PCA only (for classical) + PCA+quantum-scale (for quantum) |

**Critical design**: `scale_for_quantum` takes `data_min` / `data_max` from
training data and applies them to the test set. This prevents data leakage
from test → train distribution. A common bug is to scale train and test
independently — the test would then have different numeric ranges.

---

### `src/quantum_kernel_engine.py`

| Function | Concept | Key Detail |
|---|---|---|
| `_build_default_sampler` | Qiskit version compat | Tries StatevectorSampler first (Qiskit 1.x+), falls back to legacy Sampler |
| `create_quantum_kernel` | Fidelity kernel construction | Introspects FidelityQuantumKernel API via `inspect.signature` to handle API changes |
| `monitor_exponential_concentration` | Concentration detection | Checks off-diagonal variance; warns if < 1e-4 |
| `compute_kernel_matrix` | Kernel evaluation | Wraps `quantum_kernel.evaluate()` with monitoring |
| `analyze_kernel_properties` | Kernel diagnostics | Checks symmetry, PSD, condition number |
| `regularize_kernel_matrix` | PSD correction | Two strategies: shift (fast) or clip (exact); re-normalises diagonal |
| `compute_kernel_target_alignment` | KTA | Maps labels to {-1,+1}, builds K_y, calls `compute_kernel_alignment` |
| `compute_geometric_difference` | Advantage metric | Eigendecomposition of K_C for square root; inv of K_Q with Tikhonov reg |
| `get_git_sha` | Provenance | `subprocess("git rev-parse --short HEAD")` for result JSON stamping |

---

### `src/quantum_training.py`

| Function | Concept | Key Detail |
|---|---|---|
| `create_pegasos_qsvc` | Pegasos instantiation | Handles TWO different Qiskit ML API versions via `inspect.signature` |
| `train_qsvc` | Exact QSVC | Standard SVM solve on quantum kernel; optional GridSearchCV |
| `train_pegasos_qsvc` | Pegasos training | Passes `num_samples=len(X_train)` for correct λ-to-C conversion |

**The API version problem**: In older Qiskit ML, `PegasosQSVC` takes
`lambda_param`. In newer versions it takes `C` and `num_steps`. The code
uses `inspect.signature` to detect which API is available and map parameters
correctly. This is production-grade defensive programming.

---

### `src/evaluation_metrics.py`

| Function | Concept | Key Detail |
|---|---|---|
| `compute_metrics` | Standard metrics | Auto-detects `pos_label` as max class label if not specified |
| `bootstrap_confidence_interval` | Bootstrap CI | Uses `np.random.default_rng` (modern API) not `np.random.seed` |
| `cohens_d` | Effect size | Pooled std uses Bessel's correction (ddof=1) for unbiased estimate |
| `bonferroni_correct` | Multiple testing | Returns `corrected_alpha = α/m` and `reject` list |
| `calculate_statistical_significance` | Full comparison | Paired t-test + Wilcoxon + Cohen's d + Bootstrap CI in one call |

**Why both t-test and Wilcoxon?** Paired t-test assumes normality (not guaranteed
with 3 seeds). Wilcoxon signed-rank test is non-parametric — no normality
assumption. Publishing both is honest.

---

### `src/noise_simulation.py`

| Function | Concept | Key Detail |
|---|---|---|
| `create_ibm_noise_model` | Noise model | Tries `FakeBrisbane` first (real specs); falls back to manual T1/T2 model |
| `create_noisy_sampler` | Noisy sampler | Prefers `BackendSamplerV2`; falls back to `AerSamplerV1` |
| `simulate_noisy_kernel` | Noisy kernel eval | Calls `feature_map.decompose()` before evaluation — Aer cannot handle high-level library gates |
| `analyze_noise_effects` | ΔK analysis | Computes MAE, RMSE, Frobenius norm, spectral norm, correlation between K_ideal and K_noisy |

**The `.decompose()` call** is critical. Qiskit's `ZZFeatureMap` creates high-level
`UnitaryGate` objects that the Aer simulator cannot simulate directly. `.decompose()`
translates them to basis gates (Rz, CX, etc.) that Aer understands.

---

### `src/expressibility.py`

| Function | Concept | Key Detail |
|---|---|---|
| `sample_fidelity_distribution` | PQC fidelity distribution | Uses `Statevector.inner()` — exact, no shot noise |
| `haar_fidelity_distribution` | Reference distribution | Analytically correct: Beta(1, D-1) for D = 2^n qubits |
| `kl_divergence_histogram` | KL from samples | 75 bins, ε smoothing before normalisation to avoid log(0) |
| `compute_expressibility` | ε computation | Samples 10× more Haar points than PQC points for stable reference |
| `compute_entanglement_capability` | Meyer-Wallach Q̄ | Traces out each qubit, measures linear entropy of reduced state |

---

## Step 5 — Complete Mathematics

### M1: ZZFeatureMap Circuit

For n-qubit ZZFeatureMap with data x = [x₁, x₂, ..., xₙ]:

Single-qubit layer (Pauli-Z rotation):
```
exp(i·xⱼ·Zⱼ) = [[e^{ix_j}, 0      ],
                   [0,       e^{-ix_j}]]
```

Two-qubit ZZ interaction:
```
exp(i·xⱼ·xₖ·Zⱼ⊗Zₖ) = diag(e^{ix_jx_k}, e^{-ix_jx_k}, e^{-ix_jx_k}, e^{ix_jx_k})
```

Full circuit (one rep):
```
|φ(x)⟩ = [H^⊗n] · [Rz(2x_i)] · [Rzz(2x_ix_j for i<j)] · |0^n⟩
```

---

### M2: Fidelity Kernel

```
K(xᵢ, xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
           = Tr[ρ(xᵢ) ρ(xⱼ)]     (for pure states ρ = |φ⟩⟨φ|)
           = Pr[measure |0^n⟩ | U†(xᵢ) U(xⱼ) |0^n⟩]
```

Properties:
- K: ℝᵈ × ℝᵈ → [0, 1]
- K is symmetric: K(x, x') = K(x', x)
- K is positive semi-definite (Mercer's theorem applies)
- K(x, x) = 1 ∀x (self-overlap = 1 for normalised states)

---

### M3: Kernel-Target Alignment

```
KTA(K, y) = ⟨K, K_y⟩_F / (‖K‖_F · ‖K_y‖_F)

where: K_y = y·yᵀ,  yᵢ ∈ {-1, +1}
⟨A, B⟩_F = Σᵢⱼ Aᵢⱼ·Bᵢⱼ = Tr(AᵀB)
‖A‖_F = √⟨A,A⟩_F
```

Range: KTA ∈ [-1, 1]. In practice for binary classification, KTA ∈ [0, 1].

---

### M4: Geometric Difference

```
g(K_Q, K_C) = [‖K_C^{1/2} · K_Q^{-1} · K_C^{1/2}‖₂]^{1/2}

where ‖·‖₂ = spectral norm = σ_max (largest singular value)
```

Numerical implementation:
```python
K_c^{1/2} = V @ diag(sqrt(λ)) @ V.T   # eigendecomp
M = K_c_sqrt @ inv(K_Q + εI) @ K_c_sqrt
g = sqrt(np.linalg.norm(M, ord=2))
```

---

### M5: Expressibility (KL Divergence)

```
ε = KL(P_PQC ‖ P_Haar) = Σᵢ P_PQC(Fᵢ) · log(P_PQC(Fᵢ) / P_Haar(Fᵢ))

P_Haar ~ Beta(1, D-1)  where D = 2^n
P_PQC estimated from n_samples fidelity values {|⟨φ(θ)|φ(φ)⟩|²}
```

---

### M6: Pegasos Objective and Update

```
Primal SVM objective:
min_{w} λ/2 ‖w‖² + (1/n) Σᵢ max(0, 1 − yᵢ⟨w, φ(xᵢ)⟩)

Pegasos update (step t):
η_t = 1/(λ·t)
Violation set: Sᵢ ∈ Aₜ: yᵢ⟨wₜ, φ(xᵢ)⟩ < 1
w_{t+1} = (1−η_t·λ)·wₜ + (η_t/|Aₜ|) · Σᵢ∈Sᵢ yᵢ·φ(xᵢ)
```

In kernel form, decision function:
```
f(x) = Σᵢ αᵢ·yᵢ·K(xᵢ, x) + b
```

---

### M7: Cohen's d

```
d = (μ_Q − μ_C) / s_pooled

s_pooled = sqrt{[(n_Q-1)·σ²_Q + (n_C-1)·σ²_C] / (n_Q + n_C - 2)}
```

---

### M8: Bootstrap CI

```
For scores {s₁,...,sₙ}:
  boot_means = {mean(resample(scores)) : b=1..B}
  CI = [Q(boot_means, α/2), Q(boot_means, 1−α/2)]
where Q is the quantile function, B=2000, α=0.05
```

---

### M9: Noisy Kernel Model

With readout error rate p and gate error rate q per gate:

Single qubit readout error matrix:
```
[[1-p, p  ],
 [p,   1-p]]
```

For full circuit with gate depth d and n qubits, the noisy kernel satisfies:
```
K_noisy = K_ideal + ΔK   where ‖ΔK‖_F = O(p·d·n)
```

The noise degrades both PSD property and KTA.

---

## Step 6 — Study Notes

### 6.1 Quick Revision Notes (5-minute read)

```
QUANTUM KERNEL SVM — KEY IDEAS

1. Problem: Does quantum feature space help classify MNIST 4 vs 9?

2. Pipeline: Pixels → PCA(d) → Scale[0,π/2] → ZZFeatureMap(d qubits) →
             Fidelity K(x_i,x_j) = |<φ(x_i)|φ(x_j)>|² → SVM

3. Why [0,π/2]? Prevents kernel concentration (all values → same constant)

4. Three models: Classical RBF, Exact QSVC, Pegasos QSVC (same n for all)

5. Key metrics:
   - F1: Classification performance
   - KTA: Kernel geometry quality (before training)
   - g: Geometric difference (g>1 = quantum might help)
   - ε: Expressibility (lower = more Haar-like)

6. Statistics: Paired t-test + Bonferroni + Bootstrap CI + Cohen's d

7. Noise: IBM Brisbane model, readout 1%, measures ΔK = K_noiseless − K_noisy

8. Pegasos: λ=0.001 NOT 1.0 (λ=1.0 causes model collapse at n=100)

9. Git SHA stamps every result JSON (reproducibility)

10. Result: No quantum advantage demonstrated. g often >1, but KTA_Q < KTA_C.
```

---

### 6.2 Detailed Notes

#### Why ZZFeatureMap?

The ZZFeatureMap has a specific mathematical structure where two-qubit
interactions encode the product x_i · x_j. These quadratic feature interactions
are captured by ZZ gates. In the dual SVM space, this means the effective
feature vector includes cross-terms between all pairs of PCA components — a
structure that classical polynomial kernels also capture, but in classical space.

The hope is that the quantum circuit creates feature interactions that are
computationally hard to replicate classically — this is the quantum advantage
hypothesis. However, for n ≤ 8 qubits, statevector simulation is exact and
fast, so there is no classical hardness in this experiment (it's all simulated).
Advantage would only emerge on real hardware or at n >> 50.

#### Why PCA to exactly d = n_qubits?

Each PCA component becomes the rotation angle for one qubit. ZZFeatureMap
requires feature_dimension = number of qubits. So if you want 8 qubits, you
need exactly 8 features. PCA to d=8 then quantum-scales to [0, π/2] maps each
principal component to a qubit rotation angle.

#### The Fair Comparison Design

Both classical and quantum use:
- Same n training samples (controlled via `max_quantum_train`)
- Same preprocessed features (after PCA, using the PCA-features for classical,
  and quantum-scaled features for quantum)
- Same evaluation metrics (F1, KTA)
- Same seeds per ablation cell

This is harder than it sounds. Early quantum ML papers often used different
preprocessing for quantum and classical, making comparison meaningless.

---

### 6.3 Key Formulas Table

| Formula | Symbol | Where Used |
|---|---|---|
| K(xᵢ,xⱼ)=\|⟨φ(xᵢ)\|φ(xⱼ)⟩\|² | Fidelity kernel | kernel matrix eval |
| KTA=⟨K,K_y⟩_F/(‖K‖_F‖K_y‖_F) | Kernel-Target Alignment | quantum vs classical kernel quality |
| g=√‖K_C^{½}K_Q^{-1}K_C^{½}‖₂ | Geometric difference | quantum advantage precondition |
| ε=KL(P_PQC‖P_Haar) | Expressibility | circuit capacity analysis |
| η_t=1/(λt) | Pegasos step size | Pegasos convergence |
| d=(μ_Q−μ_C)/s_pooled | Cohen's d | effect size |
| α_corr=α/m | Bonferroni | multiple testing |
| ΔK=K_noiseless−K_noisy | Noise deviation | noise robustness |
| Q̄=(4/n)Σ_k SA(ρ_k) | Meyer-Wallach | entanglement capability |

---

### 6.4 Important Concepts Summary

| Concept | One-Line | Common Trap |
|---|---|---|
| Quantum kernel | Fidelity between encoded states | Not the same as quantum gate computation |
| KTA | Proxy for SVM performance (no training needed) | High KTA ≠ high accuracy always |
| Exponential concentration | Kernel collapses to identity at high n | Most tutorials miss this |
| g(K_Q,K_C) | Necessary, NOT sufficient, for advantage | People claim advantage from g>1 alone — wrong |
| Pegasos λ | λ = 0.001, NOT 1.0 | λ=1.0 causes model collapse at small n |
| Expressibility ε | Low = Haar-like, but barren plateaus risk | Low ε is NOT always better |
| Bootstrap CI | Quantifies uncertainty in mean estimate | Does not replace significance testing |
| Bonferroni | Divide α by number of tests | Forgetting this inflates false positive rate |
| PSD regularisation | Clip or shift negative eigenvalues | Without this, SVM solver crashes on noisy K |

---

### 6.5 Common Mistakes to Avoid

| Mistake | Correct Approach |
|---|---|
| Scale test set independently | Always use train min/max for test scaling |
| Compare quantum (n=1000) vs classical (n=100) | Use equal n |
| Claim advantage from g > 1 alone | g > 1 is necessary, not sufficient |
| Use λ=1.0 for Pegasos at small n | Use λ=0.001 (corresponds to C~10) |
| Skip Bonferroni correction | Correct α_corr = 0.05/m across m tests |
| Use [0, π] angle range | Use [0, π/2] to prevent concentration |
| Claim real noise analysis without hardware | Label it clearly as "simulated noise" |
| Trust 1-seed results | Use ≥3 seeds (10+ for publication) |
| Forget .decompose() before Aer simulation | ZZFeatureMap needs transpilation |
| Report accuracy only | Report F1 + KTA + g + ε for completeness |

---

## Step 7 — Learning Path

### Beginner (Weeks 1–3)

**Goal**: Understand classical SVM and kernel methods completely.

1. **Linear SVM**: Understand the maximum-margin hyperplane. Learn the dual formulation.
   - Resource: PRML Chapter 7 (Bishop), or sklearn docs
2. **Kernel Trick**: Understand why K(x_i,x_j) replaces ⟨φ(x_i),φ(x_j)⟩.
   - Practice: Implement RBF kernel from scratch. Compare to sklearn's output.
3. **PCA**: Understand variance maximisation, eigendecomposition of covariance matrix.
   - Practice: PCA from scratch on MNIST. Plot explained variance.
4. **Read**: Havlíček et al. (2019) — skim sections 1-2.
5. **Run**: `python run_experiment.py --fallback --disable-noise --max-quantum-train 40`
   - Look at the output CSV. Understand each column.

**Practice project**: Build a classical RBF SVM on MNIST 4 vs 9 from scratch.
Understand every line of `src/classical_models.py` and `src/preprocessing.py`.

---

### Intermediate (Weeks 4–8)

**Goal**: Understand quantum circuits, quantum kernels, and KTA.

1. **Quantum computing basics**: Qubits, gates, measurement, circuits.
   - Resource: Nielsen & Chuang chapters 1–2, or Qiskit Textbook (free online)
2. **ZZFeatureMap**: Draw the circuit by hand for n=2. Trace what happens to |00⟩.
   - Practice: Build a 2-qubit ZZFeatureMap manually in Qiskit from gates.
3. **Fidelity kernel**: Understand the compute-uncompute circuit pattern.
   - Practice: Compute K(x_1, x_2) manually for two 2D points. Verify against `quantum_kernel.evaluate()`.
4. **KTA**: Implement KTA from scratch. Verify against `compute_kernel_target_alignment()`.
5. **Read**: Thanasilp et al. (2022) — the exponential concentration paper.
6. **Experiment**: Modify `feature_range` from `pi/2` to `pi` in config. Observe KTA collapse.

**Practice project**: Build the kernel computation pipeline from scratch
using only numpy + Qiskit. Compare to the project's output for the same input.

---

### Advanced (Weeks 9–16)

**Goal**: Understand the research contributions and conduct your own experiments.

1. **Expressibility**: Derive why Haar fidelity ~ Beta(1, D-1). Implement KL divergence.
   - Practice: Test expressibility of ZZFeatureMap for reps ∈ {1, 2, 3}. Does it decrease?
2. **Geometric difference**: Derive the formula. Understand why g > 1 is necessary but not sufficient.
   - Practice: Compute g for the project outputs. What PCA dimension gives max g?
3. **Statistical testing**: Implement paired t-test and Bonferroni from scratch.
   - Practice: Run 10 seeds. Do you get p < 0.05/3 for any dimension?
4. **Noise model**: Study T1, T2, thermal relaxation. Understand what each noise source does to the kernel.
   - Practice: Sweep readout_error from 0.0 to 0.1. Plot KTA vs error rate.
5. **Read**: Huang et al. (2022) — understand proposition 3 (geometric difference bound).
6. **Modify**: Add PauliFeatureMap to the `feature_map_registry`. Run comparison against ZZFeatureMap.

**Research project**: Can you find a data distribution where g > 1 AND KTA_Q > KTA_C?
This would be the first step toward demonstrating actual quantum advantage.

---

## Step 8 — Interview & Research Questions

### Interview Questions (Technical)

**Q1**: What is a quantum kernel? How is it computed?  
**Model Answer**: A quantum kernel measures fidelity between two quantum states
encoding classical data. K(x_i,x_j) = |⟨φ(x_i)|φ(x_j)⟩|². Computed by running
U†(x_i)·U(x_j) on a QPU and measuring the |0^n⟩ probability.

**Q2**: Why is [0, π/2] used instead of [0, π] for quantum feature scaling?  
**Model Answer**: At [0, π], the ZZFeatureMap kernel suffers exponential
concentration — off-diagonal variance decays as e^{-n} with qubit count.
All kernel entries converge to a constant, making the kernel information-useless.
[0, π/2] halves the Bloch sphere rotation, preventing this collapse.

**Q3**: What is Kernel-Target Alignment and why use it?  
**Model Answer**: KTA = ⟨K, K_y⟩_F/(‖K‖_F·‖K_y‖_F). It measures how well
the kernel function's similarity aligns with actual class labels. Used here as
a training-free proxy for expected SVM performance, enabling comparison of
kernel quality across configurations without training.

**Q4**: What does g(K_Q, K_C) > 1 mean?  
**Model Answer**: It is a necessary (not sufficient) condition for quantum kernel
advantage. It means the quantum kernel cannot be expressed as a contraction of
the classical kernel in spectral norm. g ≤ 1 guarantees no quantum advantage
on any learning task. g > 1 means advantage is *possible* but not guaranteed.

**Q5**: Why is λ = 0.001 used in Pegasos instead of the default 1.0?  
**Model Answer**: λ relates to C via C = 1/(λ·n). With n=100 and λ=1.0,
C = 0.01 — extreme over-regularisation that forces the model to predict
the majority class. λ=0.001 gives C=10, which is a reasonable regularisation
for a near-separable binary classification task.

**Q6**: What is the difference between Exact QSVC and Pegasos QSVC?  
**Model Answer**: Exact QSVC solves the full quadratic program via libsvm —
exact solution, O(n³) worst-case, memory O(n²). Pegasos uses stochastic
subgradient descent with mini-batches — approximate solution, O(1/(λε))
iterations regardless of n, memory-efficient. Pegasos is needed when n is
large; at n=100 both are tractable.

---

### Conceptual Questions

**Q7**: Can quantum kernel SVM offer quantum advantage for MNIST?  
**Discussion**: MNIST is classically easy (RBF SVM achieves 99%+ full dataset).
For n=100 samples, the problem is small. Statevector simulation is exact —
no classical hardness. Quantum advantage requires: (a) classically hard kernel,
(b) g > 1, (c) the quantum feature space must contain a better separating
hyperplane. None of these are guaranteed for MNIST. The geometric difference
here is a diagnostic, not a demonstration.

**Q8**: What is exponential concentration and how does it kill quantum kernels?  
(See Concept 6 above for detailed answer)

**Q9**: Explain the trade-off between expressibility and trainability.  
**Discussion**: A highly expressive circuit (ε → 0) can reach any quantum state
— spanning the full Hilbert space. However, for L-layer circuits on n qubits,
gradients of the objective vanish as O(2^{-n}) — barren plateaus. A low-depth,
low-expressibility circuit trades coverage for trainable gradients.
ZZFeatureMap at reps=1 is deliberately shallow to avoid this.

---

### Deep Research Questions

**Q10**: Design an experiment to measure whether g(K_Q, K_C) > 1 actually implies
better quantum performance on any classification task. What data distribution
would you use and why?

**Q11**: The current experiment uses n=100. At what n would quantum kernel
computation become computationally advantageous over classical kernel computation,
assuming ideal quantum hardware?

**Q12**: How would you incorporate quantum error correction into this pipeline,
and what circuit depth would be required to maintain K accuracy within 1% of
the statevector ideal?

**Q13**: Propose a modification to ZZFeatureMap that might increase g(K_Q, K_C)
without increasing circuit depth. Justify mathematically.

**Q14**: The Bonferroni correction is conservative for correlated tests. Would
Holm-Bonferroni or BH (Benjamini-Hochberg) correction be more appropriate here?
What are the null hypotheses in this multiple testing setup?

---

## Step 9 — Weaknesses & Gaps

### Conceptual Gaps in the Current Implementation

| Gap | Description | What to Study |
|---|---|---|
| No covariant kernels | ZZFeatureMap doesn't respect MNIST's symmetry group | Glick et al. 2024 (covariant quantum kernels) |
| No kernel learning | No training of the feature map parameters | QKAT module exists but isn't connected to main loop |
| Fixed entanglement topology | "full" entanglement — no hardware-aware layout | Qiskit transpiler, coupling map optimization |
| No shot budget analysis | Statevector is exact; real hardware needs shot counts | Shot noise vs. kernel accuracy trade-off |
| Small statistical power | 3 seeds insufficient for rigorous hypothesis testing | Requires 10–30 seeds for p < 0.05 reliability |
| No feature map diversity | Only ZZFeatureMap tested — no PauliFeatureMap or hardware-efficient ansatz | Feature map comparison literature |
| No barren plateau mitigation | No local cost functions or initialisation strategies | Cerezo et al. 2021 (barren plateau mitigation) |
| Dataset scope | MNIST 4 vs. 9 only — may not generalise | Cross-dataset: Fashion-MNIST, binary CIFAR-10 |

### Design Choices to Question

1. **Why Frobenius norm for KTA?** Frobenius norm treats all kernel entries
   equally. Centering the kernel before computing KTA (centered KTA) is more
   theoretically justified. Not implemented here.

2. **Why only binary test?** The ZZFeatureMap can in principle handle
   multi-class via OvR or OvO. Only binary (4 vs. 9) is tested.

3. **Why no cross-validation on quantum model?** Classical model uses GridSearchCV
   with 3-fold CV. Quantum uses fixed hyperparameters. A CV on C for QSVC would
   make the comparison more symmetric.

4. **Noise model scope**: The IBM Brisbane noise model is an approximation.
   Real T1/T2 values vary per qubit and per run session. The fixed per-qubit
   model underestimates spatial variability.

### What to Study Next

**Priority 1** (to make research publishable):
- Centered KTA (more robust to scale)
- ≥10 seed paired experiments
- PauliFeatureMap ablation
- Wilcoxon test for small n (done) but also report it alongside T-test

**Priority 2** (to go beyond this project):
- Quantum kernel learning (training feature map via gradient)
- Hardware deployment on IBM Quantum (replace statevector)
- Quantum error mitigation: Zero Noise Extrapolation (ZNE)

**Priority 3** (research frontier):
- Quantum advantage certification via SQ lower bounds
- Covariant kernels for MNIST's translation symmetry
- Quantum transfer learning (pre-trained feature maps)

### Essential Papers Beyond the Current Reference List

| Paper | Why Essential |
|---|---|
| Cerezo et al. (2021) *Nat. Rev. Phys.* — Barren plateaus | Understand why high expressibility is dangerous |
| Liu et al. (2021) *Nat. Phys.* — Quantum advantage kernel | One of the few proven quantum advantage results in kernel ML |
| Shawe-Taylor & Cristianini — Kernel Methods for Pattern Analysis | Classical foundation; understand before quantum |
| Schuld (2021) *Phys. Rev. A* — Quantum models are kernel methods | Shows all quantum ML is secretly kernel ML |
| Sweke et al. (2021) *Quantum* — Stochastic gradient descent for quantum | Extension of Pegasos to variational settings |

---

*End of Knowledge Base — Total: 10 full steps, all 13 core concepts, all 8 algorithms, complete mathematics, study notes, 14 interview questions, and systematic weakness analysis.*
