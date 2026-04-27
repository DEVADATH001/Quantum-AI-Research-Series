# Mathematical Formulation

## Weighted Communication-Mesh Max-Cut

For a communication graph `G = (V, E)` with weighted conflict score `w_ij > 0`,
the objective is

`C(z) = sum_(i,j in E) w_ij * (1 - z_i z_j) / 2`

where `z_i in {+1, -1}` is the Ising spin associated with node `i`.

In the communication-mesh generator, each edge weight is a convex combination
of normalized latency, local interference, and reliability loss:

`w_ij = 0.45 * latency_ij + 0.35 * interference_ij + 0.20 * (1 - reliability_ij)`

with

- `latency_ij = d_ij / max_(k,l) d_kl`
- `interference_ij = |N(i) ∩ N(j)| / (|V| - 2)`
- `reliability_ij = exp(-d_ij / s)`

where `d_ij` is Euclidean separation and `s` is the reliability scale.

## QAOA Cost Hamiltonian

The weighted Max-Cut Hamiltonian is

`H_C = sum_(i,j in E) w_ij * (I - Z_i Z_j) / 2`

which is decomposed in code as

`H_C = offset + H_ZZ`

with

- `offset = sum_(i,j in E) w_ij / 2`
- `H_ZZ = - sum_(i,j in E) w_ij * Z_i Z_j / 2`

The QAOA cost unitary is therefore

`U_C(gamma) = exp(-i gamma H_C) ~ exp(+i gamma sum_(i,j) w_ij Z_i Z_j / 2)`

up to a global phase, which matches the implementation convention

`RZZ(-gamma * w_ij) = exp(+i gamma w_ij Z_i Z_j / 2)`.

## Expected-Value Objective

The standard variational objective is

`J_exp(theta) = E_theta[C] = offset + <H_ZZ>_theta`

and the classical optimizer minimizes

`L_exp(theta) = -J_exp(theta)`.

## CVaR Objective

For a measurement distribution `p_theta(x)` over bitstrings `x` with cut values
`C(x)`, define the upper-tail CVaR objective at mass `alpha in (0, 1]` as

`J_CVaR(theta; alpha) = (1 / alpha) * sum_x q_theta(x) C(x)`

where `q_theta` is obtained by sorting bitstrings in descending `C(x)` and
retaining only the best `alpha` total probability mass.

Equivalently, if `(x_1, x_2, ...)` are ordered so that

`C(x_1) >= C(x_2) >= ...`

then

`J_CVaR(theta; alpha) = (1 / alpha) * sum_k min(p_theta(x_k), r_k) * C(x_k)`

with `r_1 = alpha` and `r_(k+1) = max(0, r_k - p_theta(x_k))`.

This satisfies

- `J_CVaR(theta; 1) = J_exp(theta)`
- `J_CVaR(theta; alpha) >= J_exp(theta)` for `alpha < 1`

because it emphasizes the high-value tail of the distribution.

The optimized loss is

`L_CVaR(theta; alpha) = -J_CVaR(theta; alpha)`.

## Budget-Matched Baselines

If QAOA uses `N` objective evaluations, then a fair black-box comparison can
assign the same evaluation budget `N` to classical stochastic heuristics.

This repo therefore includes:

- `random_budget_matched`: best of `N` random cuts
- `hill_climb_budget_matched`: stochastic hill climbing with at most `N`
  objective evaluations

These methods do not claim theoretical optimality, but they provide a clearer
fairness test than comparing QAOA only against under-budgeted heuristics.
