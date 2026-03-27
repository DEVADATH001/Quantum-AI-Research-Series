# Technical Note

## Scope

This project implements a measurement-defined quantum policy benchmark under ideal, noisy, and mitigated execution. The benchmark is intended as a reproducible systems-and-evaluation artifact, not as a claim of quantum advantage.

## Policy Definition

For state `s` and trainable parameters `theta`, the quantum actor prepares

`|psi(theta, s)> = U(theta, s) |0>`

and defines the policy directly from Born probabilities on the action register:

`pi_theta(a | s) = Pr[M_action = a]`.

This is a measurement-native policy. It is not a classical softmax over expectation-value logits.

## REINFORCE Baseline Method

The legacy quantum baseline uses reward-to-go `G_t` and an action-independent baseline `b_t`:

`A_t = G_t - b_t`

with surrogate objective

`L_pg(theta) = - sum_t A_t log pi_theta(a_t | s_t) - beta sum_t H(pi_theta(. | s_t))`.

Under ideal parameter-shift-compatible gates, the action-probability derivative satisfies

`d pi_theta(a | s) / d theta_i = [pi_(theta_i + pi/2)(a | s) - pi_(theta_i - pi/2)(a | s)] / 2`.

This identity is exact only for ideal Born probabilities. Under finite shots, noise, and mitigation, the estimator is stochastic and can be biased.

## Actor-Critic Upgrade

The main upgraded method is a quantum actor with a classical value critic.

- Actor: measurement-defined quantum policy.
- Critic: classical MLP value function on one-hot state features.
- Advantage estimator: generalized advantage estimation (GAE).

With value estimates `V_phi(s_t)`, temporal-difference residuals

`delta_t = r_t + gamma V_phi(s_(t+1)) - V_phi(s_t)`

and GAE parameter `lambda`, the advantages are

`A_t = sum_l (gamma lambda)^l delta_(t+l)`.

The actor uses the policy-gradient surrogate above with `A_t` in place of reward-to-go advantages. The critic minimizes squared return error against the lambda-returns induced by GAE.

## Mitigation Assumptions

The mitigation stack is intentionally lightweight:

- qubit-wise action-register readout correction using asymmetric confusion matrices derived from the noise model
- ZNE-style folding and extrapolation in probability space
- extrapolation against realized fold factors, not requested ones

These transforms are approximations. They should be interpreted as mitigation heuristics for trend analysis, not exact inverses of hardware noise.

## Resource Metrics

The result bundle reports:

- held-out success
- held-out reward
- reward AUC
- success AUC
- runtime per seed
- estimated shots per seed
- success per runtime second
- success per million shots

These metrics are used to prevent high-cost mitigated runs from being summarized only by accuracy.
