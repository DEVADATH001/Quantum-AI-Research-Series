# Benchmark Specification

## Goal

This benchmark measures how measurement-defined quantum policies behave under ideal execution, NISQ-style noise, and lightweight mitigation, while comparing them against stronger classical baselines.

## Scenario Family

The fixed research benchmark contains eight scenarios:

1. `default_4pos`
2. `sparse_4pos`
3. `high_slip_4pos`
4. `sparse_high_slip_4pos`
5. `default_5pos`
6. `sparse_5pos`
7. `high_slip_5pos`
8. `sparse_high_slip_5pos`

These vary corridor size, reward sparsity, and action-slip probability.

## Seeds

All main benchmark runs use the fixed seed list:

`[7, 21, 33, 47, 63, 77, 91, 105, 119, 133]`

Smoke tests use only seed `7`.

## Reported Methods

- `random`
- `tabular_reinforce`
- `mlp_reinforce`
- `mlp_actor_critic`
- `quantum_reinforce_{ideal,noisy,mitigated}`
- `quantum_actor_critic_{ideal,noisy,mitigated}`

## Core Metrics

- held-out evaluation success
- held-out evaluation reward
- reward AUC
- success AUC
- runtime per seed
- estimated shots per seed
- held-out success per runtime second
- held-out success per million shots
- noise drop: `ideal - noisy`
- mitigation recovery: `mitigated - noisy`

## Intended Claim Style

This benchmark is designed for reproducible characterization and honest negative results. It should not be used to imply quantum advantage unless stronger evidence appears.
