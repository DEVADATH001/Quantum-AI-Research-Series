# QAOA-MaxCut Status

## Current State

This project is now a reproducible Max-Cut benchmark package rather than a
placeholder "production-grade" scaffold. The main upgrades in this revision are:

1. Fixed the Max-Cut objective sign so the variational loop maximizes cut value
   instead of minimizing the wrong ZZ interaction.
2. Rebuilt the optimizer bookkeeping so evaluation history, final parameters,
   and decoded bitstrings are internally consistent.
3. Repaired local execution by using exact statevector evaluation for
   unmeasured circuits and sampled Aer execution for noisy simulation.
4. Reworked RQAOA so it performs weighted reductions, keeps node mappings
   straight, and falls back to exact reduced-instance solving instead of
   returning invalid all-zero solutions.
5. Added regression tests for the previously broken optimizer, runtime, and
   RQAOA paths.
6. Added `generate_artifacts.py` and generated real outputs in `results/`.
7. Added warm-start depth scaling, config-driven SPSA hyperparameters,
   noisy-candidate re-evaluation, and plateau diagnostics in the optimizer.
8. Split representative sampled bitstrings from best-observed samples so the
   benchmark no longer overstates sampled performance.

## Reproducible Outputs

Run the full benchmark and artifact generation pipeline with:

```bash
python generate_artifacts.py
```

This writes:

- `results/metrics.csv`
- `results/approximation_ratio.png`
- `results/energy_landscape.png`
- `results/graph_cut_visualization.png`
- `results/optimization_convergence.png`
- `results/method_comparison.png`

## Remaining Research Gaps

The package is stronger as software, but it is still a benchmark repo rather
than a conference-ready research contribution. The next meaningful upgrades are:

1. Benchmark across many random seeds and graph families, not one configured
   instance.
2. Compare against stronger classical baselines than brute force on small
   graphs.
3. Add noisy and hardware-backed experiments with clear shot budgets and error
   bars.
4. Tie the "robot network" framing to a genuine domain objective instead of a
   generic D-regular graph surrogate.
5. Add backend-native hardware sampling for representative bitstrings when
   using Runtime-based execution.
