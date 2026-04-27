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
9. Added fake-backend hardware proxy fallback, serialized noise-model loading,
   and backend-aware noisy execution for NISQ-style benchmarking.
10. Added a hardware-feasibility report with transpiled depth, entangling-gate
    growth, and shot-budget estimates.
11. Switched the default artifact profile from an exact 12-qubit study to a
    smaller 6-qubit noisy hardware proxy benchmark.
12. Updated RQAOA so it can estimate elimination correlations from sampled
    counts instead of exact statevectors.
13. Added a held-out experimental study across multiple graph families with
    explicit tuning/evaluation splits, classical heuristic baselines, bootstrap
    confidence intervals, and paired significance tests.
14. The study results now show, explicitly and reproducibly, that tuned QAOA
    underperforms simple classical heuristics on the current 8-node benchmark
    families.
15. Added repeated-seed robustness analysis for the noisy benchmark so the repo
    no longer relies on a single sampled-looking result.
16. Added pairwise win/loss summaries, Holm-corrected p-values, and an
    explicit scientific verdict artifact that labels the current evidence as
    weak with medium misleading risk.
17. Refactored artifact generation behind `src/artifact_pipeline.py` so
    `generate_artifacts.py` is now a thin entrypoint instead of a monolithic
    research script.
18. Added `src/artifact_manager.py` and run-scoped `results/runs/<run_id>/`
    outputs so regenerated artifacts preserve provenance instead of silently
    overwriting prior runs.
19. Added `src/artifact_schema.py` typed records so benchmark, study, and
    robustness tables have an explicit schema rather than loose dict payloads.
20. Added `src/provenance.py` and `results/run_manifest.json` to capture git
    commit, config hash, interpreter details, and package versions for each
    artifact generation run.
21. Added project-local pytest configuration so validation behavior is scoped
    to this package instead of depending on repo-root defaults.
22. Upgraded the held-out study with stronger classical baselines, including
    Goemans-Williamson SDP rounding and budget-matched heuristics that use the
    same objective-evaluation count as tuned QAOA.
23. Added `results/study_budget_summary.csv` so the repo now exposes the
    budget-matched comparison directly instead of burying it in per-instance
    rows.
24. Added `results/publication_positioning.json` and
    `results/publication_positioning.md` so the repo states its honest
    contribution class: benchmark/negative-results artifact rather than
    algorithmic novelty.
25. Added weighted `communication_mesh` graph generation with domain-style edge
    attributes for distance, latency, reliability, interference, bandwidth,
    and derived Max-Cut weights.
26. Added CVaR-capable objective support so the same QAOA pipeline can optimize
    either the expected cut value or a risk-sensitive tail objective.
27. Extended runtime execution to expose probability distributions needed for
    sampled-objective analysis.
28. Added `docs/mathematical_formulation.md` to document the weighted graph
    construction, Hamiltonian mapping, CVaR objective, and budget-matched
    evaluation logic in one place.
29. Added publication-style figures for significance heatmaps, budget fairness,
    and sampled-vs-expected gap analysis.
30. Added `pyproject.toml` so the package can be installed as a local project
    instead of relying only on ad hoc script execution.
31. Fixed the scientific verdict logic so representative sampled bitstrings are
    not incorrectly required to outperform the expected objective.

## Reproducible Outputs

Run the full benchmark and artifact generation pipeline with:

```bash
python generate_artifacts.py
```

This writes:

- `results/metrics.csv`
- `results/hardware_feasibility.csv`
- `results/benchmark_robustness_runs.csv`
- `results/benchmark_robustness_summary.csv`
- `results/results_verdict.json`
- `results/results_verdict.md`
- `results/publication_positioning.json`
- `results/publication_positioning.md`
- `results/run_manifest.json`
- `results/sample_gap_analysis.png`
- `results/study_budget_summary.csv`
- `results/study_budget_fairness.png`
- `results/study_candidate_search.csv`
- `results/study_instance_metrics.csv`
- `results/study_method_summary.csv`
- `results/study_pairwise_summary.csv`
- `results/study_significance.csv`
- `results/study_significance_heatmap.png`
- `results/study_manifest.json`
- `results/approximation_ratio.png`
- `results/energy_landscape.png`
- `results/graph_cut_visualization.png`
- `results/optimization_convergence.png`
- `results/method_comparison.png`

Each run also preserves a full copy under `results/runs/<run_id>/`.

The current default benchmark is a 6-node weighted communication mesh on a
fake-`ibm_brisbane` noisy proxy. The latest stable metrics report:

- exact weighted cut value `4.4724`
- QAOA `p=1` expected cut value `2.9925`
- QAOA `p=1` approximation ratio `0.6691`
- robustness mean ratio `0.6856`

The held-out study now covers three graph families, and the current summary is
still unfavorable to QAOA:

- `communication_mesh_8_3`: QAOA `0.7828`, Goemans-Williamson `1.0000`
- `d_regular_8_3`: QAOA `0.8632`, Goemans-Williamson `1.0000`
- `erdos_renyi_8_0.5`: QAOA `0.8341`, Goemans-Williamson `1.0000`

The scientific verdict artifact therefore remains `weak`, with `medium`
misleading-risk.

## Remaining Research Gaps

The package is stronger as software, but it is still a benchmark repo rather
than a conference-ready research contribution. The next meaningful upgrades are:

1. Extend the repeated-seed robustness analysis from the single configured
   benchmark graph to the full multi-family study.
2. Add noisy and hardware-backed experiments with clear shot budgets and error
   bars.
3. Replace the weighted communication proxy with a fuller networking or
   robotics objective that includes real task constraints.
4. Add backend-native hardware sampling for representative bitstrings when
   using Runtime-based execution.
5. Add parameterized-circuit transpilation caching so noisy backend benchmarks
   scale beyond the current small demonstration setting.
6. Add a project-owned gradient path for controlled optimizer studies.
7. Expand the held-out study to larger weighted graphs and stronger
   approximation baselines before making any serious performance claim.
8. Add a locked environment file on top of `pyproject.toml` so external users
   can recreate the exact package environment without relying only on the run
   manifest.
