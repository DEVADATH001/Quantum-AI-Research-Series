# Scientific Results Verdict

- Overall label: `weak`
- Misleading-risk level: `high`

## Why
- The benchmark outputs are internally consistent with the method: expected objectives are tracked separately from representative sampled bitstrings, and sampled summaries obey their own bookkeeping constraints.
- Classical baselines outperform tuned QAOA on the held-out study families.
- Repeated benchmark runs show materially unstable behavior across optimizer and backend seeds.
- Sampled headline outputs can look materially better than the optimized expected objective.
- There is no corrected statistically significant evidence that QAOA improves over the included classical baselines.

## Randomness Flags
- Depth 1 shows a sampled-vs-expected gap of 0.2269, which makes single sampled outcomes look better than the optimized objective.

## Classical Outperformance
- communication_mesh_8_3: QAOA mean ratio 0.7828, better baselines: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut
- d_regular_8_3: QAOA mean ratio 0.8632, better baselines: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut
- erdos_renyi_8_0.5: QAOA mean ratio 0.8341, better baselines: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut
