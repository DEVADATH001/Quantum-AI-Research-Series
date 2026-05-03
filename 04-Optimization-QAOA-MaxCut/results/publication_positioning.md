# Publication Positioning

- Contribution type: `benchmark_negative_results_artifact`
- Algorithmic novelty: `none`
- Publishable as: `benchmark_artifact_or_negative_results_workshop_paper`
- Workshop fit: Plausible as a benchmark, software-artifact, or negative-results workshop paper.

## Assessment
- No longer just a tutorial, but still not an algorithmic research contribution.
- The repo now contributes a reproducible and honest QAOA Max-Cut benchmark showing that sampled outputs can overstate expected performance and that classical baselines remain stronger on the current study.

## Missing For Stronger Publication
- A genuine algorithmic novelty claim beyond standard QAOA/RQAOA benchmarking.
- Larger and harder instance families, including weighted graphs and medium-scale regimes.
- Live hardware evidence or broader noisy studies beyond the configured proxy benchmark.
- Budget-matched comparisons against even stronger classical methods across larger scales.
- Either theoretical analysis or decisive empirical advantages, rather than only negative or neutral results.

## Where Classical Methods Still Win
- communication_mesh_8_3: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut
- d_regular_8_3: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut
- erdos_renyi_8_0.5: goemans_williamson, greedy, hill_climb_budget_matched, local_search, random_budget_matched, random_cut

## Budget-Matched Concerns
- Budget-matched baselines matching QAOA objective-evaluation counts still remain competitive or stronger: hill_climb_budget_matched, random_budget_matched
