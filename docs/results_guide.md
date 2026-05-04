# Results Guide: How to Interpret and Cite Artifacts

This guide explains what each saved artifact means, how to read its fields, and what you can (and can't) conclude from it.

---

## General rules

1. **Always check `source_info`.** If a JSON artifact contains `"source_info": "synthetic"`, those numbers came from placeholder Hamiltonians — not real physics. Don't cite them as experimental results.
2. **Don't cross-compare modules.** Each project has different qubit counts, datasets, noise models, and optimizer budgets. A number from Project 02 doesn't mean the same thing as a number from Project 04.
3. **Smoke ≠ research.** Files in `results_smoke/` or generated with `_smoke.yaml` configs use abbreviated training and minimal seeds. They validate the pipeline, not the science.
4. **Seeds matter.** Single-seed results are anecdotal. Multi-seed results with confidence intervals are statistical. Check `n_seeds` before quoting error bars.

---

## Project 01 — Classical vs Quantum Visualization

### Key artifacts

| File | What it contains |
|---|---|
| `iris_qml_classification.ipynb` | Full Iris classification comparison: LR, RBF-SVM, Quantum SVC |
| `ghz_127_noise_benchmark.ipynb` | GHZ state fidelity under noise (fake backend + optional hardware) |

### How to read the results

- **Classification accuracy** is from sklearn's test-set evaluation. The quantum model uses a 2-qubit ZZFeatureMap — this is intentionally small.
- **GHZ fidelity** measures state preparation quality under noise. Values near 1.0 indicate good fidelity; values near 0.5 indicate the state has decohered to a classical mixture.
- `--skip-real` flag means the GHZ benchmark used a noise model, not actual hardware. That's noted in the output.

### What you can cite

> Classical baselines (Logistic Regression 97.4%, RBF-SVM 97.4%) substantially outperform the 2-qubit Quantum SVC (63.2%) on the Iris dataset.

### What you can't claim

- That quantum classification is inherently worse. The 2-qubit circuit is too small to draw general conclusions.

---

## Project 02 — Quantum Chemistry VQE

### Key artifacts

| File | What it contains |
|---|---|
| `results/raw_data/multiseed_stats_H2_warm.json` | 10-seed warm-start H₂ PES with both ansatze |
| `results/raw_data/multiseed_stats_H2.json` | Cold-start H₂ comparison |
| `results/raw_data/multiseed_stats_LIH.json` | LiH multi-seed statistics |
| `results/raw_data/multiseed_stats_BEH2.json` | BeH₂ multi-seed statistics |
| `results/raw_data/ablation_study_H2.json` | Architecture ablation (depth, entanglement, rotation gates) |
| `results/figures/pes_curve_*.png` | PES curve plots |
| `results/figures/error_*.png` | Per-point VQE error vs exact reference |

### Field reference

| Field | Meaning |
|---|---|
| `exact_energies` | Ground-state energy from exact diagonalization of the qubit Hamiltonian |
| `vqe_stats.<ansatz>.mean_energy` | VQE energy averaged over all seeds |
| `vqe_stats.<ansatz>.std_energy` | Standard deviation across seeds |
| `vqe_stats.<ansatz>.ci95_low/high` | 95% confidence interval bounds |
| `vqe_stats.<ansatz>.mean_error` | Mean `|E_vqe - E_exact|` across seeds |
| `vqe_stats.<ansatz>.chem_acc_rate` | Fraction of seeds within chemical accuracy (1.6 mHartree) |
| `hf_energies` | Hartree-Fock reference (NaN if not computed) |
| `cisd_energies` | CISD reference (NaN if not computed) |
| `warm_start` | Whether parameter transfer was used between bond lengths |
| `n_seeds` | Number of random seeds used |

### Chemical accuracy threshold

$$|E_{\text{VQE}} - E_{\text{exact}}| \leq 0.0016 \text{ Hartree} = 1.6 \text{ mHartree}$$

A `chem_acc_rate` of `1.0` means every seed, at every bond length, was within chemical accuracy.

### What you can cite

> H₂ warm-start results (10 seeds, 21 bond lengths from 0.5–2.5 Å): UCCSD achieves 100% chemical-accuracy rate across all distances with worst-case mean error of 1.57 µHa. EfficientSU2 also achieves 100% chemical-accuracy rate with worst-case mean error of 1.40 µHa. Both ansatze are well within the 1.6 mHa threshold.

### What you can't claim

- That VQE scales to larger molecules just because H₂ works. H₂ with STO-3G is a 2-qubit problem.
- That warm-start is always better — compare against the cold-start artifact to see the actual difference.

---

## Project 03 — Quantum Kernel SVM (MNIST)

### Key artifacts

| File | What it contains |
|---|---|
| `results/experiment_summary.json` | Full experiment metrics |
| `results/classical_comparison.json` | Classical RBF-SVM baseline |
| `figures/` | Confusion matrices, kernel matrices, decision boundaries |

### How to read the results

- The "MNIST" benchmark uses sklearn's 8×8 digits dataset by default (`--fallback` flag), not the full 28×28 MNIST. This matters for interpretation.
- `--max-quantum-train N` limits quantum kernel computation. Smaller N = faster but less representative.
- `--disable-noise` uses ideal simulation. That's the default for kernel computation.

### What you can cite

> On the 8×8 sklearn digits dataset: Classical RBF-SVM achieves F1 = 1.000; Quantum Pegasos SVM achieves F1 = 0.662.

### What you can't claim

- That this benchmarks "MNIST classification." The 8×8 dataset is a simplified version.
- That quantum kernels are fundamentally worse — the 4-qubit kernel has limited expressiveness by design.

---

## Project 04 — QAOA Max-Cut

### Key artifacts

| File | What it contains |
|---|---|
| `results/results_verdict.md` | Human-readable verdict with approximation ratios |
| `results/benchmark_summary.json` | Per-graph-family approximation ratios for all methods |
| `results/held_out_generalization.json` | Out-of-sample performance |
| `figures/` | Approximation ratio comparison plots |

### How to read the verdict

The verdict file contains a classification: `strong`, `moderate`, or `weak`. This rates QAOA's competitive position against classical baselines on the tested graph families.

| Verdict | Meaning |
|---|---|
| `strong` | QAOA matches or exceeds all classical baselines |
| `moderate` | QAOA is competitive on some graph families |
| `weak` | Classical baselines dominate on all tested graphs |

### What you can cite

> Verdict: **weak**. QAOA approximation ratio ~0.69 on the benchmark graph. Goemans-Williamson rounding achieves ratio 1.0 on all held-out graph families.

### What you can't claim

- That QAOA is useless. The tested graphs may be too small for quantum advantage to emerge. This is a negative result on these specific instances.

---

## Project 05 — RL Noise Mitigation

### Key artifacts

| File | What it contains |
|---|---|
| `results_benchmark_smoke/benchmark_report.md` | Smoke benchmark: 2 scenarios, short training |
| `results_benchmark_smoke/scenario_results/` | Per-scenario JSON with per-agent metrics |
| `results/benchmark_suite/` | Full benchmark output (if run) |

### How to read the benchmark report

- **Avg Eval Success**: mean success rate across evaluation episodes and scenarios. Higher is better.
- **Avg Rank**: mean rank across scenarios (1 = best). Lower is better.
- **Success / Runtime**: success rate divided by wall-clock seconds. Measures efficiency.
- **Wins**: number of scenarios where the method ranked #1.

### Smoke vs full

| Dimension | Smoke | Full |
|---|---|---|
| Scenarios | 2 | 8 |
| Seeds | 1 | 10 |
| Episodes | 50 | 1,000 |
| Mitigation | ZNE only | ZNE + PEC + readout |

### What you can cite (smoke only)

> Smoke results (pipeline validation only): Ideal quantum actor-critic achieves 1.000 success rate. Under noise, quantum REINFORCE drops ~0.250 in success. ZNE mitigation recovers ~0.062 of the degradation.

### What you can't claim

- That these are definitive findings. The smoke config uses short training and a single seed.
- That mitigation "doesn't work" based on the smoke mitigated underperforming noisy — that's a training-length artifact.

---

## How to cite this repository

Use the CITATION.cff file in the repository root:

```
DEVADATH H K. Quantum AI Research Series. 2026.
```

When citing specific results, always note the module, the exact artifact file, whether the run was smoke or research, and the `source_info` field value.

---

## Red flags to watch for

| Red flag | What it means |
|---|---|
| `source_info: synthetic` | Not real chemistry — placeholder Hamiltonians |
| `n_seeds: 1` | No statistical power — single anecdotal run |
| File in `results_smoke/` | Pipeline validation, not research data |
| `NaN` in baselines | Reference method wasn't computed for this run |
| `warm_start: false` not present | Can't tell if warm-starting was used — check the run script |
