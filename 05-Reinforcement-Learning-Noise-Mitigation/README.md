# 05-Reinforcement-Learning-Noise-Mitigation

Benchmark-first quantum RL research system for studying measurement-defined quantum policies under ideal execution, NISQ-style noise, and lightweight mitigation.

## Framing

This repository is no longer a one-task student demo. It is now organized as a reproducible benchmark-and-analysis artifact.

The intended contribution is narrow and honest:

- a fixed key-and-door benchmark family with eight named scenarios
- measurement-native quantum policies, not expectation-logit surrogates
- stronger classical baselines, including actor-critic
- explicit resource metrics, mitigation limits, and hardware-feasibility tooling
- one-command benchmark and one-command paper-style report generation

It is **not** presented as evidence of quantum advantage.

## Current Architecture

```text
05-Reinforcement-Learning-Noise-Mitigation/
|-- agent/          # quantum learners and gradient estimators
|-- benchmarks/     # named scenario registry
|-- config/         # layered YAML configs
|-- core/           # seeds, schemas, public runner APIs
|-- docs/           # benchmark spec and technical note
|-- environments/   # key-and-door environment
|-- hardware/       # hardware adapter wrappers
|-- methods/        # method-level wrappers
|-- reporting/      # paper-report builder
|-- src/            # training, benchmark, audit, sweep, report CLIs
|-- tests/          # regression and smoke tests
`-- utils/          # shared helpers
```

## Main Methods

- `quantum_reinforce`: legacy quantum baseline
- `quantum_actor_critic`: upgraded main quantum method
- `tabular_reinforce`
- `mlp_reinforce`
- `mlp_actor_critic`
- `random`

Quantum methods are reported in `ideal`, `noisy`, and `mitigated` execution modes.

## Benchmark Family

The fixed research suite contains:

1. `default_4pos`
2. `sparse_4pos`
3. `high_slip_4pos`
4. `sparse_high_slip_4pos`
5. `default_5pos`
6. `sparse_5pos`
7. `high_slip_5pos`
8. `sparse_high_slip_5pos`

The main seed list is frozen to:

`[7, 21, 33, 47, 63, 77, 91, 105, 119, 133]`

Smoke tests use seed `7`.

## Key Files

- benchmark spec: `docs/benchmark_spec.md`
- math note: `docs/technical_note.md`
- schema definitions: `core/schemas.py`
- scenario registry: `benchmarks/scenario_registry.py`
- training pipeline: `src/training_pipeline.py`
- benchmark runner: `src/benchmark_suite.py`
- paper report builder: `reporting/paper_report.py`

## Commands

Install the package locally if you want the console scripts:

```bash
pip install -e .
```

Run a smoke training pass:

```bash
python -m src.training_pipeline --config config/smoke_test.yaml
```

Run the fixed research benchmark suite:

```bash
python -m src.benchmark_suite --suite config/benchmark_suite.yaml
```

Run the smoke benchmark suite:

```bash
python -m src.benchmark_suite --suite config/benchmark_suite_smoke.yaml
```

Build the paper-style report bundle from saved benchmark results:

```bash
python -m src.paper_report --results-root results_benchmark_smoke
```

Run the scientific audit:

```bash
python -m src.experiment_design_audit --config config/training_config.yaml
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

## Output Artifacts

Training writes:

- `run_manifest.json`
- per-seed quantum logs
- per-seed baseline logs
- `summary.json`

Benchmark runs write:

- `benchmark_report.json`
- `benchmark_report.md`
- one result bundle per scenario

Paper report generation writes:

- `paper_report/paper_report.md`
- `paper_report/paper_report_bundle.json`
- `paper_report/figures/*.png`
- `paper_report/figures/*.svg`
- `paper_report/tables/*.md`
- `paper_report/tables/*.tex`

## Mathematical Notes

The actor is measurement-defined:

`pi_theta(a | s) = Pr[M_action = a]`

The upgraded main method uses a quantum actor and classical MLP critic with GAE.

Parameter-shift identities are treated as exact only in the ideal circuit setting. Under finite shots, noise, and mitigation, the gradient estimators are stochastic and may be biased.

## Hardware Position

Single forward passes are hardware-plausible on IBM-style devices. Full in-loop policy-gradient training with ZNE is not the intended hardware workflow.

The realistic path is:

1. train in ideal or fake-backend simulation
2. freeze checkpoints
3. run small hardware evaluation slices with higher shots and readout-focused mitigation

## Contribution Style

If this project is written up, the defensible framing is:

- reproducible benchmark suite
- mitigation-aware evaluation framework
- honest negative-result or systems-style study

Avoid claiming quantum advantage unless new evidence appears.
