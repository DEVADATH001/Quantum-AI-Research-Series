"""Tests for benchmark robustness and scientific verdict helpers."""

from types import SimpleNamespace

import networkx as nx

from src.results_review import BenchmarkRobustnessRunner, ScientificResultsReviewer


class _StubOptimizer:
    def __init__(self, cut_value: float):
        self.cut_value = cut_value

    def optimize(self, **kwargs):
        return SimpleNamespace(
            cut_value=self.cut_value,
            sampled_cut_value=1.0,
            best_sampled_cut_value=1.0,
            runtime=0.25,
            solution_bitstring="01",
            bitstring_probability=0.5,
            objective_std=0.01,
            objective_stderr=0.005,
            diagnostics=["Optimizer hit the iteration budget before declaring convergence."],
        )


def test_benchmark_robustness_runner_summarizes_seeded_runs():
    graph = nx.Graph()
    graph.add_edge(0, 1)

    runner = BenchmarkRobustnessRunner(
        graph=graph,
        exact_value=1.0,
        depths=[1],
        optimization_seeds=[1, 2],
        create_problem=lambda depth, seed: SimpleNamespace(
            objective_function=lambda params: -1.0,
            decode_solution=lambda params, graph: {},
        ),
        create_optimizer=lambda depth, seed: _StubOptimizer(0.7 + 0.1 * seed),
        confidence=0.95,
        n_bootstrap=200,
    )

    result = runner.run()

    assert len(result["runs"]) == 2
    assert len(result["summary"]) == 1
    assert result["summary"][0].mean_ratio > 0.0
    assert result["summary"][0].iteration_budget_hit_rate == 1.0


def test_scientific_results_reviewer_flags_weak_results_with_high_overclaim_risk():
    reviewer = ScientificResultsReviewer(
        {
            "alpha": 0.05,
            "randomness_ratio_std_threshold": 0.05,
            "representative_gap_threshold": 0.15,
        }
    )

    review = reviewer.review(
        benchmark_rows=[
            {
                "method": "qaoa_p1",
                "expected_cut_value": 5.5,
                "sampled_cut_value": 7.0,
            }
        ],
        benchmark_robustness_summary=[
            {
                "depth": 1,
                "std_ratio": 0.08,
                "mean_sample_gap": 0.30,
            }
        ],
        study_summary_rows=[
            {"family": "family_a", "method": "qaoa_tuned", "mean_ratio": 0.82},
            {"family": "family_a", "method": "greedy", "mean_ratio": 0.95},
        ],
        significance_rows=[
            {
                "family": "family_a",
                "method_b": "greedy",
                "mean_difference": -0.13,
                "p_value": 0.01,
                "p_value_holm": 0.01,
            }
        ],
        pairwise_rows=[
            {
                "family": "family_a",
                "method_b": "greedy",
                "loss_rate_a": 1.0,
            }
        ],
    )

    assert review["overall_label"] == "weak"
    assert review["misleading_risk"] == "high"
    assert review["negative_significance"]
    assert any(
        "Classical baselines outperform tuned QAOA" in reason
        for reason in review["reasoning"]
    )


def test_scientific_results_reviewer_accepts_representative_sample_below_expectation():
    reviewer = ScientificResultsReviewer()

    review = reviewer.review(
        benchmark_rows=[
            {
                "method": "qaoa_p1",
                "expected_cut_value": 3.0,
                "sampled_cut_value": 2.4,
                "best_sampled_cut_value": 4.1,
                "representative_probability": 0.12,
            }
        ],
        benchmark_robustness_summary=[
            {
                "depth": 1,
                "std_ratio": 0.01,
                "mean_sample_gap": 0.05,
            }
        ],
        study_summary_rows=[
            {"family": "family_a", "method": "qaoa_tuned", "mean_ratio": 0.81},
            {"family": "family_a", "method": "greedy", "mean_ratio": 0.92},
        ],
        significance_rows=[],
        pairwise_rows=[],
    )

    assert review["overall_label"] == "weak"
    assert any(
        "internally consistent with the method" in reason
        for reason in review["reasoning"]
    )
