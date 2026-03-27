"""Named benchmark scenarios used by the research benchmark suite."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.seeds import MAIN_BENCHMARK_SEEDS, SMOKE_SEEDS


def _base_scenario(
    *,
    name: str,
    description: str,
    n_positions: int,
    horizon: int,
    slip: float,
    progress_reward_scale: float,
    key_reward: float,
    num_qubits: int,
) -> dict[str, Any]:
    goal_position = n_positions - 1
    start_positions = tuple(range(1, goal_position))
    return {
        "name": name,
        "description": description,
        "overrides": {
            "environment": {
                "n_positions": n_positions,
                "start_positions": list(start_positions),
                "key_position": 0,
                "goal_position": goal_position,
                "max_episode_steps": horizon,
                "step_penalty": -0.02 if progress_reward_scale > 0.0 else -0.01,
                "wall_penalty": -0.05,
                "locked_goal_penalty": -0.25,
                "key_reward": key_reward,
                "goal_reward": 1.0,
                "progress_reward_scale": progress_reward_scale,
                "slip_probability": slip,
            },
            "quantum_policy": {
                "num_qubits": num_qubits,
            },
        },
    }


def research_benchmark_scenarios() -> list[dict[str, Any]]:
    """Return the fixed 8-scenario benchmark family."""

    return [
        _base_scenario(
            name="default_4pos",
            description="4-position corridor with shaping and moderate slip.",
            n_positions=4,
            horizon=8,
            slip=0.05,
            progress_reward_scale=0.05,
            key_reward=0.15,
            num_qubits=3,
        ),
        _base_scenario(
            name="sparse_4pos",
            description="4-position corridor with sparse rewards and moderate slip.",
            n_positions=4,
            horizon=8,
            slip=0.05,
            progress_reward_scale=0.0,
            key_reward=0.0,
            num_qubits=3,
        ),
        _base_scenario(
            name="high_slip_4pos",
            description="4-position corridor with shaping and high slip.",
            n_positions=4,
            horizon=8,
            slip=0.15,
            progress_reward_scale=0.05,
            key_reward=0.15,
            num_qubits=3,
        ),
        _base_scenario(
            name="sparse_high_slip_4pos",
            description="4-position corridor with sparse rewards and high slip.",
            n_positions=4,
            horizon=8,
            slip=0.15,
            progress_reward_scale=0.0,
            key_reward=0.0,
            num_qubits=3,
        ),
        _base_scenario(
            name="default_5pos",
            description="5-position corridor with shaping and moderate slip.",
            n_positions=5,
            horizon=10,
            slip=0.05,
            progress_reward_scale=0.03,
            key_reward=0.15,
            num_qubits=4,
        ),
        _base_scenario(
            name="sparse_5pos",
            description="5-position corridor with sparse rewards and moderate slip.",
            n_positions=5,
            horizon=10,
            slip=0.05,
            progress_reward_scale=0.0,
            key_reward=0.0,
            num_qubits=4,
        ),
        _base_scenario(
            name="high_slip_5pos",
            description="5-position corridor with shaping and high slip.",
            n_positions=5,
            horizon=10,
            slip=0.15,
            progress_reward_scale=0.03,
            key_reward=0.15,
            num_qubits=4,
        ),
        _base_scenario(
            name="sparse_high_slip_5pos",
            description="5-position corridor with sparse rewards and high slip.",
            n_positions=5,
            horizon=10,
            slip=0.15,
            progress_reward_scale=0.0,
            key_reward=0.0,
            num_qubits=4,
        ),
    ]


def make_research_benchmark_suite(base_config: str = "config/research_benchmark.yaml") -> dict[str, Any]:
    """Structured suite payload for the full research benchmark."""

    return {
        "suite_name": "Key-and-Door Research Benchmark",
        "description": (
            "Eight-scenario benchmark family spanning corridor size, reward sparsity, "
            "and action-slip stochasticity for measurement-defined quantum RL."
        ),
        "base_config": base_config,
        "output_root": "results/benchmark_suite",
        "experiment_overrides": {
            "seeds": MAIN_BENCHMARK_SEEDS,
            "n_eval_episodes": 64,
        },
        "scenarios": research_benchmark_scenarios(),
    }


def make_smoke_benchmark_suite(base_config: str = "config/smoke_test.yaml") -> dict[str, Any]:
    """Small suite used for integration verification."""

    scenarios = research_benchmark_scenarios()[:2]
    smoke_payload = make_research_benchmark_suite(base_config=base_config)
    smoke_payload["suite_name"] = "Key-and-Door Benchmark Smoke"
    smoke_payload["description"] = "Two-scenario smoke suite for CI and local verification."
    smoke_payload["output_root"] = "results_benchmark_smoke"
    smoke_payload["experiment_overrides"] = {
        "seeds": SMOKE_SEEDS,
        "n_eval_episodes": 8,
    }
    smoke_payload["scenarios"] = deepcopy(scenarios)
    return smoke_payload
