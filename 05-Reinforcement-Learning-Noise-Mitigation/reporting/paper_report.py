"""Build paper-style reports from saved benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from core.schemas import PaperReportBundle
from src.project_paths import path_relative_to_project, resolve_project_path
from src.benchmark_suite import METHOD_SPECS
from utils.qiskit_helpers import ensure_dir, save_json

FIGURE_FILETYPES = ("png", "svg")


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f6f4ee",
            "axes.edgecolor": "#28231d",
            "axes.labelcolor": "#28231d",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#d8d0c6",
            "grid.alpha": 0.45,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": plt.cycler(
                color=["#1f5c7a", "#ba5b2d", "#2f7d55", "#7c4d8a", "#ba8e1f", "#506f90", "#9b3d3d"]
            ),
            "savefig.bbox": "tight",
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, output_stem: Path) -> list[str]:
    saved: list[str] = []
    for ext in FIGURE_FILETYPES:
        path = output_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=180)
        saved.append(str(path.resolve()))
    plt.close(fig)
    return saved


def _write_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _tabular_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _tabular_to_latex(headers: list[str], rows: list[list[str]]) -> str:
    column_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(value) for value in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _leaderboard_rows(benchmark_report: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in benchmark_report["aggregate"]["method_leaderboard"]:
        rows.append(
            [
                entry["label"],
                _format_float(entry.get("average_eval_success")),
                _format_float(entry.get("average_rank")),
                str(int(entry.get("scenario_wins", 0))),
                _format_float(entry.get("average_eval_success_per_runtime_sec")),
                _format_float(entry.get("average_eval_success_per_million_shots")),
            ]
        )
    return rows


def _build_pairwise_rows(benchmark_report: dict[str, Any]) -> list[list[str]]:
    scenarios = benchmark_report["scenarios"]
    comparisons = [
        ("Quantum Actor-Critic mitigated", "MLP Actor-Critic", "quantum_actor_critic_mitigated", "mlp_actor_critic"),
        ("Quantum REINFORCE mitigated", "MLP REINFORCE", "quantum_reinforce_mitigated", "mlp_reinforce"),
        ("Quantum Actor-Critic mitigated", "Quantum Actor-Critic noisy", "quantum_actor_critic_mitigated", "quantum_actor_critic_noisy"),
        ("Quantum REINFORCE mitigated", "Quantum REINFORCE noisy", "quantum_reinforce_mitigated", "quantum_reinforce_noisy"),
    ]
    rows: list[list[str]] = []
    for left_label, right_label, left_key, right_key in comparisons:
        diffs = []
        for scenario in scenarios:
            left = scenario["method_metrics"].get(left_key, {}).get("eval_success")
            right = scenario["method_metrics"].get(right_key, {}).get("eval_success")
            if left is not None and right is not None:
                diffs.append(float(left) - float(right))
        if diffs:
            rows.append(
                [
                    left_label,
                    right_label,
                    _format_float(float(np.mean(diffs))),
                    _format_float(float(np.median(diffs))),
                    str(int(sum(diff > 0.0 for diff in diffs))),
                    str(len(diffs)),
                ]
            )
    return rows


def _build_mitigation_rows(benchmark_report: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for family_key, label in (
        ("quantum_reinforce", "Quantum REINFORCE"),
        ("quantum_actor_critic", "Quantum Actor-Critic"),
    ):
        noise_summary = benchmark_report["aggregate"]["noise_summary"][family_key]
        rows.append(
            [
                label,
                _format_float(noise_summary.get("average_noise_drop_eval_success")),
                _format_float(noise_summary.get("average_mitigation_recovery_eval_success")),
            ]
        )
    return rows


def _build_hardware_rows(results_root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    candidates = [
        results_root / "hardware_audit.json",
        results_root.parent / "hardware_audit.json",
        results_root.parent / "results" / "hardware_audit.json",
    ]
    hardware_report = None
    for candidate in candidates:
        if candidate.exists():
            hardware_report = _load_json(candidate)
            break
    if not hardware_report:
        return rows

    summary = hardware_report.get("hardware_realism_summary", {})
    rows.append(
        [
            "Per-circuit depth",
            str(summary.get("base_transpiled_depth", "NA")),
            "Hardware-plausible forward pass",
        ]
    )
    rows.append(
        [
            "Mitigated shot budget",
            _format_float(summary.get("estimated_shots_per_seed_mitigated", None)),
            "Full on-hardware training not realistic",
        ]
    )
    rows.append(
        [
            "Recommendation",
            "Train in sim, eval frozen checkpoints",
            "Use readout-first mitigation",
        ]
    )
    return rows


def _write_table_bundle(output_dir: Path, stem: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    markdown_path = output_dir / f"{stem}.md"
    latex_path = output_dir / f"{stem}.tex"
    markdown = _tabular_to_markdown(headers, rows)
    latex = _tabular_to_latex(headers, rows)
    return [
        _write_text(markdown_path, markdown),
        _write_text(latex_path, latex),
    ]


def _plot_leaderboard(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    leaderboard = benchmark_report["aggregate"]["method_leaderboard"]
    labels = [entry["label"] for entry in leaderboard]
    values = [float(entry["average_rank"]) if entry["average_rank"] is not None else np.nan for entry in leaderboard]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(labels)), values, color="#1f5c7a")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Average scenario rank")
    ax.set_title("Figure 1. Benchmark leaderboard by average scenario rank")
    ax.invert_yaxis()
    return _save_figure(fig, output_dir / "figure_1_leaderboard")


def _plot_heatmap(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    scenarios = benchmark_report["scenarios"]
    method_names = [name for name, _ in METHOD_SPECS if any(s["method_metrics"].get(name, {}).get("eval_success") is not None for s in scenarios)]
    matrix = np.asarray(
        [
            [scenario["method_metrics"].get(method_name, {}).get("eval_success", np.nan) for scenario in scenarios]
            for method_name in method_names
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([scenario["name"] for scenario in scenarios], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels([dict(METHOD_SPECS)[name] for name in method_names])
    ax.set_title("Figure 2. Per-scenario held-out evaluation success")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return _save_figure(fig, output_dir / "figure_2_heatmap")


def _plot_noise_forest(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    scenarios = benchmark_report["scenarios"]
    labels: list[str] = []
    drop_values: list[float] = []
    recovery_values: list[float] = []
    for scenario in scenarios:
        for family_key, prefix in (("quantum_reinforce", "QRE"), ("quantum_actor_critic", "QA2C")):
            labels.append(f"{scenario['name']} {prefix}")
            drop_values.append(float(scenario["noise_profile"][family_key]["noise_drop_eval_success"] or 0.0))
            recovery_values.append(float(scenario["noise_profile"][family_key]["mitigation_recovery_eval_success"] or 0.0))
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(labels))))
    ax.hlines(ypos, 0.0, drop_values, color="#ba5b2d", linewidth=3, label="Noise drop")
    ax.scatter(drop_values, ypos, color="#ba5b2d", s=50)
    ax.scatter(recovery_values, ypos, color="#2f7d55", s=50, label="Mitigation recovery")
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta in held-out success")
    ax.set_title("Figure 3. Noise degradation and mitigation recovery")
    ax.legend()
    return _save_figure(fig, output_dir / "figure_3_noise_forest")


def _plot_frontier(benchmark_report: dict[str, Any], output_dir: Path, *, x_key: str, stem: str, title: str) -> list[str]:
    leaderboard = benchmark_report["aggregate"]["method_leaderboard"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for entry in leaderboard:
        x_value = entry.get(x_key)
        y_value = entry.get("average_eval_success")
        if x_value is None or y_value is None:
            continue
        ax.scatter(float(x_value), float(y_value), s=90)
        ax.text(float(x_value), float(y_value), entry["label"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel(
        "Held-out success per million shots" if x_key == "average_eval_success_per_million_shots" else "Held-out success per runtime second"
    )
    ax.set_ylabel("Average held-out success")
    ax.set_title(title)
    return _save_figure(fig, output_dir / stem)


def _load_scenario_summary(scenario: dict[str, Any]) -> dict[str, Any]:
    return _load_json(Path(scenario["summary_path"]))


def _plot_learning_curves(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    scenarios = benchmark_report["scenarios"]
    if not scenarios:
        return []
    selected = [scenarios[0]]
    if len(scenarios) > 1:
        selected.append(scenarios[-1])
    fig, axes = plt.subplots(1, len(selected), figsize=(7 * len(selected), 5), squeeze=False)
    for axis, scenario in zip(axes[0], selected):
        summary = _load_scenario_summary(scenario)
        methods = summary.get("methods", {})
        curves = {
            "MLP Actor-Critic": methods.get("mlp_actor_critic", {}),
            "Quantum Actor-Critic mitigated": methods.get("quantum_actor_critic", {}).get("mitigated", {}),
            "Quantum REINFORCE mitigated": methods.get("quantum_reinforce", {}).get("mitigated", {}),
        }
        for label, aggregate in curves.items():
            reward_curve = np.asarray(aggregate.get("reward_ma_mean", []), dtype=float)
            reward_std = np.asarray(aggregate.get("reward_ma_std", []), dtype=float)
            if reward_curve.size == 0:
                continue
            episodes = np.arange(1, reward_curve.size + 1)
            axis.plot(episodes, reward_curve, linewidth=2.0, label=label)
            if reward_std.size == reward_curve.size:
                axis.fill_between(episodes, reward_curve - reward_std, reward_curve + reward_std, alpha=0.18)
        axis.set_title(scenario["name"])
        axis.set_xlabel("Episode")
        axis.set_ylabel("Moving-average return")
    axes[0][0].legend()
    fig.suptitle("Figure 6. Learning curves for representative scenarios")
    return _save_figure(fig, output_dir / "figure_6_learning_curves")


def _plot_gradient_variance(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    scenarios = benchmark_report["scenarios"]
    method_to_grad: dict[str, list[float]] = {}
    method_to_update: dict[str, list[float]] = {}
    for scenario in scenarios:
        summary = _load_scenario_summary(scenario)
        method_blocks = {
            "Quantum REINFORCE mitigated": summary.get("methods", {}).get("quantum_reinforce", {}).get("mitigated", {}),
            "Quantum Actor-Critic mitigated": summary.get("methods", {}).get("quantum_actor_critic", {}).get("mitigated", {}),
            "MLP REINFORCE": summary.get("methods", {}).get("mlp_reinforce", {}),
            "MLP Actor-Critic": summary.get("methods", {}).get("mlp_actor_critic", {}),
            "Tabular REINFORCE": summary.get("methods", {}).get("tabular_reinforce", {}),
        }
        for label, aggregate in method_blocks.items():
            grad_value = aggregate.get("grad_norm_final_mean")
            update_value = None
            per_seed = aggregate.get("per_seed_metrics", [])
            if per_seed:
                update_value = float(np.mean([seed_metric.get("total_runtime_sec", 0.0) for seed_metric in per_seed]))
            if grad_value is not None:
                method_to_grad.setdefault(label, []).append(float(grad_value))
            if update_value is not None:
                method_to_update.setdefault(label, []).append(float(update_value))

    labels = list(method_to_grad)
    x = np.arange(len(labels))
    grad_means = [float(np.mean(method_to_grad[label])) for label in labels]
    update_means = [float(np.mean(method_to_update.get(label, [0.0]))) for label in labels]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    ax.bar(x - width / 2.0, grad_means, width=width, label="Gradient norm")
    ax.bar(x + width / 2.0, update_means, width=width, label="Runtime proxy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Figure 7. Gradient-scale and stability proxy comparison")
    ax.legend()
    return _save_figure(fig, output_dir / "figure_7_gradient_variance")


def _build_policy_appendix(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    appendix_lines = ["# Policy Appendix", ""]
    for scenario in benchmark_report["scenarios"][:2]:
        policy_plot = Path(scenario["output_dir"]) / "final_policy_plot.png"
        appendix_lines.append(f"## {scenario['name']}")
        appendix_lines.append("")
        appendix_lines.append(f"- Policy plot: `{path_relative_to_project(policy_plot)}`")
        appendix_lines.append("")
    appendix_path = output_dir / "policy_appendix.md"
    return [_write_text(appendix_path, "\n".join(appendix_lines))]


def _plot_policy_panel(benchmark_report: dict[str, Any], output_dir: Path) -> list[str]:
    scenarios = benchmark_report["scenarios"][:2]
    if not scenarios:
        return []
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 5), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        policy_plot = Path(scenario["output_dir"]) / "final_policy_plot.png"
        if policy_plot.exists():
            image = plt.imread(policy_plot)
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(scenario["name"])
        else:
            axis.axis("off")
            axis.set_title(f"{scenario['name']} (missing)")
    fig.suptitle("Figure 8. Selected policy heatmaps")
    return _save_figure(fig, output_dir / "figure_8_policy_panel")


def _plot_hardware_slice(results_root: Path, output_dir: Path) -> list[str]:
    candidates = [
        results_root / "hardware_audit.json",
        results_root.parent / "hardware_audit.json",
        results_root.parent / "results" / "hardware_audit.json",
    ]
    hardware_report = None
    for candidate in candidates:
        if candidate.exists():
            hardware_report = _load_json(candidate)
            break
    if not hardware_report:
        return []

    base_circuit = hardware_report.get("base_circuit", {})
    workloads = hardware_report.get("worst_case_workloads", {})
    labels = ["Base depth", "Base 2Q gates", "Ideal/noisy shots", "Mitigated shots"]
    values = [
        float(base_circuit.get("transpiled_depth", 0.0)),
        float(base_circuit.get("transpiled_two_qubit_gates", 0.0)),
        float(workloads.get("ideal_or_noisy", {}).get("estimated_total_shots_per_seed", 0.0)) / 1000.0,
        float(workloads.get("mitigated", {}).get("estimated_total_shots_per_seed", 0.0)) / 1000.0,
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(labels)), values, color=["#1f5c7a", "#ba5b2d", "#2f7d55", "#7c4d8a"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count (shots shown in thousands)")
    ax.set_title("Figure 9. Hardware-feasibility slice")
    return _save_figure(fig, output_dir / "figure_9_hardware_slice")


def _build_main_markdown(
    benchmark_report: dict[str, Any],
    figure_paths: list[str],
    table_paths: list[str],
    appendix_paths: list[str],
) -> str:
    leaderboard = benchmark_report["aggregate"]["headline_answers"]
    lines = [
        f"# {benchmark_report['suite_name']} Report",
        "",
        benchmark_report.get("description", "").strip() or "Paper-style benchmark report generated from saved results.",
        "",
        "## Headline Answers",
        "",
        f"- Average-rank winner: {leaderboard.get('average_rank_winner', 'NA')}",
        f"- Raw performance winner: {leaderboard.get('raw_performance_winner', 'NA')}",
        f"- Runtime-efficiency winner: {leaderboard.get('runtime_efficiency_winner', 'NA')}",
        f"- Shot-efficiency winner: {leaderboard.get('shot_efficiency_winner', 'NA')}",
        "",
        "## Generated Tables",
        "",
    ]
    for path in table_paths:
        lines.append(f"- `{path_relative_to_project(path)}`")
    lines.extend(["", "## Generated Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path_relative_to_project(path)}`")
    if appendix_paths:
        lines.extend(["", "## Appendices", ""])
        for path in appendix_paths:
            lines.append(f"- `{path_relative_to_project(path)}`")
    lines.append("")
    return "\n".join(lines)


def build_paper_report(results_root: str | Path) -> dict[str, Any]:
    _configure_plot_style()
    results_root = resolve_project_path(results_root)
    benchmark_report_path = results_root / "benchmark_report.json"
    if not benchmark_report_path.exists():
        raise FileNotFoundError(f"Could not find benchmark_report.json under {results_root}")

    benchmark_report = _load_json(benchmark_report_path)
    report_dir = ensure_dir(results_root / "paper_report")
    figures_dir = ensure_dir(report_dir / "figures")
    tables_dir = ensure_dir(report_dir / "tables")
    appendix_dir = ensure_dir(report_dir / "appendix")

    table_paths: list[str] = []
    table_paths.extend(
        _write_table_bundle(
            tables_dir,
            "table_1_leaderboard",
            [
                "Method",
                "Avg eval success",
                "Avg rank",
                "Wins",
                "Success / runtime sec",
                "Success / million shots",
            ],
            _leaderboard_rows(benchmark_report),
        )
    )
    table_paths.extend(
        _write_table_bundle(
            tables_dir,
            "table_2_pairwise",
            ["Left", "Right", "Mean delta", "Median delta", "Wins", "Scenarios"],
            _build_pairwise_rows(benchmark_report),
        )
    )
    table_paths.extend(
        _write_table_bundle(
            tables_dir,
            "table_3_mitigation",
            ["Family", "Avg noise drop", "Avg mitigation recovery"],
            _build_mitigation_rows(benchmark_report),
        )
    )
    hardware_rows = _build_hardware_rows(results_root)
    if hardware_rows:
        table_paths.extend(
            _write_table_bundle(
                tables_dir,
                "table_4_hardware",
                ["Topic", "Value", "Interpretation"],
                hardware_rows,
            )
        )

    figure_paths: list[str] = []
    figure_paths.extend(_plot_leaderboard(benchmark_report, figures_dir))
    figure_paths.extend(_plot_heatmap(benchmark_report, figures_dir))
    figure_paths.extend(_plot_noise_forest(benchmark_report, figures_dir))
    figure_paths.extend(
        _plot_frontier(
            benchmark_report,
            figures_dir,
            x_key="average_eval_success_per_million_shots",
            stem="figure_4_shot_frontier",
            title="Figure 4. Held-out success versus shot-cost efficiency",
        )
    )
    figure_paths.extend(
        _plot_frontier(
            benchmark_report,
            figures_dir,
            x_key="average_eval_success_per_runtime_sec",
            stem="figure_5_runtime_frontier",
            title="Figure 5. Held-out success versus runtime efficiency",
        )
    )
    figure_paths.extend(_plot_learning_curves(benchmark_report, figures_dir))
    figure_paths.extend(_plot_gradient_variance(benchmark_report, figures_dir))
    figure_paths.extend(_plot_policy_panel(benchmark_report, figures_dir))
    figure_paths.extend(_plot_hardware_slice(results_root, figures_dir))

    appendix_paths = _build_policy_appendix(benchmark_report, appendix_dir)
    markdown_report_path = report_dir / "paper_report.md"
    markdown = _build_main_markdown(benchmark_report, figure_paths, table_paths, appendix_paths)
    _write_text(markdown_report_path, markdown)

    bundle = PaperReportBundle(
        results_root=str(results_root.resolve()),
        benchmark_report_path=str(benchmark_report_path.resolve()),
        markdown_report_path=str(markdown_report_path.resolve()),
        figure_paths=figure_paths,
        table_paths=table_paths,
        appendix_paths=appendix_paths,
        generated_files=[
            str(markdown_report_path.resolve()),
            *figure_paths,
            *table_paths,
            *appendix_paths,
        ],
    ).model_dump(mode="json")
    save_json(report_dir / "paper_report_bundle.json", bundle)
    return bundle
