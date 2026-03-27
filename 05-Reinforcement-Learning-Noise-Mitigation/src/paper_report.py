"""CLI wrapper for building the paper-style benchmark report bundle."""

from __future__ import annotations

import argparse
import json

from reporting.paper_report import build_paper_report
from src.project_paths import resolve_project_path
from utils.qiskit_helpers import configure_logging


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper-style report from saved benchmark results")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/benchmark_suite",
        help="Directory containing benchmark_report.json.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for the report builder.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    bundle = build_paper_report(resolve_project_path(args.results_root))
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
