"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Main experiment entry point — dispatches to the appropriate runner.

Usage:
    python scripts/run_experiment.py --molecule H2
    python scripts/run_experiment.py --molecule LiH --seeds 10 --mode multi_seed
    python scripts/run_experiment.py --molecule BeH2 --mode warm_start_study
    python scripts/run_experiment.py --molecule H2 --mode ablation

Modes:
    multi_seed (default) — Run N-seed PES with statistical aggregation.
    warm_start_study     — Paired warm/cold VQE, compute speedup metrics.
    ablation             — Sweep optimizer params; currently not implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQE research experiment dispatcher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--molecule", type=str, default="H2",
                        help="Molecule: H2, LiH, BeH2")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds")
    parser.add_argument("--mode", type=str, default="multi_seed",
                        choices=["multi_seed", "warm_start_study", "ablation"],
                        help="Experiment mode")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        default=True, help="Disable warm-starting (multi_seed only)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to simulation_config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.mode == "multi_seed":
        from scripts.run_multi_seed import run_multi_seed
        run_multi_seed(
            molecule=args.molecule,
            n_seeds=args.seeds,
            warm_start=args.warm_start,
            config_path=args.config,
        )

    elif args.mode == "warm_start_study":
        from scripts.run_warm_start_study import run_warm_start_study
        run_warm_start_study(
            molecule=args.molecule,
            n_seeds=args.seeds,
            config_path=args.config,
        )

    elif args.mode == "ablation":
        from scripts.run_ablation_study import run_ablation_study
        run_ablation_study(
            molecule=args.molecule,
            config_path=args.config,
        )

    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())