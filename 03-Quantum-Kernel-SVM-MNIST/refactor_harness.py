"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Developer utility — one-time refactor harness.

IMPORTANT: This script was used to perform a one-time structural refactor of
run_experiment.py to introduce the multi-seed ablation loop. It should NOT be
re-run in normal use: run_experiment.py already contains the final refactored
structure and running this script again would corrupt it.

This file is kept as a reference artefact for reproducibility. If you need to
verify the structural transformation that was applied, compare the git history
of run_experiment.py against the patterns defined in the regex blocks below.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Guard against accidental re-execution."""
    print(
        "refactor_harness.py: This is a REFERENCE-ONLY file.\n"
        "run_experiment.py already contains the refactored structure.\n"
        "Re-running this script would corrupt run_experiment.py.\n"
        "Exiting without making any changes.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Historical transformation patterns (for reference only — NOT executed)
# ---------------------------------------------------------------------------

_ORIGINAL_SIGNATURE = "def main() -> int:"
_REFACTORED_SIGNATURE = (
    "def run_single_trial(seed: int, n_components: int, args, cfg, results_dir)"
    " -> tuple[dict, dict]:"
)

_ORIGINAL_RETURN = (
    '    print("Experiment complete.")\n'
    '    print(f"Results directory: {results_dir.resolve()}")\n'
    '    print("Metrics:")\n'
    "    print(comparison_df.to_string(index=False))\n"
    "    return 0"
)

_REFACTORED_RETURN = "    return classical_metrics, pegasos_metrics"


if __name__ == "__main__":
    main()
