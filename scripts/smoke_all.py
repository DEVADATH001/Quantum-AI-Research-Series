"""
smoke_all.py — Run smoke tests for all five Quantum AI Research Series modules.

Each module runs its fastest valid entry point with bounded runtime.
Failures are captured and reported at the end — one module crashing
doesn't block the others.

Usage:
    python -m scripts.smoke_all          # full local smoke run
    python scripts/smoke_all.py          # also works
    python scripts/smoke_all.py --ci     # CI mode: faster, no pyscf required
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 output so Unicode glyphs don't crash on Windows CP1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Smoke-test command tables ─────────────────────────────────────────────────
# Each entry: (label, working_dir_relative_to_repo_root, command_list)

# Full local smoke tests (default).
SMOKE_TESTS_LOCAL = [
    (
        "01 — Iris Classification",
        "01-Classical-vs-Quantum-Visualization",
        [sys.executable, "Quantum_ML_-_Iris_Classification.py", "--no-show"],
    ),
    (
        "02 — VQE Verification",
        "02-Quantum-Chemistry-VQE",
        [sys.executable, "scripts/run_verification.py"],
    ),
    (
        "03 — Quantum Kernel SVM",
        "03-Quantum-Kernel-SVM-MNIST",
        [
            sys.executable,
            "run_experiment.py",
            "--config", "config/smoke_config.yaml",
            "--fallback",
            "--max-quantum-train", "15",
            "--max-kernel-samples", "15",
            "--disable-noise",
            "--results-dir", "results/smoke",
        ],
    ),
    (
        "04 — QAOA Max-Cut",
        "04-Optimization-QAOA-MaxCut",
        [sys.executable, "generate_artifacts.py"],
    ),
    (
        "05 — RL Noise Mitigation (smoke benchmark)",
        "05-Reinforcement-Learning-Noise-Mitigation",
        [
            sys.executable, "-m", "src.benchmark_suite",
            "--suite", "config/benchmark_suite_smoke.yaml",
        ],
    ),
]

# CI smoke tests — optimised for speed and minimal dependencies.
#
# Key differences vs local:
#   01: --quantum-max-kernel-evals 500 → tiny QSVC decision-boundary grid
#       (keeps Iris under ~60 s on a 2-core runner vs ~350 s locally)
#   02: --no-pyscf fallback is automatic in run_verification.py when pyscf
#       is absent; nothing extra needed here
#   03–05: unchanged (already fast)
SMOKE_TESTS_CI = [
    (
        "01 — Iris Classification (CI fast)",
        "01-Classical-vs-Quantum-Visualization",
        [
            sys.executable,
            "Quantum_ML_-_Iris_Classification.py",
            "--no-show",
            "--quantum-max-kernel-evals", "500",
            "--classical-grid", "40",
        ],
    ),
    (
        "02 — VQE Verification",
        "02-Quantum-Chemistry-VQE",
        [sys.executable, "scripts/run_verification.py"],
    ),
    (
        "03 — Quantum Kernel SVM",
        "03-Quantum-Kernel-SVM-MNIST",
        [
            sys.executable,
            "run_experiment.py",
            "--config", "config/smoke_config.yaml",
            "--fallback",
            "--max-quantum-train", "15",
            "--max-kernel-samples", "15",
            "--disable-noise",
            "--results-dir", "results/smoke",
        ],
    ),
    (
        "04 — QAOA Max-Cut",
        "04-Optimization-QAOA-MaxCut",
        [sys.executable, "generate_artifacts.py"],
    ),
    (
        "05 — RL Noise Mitigation (smoke benchmark)",
        "05-Reinforcement-Learning-Noise-Mitigation",
        [
            sys.executable, "-m", "src.benchmark_suite",
            "--suite", "config/benchmark_suite_smoke.yaml",
        ],
    ),
]

# Hard timeout per module (seconds).
# Local: generous (10 min). CI: tighter (5 min — if a module can't finish in 5 min on a
# 2-core runner it's not a valid smoke test).
MODULE_TIMEOUT_LOCAL = 600
MODULE_TIMEOUT_CI = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str, char: str = "=") -> None:
    width = max(len(msg) + 4, 60)
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ci",
        action="store_true",
        help=(
            "CI mode: use faster command variants and a tighter per-module timeout. "
            "Automatically enabled when the CI environment variable is set."
        ),
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Also auto-detect GitHub Actions / generic CI environments.
    is_ci = args.ci or bool(os.environ.get("CI"))

    smoke_tests = SMOKE_TESTS_CI if is_ci else SMOKE_TESTS_LOCAL
    module_timeout = MODULE_TIMEOUT_CI if is_ci else MODULE_TIMEOUT_LOCAL

    results: list[tuple[str, bool, float, str]] = []

    mode_label = "CI mode" if is_ci else "local mode"
    _banner(f"Quantum AI Research Series — Smoke All  [{mode_label}]", "#")
    print(f"Python:    {sys.executable}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Timeout:   {module_timeout}s per module\n")

    # Force non-interactive matplotlib backend for all child processes.
    # This prevents Tkinter threading crashes when running under subprocess.
    child_env = {**os.environ, "MPLBACKEND": "Agg"}

    for label, rel_dir, cmd in smoke_tests:
        cwd = REPO_ROOT / rel_dir
        _banner(f"Running: {label}")
        print(f"  cwd:  {cwd}")
        print(f"  cmd:  {' '.join(cmd)}\n")

        if not cwd.is_dir():
            msg = f"Directory not found: {cwd}"
            print(f"  [SKIP] {msg}")
            results.append((label, False, 0.0, msg))
            continue

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                timeout=module_timeout,
                capture_output=True,
                text=True,
                env=child_env,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.perf_counter() - t0

            if proc.returncode == 0:
                print(f"  [PASS]  ({elapsed:.1f}s)")
                results.append((label, True, elapsed, ""))
            else:
                # Show last 30 lines of combined output for diagnosis.
                combined = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
                tail = "\n".join(combined[-30:])
                print(f"  [FAIL]  (exit {proc.returncode}, {elapsed:.1f}s)")
                if tail:
                    print(f"\n  --- output (last 30 lines) ---\n{tail}\n")
                results.append(
                    (label, False, elapsed, f"exit {proc.returncode}")
                )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"  [TIMEOUT] after {module_timeout}s")
            results.append((label, False, elapsed, "timeout"))

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [ERROR] {exc}")
            results.append((label, False, elapsed, str(exc)))

    # ── Summary ──────────────────────────────────────────────
    _banner("Summary", "#")
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    total_time = sum(t for _, _, t, _ in results)

    for label, ok, elapsed, err in results:
        status = "PASS" if ok else f"FAIL ({err})"
        print(f"  [{elapsed:6.1f}s] {status:30s}  {label}")

    print(f"\n  {passed}/{total} passed  —  total wall-clock: {total_time:.1f}s\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
