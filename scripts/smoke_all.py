"""
smoke_all.py — Run smoke tests for all five Quantum AI Research Series modules.

Each module runs its fastest valid entry point with bounded runtime.
Failures are captured and reported at the end — one module crashing
doesn't block the others.

Usage:
    python -m scripts.smoke_all          # from repo root
    python scripts/smoke_all.py          # also works
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry: (label, working_dir relative to repo root, command)
SMOKE_TESTS = [
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
            "--fallback",
            "--max-quantum-train", "20",
            "--disable-noise",
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

# Hard timeout per module (seconds). Generous — smoke runs should be
# well under this, but PQC training can be slow on weak hardware.
MODULE_TIMEOUT = 600  # 10 minutes


def _banner(msg: str, char: str = "=") -> None:
    width = max(len(msg) + 4, 60)
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}\n")


def main() -> int:
    results: list[tuple[str, bool, float, str]] = []

    _banner("Quantum AI Research Series — Smoke All", "#")
    print(f"Python:    {sys.executable}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Timeout:   {MODULE_TIMEOUT}s per module\n")

    for label, rel_dir, cmd in SMOKE_TESTS:
        cwd = REPO_ROOT / rel_dir
        _banner(f"Running: {label}")
        print(f"  cwd:  {cwd}")
        print(f"  cmd:  {' '.join(cmd)}\n")

        if not cwd.is_dir():
            msg = f"Directory not found: {cwd}"
            print(f"  ✗ SKIP — {msg}")
            results.append((label, False, 0.0, msg))
            continue

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                timeout=MODULE_TIMEOUT,
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - t0

            if proc.returncode == 0:
                print(f"  ✓ PASS  ({elapsed:.1f}s)")
                results.append((label, True, elapsed, ""))
            else:
                # Show last 20 lines of stderr for diagnosis
                tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                print(f"  ✗ FAIL  (exit {proc.returncode}, {elapsed:.1f}s)")
                if tail:
                    print(f"\n  --- stderr (last 20 lines) ---\n{tail}\n")
                results.append(
                    (label, False, elapsed, f"exit {proc.returncode}")
                )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"  ✗ TIMEOUT after {MODULE_TIMEOUT}s")
            results.append((label, False, elapsed, "timeout"))

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ✗ ERROR — {exc}")
            results.append((label, False, elapsed, str(exc)))

    # ── Summary ──────────────────────────────────────────────
    _banner("Summary", "#")
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    total_time = sum(t for _, _, t, _ in results)

    for label, ok, elapsed, err in results:
        status = "✓ PASS" if ok else f"✗ FAIL ({err})"
        print(f"  [{elapsed:6.1f}s] {status:30s}  {label}")

    print(f"\n  {passed}/{total} passed  —  total wall-clock: {total_time:.1f}s\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
