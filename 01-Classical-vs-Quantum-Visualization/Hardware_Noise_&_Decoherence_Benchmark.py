"""Compatibility entrypoint for GHZ decoherence benchmarking.

This script intentionally delegates to compare_ghz_three_way.py so there is a
single benchmark implementation and a single output schema.
"""

from __future__ import annotations

from compare_ghz_three_way import main


if __name__ == "__main__":
    main()
