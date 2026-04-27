"""CLI entrypoint for artifact generation."""

from __future__ import annotations

import logging
from pathlib import Path

from src.artifact_pipeline import ArtifactPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.WARNING)
logging.getLogger("stevedore").setLevel(logging.WARNING)


def main() -> None:
    """Run the full benchmark and artifact pipeline."""
    ArtifactPipeline(Path(__file__).resolve().parent).run()


if __name__ == "__main__":
    main()
