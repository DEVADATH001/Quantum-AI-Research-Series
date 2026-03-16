"""Author: DEVADATH H K

Project: Quantum AI Research Series

Repository-level package metadata for the Quantum AI Research Series."""

from pathlib import Path

from setuptools import setup

setup(
    name="quantum-ai-research-series",
    version="2026.3.16",
    description="Repository metadata for the Quantum AI Research Series research portfolio.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    py_modules=[],
    author="DEVADATH H K",
    license="MIT",
)
