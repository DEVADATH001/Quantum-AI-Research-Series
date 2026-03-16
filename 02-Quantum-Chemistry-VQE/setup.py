"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Setup configuration for quantum chemistry VQE project."""

from pathlib import Path

from setuptools import find_packages, setup

def _requirements() -> list[str]:
    text = (Path(__file__).parent / "requirements.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]

setup(
    name="quantum-chemistry-vqe",
    version="0.3.0",
    description="Modular VQE PES benchmarking platform for quantum chemistry.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=_requirements(),
    python_requires=">=3.10",
)
