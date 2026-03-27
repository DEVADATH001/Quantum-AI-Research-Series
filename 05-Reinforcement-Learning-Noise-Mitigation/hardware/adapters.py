"""Thin adapters around existing hardware audit/evaluation utilities."""

from __future__ import annotations

from src.hardware_audit import build_hardware_feasibility_report
from src.hardware_evaluation import run_hardware_evaluation

__all__ = ["build_hardware_feasibility_report", "run_hardware_evaluation"]
