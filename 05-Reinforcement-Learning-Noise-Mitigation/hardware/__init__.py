"""Hardware adapter re-exports for fake and real backend workflows."""

from hardware.adapters import build_hardware_feasibility_report, run_hardware_evaluation

__all__ = ["build_hardware_feasibility_report", "run_hardware_evaluation"]
