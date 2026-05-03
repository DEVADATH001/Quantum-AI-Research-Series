"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Experiment Tracker for structured logging."""

import json
import os
import time
from typing import Dict, Any, List
from pathlib import Path

class ExperimentTracker:
    """Logs experiment parameters and results to JSON lines format for tracking."""

    def __init__(self, log_dir: str = "results/tracking"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"run_{int(time.time())}"
        self.log_file = self.log_dir / f"{self.run_id}.jsonl"
        self.metadata: Dict[str, Any] = {}
        
    def log_metadata(self, key: str, value: Any):
        """Log static metadata (e.g., molecule, hardware, parameters)."""
        self.metadata[key] = value
        
    def log_metric(self, step: int, metrics: Dict[str, Any]):
        """Log a time-series metric (e.g., VQE energy per iteration)."""
        record = {
            "run_id": self.run_id,
            "type": "metric",
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        self._write(record)
        
    def log_result(self, result: Dict[str, Any]):
        """Log final experiment result."""
        record = {
            "run_id": self.run_id,
            "type": "result",
            "metadata": self.metadata,
            "timestamp": time.time(),
            **result
        }
        self._write(record)
        
    def _write(self, record: dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
