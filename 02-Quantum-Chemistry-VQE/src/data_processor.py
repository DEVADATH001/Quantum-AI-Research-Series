"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Data persistence helpers."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy objects."""

    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)

def save_results(data: Dict, filename: str, output_dir: str = "results/raw_data") -> str:
    """Persist dictionary payload as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, cls=NumpyEncoder)
    return path

def load_results(filename: str, input_dir: str = "results/raw_data") -> Dict:
    """Load JSON payload."""
    path = os.path.join(input_dir, filename)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def save_energy_table(rows: Iterable[Dict], filename: str, output_dir: str = "results/raw_data") -> str:
    """Save flattened energy rows to CSV."""
    rows_list: List[Dict] = list(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if not rows_list:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return path
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)
    return path

from qiskit.qasm2 import dumps as dump_qasm2

def save_qasm_circuit(circuit: Any, filename: str, output_dir: str = "results/figures") -> str:
    """Export a Qiskit circuit to OpenQASM format using qasm2.dumps."""
    os.makedirs(output_dir, exist_ok=True)
    if not filename.endswith(".qasm"):
        filename += ".qasm"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dump_qasm2(circuit))
    return path
