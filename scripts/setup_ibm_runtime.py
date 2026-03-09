"""
Save IBM Quantum credentials from environment variable.

Usage:
    setx IBM_QUANTUM_TOKEN "YOUR_NEW_TOKEN"
    python scripts/setup_ibm_runtime.py
"""

import os

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN is not set. Set it first, then rerun this script."
        )

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=token,
        overwrite=True,
    )
    print("IBM Quantum account saved successfully.")


if __name__ == "__main__":
    main()

