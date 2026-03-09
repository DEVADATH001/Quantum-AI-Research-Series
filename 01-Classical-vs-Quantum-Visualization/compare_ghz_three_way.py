from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def build_ghz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def extract_counts(pub_result):
    data = pub_result.data
    for key in ("meas", "c", "cr"):
        if hasattr(data, key):
            reg = getattr(data, key)
            if hasattr(reg, "get_counts"):
                return reg.get_counts()

    for key in dir(data):
        if key.startswith("_"):
            continue
        reg = getattr(data, key)
        if hasattr(reg, "get_counts"):
            return reg.get_counts()

    raise RuntimeError("No classical register counts found in SamplerV2 result.")


def normalize_counts(raw_counts, width: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in raw_counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{width}b")
        else:
            bitstring = str(key)
        out[bitstring] = out.get(bitstring, 0) + int(val)
    return out


def short_state(state: str, keep: int = 8) -> str:
    if len(state) <= 2 * keep:
        return state
    return f"{state[:keep]}...{state[-keep:]}"


def pending_jobs(backend) -> int:
    try:
        return int(getattr(backend.status(), "pending_jobs", -1))
    except Exception:
        return -1


def build_service() -> tuple[QiskitRuntimeService, str]:
    last_error = None
    for channel in ("ibm_cloud", "ibm_quantum_platform"):
        try:
            return QiskitRuntimeService(channel=channel), channel
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Unable to build QiskitRuntimeService from saved credentials.") from last_error


def choose_backend(service: QiskitRuntimeService, min_qubits: int, backend_name: str | None):
    if backend_name:
        backend = service.backend(backend_name)
        if backend.num_qubits < min_qubits:
            raise ValueError(f"Backend {backend.name} has {backend.num_qubits} qubits, expected >= {min_qubits}.")
        return backend

    candidates = service.backends(simulator=False, operational=True)
    candidates = [b for b in candidates if b.num_qubits >= min_qubits]
    if not candidates:
        raise RuntimeError(f"No operational IBM backend with >= {min_qubits} qubits.")
    return sorted(candidates, key=pending_jobs)[0]


def run_sampler(mode_label: str, mode_object, circuit: QuantumCircuit, shots: int) -> dict:
    sampler = Sampler(mode=mode_object)
    t0 = time.perf_counter()
    job = sampler.run([(circuit, [])], shots=shots)
    pub_result = job.result()[0]
    elapsed = time.perf_counter() - t0

    counts = normalize_counts(extract_counts(pub_result), circuit.num_clbits)
    total = sum(counts.values())
    probs = {state: count / total for state, count in counts.items()}
    all_zero = "0" * circuit.num_clbits
    all_one = "1" * circuit.num_clbits

    top_states = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:10]
    try:
        job_id = job.job_id()
    except Exception:
        job_id = None

    return {
        "mode": mode_label,
        "job_id": job_id,
        "shots": shots,
        "elapsed_seconds": round(elapsed, 3),
        "unique_states": len(counts),
        "p_all_zero": probs.get(all_zero, 0.0),
        "p_all_one": probs.get(all_one, 0.0),
        "p_ghz_subspace": probs.get(all_zero, 0.0) + probs.get(all_one, 0.0),
        "top_states": [
            {
                "state_short": short_state(state),
                "state_full": state,
                "probability": probability,
                "count": counts[state],
            }
            for state, probability in top_states
        ],
    }


def build_noisy_simulator(real_backend):
    try:
        simulator = AerSimulator.from_backend(real_backend)
        simulator.set_options(method="matrix_product_state")
        return simulator, "from_backend"
    except Exception:
        noise_model = NoiseModel.from_backend(real_backend)
        simulator = AerSimulator(method="matrix_product_state", noise_model=noise_model)
        return simulator, "noise_model_fallback"


def save_chart(report: dict, out_path: Path) -> None:
    labels = ["Local Ideal", "Simulated Noisy", "Real IBM"]
    ghz_subspace = [
        report["local"]["p_ghz_subspace"],
        report["simulated"]["p_ghz_subspace"],
        report["real"]["p_ghz_subspace"],
    ]
    elapsed = [
        report["local"]["elapsed_seconds"],
        report["simulated"]["elapsed_seconds"],
        report["real"]["elapsed_seconds"],
    ]
    unique_states = [
        report["local"]["unique_states"],
        report["simulated"]["unique_states"],
        report["real"]["unique_states"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].bar(labels, ghz_subspace, color=["#3b82f6", "#f59e0b", "#ef4444"])
    axes[0].set_title("GHZ Subspace Probability")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("P(|0...0>) + P(|1...1>)")

    axes[1].bar(labels, elapsed, color=["#3b82f6", "#f59e0b", "#ef4444"])
    axes[1].set_title("Execution Time")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(labels, unique_states, color=["#3b82f6", "#f59e0b", "#ef4444"])
    axes[2].set_title("Unique Output States")
    axes[2].set_ylabel("Count")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GHZ-127 local vs simulated vs real IBM comparison.")
    parser.add_argument("--backend", default=None, help="Optional explicit IBM backend name.")
    parser.add_argument("--local-shots", type=int, default=1024, help="Shots for local ideal simulation.")
    parser.add_argument("--sim-shots", type=int, default=512, help="Shots for noisy simulation.")
    parser.add_argument("--real-shots", type=int, default=256, help="Shots for real hardware.")
    parser.add_argument(
        "--output",
        default="assets/three_way_ghz127_comparison.json",
        help="Path to JSON output report.",
    )
    parser.add_argument(
        "--chart",
        default="assets/three_way_ghz127_comparison.png",
        help="Path to comparison chart image.",
    )
    args = parser.parse_args()

    service, channel = build_service()
    backend = choose_backend(service, min_qubits=127, backend_name=args.backend)

    candidates = service.backends(simulator=False, operational=True)
    candidates = sorted([b for b in candidates if b.num_qubits >= 127], key=pending_jobs)
    candidate_snapshot = [
        {
            "name": b.name,
            "num_qubits": b.num_qubits,
            "pending_jobs": pending_jobs(b),
        }
        for b in candidates
    ]

    ghz = build_ghz(127)
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_ghz = pass_manager.run(ghz)

    local_result = run_sampler(
        "local_aer_mps_ideal",
        AerSimulator(method="matrix_product_state"),
        isa_ghz,
        args.local_shots,
    )

    noisy_backend, noisy_backend_source = build_noisy_simulator(backend)
    simulated_result = run_sampler(
        f"simulated_aer_noisy_{backend.name}",
        noisy_backend,
        isa_ghz,
        args.sim_shots,
    )

    real_result = run_sampler(
        f"real_{backend.name}",
        backend,
        isa_ghz,
        args.real_shots,
    )

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_channel": channel,
        "isa_backend_target": {
            "name": backend.name,
            "num_qubits": backend.num_qubits,
            "pending_jobs_at_selection": pending_jobs(backend),
        },
        "candidate_backends": candidate_snapshot,
        "isa_circuit": {
            "num_qubits": isa_ghz.num_qubits,
            "num_clbits": isa_ghz.num_clbits,
            "depth": isa_ghz.depth(),
            "size": isa_ghz.size(),
        },
        "local": local_result,
        "simulated": {**simulated_result, "simulator_source": noisy_backend_source},
        "real": real_result,
        "delta_real_minus_local": {
            "p_ghz_subspace": real_result["p_ghz_subspace"] - local_result["p_ghz_subspace"],
            "p_all_zero": real_result["p_all_zero"] - local_result["p_all_zero"],
            "p_all_one": real_result["p_all_one"] - local_result["p_all_one"],
            "elapsed_seconds": real_result["elapsed_seconds"] - local_result["elapsed_seconds"],
        },
        "delta_real_minus_simulated": {
            "p_ghz_subspace": real_result["p_ghz_subspace"] - simulated_result["p_ghz_subspace"],
            "p_all_zero": real_result["p_all_zero"] - simulated_result["p_all_zero"],
            "p_all_one": real_result["p_all_one"] - simulated_result["p_all_one"],
            "elapsed_seconds": real_result["elapsed_seconds"] - simulated_result["elapsed_seconds"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    chart_path = Path(args.chart)
    save_chart(report, chart_path)

    print(f"Backend: {backend.name}")
    print(f"Local job: {local_result['job_id']}")
    print(f"Simulated job: {simulated_result['job_id']}")
    print(f"Real job: {real_result['job_id']}")
    print(f"Wrote JSON: {output_path}")
    print(f"Wrote chart: {chart_path}")


if __name__ == "__main__":
    main()

