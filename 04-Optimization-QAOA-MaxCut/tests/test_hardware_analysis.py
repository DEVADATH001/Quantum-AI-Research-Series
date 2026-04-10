"""Regression tests for hardware-feasibility analysis."""

from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.graph_generator import GraphGenerator
from src.hardware_analysis import HardwareFeasibilityAnalyzer, HardwareFeasibilityThresholds
from src.qaoa_circuit import QAOACircuitBuilder


def test_hardware_feasibility_analyzer_reports_transpilation_growth():
    graph = GraphGenerator(seed=42).generate_d_regular_graph(n_nodes=8, degree=3, seed=42)
    circuit = QAOACircuitBuilder(n_qubits=8, p=2).build_qaoa_circuit_multilayer(
        graph,
        gammas=[0.5, 0.4],
        betas=[0.2, 0.3],
    )
    analyzer = HardwareFeasibilityAnalyzer(
        backend=FakeBrisbane(),
        optimization_level=1,
        seed=42,
        thresholds=HardwareFeasibilityThresholds(
            max_transpiled_depth=200,
            max_two_qubit_gates=120,
            max_total_shots=150000,
        ),
    )

    report = analyzer.analyze(
        circuit,
        shots_per_evaluation=2048,
        n_evaluations=60,
        objective_repetitions=2,
        report_repetitions=4,
    )

    assert report["logical_qubits"] == 8
    assert report["logical_two_qubit_gates"] > 0
    assert report["transpiled_depth"] >= report["logical_depth"]
    assert report["transpiled_two_qubit_gates"] >= report["logical_two_qubit_gates"]
    assert report["estimated_total_shots"] > 0
    assert report["status"] in {
        "small_scale_feasible",
        "possible_but_fragile",
        "unlikely_without_major_noise",
    }
