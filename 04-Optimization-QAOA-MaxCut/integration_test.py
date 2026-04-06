"""Lightweight end-to-end sanity test for the QAOA Max-Cut pipeline."""

import logging

from src.classical_solver import ClassicalSolver
from src.graph_generator import GraphGenerator
from src.qaoa_optimizer import MaxCutQAOAProblem, QAOAOptimizer
from src.runtime_executor import RuntimeExecutor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    print("--- 1. Generating Graph ---")
    graph = GraphGenerator(seed=42).generate_d_regular_graph(n_nodes=6, degree=3, seed=42)
    print(f"Graph generated: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("\n--- 2. Finding Classical Exact Solution ---")
    classical_result = ClassicalSolver().solve_exact(graph)
    optimal_cut = classical_result.optimal_value
    print(f"Optimal cut value: {optimal_cut}")

    print("\n--- 3. Building Backend-Aware QAOA Problem ---")
    problem = MaxCutQAOAProblem(
        graph=graph,
        p=1,
        executor=RuntimeExecutor(mode="local", shots=1024, seed=42),
        seed=42,
        analysis_shots=1024,
        objective_repetitions=1,
        report_repetitions=1,
        analysis_mode="same_backend",
    )

    print("\n--- 4. Optimizing Expected Max-Cut Objective ---")
    optimizer = QAOAOptimizer(
        p=1,
        optimizer_type="COBYLA",
        maxiter=50,
        seed=42,
        selection_repetitions=2,
        plateau_window=10,
        plateau_tolerance=1e-5,
    )
    opt_result = optimizer.optimize(
        objective_function=problem.objective_function,
        n_qubits=graph.number_of_nodes(),
        graph=graph,
        solution_decoder=problem.decode_solution,
        selection_objective_function=problem.objective_function,
    )

    print(f"Optimization finished in {opt_result.runtime:.2f}s")
    print(f"Optimal parameters: {opt_result.optimal_params}")
    print(f"Best minimized objective: {opt_result.optimal_value:.4f}")
    print(f"Expected cut value: {opt_result.cut_value:.4f}")
    print(f"Objective standard error: {opt_result.objective_stderr:.6f}")

    print("\n--- 5. Representative Bitstring Output ---")
    print(f"Representative sampled bitstring: {opt_result.solution_bitstring}")
    print(f"Representative sampled cut value: {opt_result.sampled_cut_value}")
    print(f"Most likely bitstring: {opt_result.most_likely_bitstring}")
    print(f"Most likely cut value: {opt_result.most_likely_cut_value}")
    print(f"Most likely probability: {opt_result.bitstring_probability}")
    print(f"Best sampled bitstring: {opt_result.best_sampled_bitstring}")
    print(f"Best sampled cut value: {opt_result.best_sampled_cut_value}")

    print("\n--- 6. Diagnostics ---")
    if opt_result.diagnostics:
        for message in opt_result.diagnostics:
            print(f"- {message}")
    else:
        print("No plateau or iteration-budget warnings triggered.")

    expected_ratio = opt_result.cut_value / optimal_cut if optimal_cut > 0 else None
    sampled_ratio = (
        opt_result.sampled_cut_value / optimal_cut
        if optimal_cut > 0 and opt_result.sampled_cut_value is not None
        else None
    )
    best_sampled_ratio = (
        opt_result.best_sampled_cut_value / optimal_cut
        if optimal_cut > 0 and opt_result.best_sampled_cut_value is not None
        else None
    )

    print("\n--- 7. Solution Quality ---")
    if expected_ratio is not None:
        print(f"Expected approximation ratio: {expected_ratio:.4f}")
    else:
        print("Expected approximation ratio: N/A")

    if sampled_ratio is not None:
        print(f"Representative sampled approximation ratio: {sampled_ratio:.4f}")
    else:
        print("Representative sampled approximation ratio: N/A")

    if best_sampled_ratio is not None:
        print(f"Best sampled approximation ratio: {best_sampled_ratio:.4f}")
    else:
        print("Best sampled approximation ratio: N/A")


if __name__ == "__main__":
    main()
