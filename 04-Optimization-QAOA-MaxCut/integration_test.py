import numpy as np
import logging

from src.graph_generator import GraphGenerator
from src.hamiltonian_builder import HamiltonianBuilder
from src.qaoa_circuit import QAOACircuitBuilder
from src.qaoa_optimizer import QAOAOptimizer
from src.classical_solver import ClassicalSolver

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("--- 1. Generating Graph ---")
    gen = GraphGenerator(seed=42)
    graph = gen.generate_d_regular_graph(n_nodes=6, degree=3, seed=42)
    n_qubits = graph.number_of_nodes()
    print(f"Graph generated: {n_qubits} nodes, {graph.number_of_edges()} edges")

    print("\n--- 2. Finding Classical Exact Solution ---")
    classical = ClassicalSolver()
    classical_result = classical.solve_exact(graph)
    optimal_cut = classical_result.optimal_value
    print(f"Optimal cut value: {optimal_cut}")

    print("\n--- 3. Building Hamiltonian ---")
    builder = HamiltonianBuilder()
    hamiltonian, offset = builder.build_maxcut_hamiltonian(graph)
    print(f"Hamiltonian terms: {len(hamiltonian)}")

    print("\n--- 4. Setting up QAOA Objective ---")
    p = 1
    circuit_builder = QAOACircuitBuilder(n_qubits=n_qubits, p=p)
    
    def objective_function(params):
        # Build parameterized circuit
        gamma, beta = params
        qc = circuit_builder.build_qaoa_circuit_simple(graph, gamma, beta)
        
        # Get statevector
        sv = Statevector(qc)
        
        # Compute expectation value
        exp_value = sv.expectation_value(hamiltonian).real
        return exp_value

    print("\n--- 5. Optimizing Parameters ---")
    optimizer = QAOAOptimizer(p=p, optimizer_type='COBYLA', maxiter=50)
    opt_result = optimizer.optimize(
        objective_function=objective_function,
        n_qubits=n_qubits,
        graph=graph
    )
    
    print(f"Optimization finished in {opt_result.runtime:.2f}s")
    print(f"Optimal parameters: {opt_result.optimal_params}")
    print(f"Optimal value (energy): {opt_result.optimal_value}")
    
    print("\n--- 6. Solution Quality ---")
    qaoa_cut = opt_result.cut_value
    qaoa_bitstring = opt_result.solution_bitstring
    print(f"QAOA selected bitstring: {qaoa_bitstring}")
    print(f"QAOA Cut Value: {qaoa_cut}")
    
    if optimal_cut > 0:
        ratio = qaoa_cut / optimal_cut
        print(f"Approximation Ratio: {ratio:.4f}")
    else:
        print("Approximation Ratio: N/A (optimal cut is 0)")

if __name__ == "__main__":
    main()
