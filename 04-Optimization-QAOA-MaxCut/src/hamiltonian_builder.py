"""Author: DEVADATH H K

Hamiltonian Builder Module

Constructs Ising Hamiltonians for the Max-Cut problem using Pauli operators.

The Max-Cut problem is formulated as:
C = 1/2 Σ_{(i,j) ∈ E} (1 - Z_i Z_j)

Where:
- Z_i is the Pauli-Z operator on qubit i
- Z_i Z_j represents the ZZ interaction term
- Edges crossing the cut contribute +1 to the cost"""

import logging
from typing import Tuple, Optional
import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)

class HamiltonianBuilder:
    """
    Builds Ising Hamiltonians for combinatorial optimization problems.
    
    Converts Max-Cut problems to quantum Hamiltonians using the Ising model:
    H = Σ w_ij (1 - Z_i Z_j) / 2
    
    The ground state of this Hamiltonian corresponds to the Max-Cut solution.
    """
    
    def __init__(self) -> None:
        """Initialize the Hamiltonian builder."""
        logger.info("HamiltonianBuilder initialized")
    
    def build_maxcut_hamiltonian(
        self,
        graph: nx.Graph
    ) -> Tuple[SparsePauliOp, np.ndarray]:
        """
        Build the Ising Hamiltonian for Max-Cut.
        
        The Hamiltonian is:
            H = Σ_{(i,j) ∈ E} w_ij * (1 - Z_i Z_j) / 2
            
        For unweighted graphs, w_ij = 1 for all edges.
        
        Args:
            graph: NetworkX graph with optional edge weights
            
        Returns:
            Tuple of (Hamiltonian, offset)
                - Hamiltonian: SparsePauliOp representation
                - offset: Classical offset term
        """
        n_qubits = graph.number_of_nodes()
        edges = list(graph.edges())
        
        # Get edge weights (default to 1 if not specified)
        if nx.is_weighted(graph):
            weights = nx.get_edge_attributes(graph, 'weight')
        else:
            weights = {edge: 1.0 for edge in edges}
        
        logger.info(
            f"Building Max-Cut Hamiltonian for {n_qubits} qubits, "
            f"{len(edges)} edges"
        )
        
        # Build the Hamiltonian as a sum of ZZ terms
        # H = Σ w_ij * (1 - Z_i Z_j) / 2
        #   = Σ w_ij/2 - Σ w_ij * Z_i Z_j / 2
        #   
        # The constant term Σ w_ij/2 becomes the offset
        
        pauli_list = []
        coeffs = []
        
        for i, j in edges:
            w = weights.get((i, j), weights.get((j, i), 1.0))
            
            # Create ZZ operator for edge (i, j)
            # Z_i Z_j term: -w/2 * Z_i Z_j
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_str = ''.join(reversed(pauli_str))  # Qiskit uses MSB first
            
            pauli_list.append(pauli_str)
            coeffs.append(-w / 2)
        
        # Build constant offset term: Σ w_ij / 2
        offset = sum(weights.values()) / 2
        
        # Create the SparsePauliOp
        if pauli_list:
            hamiltonian = SparsePauliOp(pauli_list, coeffs=coeffs)
            # Simplify by grouping
            hamiltonian = hamiltonian.simplify()
        else:
            hamiltonian = SparsePauliOp(['I' * n_qubits], coeffs=[0.0])
        
        logger.info(
            f"Hamiltonian built with {len(hamiltonian)} terms, "
            f"offset={offset:.4f}"
        )
        
        return hamiltonian, offset
    
    def build_maxcut_hamiltonian_qubo(
        self,
        graph: nx.Graph
    ) -> Tuple[SparsePauliOp, float]:
        """
        Alternative: Build QUBO-formatted Hamiltonian.
        
        Uses the transformation:
            x_i ∈ {0,1} → z_i = 2*x_i - 1
            
        Args:
            graph: NetworkX graph
            
        Returns:
            Tuple of (Hamiltonian, offset)
        """
        n_qubits = graph.number_of_nodes()
        edges = list(graph.edges())
        
        # Get weights
        if nx.is_weighted(graph):
            weights = nx.get_edge_attributes(graph, 'weight')
        else:
            weights = {edge: 1.0 for edge in edges}
        
        logger.info(f"Building QUBO Hamiltonian for {n_qubits} qubits")
        
        # For QUBO: minimize Σ w_ij * (x_i + x_j - 2*x_i*x_j)
        # Equivalent to Ising after transformation
        
        pauli_list = []
        coeffs = []
        
        # Linear terms: Σ w_ij * x_i
        for i in range(n_qubits):
            w_sum = sum(
                weights.get((i, j), weights.get((j, i), 1.0))
                for j in graph.neighbors(i)
            ) / 2
            if w_sum != 0:
                pauli_str = ['I'] * n_qubits
                pauli_str[i] = 'Z'
                pauli_str = ''.join(reversed(pauli_str))
                pauli_list.append(pauli_str)
                coeffs.append(w_sum * 2)
        
        # Quadratic terms: -2 * w_ij * x_i * x_j → Z_i Z_j term
        for i, j in edges:
            w = weights.get((i, j), weights.get((j, i), 1.0))
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_str = ''.join(reversed(pauli_str))
            pauli_list.append(pauli_str)
            coeffs.append(-w)
        
        offset = sum(weights.values())
        
        if pauli_list:
            hamiltonian = SparsePauliOp(pauli_list, coeffs=coeffs)
            hamiltonian = hamiltonian.simplify()
        else:
            hamiltonian = SparsePauliOp(['I' * n_qubits], coeffs=[0.0])
        
        return hamiltonian, offset
    
    def evaluate_expectation(
        self,
        hamiltonian: SparsePauliOp,
        statevector: np.ndarray
    ) -> float:
        """
        Evaluate expectation value of Hamiltonian for a given state.
        
        Args:
            hamiltonian: The Hamiltonian operator
            statevector: Quantum state as numpy array
            
        Returns:
            Expectation value
        """
        # For simulation, we can compute directly
        return np.real(np.conj(statevector) @ hamiltonian.to_matrix() @ statevector)
    
    def get_cost_from_bitstring(
        self,
        graph: nx.Graph,
        bitstring: str
    ) -> float:
        """
        Compute cost for a given bitstring solution.
        
        Args:
            graph: NetworkX graph
            bitstring: Binary string (e.g., '01011')
            
        Returns:
            Cost value (number of edges in cut)
        """
        n = len(bitstring)
        cost = 0
        
        for i, j in graph.edges():
            if i < n and j < n:
                # Edge contributes if nodes are in different partitions
                if bitstring[i] != bitstring[j]:
                    cost += 1
        
        return cost
    
    def bitstring_to_partition(
        self,
        bitstring: str
    ) -> Tuple[set, set]:
        """
        Convert bitstring to node partitions.
        
        Args:
            bitstring: Binary string where 0→partition A, 1→partition B
            
        Returns:
            Tuple of (partition_A, partition_B) as sets
        """
        partition_a = set()
        partition_b = set()
        
        for i, bit in enumerate(bitstring):
            if bit == '0':
                partition_a.add(i)
            else:
                partition_b.add(i)
        
        return partition_a, partition_b
    
    def create_mixer_hamiltonian(
        self,
        n_qubits: int
    ) -> SparsePauliOp:
        """
        Create the mixer Hamiltonian for QAOA.
        
        Mixer Hamiltonian: H_M = Σ X_i
        
        This promotes exploration of the solution space.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            Mixer Hamiltonian as SparsePauliOp
        """
        logger.info(f"Creating mixer Hamiltonian for {n_qubits} qubits")
        
        pauli_list = []
        coeffs = []
        
        for i in range(n_qubits):
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'X'
            pauli_str = ''.join(reversed(pauli_str))
            pauli_list.append(pauli_str)
            coeffs.append(1.0)
        
        mixer = SparsePauliOp(pauli_list, coeffs=coeffs)
        
        return mixer
    
    def compute_classical_cost(
        self,
        graph: nx.Graph,
        x: np.ndarray
    ) -> float:
        """
        Compute classical cost function for QAOA parameters.
        
        Uses the mapping: z_i = 2*x_i - 1 where x_i ∈ {0,1}
        
        Cost = Σ w_ij * (1 - z_i * z_j) / 2
        
        Args:
            graph: NetworkX graph
            x: Binary variables (0 or 1)
            
        Returns:
            Cost value
        """
        # Convert to ±1
        z = 2 * x - 1
        
        cost = 0.0
        for i, j in graph.edges():
            w = graph[i].get(j, {}).get('weight', 1.0)
            cost += w * (1 - z[i] * z[j]) / 2
        
        return cost

