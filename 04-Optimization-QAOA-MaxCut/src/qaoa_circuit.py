"""
QAOA Circuit Builder Module

Constructs parameterized QAOA quantum circuits for Max-Cut optimization.

A QAOA circuit consists of:
1. Initial state (uniform superposition)
2. p layers of:
   - Cost Hamiltonian unitary: U_C(γ) = exp(-iγH_C)
   - Mixer Hamiltonian unitary: U_M(β) = exp(-iβH_M)

Where:
   - H_C = Σ w_ij (1 - Z_i Z_j) / 2  (Cost Hamiltonian)
   - H_M = Σ X_i  (Mixer Hamiltonian)
   - γ, β are variational parameters

Author: Quantum AI Research Team
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class QAOACircuitBuilder:
    """
    Builds parameterized QAOA circuits for combinatorial optimization.
    
    Implements both the standard QAOA circuit and variants
    with custom mixer Hamiltonians.
    """
    
    def __init__(self, n_qubits: int, p: int = 1) -> None:
        """
        Initialize the QAOA circuit builder.
        
        Args:
            n_qubits: Number of qubits (problem size)
            p: Number of QAOA layers
        """
        self.n_qubits = n_qubits
        self.p = p
        self.rng = np.random.default_rng()
        
        # Create parameter vectors
        self.gammas = ParameterVector('gamma', p)
        self.betas = ParameterVector('beta', p)
        
        logger.info(
            f"QAOACircuitBuilder initialized: n_qubits={n_qubits}, p={p}"
        )
    
    def build_qaoa_circuit(
        self,
        cost_hamiltonian: SparsePauliOp,
        initial_state: Optional[QuantumCircuit] = None
    ) -> QuantumCircuit:
        """
        Build a complete QAOA circuit.
        
        Args:
            cost_hamiltonian: Cost Hamiltonian (SparsePauliOp)
            initial_state: Optional custom initial state
            
        Returns:
            Parameterized QAOA circuit
        """
        logger.info(f"Building QAOA circuit with p={self.p} layers")
        
        qc = QuantumCircuit(self.n_qubits)
        
        # Step 1: Initialize to uniform superposition
        if initial_state is None:
            # Apply Hadamard to all qubits
            for i in range(self.n_qubits):
                qc.h(i)
        else:
            qc.compose(initial_state, inplace=True)
        
        # Step 2: Apply p layers of QAOA
        for layer in range(self.p):
            gamma = self.gammas[layer]
            beta = self.betas[layer]
            
            # Cost unitary: exp(-iγH_C)
            cost_circuit = self._build_cost_unitary(cost_hamiltonian, gamma)
            qc.compose(cost_circuit, inplace=True)
            
            # Mixer unitary: exp(-iβH_M)
            mixer_circuit = self._build_mixer_unitary(beta)
            qc.compose(mixer_circuit, inplace=True)
        
        logger.info(f"QAOA circuit built with {qc.num_qubits} qubits, "
                   f"{qc.depth()} gates")
        
        return qc
    
    def _build_cost_unitary(
        self,
        hamiltonian: SparsePauliOp,
        gamma: float
    ) -> QuantumCircuit:
        """
        Build the cost Hamiltonian unitary.
        
        U_C(γ) = exp(-iγH_C)
        
        For Max-Cut: H_C = Σ w_ij (1 - Z_i Z_j) / 2
        
        This decomposes into RZZ gates for each edge.
        
        Args:
            hamiltonian: Cost Hamiltonian
            gamma: Parameter γ
            
        Returns:
            Circuit implementing cost unitary
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            pauli_str = pauli.to_label()
            
            # Qiskit uses MSB first, so qubit 0 is at the end of the string
            # Reversing it maps index i to qubit i
            indices = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
            
            # Use np.real(coeff) because coeffs are complex
            if len(indices) == 2:
                i, j = indices
                qc.rzz(2 * gamma * np.real(coeff), i, j)
            elif len(indices) == 1:
                i = indices[0]
                qc.rz(2 * gamma * np.real(coeff), i)
        
        return qc
    
    def _build_mixer_unitary(self, beta: float) -> QuantumCircuit:
        """
        Build the mixer Hamiltonian unitary.
        
        U_M(β) = exp(-iβH_M)
        
        For H_M = Σ X_i, this is simply R_x rotations:
        exp(-iβX) = R_x(2β)
        
        Args:
            beta: Parameter β
            
        Returns:
            Circuit implementing mixer unitary
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply RX rotation to each qubit
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
        
        return qc
    
    def build_qaoa_circuit_simple(
        self,
        graph: nx.Graph,
        gamma: float,
        beta: float
    ) -> QuantumCircuit:
        """
        Build a simplified single-layer QAOA circuit.
        
        This is a more direct implementation for Max-Cut.
        
        Args:
            graph: NetworkX graph
            gamma: Cost parameter
            beta: Mixer parameter
            
        Returns:
            QAOA circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize to superposition
        for i in range(self.n_qubits):
            qc.h(i)
        
        # Cost layer: RZZ gates for each edge
        for i, j in graph.edges():
            qc.rzz(2 * gamma, i, j)
        
        # Mixer layer: RX gates
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
        
        return qc
    
    def build_qaoa_circuit_multilayer(
        self,
        graph: nx.Graph,
        gammas: List[float],
        betas: List[float]
    ) -> QuantumCircuit:
        """
        Build a multi-layer QAOA circuit.
        
        Args:
            graph: NetworkX graph
            gammas: List of gamma parameters (length p)
            betas: List of beta parameters (length p)
            
        Returns:
            Multi-layer QAOA circuit
        """
        if len(gammas) != len(betas):
            raise ValueError(
                f"gammas and betas must have same length, "
                f"got {len(gammas)} and {len(betas)}"
            )
        
        p = len(gammas)
        qc = QuantumCircuit(self.n_qubits)
        
        # Initialize
        for i in range(self.n_qubits):
            qc.h(i)
        
        # Apply p layers
        for gamma, beta in zip(gammas, betas):
            # Cost layer
            for i, j in graph.edges():
                qc.rzz(2 * gamma, i, j)
            
            # Mixer layer
            for i in range(self.n_qubits):
                qc.rx(2 * beta, i)
        
        return qc
    
    def get_initial_parameters(
        self,
        strategy: str = "random"
    ) -> np.ndarray:
        """
        Generate initial parameters for QAOA.
        
        Args:
            strategy: Initialization strategy
                - "random": Random parameters in [0, 2π]
                - "linear": γ_i = π/2, β_i = π/4
                - "custom": Use provided values
                
        Returns:
            Array of parameters [gamma0, beta0, gamma1, beta1, ...]
        """
        params = np.zeros(2 * self.p)
        
        if strategy == "random":
            params = self.rng.uniform(0, 2 * np.pi, size=2 * self.p)
        elif strategy == "linear":
            for i in range(self.p):
                params[2 * i] = np.pi / 2      # gamma
                params[2 * i + 1] = np.pi / 4  # beta
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Generated initial parameters: {strategy}")
        return params
    
    def parameter_bounds(
        self
    ) -> Tuple[List[float], List[float]]:
        """
        Get parameter bounds for optimization.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = [0.0] * (2 * self.p)
        upper = [2 * np.pi] * (2 * self.p)
        
        return lower, upper
    
    def circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """
        Export circuit to QASM format.
        
        Args:
            circuit: QAOA circuit
            
        Returns:
            QASM string
        """
        return circuit.qasm()
    
    def count_gates(self, circuit: QuantumCircuit) -> dict:
        """
        Count gates in the circuit.
        
        Args:
            circuit: QAOA circuit
            
        Returns:
            Dictionary of gate counts
        """
        from collections import Counter
        
        gate_counts = Counter()
        for instruction in circuit.data:
            gate_name = instruction[0].name
            gate_counts[gate_name] += 1
        
        return dict(gate_counts)


class QAOACircuitFactory:
    """
    Factory class for creating different QAOA circuit variants.
    """
    
    @staticmethod
    def create_xy_mixer(
        n_qubits: int,
        graph: nx.Graph
    ) -> QuantumCircuit:
        """
        Create QAOA circuit with XY mixer Hamiltonian.
        
        XY mixer: H_M = Σ (X_i X_j + Y_i Y_j)
        
        Args:
            n_qubits: Number of qubits
            graph: Problem graph
            
        Returns:
            QAOA circuit with XY mixer
        """
        qc = QuantumCircuit(n_qubits)
        
        # Initialize
        for i in range(n_qubits):
            qc.h(i)
        
        # XY mixer on edges
        for i, j in graph.edges():
            # Ising-like mixer (simplified)
            qc.cx(i, j)
            qc.rz(0.5, j)
            qc.cx(i, j)
        
        return qc
    
    @staticmethod
    def create_hea_qaoa(
        n_qubits: int,
        p: int,
        layers: Optional[List[int]] = None
    ) -> QuantumCircuit:
        """
        Create QAOA with Hardware-Efficient Ansatz style mixer.
        
        Args:
            n_qubits: Number of qubits
            p: Number of layers
            layers: Optional custom layer configuration
            
        Returns:
            HEA-style QAOA circuit
        """
        qc = QuantumCircuit(n_qubits)
        
        # Initialize
        for i in range(n_qubits):
            qc.ry(np.pi / 4, i)
        
        # HEA layers
        for _ in range(p):
            # Entangling layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            
            # Single-qubit rotations
            for i in range(n_qubits):
                qc.ry(np.random.uniform(0, 2 * np.pi), i)
        
        return qc

