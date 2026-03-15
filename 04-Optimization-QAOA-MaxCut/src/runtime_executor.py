"""
Runtime Executor Module

Implements Qiskit Runtime V2 execution engine for QAOA.

Execution Modes:
- Local Aer simulator ( noiseless )
- Noisy simulator (with noise models)
- IBM Quantum hardware (real quantum computers)

Features:
- EstimatorV2 for expectation value calculation
- Noise mitigation (T-REX - Twirled Readout Error eXtinguishing)
- Circuit transpilation and optimization
- Resilience levels for different accuracy/speed tradeoffs

Author: Quantum AI Research Team
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """
    Container for quantum execution results.
    """
    # Expectation value
    expectation_value: float
    
    # Variance/standard deviation
    variance: float
    
    # Number of shots used
    n_shots: int
    
    # Circuit depth
    circuit_depth: int
    
    # Execution time
    runtime: float
    
    # Backend used
    backend_name: str
    
    # Solution bitstring (sampled)
    sampled_bitstring: Optional[str] = None
    
    # Measurement counts
    measurement_counts: Optional[Dict[str, int]] = None


class RuntimeExecutor:
    """
    Quantum execution engine using Qiskit Runtime.
    
    Supports multiple execution modes and provides
    unified interface for QAOA evaluation.
    """
    
    def __init__(
        self,
        mode: str = "local",
        backend_name: Optional[str] = None,
        shots: int = 1024,
        resilience_level: int = 1,
        optimization_level: int = 1,
        seed: Optional[int] = 42
    ) -> None:
        """
        Initialize runtime executor.
        
        Args:
            mode: Execution mode ("local", "noisy_simulator", "ibm_hardware")
            backend_name: Backend name (for IBM or simulator)
            shots: Number of measurement shots
            resilience_level: Noise resilience (0-2)
            optimization_level: Transpiler optimization (0-3)
            seed: Random seed
        """
        self.mode = mode
        self.backend_name = backend_name
        self.shots = shots
        self.resilience_level = resilience_level
        self.optimization_level = optimization_level
        self.seed = seed
        
        self._initialize_backend()
        
        logger.info(
            f"RuntimeExecutor initialized: mode={mode}, "
            f"backend={backend_name}, shots={shots}"
        )
    
    def _initialize_backend(self) -> None:
        """
        Initialize the quantum backend based on mode.
        """
        if self.mode == "local":
            from qiskit_aer import Aer
            
            self.backend = Aer.get_backend("aer_simulator")
            self.primitive = None
            
        elif self.mode == "noisy_simulator":
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            
            # Create noise model from IBM backend if specified
            if self.backend_name:
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    
                    service = QiskitRuntimeService()
                    real_backend = service.backend(self.backend_name)
                    noise_model = NoiseModel.from_backend(real_backend)
                except Exception as e:
                    logger.warning(f"Could not load noise model: {e}")
                    noise_model = None
            else:
                # Default noise model
                from qiskit_aer.noise import NoiseModel
                noise_model = NoiseModel()
            
            self.backend = AerSimulator(noise_model=noise_model)
            self.primitive = None
            
        elif self.mode == "ibm_hardware":
            if not self.backend_name:
                raise ValueError(
                    "backend_name required for IBM hardware execution"
                )
            
            # Import Runtime Service
            from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
            
            # Get service (uses saved credentials)
            try:
                service = QiskitRuntimeService()
            except Exception as e:
                logger.warning(f"Could not connect to IBM Quantum: {e}")
                service = None
                self.backend = None
                self.primitive = None
                return
            
            # Get backend
            self.backend = service.backend(self.backend_name)
            
            # Create EstimatorV2
            self.primitive = EstimatorV2(
                backend=self.backend,
                options={
                    'default_shots': self.shots,
                    'optimization_level': self.optimization_level,
                    'resilience_level': self.resilience_level
                }
            )
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def execute_circuit(
        self,
        circuit,
        hamiltonian,
        parameter_values: Optional[np.ndarray] = None
    ) -> ExecutionResult:
        """
        Execute QAOA circuit and compute expectation value.
        
        Args:
            circuit: QAOA quantum circuit
            hamiltonian: Cost Hamiltonian (SparsePauliOp)
            parameter_values: Optional parameter binding
            
        Returns:
            ExecutionResult
        """
        import time
        start_time = time.time()
        
        logger.info(f"Executing circuit on {self.mode}")
        
        # Bind parameters if provided
        if parameter_values is not None:
            from qiskit import transpile
            
            # Convert to list if needed
            if hasattr(parameter_values, 'tolist'):
                param_list = parameter_values.tolist()
            else:
                param_list = list(parameter_values)
            
            # Bind to circuit
            from qiskit.circuit import Parameter
            bound_circuit = circuit.assign_parameters(param_list)
        else:
            bound_circuit = circuit
        
        # Execute based on mode
        if self.mode == "local" or self.mode == "noisy_simulator":
            result = self._execute_aer(bound_circuit, hamiltonian)
        else:
            result = self._execute_estimator(bound_circuit, hamiltonian)
        
        result.runtime = time.time() - start_time
        result.circuit_depth = bound_circuit.depth()
        
        return result
    
    def _execute_aer(
        self,
        circuit,
        hamiltonian
    ) -> ExecutionResult:
        """
        Execute using Aer simulator.
        
        Args:
            circuit: Bound quantum circuit
            hamiltonian: Cost Hamiltonian
            
        Returns:
            ExecutionResult
        """
        from qiskit_aer import AerSimulator
        from qiskit.quantum_info import Statevector
        
        # Get backend
        if isinstance(self.backend, AerSimulator):
            # Run simulation
            job = self.backend.run(circuit, shots=self.shots)
            result = job.result()
            
            # Get measurement counts
            counts = result.get_counts()
            
            # Compute expectation value from counts
            exp_value = self._compute_expectation_from_counts(
                counts, hamiltonian, circuit.num_qubits
            )
            
            # Get sampled bitstring (most frequent)
            sampled = max(counts, key=counts.get)
            
            return ExecutionResult(
                expectation_value=exp_value,
                variance=np.sqrt(exp_value / self.shots) if exp_value > 0 else 0,
                n_shots=self.shots,
                circuit_depth=circuit.depth(),
                runtime=0.0,
                backend_name=self.backend.name(),
                sampled_bitstring=sampled,
                measurement_counts=counts
            )
        else:
            raise RuntimeError("Invalid backend for Aer execution")
    
    def _execute_estimator(
        self,
        circuit,
        hamiltonian
    ) -> ExecutionResult:
        """
        Execute using EstimatorV2.
        
        Args:
            circuit: Bound quantum circuit
            hamiltonian: Cost Hamiltonian
            
        Returns:
            ExecutionResult
        """
        if self.primitive is None:
            raise RuntimeError("Estimator not initialized")
        
        # Run estimator
        pub = (circuit, [hamiltonian])
        job = self.primitive.run([pub])
        result = job.result()
        
        # Extract expectation value
        exp_value = result[0].data.evs[0]
        
        return ExecutionResult(
            expectation_value=exp_value,
            variance=result[0].data.evs[0],  # Approximate
            n_shots=self.shots,
            circuit_depth=circuit.depth(),
            runtime=0.0,
            backend_name=self.backend.name,
            sampled_bitstring=None,
            measurement_counts=None
        )
    
    def _compute_expectation_from_counts(
        self,
        counts: Dict[str, int],
        hamiltonian,
        n_qubits: int
    ) -> float:
        """
        Compute expectation value from measurement counts.
        
        Args:
            counts: Measurement outcome counts
            hamiltonian: Hamiltonian operator
            n_qubits: Number of qubits
            
        Returns:
            Expectation value
        """
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            return 0.0
        
        # Compute cost for each bitstring
        total_cost = 0.0
        
        for bitstring, count in counts.items():
            # Convert bitstring to +1/-1 values
            z_values = np.array([1 if b == '0' else -1 for b in bitstring])
            
            # Compute cost using Hamiltonian
            cost = self._evaluate_hamiltonian(hamiltonian, z_values)
            
            total_cost += cost * count
        
        return total_cost / total_shots
    
    def _evaluate_hamiltonian(
        self,
        hamiltonian,
        z_values: np.ndarray
    ) -> float:
        """
        Evaluate Hamiltonian for a given z assignment.
        
        Args:
            hamiltonian: SparsePauliOp
            z_values: +1/-1 assignment
            
        Returns:
            Cost value
        """
        cost = 0.0
        # Evaluate each Pauli term directly to avoid matrix explosion
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            pauli_str = pauli.to_label()
            
            term_val = 1.0
            # Qiskit uses MSB first. Reversing maps string index i to qubit i
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'Z':
                    term_val *= z_values[i]
            
            cost += np.real(coeff) * term_val
            
        return cost
    
    def get_backend_info(self) -> Dict:
        """
        Get information about the current backend.
        
        Returns:
            Dictionary with backend information
        """
        info = {
            'mode': self.mode,
            'backend_name': self.backend_name,
            'shots': self.shots,
            'resilience_level': self.resilience_level,
            'optimization_level': self.optimization_level
        }
        
        if self.backend is not None:
            try:
                info['num_qubits'] = self.backend.num_qubits
                info['coupling_map'] = self.backend.coupling_map
            except:
                pass
        
        return info


class BatchExecutor:
    """
    Batch executor for running multiple QAOA evaluations.
    
    Useful for parameter sweeps and optimization.
    """
    
    def __init__(self, executor: RuntimeExecutor) -> None:
        """
        Initialize batch executor.
        
        Args:
            executor: RuntimeExecutor instance
        """
        self.executor = executor
        logger.info("BatchExecutor initialized")
    
    def execute_parameter_sweep(
        self,
        circuit_factory,
        hamiltonian,
        parameter_grid: List[np.ndarray]
    ) -> List[float]:
        """
        Execute circuit over parameter grid.
        
        Args:
            circuit_factory: Function that creates circuit with params
            hamiltonian: Cost Hamiltonian
            parameter_grid: List of parameter arrays
            
        Returns:
            List of expectation values
        """
        results = []
        
        for params in parameter_grid:
            circuit = circuit_factory(params)
            result = self.executor.execute_circuit(circuit, hamiltonian, params)
            results.append(result.expectation_value)
        
        return results
    
    def execute_batch(
        self,
        circuits: List,
        hamiltonians: List
    ) -> List[ExecutionResult]:
        """
        Execute multiple circuits in batch.
        
        Args:
            circuits: List of QAOA circuits
            hamiltonians: List of Hamiltonians
            
        Returns:
            List of ExecutionResults
        """
        results = []
        
        for circuit, hamiltonian in zip(circuits, hamiltonians):
            result = self.executor.execute_circuit(circuit, hamiltonian)
            results.append(result)
        
        return results


def create_executor(
    mode: str = "local",
    backend_name: Optional[str] = None,
    **kwargs
) -> RuntimeExecutor:
    """
    Factory function to create RuntimeExecutor.
    
    Args:
        mode: Execution mode
        backend_name: Backend name
        **kwargs: Additional arguments
        
    Returns:
        RuntimeExecutor instance
    """
    return RuntimeExecutor(
        mode=mode,
        backend_name=backend_name,
        **kwargs
    )

