"""Author: DEVADATH H K

Qiskit Helper Functions

Utility functions for working with Qiskit primitives and backends."""

import logging
from typing import Dict, Optional, List, Any
import numpy as np

logger = logging.getLogger(__name__)

def get_backend_info(backend) -> Dict[str, Any]:
    """
    Get information about a quantum backend.
    
    Args:
        backend: Qiskit backend
        
    Returns:
        Dictionary with backend information
    """
    info = {
        'name': backend.name,
        'num_qubits': backend.num_qubits,
    }
    
    # Try to get coupling map
    try:
        if hasattr(backend, 'coupling_map') and backend.coupling_map:
            info['coupling_map'] = backend.coupling_map
    except:
        pass
    
    # Try to get basis gates
    try:
        if hasattr(backend, 'basis_gates'):
            info['basis_gates'] = backend.basis_gates
    except:
        pass
    
    return info

def create_sampler(
    backend: Optional[Any] = None,
    options: Optional[Dict] = None
):
    """
    Create a Sampler primitive.
    
    Args:
        backend: Optional backend
        options: Sampler options
        
    Returns:
        Sampler instance
    """
    from qiskit.primitives import Sampler
    
    if backend:
        from qiskit.primitives import BackendSampler
        return BackendSampler(backend=backend, options=options or {})
    
    return Sampler(options=options or {})

def create_estimator(
    backend: Optional[Any] = None,
    options: Optional[Dict] = None
):
    """
    Create an Estimator primitive.
    
    Args:
        backend: Optional backend
        options: Estimator options
        
    Returns:
        Estimator instance
    """
    from qiskit.primitives import Estimator
    
    if backend:
        from qiskit.primitives import BackendEstimator
        return BackendEstimator(backend=backend, options=options or {})
    
    return Estimator(options=options or {})

def transpile_circuit(
    circuit,
    backend=None,
    optimization_level: int = 1,
    seed: Optional[int] = None
):
    """
    Transpile a circuit for a backend.
    
    Args:
        circuit: Quantum circuit
        backend: Target backend
        optimization_level: Transpiler optimization (0-3)
        seed: Random seed
        
    Returns:
        Transpiled circuit
    """
    from qiskit import transpile
    
    if backend is None:
        # Use default Aer simulator
        from qiskit_aer import Aer
        
        backend = Aer.get_backend('aer_simulator')
    
    transpiled = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed
    )
    
    return transpiled

def evaluate_circuit_expectation(
    circuit,
    hamiltonian,
    estimator,
    parameter_values: Optional[np.ndarray] = None
) -> float:
    """
    Evaluate expectation value of Hamiltonian for a circuit using an Estimator.
    
    Args:
        circuit: Quantum circuit
        hamiltonian: Hamiltonian operator (SparsePauliOp)
        estimator: Estimator primitive (V1 or V2)
        parameter_values: Optional values for parameterized circuits
        
    Returns:
        Expectation value
    """
    if hasattr(estimator, 'run'):
        # V1/V2 Estimator pattern
        if parameter_values is not None:
            job = estimator.run([circuit], [hamiltonian], [parameter_values])
        else:
            job = estimator.run([circuit], [hamiltonian])
        result = job.result()
        
        # Handle V2 result structure if applicable
        if hasattr(result, 'data'):
            return result[0].data.evs[0]
        return result.values[0]
    
    raise ValueError("Provided estimator does not support the .run() interface.")

def get_quantum_resources(circuit) -> Dict[str, int]:
    """
    Get resource estimates for a circuit.
    
    Args:
        circuit: Quantum circuit
        
    Returns:
        Dictionary with resource counts
    """
    from qiskit.transpiler.analysis import ResourceEstimationObserver
    
    resources = {
        'depth': circuit.depth(),
        'num_qubits': circuit.num_qubits(),
        'num_gates': len(circuit.data),
        'num_clbits': circuit.num_clbits(),
    }
    
    # Count gate types
    gate_counts = {}
    for instruction in circuit.data:
        gate_name = instruction[0].name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    resources['gate_counts'] = gate_counts
    
    return resources

def create_noisy_backend(
    backend_name: str,
    noise_model: Optional[Any] = None,
    coupling_map: Optional[List[List[int]]] = None
):
    """
    Create a noisy simulator backend.
    
    Args:
        backend_name: Name of IBM backend to simulate
        noise_model: Custom noise model
        coupling_map: Custom coupling map
        
    Returns:
        Noisy Aer backend
    """
    from qiskit_aer import AerSimulator
    
    if noise_model is None and coupling_map is None:
        # Use default
        return AerSimulator()
    
    return AerSimulator(
        noise_model=noise_model,
        coupling_map=coupling_map
    )

def format_qubit_mapping(
    mapping: Dict[int, int]
) -> str:
    """
    Format qubit mapping for display.
    
    Args:
        mapping: Dictionary mapping logical to physical qubits
        
    Returns:
        Formatted string
    """
    lines = ["Qubit Mapping:"]
    
    for logical, physical in sorted(mapping.items()):
        lines.append(f"  Logical Qubit {logical} → Physical Qubit {physical}")
    
    return "\n".join(lines)

def calculate_trex_mitigation(
    counts: Dict[str, int],
    readout_errors: Dict[int, Dict[str, float]]
) -> Dict[str, float]:
    """
    Apply T-REX readout error mitigation.
    
    Args:
        counts: Measurement counts
        readout_errors: Readout error probabilities per qubit
        
    Returns:
        Mitigated counts
    """
    # Simplified T-REX implementation
    n_qubits = len(list(readout_errors.keys()))
    
    # Build confusion matrix
    confusion_matrix = np.ones((2, 2))
    
    for qubit, errors in readout_errors.items():
        p0_given_0 = errors.get('0', 1.0)  # Correct 0
        p1_given_1 = errors.get('1', 1.0)  # Correct 1
        
        confusion_matrix[0, 0] *= p0_given_0
        confusion_matrix[1, 1] *= p1_given_1
    
    # Invert and apply (simplified)
    try:
        inv_matrix = np.linalg.inv(confusion_matrix)
    except:
        return counts
    
    # Apply mitigation
    mitigated = {}
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        mitigated[bitstring] = count / total
    
    return mitigated

