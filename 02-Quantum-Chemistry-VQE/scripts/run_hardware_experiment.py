"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Hardware deployment bridge and orchestration."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
from src.backend_manager import BackendManager
from src.experiment_tracker import ExperimentTracker
from src.molecule_driver import MoleculeDriver
from src.hamiltonian_optimizer import HamiltonianOptimizer
from src.ansatz_factory import AnsatzFactory
from src.vqe_engine import VQEEngine
from qiskit_ibm_runtime import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Hardware Deployment Bridge")
    tracker = ExperimentTracker()
    
    # 1. Prepare Problem (e.g., LiH or BeH2)
    driver = MoleculeDriver()
    problem, metadata = driver.get_problem("H2", 0.735, allow_synthetic_fallback=True)
    tracker.log_metadata("molecule", metadata.__dict__)
    
    # 2. Hardware configuration
    manager = BackendManager()
    
    try:
        available_backends = manager.service.backends()
        if not available_backends:
            logger.error("No backends available on this IBM Quantum account.")
            return
            
        backend = manager.service.least_busy(simulator=False, operational=True)
        backend_name = backend.name
        logger.info(f"Selected least busy operational backend: {backend_name}")
    except Exception as e:
        logger.warning(f"Could not automatically select least busy backend: {e}")
        # Fallback to the first available if least_busy fails
        backend = manager.service.backends()[0]
        backend_name = backend.name
        logger.info(f"Fallback to available backend: {backend_name}")

    optimization_level = 3
    resilience_level = 2  # ZNE + TREX
    
    try:
        # Open plan users are not authorized to run sessions, so we run sessionless
        estimator = manager.get_estimator(
            backend_name=backend_name,
            optimization_level=optimization_level,
            resilience_level=resilience_level,
            shots=4096,
            session=None
        )
        
        tracker.log_metadata("backend", backend_name)
        tracker.log_metadata("optimization_level", optimization_level)
        tracker.log_metadata("resilience_level", resilience_level)
        
        from src.problem_builder import build_mapped_hamiltonian
        from qiskit_algorithms.optimizers import SPSA
        from src.classical_solver import get_exact_energy_from_qubit_operator
        
        # 3. Setup Problem Mapping & Tapering
        mapping = build_mapped_hamiltonian(problem, two_qubit_reduction=True, mapping_name="parity")
        qubit_op, tapered_mapper = HamiltonianOptimizer.apply_z2_tapering(problem, mapping.mapper)
        
        # Setup Ansatz
        ansatz_factory = AnsatzFactory()
        ansatz = ansatz_factory.build("RYRZ", problem, mapper=tapered_mapper)
        
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        
        logger.info(f"Transpiling ansatz and operator to ISA for {backend_name}...")
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level)
        isa_ansatz = pm.run(ansatz)
        isa_qubit_op = qubit_op.apply_layout(isa_ansatz.layout)
        
        # 4. Configure Optimizer and Engine
        optimizer = SPSA(maxiter=50) # SPSA is resilient to hardware noise
        vqe = VQEEngine(estimator, optimizer=optimizer)
        
        total_constant = sum(problem.hamiltonian.constants.values())
        vqe.initialize_vqe(ansatz=isa_ansatz, energy_shift=total_constant)
        
        exact = get_exact_energy_from_qubit_operator(qubit_op) + total_constant
        logger.info(f"Exact Energy for Reference: {exact:.8f} Ha")
        
        logger.info("Starting hardware VQE loop...")
        result = vqe.run_vqe_qubit_with_retry(
            qubit_operator=isa_qubit_op,
            ansatz=isa_ansatz,
            exact_energy=exact,
            threshold=0.0016
        )
        
        tracker.log_result({"energy": result.energy, "iterations": len(result.history)})
        
        logger.info(f"Hardware execution complete. VQE Energy: {result.energy:.8f} Ha")
            
    except Exception as e:
        import traceback
        logger.error(f"Hardware deployment failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
