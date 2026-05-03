"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Backend Manager for modular execution of Qiskit Primitives."""

from typing import Optional, Any, Dict
import logging

from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit_ibm_runtime.options import EstimatorOptions
from qiskit.primitives import StatevectorEstimator

logger = logging.getLogger(__name__)

class BackendManager:
    """Manages Qiskit execution backends and primitives (EstimatorV2)."""

    def __init__(self):
        self.service: Optional[QiskitRuntimeService] = None
        self._initialize_service()

    def _initialize_service(self):
        """Attempts to initialize the IBM Quantum Runtime service."""
        try:
            self.service = QiskitRuntimeService()
            logger.info("Successfully connected to Qiskit Runtime Service.")
        except Exception as e:
            logger.warning(f"Could not connect to Qiskit Runtime Service: {e}")

    def get_estimator(
        self, 
        backend_name: str = "statevector", 
        optimization_level: int = 1,
        resilience_level: int = 1,
        shots: int = 4096,
        session: Optional[Session] = None
    ) -> Any:
        """
        Get an Estimator primitive based on the requested backend.
        
        Args:
            backend_name: "statevector", "aer_simulator", or the name of an IBM backend (e.g. "ibm_kyiv")
            optimization_level: Qiskit transpile optimization level
            resilience_level: Level of error mitigation (0=None, 1=TREX, 2=ZNE)
            shots: Number of shots for hardware/simulators
            session: Optional IBM Runtime session for hardware execution
            
        Returns:
            An EstimatorV2 (for hardware/Aer) or StatevectorEstimator.
        """
        if backend_name.lower() == "statevector":
            logger.info("Using Exact StatevectorEstimator (Ideal, No Shots)")
            return StatevectorEstimator()
            
        if self.service is None:
            raise RuntimeError("QiskitRuntimeService is not initialized. Cannot run on hardware.")
            
        backend = self.service.backend(backend_name)
        
        options = EstimatorOptions()
        options.resilience_level = resilience_level
        
        # Configure advanced error mitigation if resilience is requested
        if resilience_level >= 1:
            options.resilience.measure_mitigation = True # TREX
        if resilience_level >= 2:
            options.resilience.zne_mitigation = True # ZNE
            
        options.default_shots = shots
        
        if session is not None:
            logger.info(f"Using EstimatorV2 on {backend_name} with session.")
            return EstimatorV2(mode=session, options=options)
            
        logger.info(f"Using EstimatorV2 on {backend_name}.")
        return EstimatorV2(mode=backend, options=options)
