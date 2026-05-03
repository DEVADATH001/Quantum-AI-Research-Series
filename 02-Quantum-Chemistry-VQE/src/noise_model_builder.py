"""Author: DEVADATH H K

Quantum AI Research Series

Project 02: Quantum Chemistry VQE
Task: Noise Model Builder."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class NoiseModelBuilder:
    """Builds Qiskit Aer noise models from real IBM backend properties."""
    
    @staticmethod
    def from_backend(backend_name: str = "ibm_kyiv") -> Optional[Any]:
        """
        Creates a noise model from a real IBM backend.
        
        Args:
            backend_name: Name of the IBM backend
            
        Returns:
            NoiseModel object if successful, None otherwise.
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_aer.noise import NoiseModel
            
            service = QiskitRuntimeService()
            backend = service.get_backend(backend_name)
            noise_model = NoiseModel.from_backend(backend)
            logger.info(f"Successfully loaded noise model for {backend_name}")
            return noise_model
        except Exception as e:
            logger.warning(f"Could not load noise model for {backend_name}: {e}")
            return None
            
    @staticmethod
    def get_aer_simulator_with_noise(backend_name: str = "ibm_kyiv") -> Optional[Any]:
        """Returns an AerSimulator configured with the noise model."""
        try:
            from qiskit_aer import AerSimulator
            noise_model = NoiseModelBuilder.from_backend(backend_name)
            if noise_model is None:
                return None
            return AerSimulator(noise_model=noise_model)
        except Exception as e:
            logger.warning(f"Could not initialize AerSimulator with noise: {e}")
            return None
