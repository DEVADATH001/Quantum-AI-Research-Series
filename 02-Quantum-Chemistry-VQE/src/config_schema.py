"""Pydantic schemas for simulation configuration validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    random_seed: int = 7
    allow_synthetic_fallback: bool = True


class DistanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
    step: float

    @model_validator(mode="after")
    def validate_range(self):
        if self.step <= 0:
            raise ValueError("distances.step must be > 0")
        if self.end < self.start:
            raise ValueError("distances.end must be >= distances.start")
        return self


class ActiveSpaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    freeze_core: bool = False
    active_electrons: Optional[int] = None
    active_spatial_orbitals: Optional[int] = None


class MoleculeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    distances: DistanceConfig
    charge: int = 0
    spin: int = 0
    basis: str = "sto3g"
    active_space: Optional[ActiveSpaceConfig] = None


class AnsatzConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    reps: int = 3
    entanglement: str = "circular"
    su2_gates: List[str] = ["ry"]


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "SLSQP"
    maxiter: int = 100


class VQEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ansatz: List[AnsatzConfig]
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    mapping: str = "parity"  # parity, jordan_wigner, bravyi_kitaev
    warm_start: bool = True  # Use previous optimal parameters as next initial point


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "local"
    resilience_level: int = 1
    optimization_level: int = 1
    shots: int = 4096


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chemical_accuracy_mhartree: float = 1.6


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    molecules: Dict[str, MoleculeConfig]
    vqe: VQEConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize config dictionary."""
    return SimulationConfig.model_validate(config).model_dump()
