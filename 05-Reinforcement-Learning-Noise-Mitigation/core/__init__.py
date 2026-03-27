"""Core interfaces and schemas for the research benchmark system."""

from core.schemas import (
    DEFAULT_SCHEMA_VERSION,
    BenchmarkSuiteResult,
    EvalResult,
    MethodSpec,
    PaperReportBundle,
    ResourceMetrics,
    RunResult,
    ScenarioSpec,
)

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "BenchmarkSuiteResult",
    "EvalResult",
    "MethodSpec",
    "PaperReportBundle",
    "ResourceMetrics",
    "RunResult",
    "ScenarioSpec",
]
