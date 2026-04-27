"""Tests for run-scoped artifact management and provenance capture."""

import shutil
from pathlib import Path

from src.artifact_manager import ArtifactManager
from src.artifact_schema import BenchmarkMetricRecord
from src.provenance import collect_run_provenance, config_hash


def test_artifact_manager_writes_run_scoped_and_root_mirrors():
    tmp_path = Path("04-Optimization-QAOA-MaxCut/tests/.tmp_artifacts")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    manager = ArtifactManager(project_root=tmp_path, output_dir="results", run_id="test-run")
    record = BenchmarkMetricRecord(
        method="qaoa_p1",
        depth=1,
        expected_cut_value=1.0,
        sampled_cut_value=1.0,
        best_sampled_cut_value=1.0,
        approximation_ratio=1.0,
        minimization_objective=-1.0,
        reevaluated_minimization_objective=-1.0,
        objective_std=0.0,
        objective_stderr=0.0,
        n_evaluations=1,
        runtime_sec=0.1,
        representative_bitstring="01",
        representative_probability=0.5,
        best_sampled_bitstring="01",
        analysis_mode="same_backend",
        diagnostics="",
        most_likely_bitstring="01",
    )

    manager.write_csv("metrics.csv", [record])
    manager.write_json("run_manifest.json", {"run_id": "test-run"})
    manager.write_text("notes.md", "hello")

    assert manager.run_path("metrics.csv").exists()
    assert manager.root_path("metrics.csv").exists()
    assert manager.run_path("run_manifest.json").exists()
    assert manager.root_path("notes.md").read_text(encoding="utf-8") == "hello"

    shutil.rmtree(tmp_path)


def test_collect_run_provenance_contains_config_hash():
    config = {"graph": {"n_nodes": 6}, "optimizer": {"seed": 42}}
    provenance = collect_run_provenance(Path("."), config, run_id="unit-test")

    assert provenance["run_id"] == "unit-test"
    assert provenance["config_hash"] == config_hash(config)
    assert "python_version" in provenance
