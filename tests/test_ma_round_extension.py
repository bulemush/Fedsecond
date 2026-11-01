import copy
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from fedproxy.ma import extend, homogeneous as suite
from fedproxy.ma.round_budget import config_digest, target_rounds
from test_ma_homogeneous import _candidate, _evaluation, _fixture, _fusion, _rounds, _write


@pytest.fixture(autouse=True)
def isolate_runtime_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def test_extension_preserves_frozen_config_hash_checkpoints_and_old_scores(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    frozen = (run / "execution_config.yaml").read_bytes()
    checkpoint = (run / "checkpoints/round_0010/metadata.json").read_bytes()
    assert extend.preflight("obqa", run.parent, 20) == cfg
    extend.register_extension(cfg, 20)
    assert target_rounds(cfg) == 20
    assert suite.completed_rounds(run, cfg, ma=True) == 10
    assert (run / "execution_config.yaml").read_bytes() == frozen
    assert (run / "checkpoints/round_0010/metadata.json").read_bytes() == checkpoint
    archived = run / "round_results/round_0010/evaluation/fused_summary.json"
    assert suite._json(archived)["task_accuracy"] == pytest.approx(0.6)
    assert suite._json(run / "round_extension.json")["config_hash"] == config_digest(cfg)


def test_extension_uses_saved_epochs_instead_of_current_yaml_defaults(tmp_path):
    cfg, _, source = _fixture(tmp_path)
    cfg["training"]["local_epochs"] = 10
    run = _candidate(cfg, "obqa", source)
    assert extend.preflight("obqa", run.parent, 20)["training"]["local_epochs"] == 10


def test_extension_requires_committed_residuals_even_at_original_final_round(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    (run / "clients/client_0/round_0010/residual.pt").unlink()
    with pytest.raises(FileNotFoundError, match="residual states"):
        extend.preflight("obqa", run.parent, 20)
    assert not (run / "round_extension.json").exists()


def test_extension_rejects_changed_original_config_and_invalid_horizon(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    extend.register_extension(cfg, 20)
    changed = copy.deepcopy(cfg)
    changed["training"]["local_epochs"] += 1
    with pytest.raises(ValueError, match="changed original configuration"):
        target_rounds(changed)
    path = Path(cfg["run"]["output_dir"]) / "round_extension.json"
    value = suite._json(path)
    value["target_rounds"] = 9
    _write(path, value)
    with pytest.raises(ValueError, match="Invalid round extension"):
        target_rounds(cfg)


def test_extension_can_resume_partial_extended_run_without_rewriting_history(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    extend.register_extension(cfg, 20)
    archive = (run / "round_results/round_0010/evaluation/fused_summary.json").read_bytes()
    _rounds(run, cfg, ma=True, start=11, end=12)
    assert extend.preflight("obqa", run.parent, 20) == cfg
    extend.register_extension(cfg, 20)
    assert suite.completed_rounds(run, cfg, ma=True) == 12
    assert (run / "round_results/round_0010/evaluation/fused_summary.json").read_bytes() == archive
    with pytest.raises(ValueError, match="Complete the current 20-round budget"):
        extend.preflight("obqa", run.parent, 30)


def test_existing_ten_round_run_continues_then_refuses_old_fusion_and_evaluates_twenty(tmp_path, monkeypatch):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    extend.register_extension(cfg, 20)
    original = (run / "checkpoints/round_0010/metadata.json").read_bytes()
    calls = []
    def stage(name, frozen, gpu_ids):
        calls.append(name)
        assert suite._yaml(frozen) == cfg
        if name == "resume":
            _rounds(run, cfg, ma=True, start=11, end=20)
        elif name == "evaluate-fused":
            _evaluation(run, "obqa", score=0.7)
    def command(arguments, **kwargs):
        calls.append("fuse")
        assert arguments[arguments.index("--checkpoint") + 1] == str(run / "checkpoints/round_0020")
        assert "--overwrite" in arguments
        _fusion(run)
        _write(run / "exports/llm_fused/fusion_manifest.json", {"checkpoint": str(run / "checkpoints/round_0020")})
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", command)
    result = suite.execute_task("obqa", cfg, "2,3")
    assert calls == ["reuse-audit", "dry-run", "resume", "fuse", "evaluate-fused"]
    assert result["rounds"] == 20 and result["task_accuracy"] == pytest.approx(0.7)
    assert (run / "checkpoints/round_0010/metadata.json").read_bytes() == original
    calls.clear()
    suite.execute_task("obqa", cfg, "2,3")
    assert calls == []


def test_failed_later_task_preflight_does_not_register_any_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "TASKS", ("obqa", "arc_easy"))
    cfg, _, _ = _fixture(tmp_path, "obqa", candidate=True)
    other, _, _ = _fixture(tmp_path, "arc_easy", candidate=True)
    run = Path(cfg["run"]["output_dir"])
    (Path(other["run"]["output_dir"]) / "clients/client_0/round_0010/residual.pt").unlink()
    with pytest.raises(FileNotFoundError, match="residual states"):
        extend.main(["--root", str(run.parent), "--rounds", "20", "--gpu-ids", "2,3"])
    assert not (run / "round_extension.json").exists()
    assert not (run / "round_results").exists()


def test_validation_only_does_not_change_round_budget(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    extend.main(["--root", str(run.parent), "--task", "obqa", "--rounds", "20", "--validate-only"])
    assert target_rounds(cfg) == 10
    assert not (run.parent / ".suite.lock").exists()


def test_extension_refuses_incomplete_original_run_and_nonincreasing_target(tmp_path):
    cfg, _, source = _fixture(tmp_path)
    run = _candidate(cfg, "obqa", source, rounds=5, evaluated=False)
    with pytest.raises(ValueError, match="Complete the current 10-round budget"):
        extend.preflight("obqa", run.parent, 20)
    with pytest.raises(ValueError, match="extend the original budget"):
        extend.preflight("obqa", run.parent, 10)


def test_extension_cli_finishes_all_tasks_and_repeat_launch_does_not_retrain(tmp_path, monkeypatch):
    for task in suite.TASKS:
        _fixture(tmp_path, task, candidate=True)
    root = tmp_path / "MA_homogeneous"
    suite.write_suite_summary(root)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    calls = []
    def stage(name, frozen, gpu_ids):
        cfg = suite._yaml(frozen)
        run = Path(cfg["run"]["output_dir"])
        task = cfg["data"]["tasks"][0]
        calls.append((task, name))
        if name == "resume":
            _rounds(run, cfg, ma=True, start=11, end=20)
        elif name == "evaluate-fused":
            _evaluation(run, task, score=0.7)
    def command(arguments, **kwargs):
        cfg = suite._yaml(Path(arguments[arguments.index("--config") + 1]))
        run = Path(cfg["run"]["output_dir"])
        _write(run / "exports/llm_fused/fusion_manifest.json", {"checkpoint": str(run / "checkpoints/round_0020")})
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", command)
    arguments = ["--root", str(root), "--rounds", "20", "--gpu-ids", "2,3"]
    extend.main(arguments)
    assert len([item for item in calls if item[1] == "resume"]) == 8
    summary = suite._json(root / "summary.json")
    assert summary["rounds"] == 20 and summary["all_avg"] == pytest.approx(0.7)
    assert suite._json(root / "round_results/round_0010/summary.json")["all_avg"] == pytest.approx(0.6)
    calls.clear()
    extend.main(arguments)
    assert calls == []
