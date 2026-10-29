import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fedproxy.config import load_config, save_resolved_config
from fedproxy.data.partition import partition_hash
from fedproxy.ma import homogeneous as suite


@pytest.fixture(autouse=True)
def isolate_runtime_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _model(path, layers=32):
    _write(path / "config.json", {"model_type": "llama", "num_hidden_layers": layers, "hidden_size": 4096})
    (path / "model.safetensors").write_bytes(b"test fixture, never loaded")


def _rounds(path, cfg, ma, start=1, end=10):
    for number in range(start, end + 1):
        metrics = {f"client_{i}": {"elapsed_seconds": 10} for i in range(4)}
        meta = {"complete": True, "completed_round": number, "client_metrics": metrics,
                "communication_bytes": {"uplink": 60, "downlink": 110, "downlink_fp32_state_estimate": 200}
                if ma else {"client_upload_total": 100, "server_down_total_including_conflict": 200}}
        if ma:
            meta.update(ma_protocol_version=1, config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                        analysis={"observed_coordinate_count": 200, "multi_contributor_coordinate_count": 150},
                        partition_hash=json.loads((path / "data_manifest.json").read_text())["partition_hash"],
                        compression_manifest_hash=hashlib.sha256((path / "compression/compression_manifest.json").read_bytes()).hexdigest(),
                        client_states={})
            for client in metrics:
                residual = path / "clients" / client / f"round_{number:04d}" / "residual.pt"
                residual.parent.mkdir(parents=True, exist_ok=True)
                residual.write_bytes(b"test residual, never loaded")
                meta["client_states"][client] = str(residual)
            _write(path / "metrics" / f"round_{number:04d}.json", {"round_wall_seconds": 25})
        checkpoint = path / "checkpoints" / f"round_{number:04d}"
        _write(checkpoint / "metadata.json", meta)
        (checkpoint / "adapter.safetensors").write_bytes(b"test adapter, never loaded")
        (checkpoint / "conflict.safetensors").write_bytes(b"test conflict, never loaded")


def _fusion(path):
    _model(path / "exports/llm_fused")
    _write(path / "exports/llm_fused/fusion_manifest.json", {"checkpoint": str(path / "checkpoints/round_0010")})


def _evaluation(path, task, score=0.6):
    manifest = path / "exports/llm_fused/fusion_manifest.json"
    _write(path / "evaluation/fused_summary.json", {
        "model_kind": "fused", "synthetic": False, "task": task,
        "tasks": {task: score}, "task_accuracy": score,
        "protocol_fingerprint": task,
        "fusion_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    _write(path / "evaluation/raw/fused_lm_eval.json", {"versions": {task: 1}, "configs": {task: {}}, "n-shot": {task: 0}})


def _fixture(tmp_path, task="obqa", candidate=False):
    model = tmp_path / "model"
    _model(model)
    source = tmp_path / "baseline" / suite.run_name(task)
    cfg = suite.task_config(task, root=tmp_path / "MA_homogeneous", baseline_root=tmp_path / "baseline")
    cfg["model"].update(name_or_path=str(model), tokenizer_name_or_path=str(model))
    original = load_config(suite.PROJECT / "configs/experiments" / f"llama2_homogeneous_{task}_len320_mb4.yaml")
    original["model"] = copy.deepcopy(cfg["model"])
    original["run"]["output_dir"] = str(source)
    save_resolved_config(original, source / "resolved_config.train.yaml")
    manifest = {"scenario": "homogeneous", "seed": 42, "dataset_fingerprints": {task: "test-data"},
                "clients": {f"client_{i}": list(range(i, 12, 4)) for i in range(4)}}
    manifest["partition_hash"] = partition_hash(manifest)
    _write(source / "data_manifest.json", manifest)
    _model(source / "compression/proxy", layers=16)
    _write(source / "compression/compression_manifest.json", {
        "source_model": str(model), "proxy_layers": 16, "original_layers": 32, "tokenizer_sha256": "vocab",
    })
    _rounds(source, original, ma=False)
    _fusion(source)
    _evaluation(source, task, score=0.5)
    if candidate:
        _candidate(cfg, task, source)
    return cfg, original, source


def _candidate(cfg, task, source, rounds=10, evaluated=True):
    path = Path(cfg["run"]["output_dir"])
    save_resolved_config(cfg, path / "execution_config.yaml")
    save_resolved_config(cfg, path / "resolved_config.train.yaml")
    shutil.copytree(source / "compression", path / "compression")
    shutil.copyfile(source / "data_manifest.json", path / "data_manifest.json")
    _rounds(path, cfg, ma=True, end=rounds)
    if evaluated:
        _fusion(path)
        _evaluation(path, task)
    return path


@pytest.mark.parametrize("task", suite.TASKS)
def test_configs_match_approved_budget_and_inherit_all_ma_fields(task):
    cfg = suite.task_config(task)
    original = load_config(suite.PROJECT / "configs/experiments" / f"llama2_homogeneous_{task}_len320_mb4.yaml")
    suite._check_contract(task, cfg)
    for section in ("data", "training", "federated", "evaluation"):
        assert cfg[section] == original[section]
    assert cfg["ma"] == load_config(suite.PROJECT / "configs/ma/base.yaml")["ma"]
    assert Path(cfg["run"]["output_dir"]) == suite.DEFAULT_ROOT / suite.run_name(task)
    assert cfg["run"]["reuse_model_relocation"] is None
    assert cfg["run"]["reuse_artifacts_from"] == str(Path("runs") / suite.run_name(task)).replace("\\", "/")


def test_preflight_is_read_only_and_checks_existing_baseline(tmp_path):
    cfg, _, _ = _fixture(tmp_path)
    assert suite.preflight("obqa", cfg) == cfg
    assert not Path(cfg["run"]["output_dir"]).exists()


@pytest.mark.parametrize("section,key,value", [
    ("training", "local_epochs", 10), ("training", "micro_batch_size", 2),
    ("data", "max_input_length", 64), ("data", "max_train_samples_per_task", 5000),
])
def test_preflight_rejects_changed_baseline_budget(tmp_path, section, key, value):
    cfg, original, source = _fixture(tmp_path)
    original[section][key] = value
    save_resolved_config(original, source / "resolved_config.train.yaml")
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        suite.preflight("obqa", cfg)
    assert not Path(cfg["run"]["output_dir"]).exists()


def test_preflight_rejects_partition_content_corruption(tmp_path):
    cfg, _, source = _fixture(tmp_path)
    path = source / "data_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["clients"]["client_0"][0] = 100
    _write(path, manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        suite.preflight("obqa", cfg)


def test_preflight_rejects_other_existing_artifacts_and_changed_snapshot(tmp_path):
    cfg, _, _ = _fixture(tmp_path)
    path = Path(cfg["run"]["output_dir"])
    (path / "compression").mkdir(parents=True)
    with pytest.raises(ValueError, match="does not link"):
        suite.preflight("obqa", cfg)
    (path / "compression").rmdir()
    old = copy.deepcopy(cfg)
    old["training"]["local_epochs"] = 10
    save_resolved_config(old, path / "resolved_config.train.yaml")
    with pytest.raises(ValueError, match="Existing MA settings changed"):
        suite.preflight("obqa", cfg)


def test_known_model_relocation_is_checked_and_recorded(tmp_path, monkeypatch):
    cfg, original, source = _fixture(tmp_path)
    move = {"original": str(tmp_path / "deleted_model"), "current": cfg["model"]["name_or_path"]}
    original["model"].update(name_or_path=move["original"], tokenizer_name_or_path=move["original"])
    save_resolved_config(original, source / "resolved_config.train.yaml")
    manifest = suite._json(source / "compression/compression_manifest.json")
    manifest["source_model"] = move["original"]
    _write(source / "compression/compression_manifest.json", manifest)
    real_load = suite.load_config
    def load(path):
        result = real_load(path)
        if Path(path) == suite.PROJECT / "configs/ma/base.yaml":
            result["run"]["reuse_model_relocation"] = move
        return result
    monkeypatch.setattr(suite, "load_config", load)
    monkeypatch.setattr("fedproxy.reuse._tokenizer_vocab_sha256", lambda _: "vocab")
    assert suite.preflight("obqa", cfg)["run"]["reuse_model_relocation"] == move


def test_comparison_uses_single_task_accuracy_and_real_downlink(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    pair = suite.compare_task("obqa", cfg)
    assert pair["accuracy_delta_pp"] == pytest.approx(10)
    assert pair["uplink_ratio"] == pytest.approx(0.6)
    assert pair["downlink_ratio"] == pytest.approx(0.55)
    assert pair["candidate"]["local_epochs"] == 5


def test_stale_fusion_or_wrong_task_cannot_be_reported(tmp_path):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    path = run / "evaluation/fused_summary.json"
    evaluation = suite._json(path)
    evaluation["fusion_manifest_sha256"] = "stale"
    _write(path, evaluation)
    with pytest.raises(ValueError, match="predates"):
        suite.compare_task("obqa", cfg)
    evaluation["task"] = "rte"
    _write(path, evaluation)
    with pytest.raises(ValueError, match="Invalid single-task"):
        suite.compare_task("obqa", cfg)


def test_resume_rejects_missing_residual_or_replaced_compression(tmp_path):
    cfg, _, source = _fixture(tmp_path)
    run = _candidate(cfg, "obqa", source, rounds=2, evaluated=False)
    assert suite.completed_rounds(run, cfg, ma=True) == 2
    residual = run / "clients/client_0/round_0002/residual.pt"
    residual.unlink()
    with pytest.raises(FileNotFoundError, match="residual states"):
        suite.completed_rounds(run, cfg, ma=True)
    residual.write_bytes(b"test")
    _write(run / "compression/compression_manifest.json", {"changed": True})
    with pytest.raises(ValueError, match="compression_manifest_hash"):
        suite.completed_rounds(run, cfg, ma=True)


def test_stale_evaluation_is_refreshed_without_retraining_or_refusion(tmp_path, monkeypatch):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    evaluation = suite._json(run / "evaluation/fused_summary.json")
    evaluation["fusion_manifest_sha256"] = "stale"
    _write(run / "evaluation/fused_summary.json", evaluation)
    calls = []
    def stage(name, *args):
        calls.append(name)
        if name == "evaluate-fused":
            _evaluation(run, "obqa")
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unnecessary refusion"))
    suite.execute_task("obqa", cfg, "0,1")
    assert calls == ["reuse-audit", "evaluate-fused"]


def test_address_only_harness_difference_skips_completed_training_and_evaluation(tmp_path, monkeypatch):
    cfg, _, source = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    for directory, address in ((source, "0x123abc"), (run, "0x456def")):
        raw = suite._json(directory / "evaluation/raw/fused_lm_eval.json")
        raw["configs"]["obqa"]["process_docs"] = f"<function process_docs at {address}>"
        _write(directory / "evaluation/raw/fused_lm_eval.json", raw)
    monkeypatch.setattr(suite, "_stage", lambda *args: pytest.fail("Already completed task was rerun"))
    result = suite.execute_task("obqa", cfg, "0,1")
    assert result["accuracy_delta_pp"] == pytest.approx(10)
    assert (run / "comparison.json").is_file()


def test_real_harness_difference_does_not_trigger_futile_candidate_reevaluation(tmp_path, monkeypatch):
    cfg, _, _ = _fixture(tmp_path, candidate=True)
    run = Path(cfg["run"]["output_dir"])
    raw = suite._json(run / "evaluation/raw/fused_lm_eval.json")
    raw["versions"]["obqa"] = 2
    _write(run / "evaluation/raw/fused_lm_eval.json", raw)
    monkeypatch.setattr(suite, "_stage", lambda *args: pytest.fail("Real task mismatch was ignored"))
    with pytest.raises(suite.HarnessMismatchError, match="versions.obqa"):
        suite.execute_task("obqa", cfg, "0,1")


def test_collect_all_eight_models_and_refuse_partial_average(tmp_path):
    for task in suite.TASKS:
        _fixture(tmp_path, task, candidate=True)
    root = tmp_path / "MA_homogeneous"
    result = suite.write_suite_summary(root)
    assert result["independent_models"] == 8
    assert result["all_avg"] == pytest.approx(0.6)
    assert result["baseline"]["all_avg"] == pytest.approx(0.5)
    assert result["all_delta_pp"] == pytest.approx(10)
    assert (root / "summary.json").is_file() and (root / "comparison.csv").is_file()
    (root / suite.run_name("mnli") / "evaluation/fused_summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        suite.collect_suite(root)


@pytest.mark.parametrize("finished,expected_training", [(0, "train"), (2, "resume"), (10, None)])
def test_execute_runs_or_resumes_then_fuses_and_evaluates(tmp_path, monkeypatch, finished, expected_training):
    cfg, _, source = _fixture(tmp_path)
    run = Path(cfg["run"]["output_dir"])
    if finished:
        _candidate(cfg, "obqa", source, rounds=finished, evaluated=False)
    calls = []
    def stage(name, config_path, gpu_ids):
        calls.append(name)
        assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == cfg
        if name == "train":
            _candidate_artifacts(run, source)
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            _rounds(run, cfg, ma=True)
        elif name == "resume":
            _rounds(run, cfg, ma=True, start=finished + 1)
        elif name == "evaluate-fused":
            _evaluation(run, "obqa")
    def command(arguments, **kwargs):
        calls.append("fuse")
        assert "fedproxy.cli" in arguments and "fuse" in arguments
        _fusion(run)
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", command)
    result = suite.execute_task("obqa", cfg, "0,1")
    expected = ["reuse-audit"] + (["dry-run", expected_training] if expected_training else []) + ["fuse", "evaluate-fused"]
    assert calls == expected
    assert result["accuracy_delta_pp"] == pytest.approx(10)
    assert (run / "comparison.json").is_file()
    calls.clear()
    suite.execute_task("obqa", cfg, "0,1")
    assert calls == []  # Completion is verified rather than inferred from a filename.


def _candidate_artifacts(run, source):
    shutil.copytree(source / "compression", run / "compression")
    shutil.copyfile(source / "data_manifest.json", run / "data_manifest.json")


def test_invalid_later_preflight_prevents_every_training_job(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "task_config", lambda task, *args: {"run": {"output_dir": str(tmp_path / task)}})
    def check(task, cfg):
        if task == "arc_challenge":
            raise ValueError("baseline budget mismatch")
        return cfg
    monkeypatch.setattr(suite, "preflight", check)
    monkeypatch.setattr(suite, "execute_task", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="budget mismatch"):
        suite.main(["run", "--root", str(tmp_path)])
    assert not calls


@pytest.mark.parametrize("gpu_ids", ["0", "0,0", "0,1,2"])
def test_suite_rejects_gpu_selection_other_than_two_distinct_ids(monkeypatch, gpu_ids):
    monkeypatch.setattr(suite, "preflight", lambda *args: pytest.fail("Invalid GPU IDs reached preflight"))
    with pytest.raises(SystemExit) as error:
        suite.main(["run", "--gpu-ids", gpu_ids])
    assert error.value.code == 2


@pytest.mark.parametrize("visible", [1, 2])
def test_suite_exports_two_gpu_requirement_before_preflight_and_checks_visibility(tmp_path, monkeypatch, visible):
    cfg = suite.task_config("obqa", tmp_path)
    saved = copy.deepcopy(cfg)
    monkeypatch.setattr(suite, "task_config", lambda *args: cfg)
    def check(task, config):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert os.environ["FEDPROXY_REQUIRED_PARALLEL_CLIENTS"] == "2"
        return config
    monkeypatch.setattr(suite, "preflight", check)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: visible)))
    calls = []
    def execute(task, config, gpu_ids):
        assert gpu_ids == "2,3"
        assert os.environ["FEDPROXY_GPU_IDS"] == "2,3"
        calls.append(task)
    monkeypatch.setattr(suite, "execute_task", execute)
    arguments = ["run", "--task", "obqa", "--root", str(tmp_path), "--gpu-ids", "2,3"]
    if visible == 1:
        with pytest.raises(ValueError, match="requires 2 parallel clients"):
            suite.main(arguments)
        assert not (Path(cfg["run"]["output_dir"]) / "execution_config.yaml").exists()
        assert calls == []
    else:
        suite.main(arguments)
        assert calls == ["obqa"]
        assert suite._yaml(Path(cfg["run"]["output_dir"]) / "execution_config.yaml") == saved
    assert cfg == saved


def test_suite_lock_blocks_another_launcher_and_releases(tmp_path):
    with suite._suite_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another MA"):
            with suite._suite_lock(tmp_path):
                pytest.fail("A second suite acquired the same lock")
    with suite._suite_lock(tmp_path):
        pass
