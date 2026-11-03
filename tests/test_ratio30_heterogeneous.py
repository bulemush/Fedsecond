import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedproxy import ratio30_heterogeneous as experiment
from fedproxy.config import save_resolved_config
from fedproxy.data.partition import heterogeneous_partition, partition_hash
from fedproxy.evaluation.report import summarize
from fedproxy.ma import homogeneous as suite
from fedproxy.ma.round_budget import config_digest
from test_ma_homogeneous import _fusion, _model, _write


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    model = tmp_path / "model"
    _model(model)
    original_method, original_shared = experiment.method_config, experiment.shared_config
    def config(method, root=experiment.ROOT):
        cfg = original_method(method, root)
        cfg["model"].update(name_or_path=str(model), tokenizer_name_or_path=str(model))
        return cfg
    def shared(root=experiment.ROOT):
        cfg = original_shared(root)
        cfg["model"].update(name_or_path=str(model), tokenizer_name_or_path=str(model))
        return cfg
    monkeypatch.setattr(experiment, "method_config", config)
    monkeypatch.setattr(experiment, "shared_config", shared)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    real_run = subprocess.run
    if os.name == "nt":
        def link(path, target):
            assert path.resolve().is_relative_to(tmp_path.resolve())
            assert target.resolve().is_relative_to(tmp_path.resolve())
            if target.is_file():
                os.link(target, path)
            else:
                result = real_run(["cmd.exe", "/d", "/c", "mklink", "/J", str(path), str(target.resolve())],
                                  capture_output=True, timeout=10)
                assert result.returncode == 0, result.stderr
        monkeypatch.setattr(experiment.common, "_create_link", link)
    calls = []

    def rounds(run, cfg, start, end):
        ma = cfg["ma"]["enabled"]
        manifest = suite._json(run / "data_manifest.json")
        for number in range(start, end + 1):
            metrics = {f"client_{i}": {"elapsed_seconds": 10 + i} for i in range(8)}
            meta = {"complete": True, "completed_round": number, "client_metrics": metrics,
                    "config_hash": config_digest(cfg), "source_model": cfg["model"]["name_or_path"],
                    "partition_hash": manifest["partition_hash"], "communication_bytes":
                    {"uplink": 60, "downlink": 110, "downlink_fp32_state_estimate": 200}
                    if ma else {"client_upload_total": 100, "server_down_total_including_conflict": 200}}
            if ma:
                meta.update(ma_protocol_version=1, client_states={},
                            compression_manifest_hash=hashlib.sha256((run / "compression/compression_manifest.json").read_bytes()).hexdigest(),
                            analysis={"observed_coordinate_count": 200, "multi_contributor_coordinate_count": 150})
                for client in metrics:
                    residual = run / "clients" / client / f"round_{number:04d}" / "residual.pt"
                    residual.parent.mkdir(parents=True, exist_ok=True)
                    residual.write_bytes(b"fixture residual, not loaded")
                    meta["client_states"][client] = str(residual)
                _write(run / "metrics" / f"round_{number:04d}.json", {"round_wall_seconds": 25})
            checkpoint = run / "checkpoints" / f"round_{number:04d}"
            _write(checkpoint / "metadata.json", meta)
            (checkpoint / "adapter.safetensors").write_bytes(b"fixture adapter, not loaded")
            (checkpoint / "conflict.safetensors").write_bytes(b"fixture conflict, not loaded")

    def evaluate(run, cfg):
        scores = {task: 0.5 + i * 0.01 + (0.02 if cfg["ma"]["enabled"] else 0)
                  for i, task in enumerate(experiment.TASKS)}
        protocol = {"tasks": experiment.HARNESS_TASKS, "num_fewshot": 0, "batch_size": 1,
                    "primary_metric": "acc", "dtype": "bfloat16"}
        _write(run / "evaluation/fused_summary.json", {
            **summarize(scores), "protocol": protocol,
            "protocol_fingerprint": hashlib.sha256(experiment.json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            "model_kind": "fused", "synthetic": False, "diagnostic": True,
            "fusion_manifest_sha256": hashlib.sha256((run / "exports/llm_fused/fusion_manifest.json").read_bytes()).hexdigest(),
        })
        _write(run / "evaluation/raw/fused_lm_eval.json", {
            "versions": {task: 1 for task in experiment.HARNESS_TASKS},
            "configs": {task: {"metadata": {"pretrained": str(run / "exports/llm_fused")}}
                        for task in experiment.HARNESS_TASKS},
            "n-shot": {task: 0 for task in experiment.HARNESS_TASKS},
        })

    def stage(name, frozen, gpu_ids):
        cfg = suite._yaml(frozen)
        run = Path(cfg["run"]["output_dir"])
        calls.append((run, name))
        if name == "compress":
            _model(run / "compression/proxy", layers=23)
            _write(run / "compression/compression_manifest.json", {
                "original_layers": 32, "proxy_layers": 23, "kept_indices": list(range(23)),
                "layer_map": {str(i): i for i in range(23)}, "source_model": str(model),
                "requested_removed_block_ratio": 0.30, "actual_removed_block_ratio": 1 - 23 / 32,
            })
            save_resolved_config(cfg, run / "resolved_config.compress.yaml")
        elif name == "prepare":
            selected = {task: list(range(12 + i)) for i, task in enumerate(experiment.TASKS)}
            manifest = {"scenario": "heterogeneous", "seed": 42,
                        "dataset_fingerprints": {task: "fixture" for task in experiment.TASKS},
                        "clients": heterogeneous_partition(list(experiment.TASKS), selected)}
            manifest["partition_hash"] = partition_hash(manifest)
            _write(run / "data_manifest.json", manifest)
            save_resolved_config(cfg, run / "resolved_config.prepare-data.yaml")
        elif name in ("train", "resume"):
            first = suite.completed_rounds(run, cfg, ma=cfg["ma"]["enabled"]) + 1
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            rounds(run, cfg, first, 10)
        elif name == "evaluate-fused":
            evaluate(run, cfg)

    def command(arguments, **kwargs):
        cfg = suite._yaml(Path(arguments[arguments.index("--config") + 1]))
        run = Path(cfg["run"]["output_dir"])
        assert "fuse" in arguments and arguments[arguments.index("--checkpoint") + 1] == str(run / "checkpoints/round_0010")
        calls.append((run, "fuse"))
        _fusion(run)
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", command)
    monkeypatch.setattr("fedproxy.ma.report.compare", lambda *args: pytest.fail("Methods must be evaluated independently"))
    return SimpleNamespace(root=tmp_path / "heterogeneous", calls=calls, rounds=rounds, stage=stage)


def test_validation_is_read_only_and_checks_eight_task_scheme_b(simulation):
    experiment.main(["validate", "--root", str(simulation.root)])
    assert not simulation.root.exists() and simulation.calls == []


def test_full_pipeline_trains_two_global_models_and_skips_verified_completion(simulation):
    root = simulation.root
    args = ["run", "--root", str(root), "--gpu-ids", "2,3"]
    experiment.main(args)
    counts = {stage: sum(name == stage for _, name in simulation.calls)
              for stage in ("compress", "prepare", "train", "fuse", "evaluate-fused")}
    assert counts == {"compress": 1, "prepare": 1, "train": 2, "fuse": 2, "evaluate-fused": 2}
    assert (root / "MA/data_manifest.json").samefile(root / "FedProxy/data_manifest.json")
    for method, folder in experiment.common.FOLDERS.items():
        run = root / folder
        assert (run / "compression").samefile(root / "shared/compression")
        result = suite._json(run / "summary.json")
        assert result["scenario"] == "heterogeneous" and result["num_clients"] == 8
        assert result["rounds"] == 10 and set(result["tasks"]) == set(experiment.TASKS)
        assert result["all_avg"] == pytest.approx(0.555 if method == "ma" else 0.535)
        assert result["evaluation_checkpoint_verified"]
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 11
        assert not (run / "comparison.json").exists()
        meta = suite._json(run / "checkpoints/round_0010/metadata.json")
        assert len(meta["client_metrics"]) == 8
        if method == "ma":
            assert len(meta["client_states"]) == 8
    simulation.calls.clear()
    experiment.main(args)
    assert simulation.calls == []


def test_ma_only_does_not_require_a_trained_fedproxy_model(simulation):
    root = simulation.root
    experiment.main(["run", "--root", str(root), "--method", "ma"])
    assert [run for run, stage in simulation.calls if stage == "train"] == [root / "MA"]
    assert not (root / "FedProxy/checkpoints").exists()
    assert (root / "FedProxy/resolved_config.prepare-data.yaml").is_file()
    assert (root / "MA/summary.json").is_file()


def test_fedproxy_and_ma_can_be_launched_separately(simulation):
    root = simulation.root
    for method in ("fedproxy", "ma"):
        experiment.main(["run", "--root", str(root), "--method", method])
    assert sum(stage == "compress" for _, stage in simulation.calls) == 1
    assert sum(stage == "prepare" for _, stage in simulation.calls) == 1


@pytest.mark.parametrize("method", ["fedproxy", "ma"])
def test_partial_round_history_is_preserved_when_resuming(simulation, monkeypatch, method):
    failed = False
    def interrupt(name, frozen, gpu_ids):
        nonlocal failed
        if name == "train" and not failed:
            failed = True
            cfg = suite._yaml(frozen)
            run = Path(cfg["run"]["output_dir"])
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            simulation.rounds(run, cfg, 1, 3)
            raise RuntimeError("simulated interruption")
        simulation.stage(name, frozen, gpu_ids)
    monkeypatch.setattr(suite, "_stage", interrupt)
    args = ["run", "--root", str(simulation.root), "--method", method]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.main(args)
    run = simulation.root / experiment.common.FOLDERS[method]
    first = (run / "checkpoints/round_0001/metadata.json").read_bytes()
    experiment.main(args)
    assert (run / "checkpoints/round_0001/metadata.json").read_bytes() == first
    assert (run, "resume") in simulation.calls
    assert suite._json(run / "summary.json")["rounds"] == 10


def test_missing_eighth_client_residual_stops_resume(simulation, monkeypatch):
    def interrupt(name, frozen, gpu_ids):
        if name == "train":
            cfg = suite._yaml(frozen)
            run = Path(cfg["run"]["output_dir"])
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            simulation.rounds(run, cfg, 1, 3)
            raise RuntimeError("simulated interruption")
        simulation.stage(name, frozen, gpu_ids)
    monkeypatch.setattr(suite, "_stage", interrupt)
    args = ["run", "--root", str(simulation.root), "--method", "ma"]
    with pytest.raises(RuntimeError):
        experiment.main(args)
    (simulation.root / "MA/clients/client_7/round_0003/residual.pt").unlink()
    simulation.calls.clear()
    with pytest.raises(FileNotFoundError, match="residual states"):
        experiment.main(args)
    assert simulation.calls == []


@pytest.mark.parametrize("damage", ["binding", "missing_task", "average", "protocol", "null_scores", "null_protocol"])
def test_stale_or_incomplete_evaluation_is_refreshed_without_retraining(simulation, damage):
    root = simulation.root
    args = ["run", "--root", str(root), "--method", "ma"]
    experiment.main(args)
    path = root / "MA/evaluation/fused_summary.json"
    value = suite._json(path)
    if damage == "binding":
        value["fusion_manifest_sha256"] = "stale"
    elif damage == "missing_task":
        del value["tasks"]["mnli"]
    elif damage == "average":
        value["all_avg"] = 0.99
    elif damage == "protocol":
        value["protocol"]["num_fewshot"] = 5
    elif damage == "null_scores":
        value["tasks"] = None
    else:
        value["protocol"] = None
    _write(path, value)
    simulation.calls.clear()
    experiment.main(args)
    assert simulation.calls == [(root / "MA", "audit"), (root / "MA", "evaluate-fused")]


@pytest.mark.parametrize("damage", ["duplicate", "too_many", "wrong_task", "iid_format"])
def test_heterogeneous_partition_validator_rejects_invalid_assignment(simulation, damage):
    root = simulation.root
    experiment.main(["run", "--root", str(root), "--method", "fedproxy"])
    path = root / "FedProxy/data_manifest.json"
    value = suite._json(path)
    client = value["clients"]["client_0"]
    if damage == "duplicate":
        client["indices"] = [0, 0]
    elif damage == "too_many":
        client["indices"] = list(range(3001))
    elif damage == "wrong_task":
        client["task"] = "mnli"
    else:
        value["clients"]["client_0"] = client["indices"]
    value["partition_hash"] = partition_hash({key: item for key, item in value.items() if key != "partition_hash"})
    _write(path, value)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="assignment|indices"):
        experiment.main(["run", "--root", str(root), "--method", "fedproxy"])
    assert simulation.calls == []


def test_cross_task_index_reuse_is_valid(simulation):
    root = simulation.root
    experiment.main(["run", "--root", str(root), "--method", "ma"])
    manifest = experiment.check_partition(root / "MA/data_manifest.json")
    assert all(0 in value["indices"] for value in manifest["clients"].values())


@pytest.mark.parametrize("visible", [0, 1])
def test_two_visible_gpus_are_required_before_preparation(simulation, monkeypatch, visible):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: visible)))
    with pytest.raises(ValueError, match="parallel|visible"):
        experiment.main(["run", "--root", str(simulation.root)])
    assert simulation.calls == []
    assert not (simulation.root / "shared/execution_config.yaml").exists()


def test_frozen_budget_changes_are_rejected_without_overwriting(simulation):
    args = ["run", "--root", str(simulation.root), "--method", "ma"]
    experiment.main(args)
    path = simulation.root / "MA/execution_config.yaml"
    cfg = suite._yaml(path)
    cfg["training"]["local_epochs"] = 10
    save_resolved_config(cfg, path)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="Existing configuration changed"):
        experiment.main(args)
    assert simulation.calls == [] and suite._yaml(path)["training"]["local_epochs"] == 10


def test_summary_can_be_regenerated_without_training(simulation):
    args = ["--root", str(simulation.root)]
    experiment.main(["run", *args])
    simulation.calls.clear()
    experiment.main(["summarize", *args])
    assert simulation.calls == []
