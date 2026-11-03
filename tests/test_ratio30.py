import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedproxy import ratio30 as ratio
from fedproxy.config import load_config, save_resolved_config
from fedproxy.data.partition import partition_hash
from fedproxy.ma import homogeneous as suite
from fedproxy.ma.round_budget import config_digest
from fedproxy.ma.scheduler import audit_residual_delivery, plan_round
from test_ma_homogeneous import _evaluation, _fusion, _model, _rounds, _write


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.mark.parametrize("task", ratio.TASKS)
def test_both_configurations_match_approved_budget_and_original_common_settings(task):
    fed = ratio.task_config("fedproxy", task)
    ma = ratio.task_config("ma", task)
    original = load_config(ratio.PROJECT / "configs/experiments" / f"llama2_homogeneous_{task}_len320_mb4.yaml")
    for method, cfg in (("fedproxy", fed), ("ma", ma)):
        ratio.check_contract(method, task, cfg)
        for key in ("model", "data", "training", "lora", "federated", "pcr", "h_ties", "evaluation"):
            assert cfg[key] == original[key]
        assert cfg["compression"] == {**original["compression"], "remove_ratio": 0.30}
    expected = copy.deepcopy(load_config(ratio.PROJECT / "configs/ma/base.yaml")["ma"])
    expected["schedule"].update(train_blocks=17, upload_blocks=14)
    assert ma["ma"] == expected
    assert fed["ma"]["enabled"] is False
    assert fed["training"]["local_epochs"] == ma["training"]["local_epochs"] == 5


def test_scheme_b_real_scheduler_covers_all_23_layers_and_meets_residual_ttl():
    # Scheduling depends on the proxy block IDs, not on GPU tensors.
    registry = SimpleNamespace(blocks=tuple(range(23)))
    clients = [f"client_{i}" for i in range(4)]
    cfg = ratio.task_config("ma", "obqa")
    audit = audit_residual_delivery(registry, clients, 10, cfg)
    assert audit["max_observed_wait_rounds"] <= cfg["ma"]["residual"]["max_age"]
    for round_id in range(10):
        plans = plan_round(registry, clients, round_id, cfg)
        anchors = next(iter(plans.values())).anchor_blocks
        for plan in plans.values():
            assert len(plan.train_blocks) == 17 and len(plan.upload_blocks) == 14
            assert plan.anchor_blocks == anchors and len(anchors) == 1
        for block in set(registry.blocks) - anchors:
            assert sum(block in plan.upload_blocks for plan in plans.values()) >= 2


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    model = tmp_path / "model"
    _model(model)
    original_config = ratio.task_config
    def config(method, task, root=ratio.ROOT):
        cfg = original_config(method, task, root)
        cfg["model"].update(name_or_path=str(model), tokenizer_name_or_path=str(model))
        return cfg
    monkeypatch.setattr(ratio, "task_config", config)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    real_run = subprocess.run
    if os.name == "nt":
        # Directory junctions and hard links need no Windows symlink privilege.
        # They have the same artifact identity checked by the production code.
        def link(path, target):
            assert path.resolve().is_relative_to(tmp_path.resolve())
            assert target.resolve().is_relative_to(tmp_path.resolve())
            if target.is_file():
                os.link(target, path)
            else:
                result = real_run(["cmd.exe", "/d", "/c", "mklink", "/J", str(path), str(target.resolve())],
                                  capture_output=True, timeout=10)
                assert result.returncode == 0, result.stderr
        monkeypatch.setattr(ratio, "_create_link", link)
    calls = []
    def stage(name, frozen, gpu_ids):
        cfg = suite._yaml(frozen)
        run = Path(cfg["run"]["output_dir"])
        task = cfg["data"]["tasks"][0]
        ma = cfg.get("ma", {}).get("enabled", False)
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
            manifest = {"scenario": "homogeneous", "seed": 42, "dataset_fingerprints": {task: "fixture"},
                        "clients": {f"client_{i}": list(range(i, 12, 4)) for i in range(4)}}
            manifest["partition_hash"] = partition_hash(manifest)
            _write(run / "data_manifest.json", manifest)
            save_resolved_config(cfg, run / "resolved_config.prepare-data.yaml")
        elif name in ("train", "resume"):
            finished = suite.completed_rounds(run, cfg, ma=ma)
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            rounds(run, cfg, finished + 1, 10)
        elif name == "evaluate-fused":
            _evaluation(run, task, score=0.6 if ma else 0.55)
            raw = suite._json(run / "evaluation/raw/fused_lm_eval.json")
            raw["configs"][task]["metadata"] = {"pretrained": str(run / "exports/llm_fused")}
            _write(run / "evaluation/raw/fused_lm_eval.json", raw)
    def rounds(run, cfg, start, end):
        ma = cfg.get("ma", {}).get("enabled", False)
        _rounds(run, cfg, ma=ma, start=start, end=end)
        for number in range(start, end + 1):
            path = run / "checkpoints" / f"round_{number:04d}" / "metadata.json"
            meta = suite._json(path)
            meta.update(config_hash=config_digest(cfg), source_model=cfg["model"]["name_or_path"],
                        partition_hash=suite._json(run / "data_manifest.json")["partition_hash"])
            _write(path, meta)
    def command(arguments, **kwargs):
        cfg = suite._yaml(Path(arguments[arguments.index("--config") + 1]))
        run = Path(cfg["run"]["output_dir"])
        calls.append((run, "fuse"))
        assert "fuse" in arguments
        _fusion(run)
    monkeypatch.setattr(suite, "_stage", stage)
    monkeypatch.setattr(suite.subprocess, "run", command)
    monkeypatch.setattr("fedproxy.ma.report.compare", lambda *args: pytest.fail("Results must not be compared"))
    return SimpleNamespace(root=tmp_path / "ratio30", calls=calls, stage=stage, rounds=rounds)


def test_validate_is_read_only_and_requires_no_old_baseline_results(simulation):
    root = simulation.root
    ratio.main(["validate", "--root", str(root)])
    assert not root.exists()
    assert simulation.calls == []


def test_both_methods_share_one_proxy_and_exact_partitions_and_skip_finished_runs(simulation):
    root = simulation.root
    arguments = ["run", "--root", str(root), "--gpu-ids", "2,3"]
    ratio.main(arguments)
    assert len([call for call in simulation.calls if call[1] == "compress"]) == 1
    assert len([call for call in simulation.calls if call[1] == "prepare"]) == 8
    assert len([call for call in simulation.calls if call[1] == "train"]) == 16
    for task in ratio.TASKS:
        base = root / "FedProxy" / suite.run_name(task)
        ma = root / "MA" / suite.run_name(task)
        assert base.joinpath("data_manifest.json").samefile(ma / "data_manifest.json")
        assert (base / "compression").samefile(root / "shared/compression")
        assert (ma / "compression").samefile(root / "shared/compression")
        assert suite._json(base / "result.json")["task_accuracy"] == pytest.approx(0.55)
        assert suite._json(ma / "result.json")["task_accuracy"] == pytest.approx(0.6)
    for folder in ("FedProxy", "MA"):
        assert suite._json(root / folder / "summary.json")["rounds"] == 10
        assert (root / folder / "results.csv").is_file()
        assert not (root / folder / "comparison.csv").exists()
    simulation.calls.clear()
    ratio.main(arguments)
    assert simulation.calls == []


def test_ma_only_prepares_inputs_without_training_fedproxy(simulation):
    root = simulation.root
    ratio.main(["run", "--root", str(root), "--method", "ma", "--task", "arc_easy"])
    trained = [run for run, stage in simulation.calls if stage == "train"]
    assert len(trained) == 1 and trained[0].parent.name == "MA"
    base = root / "FedProxy" / suite.run_name("arc_easy")
    assert (base / "resolved_config.prepare-data.yaml").is_file()
    assert not (base / "checkpoints").exists()


def test_fedproxy_then_ma_can_be_launched_independently(simulation):
    root = simulation.root
    for method in ("fedproxy", "ma"):
        ratio.main(["run", "--root", str(root), "--method", method, "--task", "obqa"])
    assert len([call for call in simulation.calls if call[1] == "compress"]) == 1
    assert len([call for call in simulation.calls if call[1] == "prepare"]) == 1


@pytest.mark.parametrize("method", ["fedproxy", "ma"])
def test_partial_training_resumes_only_unfinished_rounds(simulation, monkeypatch, method):
    root = simulation.root
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
        return simulation.stage(name, frozen, gpu_ids)
    monkeypatch.setattr(suite, "_stage", interrupt)
    args = ["run", "--root", str(root), "--method", method, "--task", "rte"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ratio.main(args)
    run = root / ratio.FOLDERS[method] / suite.run_name("rte")
    first = (run / "checkpoints/round_0001/metadata.json").read_bytes()
    ratio.main(args)
    assert (run / "checkpoints/round_0001/metadata.json").read_bytes() == first
    assert suite._json(run / "result.json")["rounds"] == 10
    assert (run, "resume") in simulation.calls


def test_shared_proxy_or_partition_changes_stop_before_more_training(simulation):
    root = simulation.root
    arguments = ["run", "--root", str(root), "--method", "fedproxy", "--task", "obqa"]
    ratio.main(arguments)
    path = root / "shared/compression/compression_manifest.json"
    value = suite._json(path)
    value["calibration_fingerprint"] = "changed"
    _write(path, value)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="provenance changed"):
        ratio.main(arguments)
    assert simulation.calls == []


def test_wrong_ratio_proxy_and_configuration_changes_are_rejected(simulation):
    root = simulation.root
    arguments = ["run", "--root", str(root), "--method", "ma", "--task", "obqa"]
    ratio.main(arguments)
    path = root / "shared/compression/compression_manifest.json"
    value = suite._json(path)
    value["proxy_layers"] = 16
    _write(path, value)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="23/32-layer proxy"):
        ratio.main(arguments)
    assert simulation.calls == []


def test_recomputed_but_different_partition_cannot_replace_recorded_inputs(simulation):
    root = simulation.root
    arguments = ["run", "--root", str(root), "--method", "fedproxy", "--task", "obqa"]
    ratio.main(arguments)
    path = root / "FedProxy" / suite.run_name("obqa") / "data_manifest.json"
    manifest = suite._json(path)
    manifest["clients"]["client_0"][0] = 100
    manifest["partition_hash"] = partition_hash({key: value for key, value in manifest.items() if key != "partition_hash"})
    _write(path, manifest)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="Checkpoint config/data/model changed|Recorded partition changed"):
        ratio.main(arguments)
    assert simulation.calls == []


def test_changed_frozen_configuration_is_not_overwritten(simulation):
    root = simulation.root
    arguments = ["run", "--root", str(root), "--method", "ma", "--task", "obqa"]
    ratio.main(arguments)
    path = root / "MA" / suite.run_name("obqa") / "execution_config.yaml"
    saved = suite._yaml(path)
    saved["training"]["local_epochs"] = 10
    save_resolved_config(saved, path)
    simulation.calls.clear()
    with pytest.raises(ValueError, match="Existing configuration changed"):
        ratio.main(arguments)
    assert simulation.calls == []
    assert suite._yaml(path)["training"]["local_epochs"] == 10


def test_later_invalid_task_prevents_all_training_and_compression(simulation, monkeypatch):
    real_config = ratio.task_config
    def changed(method, task, root=ratio.ROOT):
        cfg = real_config(method, task, root)
        if task == "mnli":
            cfg["training"]["local_epochs"] = 10
        return cfg
    monkeypatch.setattr(ratio, "task_config", changed)
    with pytest.raises(ValueError, match="training.local_epochs must be 5"):
        ratio.main(["run", "--root", str(simulation.root)])
    assert simulation.calls == []
    assert not (simulation.root / "shared").exists()


@pytest.mark.parametrize("visible", [0, 1])
def test_two_actual_gpus_are_required_before_preparing_inputs(simulation, monkeypatch, visible):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: visible)))
    with pytest.raises(ValueError, match="parallel|visible"):
        ratio.main(["run", "--root", str(simulation.root)])
    assert simulation.calls == []
    assert not (simulation.root / "shared/execution_config.yaml").exists()
