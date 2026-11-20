import copy
import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from fedproxy.config import ConfigError, load_config, save_resolved_config, validate_config
from fedproxy.data.partition import heterogeneous_partition, partition_hash
from fedproxy.evaluation.protocol import protocol_fingerprint, task_overrides
from fedproxy.evaluation.report import summarize
from fedproxy.federated.state import ParameterSpec, flatten_state
from fedproxy.ma import ablations as experiment
from fedproxy.ma.aggregation import aggregate
from fedproxy.ma.client import build_update
from fedproxy.ma.codec import Observation
from fedproxy.ma.schema import BlockRegistry, ClientPlan
from fedproxy.utils.checkpoint import save_checkpoint


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def small_config():
    cfg = load_config("configs/ma/heterogeneous_ratio30.yaml")
    cfg["ma"]["aggregation"].update(q_min=1, q0=1.0, beta=0.0)
    cfg["ma"]["compression"].update(keep_a=0.5, keep_b=0.5)
    return cfg


def tensor_state():
    return {f"model.layers.{layer}.q_proj.lora_A.weight": torch.zeros(64) for layer in (0, 1)}


def test_a1_weighted_mean_preserves_conflict_and_ignores_missing_and_observed_zero():
    cfg = small_config()
    state = tensor_state()
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(state))
    name = next(iter(state))
    observations = [
        Observation(str(i), 1, {name: (torch.tensor([0, 1]), torch.tensor([x, y]), 1.0)}, {}, 0)
        for i, (x, y) in enumerate(((0.3, 0.6), (0.2, 0.0), (-0.1, 0.0)))
    ]
    reference, conflict, _, _ = aggregate(observations, registry, state, cfg)
    cfg["ma"]["aggregation"].update(merge_rule="observed_mean", similarity_scope="union")
    merged, actual_conflict, _, _ = aggregate(observations, registry, state, cfg)
    assert reference[name][0] == pytest.approx(0.25)
    assert merged[name][0] == pytest.approx((0.3 + 0.2 - 0.1) / 3)
    assert merged[name][1] == pytest.approx(0.6)
    assert merged[name][2] == 0
    for key in conflict:
        torch.testing.assert_close(conflict[key], actual_conflict[key], rtol=0, atol=0)


def test_a1_union_norm_keeps_intersection_validity_and_reduces_consensus():
    cfg = small_config()
    cfg["ma"]["aggregation"]["beta"] = 1.0
    state = tensor_state()
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(state))
    name = next(iter(state))
    obs = [
        Observation("a", 1, {name: (torch.tensor([0, 1]), torch.tensor([1.0, 10.0]), 1.0)}, {}, 0),
        Observation("b", 1, {name: (torch.tensor([0, 2]), torch.tensor([1.0, 10.0]), 1.0)}, {}, 0),
        Observation("c", 1, {name: (torch.tensor([3]), torch.tensor([1.0]), 1.0)}, {}, 0),
    ]
    _, _, _, original = aggregate(obs, registry, state, cfg)
    cfg["ma"]["aggregation"]["similarity_scope"] = "union"
    _, _, _, actual = aggregate(obs, registry, state, cfg)
    assert original["valid_similarity_pairs"] == actual["valid_similarity_pairs"] == 1
    assert actual["client_weights"]["a"] < original["client_weights"]["a"]
    # Dense observations make the two supports identical.
    dense = [Observation(x.client_id, 1, {name: (torch.arange(64), torch.ones(64), 1.0)}, {}, 0)
             for x in obs]
    union = aggregate(dense, registry, state, cfg)
    cfg["ma"]["aggregation"]["similarity_scope"] = "intersection"
    intersection = aggregate(dense, registry, state, cfg)
    assert union[3] == intersection[3]


def test_a2_first_round_matches_but_second_round_drops_prior_residual():
    cfg = small_config()
    base = tensor_state()
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(base))
    plan = ClientPlan("a", 0, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    local = {name: torch.arange(64).float() / 100 for name in base}
    first, residual, _ = build_update(local, base, {}, registry, plan, cfg)
    off = copy.deepcopy(cfg)
    off["ma"]["residual"]["enabled"] = False
    without, empty, _ = build_update(local, base, {}, registry, plan, off)
    assert residual and not empty
    for name in first:
        assert first[name][2] == without[name][2] == 1.0
        torch.testing.assert_close(first[name][1], without[name][1], rtol=0, atol=0)
    plan = replace(plan, round_id=1)
    later, _, _ = build_update(base, base, residual, registry, plan, cfg)
    dropped, _, _ = build_update(base, base, {}, registry, plan, off)
    assert sum(float(v[1].abs().sum()) for v in later.values()) > 0
    assert all(torch.count_nonzero(v[1]) == 0 for v in dropped.values())


def test_a4_random_is_client_specific_keeps_k_anchor_and_global_rng():
    cfg = small_config()
    cfg["ma"]["compression"]["selection"] = "random"
    base = tensor_state()
    local = {name: torch.arange(64).float() for name in base}
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(base))
    def select(client):
        plan = ClientPlan(client, 2, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
        return build_update(local, base, {}, registry, plan, cfg)[0]
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    a, repeated, b = select("a"), select("a"), select("b")
    assert torch.equal(before, torch.get_rng_state())
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    for name in base:
        assert torch.equal(a[name][0], repeated[name][0])
        assert len(a[name][0]) == (64 if registry.tensor_block[name] == 0 else 32)
        assert len(a[name][0].unique()) == len(a[name][0])
    nonanchor = next(name for name in base if registry.tensor_block[name] == 1)
    assert not torch.equal(a[nonanchor][0], b[nonanchor][0])
    assert not torch.equal(a[nonanchor][0], torch.arange(32, 64))


def test_a3_zero_lambda_matches_zero_conflict_and_changes_nonzero_conflict():
    from transformers import LlamaConfig, LlamaForCausalLM
    from fedproxy.federated.client import train_client
    from fedproxy.models.lora import export_adapter, inject_lora
    from fedproxy.smoke import _BatchList, _batch

    cfg = load_config("configs/smoke.yaml")
    cfg["training"].update(local_epochs=3, learning_rate=0.03)
    cfg["pcr"].update(enabled=True, lambda_reg=10.0)
    torch.manual_seed(42)
    model_cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    backbone = copy.deepcopy(LlamaForCausalLM(model_cfg).state_dict())
    def factory():
        model = LlamaForCausalLM(model_cfg)
        model.load_state_dict(backbone)
        return inject_lora(model, cfg["lora"])
    initial = export_adapter(factory())
    def run(lambda_reg, conflict_value):
        selected = copy.deepcopy(cfg)
        selected["pcr"]["lambda_reg"] = lambda_reg
        torch.manual_seed(55)
        return train_client("a", factory, initial,
                            {key: torch.full_like(value, conflict_value) for key, value in initial.items()},
                            _BatchList([_batch(3, 64)]), selected)
    zero = run(0.0, 0.0)
    first = run(10.0, 0.0)
    disabled = run(0.0, 1.0)
    enabled = run(10.0, 1.0)
    for key in initial:
        assert torch.equal(zero.adapter_state[key], first.adapter_state[key])
        assert torch.equal(zero.adapter_state[key], disabled.adapter_state[key])
    assert disabled.metrics["weighted_pcr_loss"] == 0.0
    assert enabled.metrics["weighted_pcr_loss"] > 0.0
    assert any(not torch.equal(enabled.adapter_state[key], disabled.adapter_state[key]) for key in initial)


@pytest.mark.parametrize("path,value", [
    ("ma.compression.selection", "bad"), ("ma.aggregation.merge_rule", "bad"),
    ("ma.aggregation.similarity_scope", "bad"),
])
def test_new_modes_reject_typos(path, value):
    cfg = small_config()
    experiment._set(cfg, path, value)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def create_rounds(run, cfg, schedules, first=1, last=10):
    state = tensor_state()
    for number in range(first, last + 1):
        clients = experiment.suite._json(run / "data_manifest.json")["clients"]
        metrics = {client: {"elapsed_seconds": 1.0, "optimizer_steps": 5,
                            "effective_target_tokens": 32, "transmitted_values": 96}
                   for client in clients}
        states = {}
        for client in clients:
            path = run / "clients" / client / f"round_{number:04d}/residual.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({}, path)
            states[client] = str(path)
        meta = {"config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "completed_round": number, "ma_protocol_version": 1,
                "client_metrics": metrics, "client_states": states,
                "partition_hash": experiment.suite._json(run / "data_manifest.json")["partition_hash"],
                "compression_manifest_hash": experiment._sha(run / "compression/compression_manifest.json"),
                "analysis": {"observed_coordinate_count": 100, "multi_contributor_coordinate_count": 50},
                "communication_bytes": {"uplink": 100, "downlink": 200,
                                         "downlink_fp32_state_estimate": 300}}
        save_checkpoint(run / "checkpoints" / f"round_{number:04d}", state, state, meta)
        write(run / "schedules" / f"round_{number:04d}.json", schedules[number - 1])


def create_evaluation(run, score=0.6):
    fusion = run / "exports/llm_fused/fusion_manifest.json"
    write(fusion, {"checkpoint": str(run / "checkpoints/round_0010")})
    write(fusion.parent / "config.json", {})
    (fusion.parent / "model.safetensors").write_bytes(b"fixture")
    tasks = ["openbookqa" if x == "obqa" else x for x in experiment.TASKS]
    protocol = {"tasks": tasks, "num_fewshot": 0, "batch_size": 1,
                "primary_metric": "acc", "task_overrides": task_overrides(tasks)}
    write(run / "evaluation/fused_summary.json", {
        **summarize(dict.fromkeys(experiment.TASKS, score)), "model_kind": "fused",
        "protocol": protocol, "protocol_fingerprint": protocol_fingerprint(protocol),
        "fusion_manifest_sha256": experiment._sha(fusion),
    })
    write(run / "evaluation/raw/fused_lm_eval.json", {"versions": dict.fromkeys(tasks, 1)})


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    run = tmp_path / "MA"
    cfg = load_config("configs/ma/heterogeneous_ratio30.yaml")
    cfg["run"]["output_dir"] = str(run)
    save_resolved_config(cfg, run / "resolved_config.train.yaml")
    clients = heterogeneous_partition(list(experiment.TASKS), {x: [0, 1] for x in experiment.TASKS})
    manifest = {"scenario": "heterogeneous", "seed": 42, "clients": clients}
    manifest["partition_hash"] = partition_hash(manifest)
    write(run / "data_manifest.json", manifest)
    write(run / "compression/compression_manifest.json", {"proxy_layers": 23})
    (run / "compression/proxy").mkdir()
    registry = BlockRegistry(None, tuple(range(23)), {}, {})
    schedules = experiment.schedule_records(registry, list(clients), cfg)
    create_rounds(run, cfg, schedules)
    create_evaluation(run)
    return run


def test_preflight_is_read_only_and_uses_only_four_declared_interventions(reference, tmp_path):
    root = tmp_path / "ablations"
    experiment.main(["validate", "--reference", str(reference), "--root", str(root)])
    assert not root.exists()
    context = experiment.reference_context(reference)
    for group in experiment.GROUPS:
        cfg = experiment.variant_config(context, group, root / group)
        experiment.verify_variant(cfg, context)
        cfg["training"]["local_epochs"] = 10
        with pytest.raises(ValueError, match="Unexpected changes"):
            experiment.verify_variant(cfg, context)


def test_changed_reference_schedule_is_rejected_before_training(reference):
    path = reference / "schedules/round_0002.json"
    value = experiment.suite._json(path)
    value["client_0"]["upload"] = []
    write(path, value)
    with pytest.raises(ValueError, match="schedule differs"):
        experiment.reference_context(reference)


def test_initial_adapter_hash_is_checked_against_retained_packet(reference):
    context = experiment.reference_context(reference)
    cfg = experiment.variant_config(context, "A1", reference.parent / "A1")
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(tensor_state()))
    digest = hashlib.sha256(flatten_state(tensor_state(), registry.spec).numpy().tobytes()).hexdigest()
    header = json.dumps({"schema_hash": registry.spec.schema_hash, "round_id": 0, "base_hash": digest}).encode()
    path = reference / "clients/client_0/round_0001/update.bin"
    path.write_bytes(struct.pack("<I", len(header)) + header)
    assert experiment.verify_initialization(cfg, registry, digest)["reference_packet_headers_checked"] == 1
    with pytest.raises(ValueError, match="Initial adapter differs"):
        experiment.verify_initialization(cfg, registry, "different")


def test_train_entry_checks_reference_schedules_and_initialization_before_dry_run(
    reference, tmp_path, monkeypatch,
):
    import shutil
    from fedproxy import workflows
    from fedproxy.ma import server

    context = experiment.reference_context(reference)
    cfg = experiment.variant_config(context, "A4", tmp_path / "A4")
    run = Path(cfg["run"]["output_dir"])
    run.mkdir()
    shutil.copytree(reference / "compression", run / "compression")
    shutil.copyfile(reference / "data_manifest.json", run / "data_manifest.json")
    initial = {f"model.layers.{i}.q_proj.lora_A.weight": torch.zeros(64) for i in range(23)}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(workflows, "_load_train_model", lambda *args: object())
    monkeypatch.setattr(workflows, "audit_training_prompts", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "export_adapter", lambda model: initial)
    result = server.train_ma(cfg, dry_run=True)
    assert result["proxy_blocks"] == 23
    assert result["round_1_schedule"] == context["schedules"][0]
    assert not (run / "metrics").exists()
    cfg["ma"]["schedule"]["seed"] = 11
    with pytest.raises(ValueError, match="Unexpected changes"):
        server.train_ma(cfg, dry_run=True)


def test_suite_runs_four_variants_resumes_and_summarizes_a3(reference, tmp_path, monkeypatch):
    root = tmp_path / "ablations"
    original_bytes = (reference / "resolved_config.train.yaml").read_bytes()
    calls = []
    interrupted = False

    def stage(name, frozen, gpu_ids):
        nonlocal interrupted
        cfg = experiment.suite._yaml(frozen)
        run = Path(cfg["run"]["output_dir"])
        calls.append((run.name, name))
        assert run != reference
        if name in ("train", "resume"):
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            context = experiment.reference_context(reference)
            first = experiment.suite.completed_rounds(run, cfg, ma=True) + 1
            if run.name == "A2" and not interrupted:
                interrupted = True
                create_rounds(run, cfg, context["schedules"], first, 3)
                raise RuntimeError("interrupt at A2")
            create_rounds(run, cfg, context["schedules"], first)
            write(run / "metrics/ablation_initialization.json", {"evidence": "fixture"})
        if name == "evaluate-fused":
            create_evaluation(run, score=0.58)

    def fuse(arguments, **kwargs):
        cfg = experiment.suite._yaml(Path(arguments[arguments.index("--config") + 1]))
        run = Path(cfg["run"]["output_dir"])
        assert run != reference and "fuse" in arguments
        create_evaluation(run, score=0.58)

    # Exercise orchestration without 7B training or requiring symlink privileges on Windows.
    import subprocess
    import fedproxy.ratio30
    real_subprocess_run = subprocess.run
    def link(path, target):
        if path.exists():
            return
        if target.is_dir():
            if os.name == "nt":
                result = real_subprocess_run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(path), str(target)],
                    capture_output=True, timeout=10,
                )
                assert result.returncode == 0, result.stderr
            else:
                path.symlink_to(target, target_is_directory=True)
        else:
            path.symlink_to(target) if os.name != "nt" else os.link(target, path)
    monkeypatch.setattr(fedproxy.ratio30, "_link", link)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(experiment.suite, "_stage", stage)
    monkeypatch.setattr(experiment.suite.subprocess, "run", fuse)
    args = ["run", "--reference", str(reference), "--root", str(root)]
    with pytest.raises(RuntimeError, match="interrupt at A2"):
        experiment.main(args)
    first = (root / "A2/checkpoints/round_0001/metadata.json").read_bytes()
    experiment.main(args)
    assert (root / "A2/checkpoints/round_0001/metadata.json").read_bytes() == first
    assert ("A2", "resume") in calls
    assert sum(name == "train" for _, name in calls) == 4
    report = experiment.suite._json(root / "summary.json")
    assert len(report["rows"]) == 5 and report["pending"] == []
    assert report["rows"][3]["group"] == "A3"
    assert report["rows"][3]["contribution_pp"] == pytest.approx(2.0)
    assert (reference / "resolved_config.train.yaml").read_bytes() == original_bytes
    calls.clear()
    experiment.main(args)
    assert calls == []
    stale_path = root / "A4/evaluation/fused_summary.json"
    stale = experiment.suite._json(stale_path)
    stale["fusion_manifest_sha256"] = "stale"
    write(stale_path, stale)
    partial = experiment.write_report(experiment.reference_context(reference), root)
    assert partial["pending"] == ["A4"] and "A4" in partial["invalid"]
    experiment.main(args)
    assert calls == [("A4", "audit"), ("A4", "evaluate-fused")]
    cfg = experiment.suite._yaml(root / "A4/execution_config.yaml")
    cfg["training"]["local_epochs"] = 10
    save_resolved_config(cfg, root / "A4/execution_config.yaml")
    with pytest.raises(ValueError, match="configuration differs"):
        experiment.main(args)
