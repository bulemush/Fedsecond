"""Exercise the real server, worker, packets and checkpoints with a cheap trainer."""
import json

import pytest
import torch

from fedproxy.config import load_config
from fedproxy.federated.state import ClientResult
from fedproxy.ma import client, server
from fedproxy.utils.checkpoint import load_checkpoint


@pytest.mark.parametrize("optimized", [False, True])
def test_transport_accounting_and_partial_round_resume(tmp_path, monkeypatch, optimized):
    from fedproxy import workflows

    cfg = load_config("configs/ma/base.yaml")
    cfg["run"].update(output_dir=str(tmp_path), reuse_artifacts_from=None)
    cfg["data"]["truncation_audit"] = False
    cfg["federated"].update(rounds=2, client_execution="sequential", max_parallel_clients=1)
    cfg["ma"]["schedule"].update(train_blocks=2, upload_blocks=2, q_block=0)
    if not optimized:
        cfg["ma"].pop("communication")
        cfg["ma"]["compression"]["index_codec"] = "varint"
    proxy = tmp_path / "compression/proxy"
    proxy.mkdir(parents=True)
    (proxy.parent / "compression_manifest.json").write_text("{}")
    (tmp_path / "data_manifest.json").write_text(json.dumps({
        "clients": ["a", "b"], "partition_hash": "test",
    }))
    initial = {f"model.layers.{block}.q_proj.lora_A.weight": torch.arange(32).float()
               for block in range(3)}
    monkeypatch.setattr(workflows, "_load_train_model", lambda *args: object())
    monkeypatch.setattr(server, "export_adapter", lambda _: {k: v.clone() for k, v in initial.items()})
    monkeypatch.setattr(client, "load_tokenizer", lambda _: object())
    monkeypatch.setattr(workflows, "_load_client_examples", lambda *args: [])
    monkeypatch.setattr(workflows, "_make_client_loader", lambda *args, **kwargs: object())
    calls = []

    def fake_train(client_id, factory, base, conflict, loader, config, **kwargs):
        active = kwargs["active_names"]
        assert set(conflict) == (active if optimized else set(base))
        assert set(base) == set(initial)
        calls.append((kwargs["round_id"], client_id))
        local = {name: value.clone() for name, value in base.items()}
        for name in active:
            local[name][::2] += 0.125
        return ClientResult(client_id, local, 2, {"task_loss": 0.5, "elapsed_seconds": 0.0})

    monkeypatch.setattr(client, "train_client", fake_train)
    real_job = client.run_client_job
    down_sizes = {}

    def interrupted_job(payload):
        if optimized:
            assert "base_vector" not in payload and "conflict_vector" not in payload
            down_sizes[payload["round_id"], payload["client_id"]] = len(payload["downlink_wire"])
        if payload["round_id"] == 1 and payload["client_id"] == "b":
            raise RuntimeError("simulated disconnect")
        return real_job(payload)

    monkeypatch.setattr(server, "run_client_job", interrupted_job)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        server.train_ma(cfg)
    staged = tmp_path / "clients/a/round_0002/update.bin"
    saved = staged.read_bytes()

    def resumed_job(payload):
        if optimized:
            down_sizes[payload["round_id"], payload["client_id"]] = len(payload["downlink_wire"])
        return real_job(payload)

    monkeypatch.setattr(server, "run_client_job", resumed_job)
    result = server.train_ma(cfg, resume=str(tmp_path / "checkpoints/round_0001"))
    assert result["completed_rounds"] == 2
    assert calls == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]
    assert staged.read_bytes() == saved
    history = json.loads((tmp_path / "metrics/aggregation.json").read_text(encoding="utf-8"))
    for round_id, metadata in enumerate(history):
        comm = metadata["communication_bytes"]
        assert comm["uplink"] == sum(
            (tmp_path / f"clients/{name}/round_{round_id + 1:04d}/update.bin").stat().st_size
            for name in ("a", "b")
        )
        assert comm["downlink_fp32_state_estimate"] == 2 * 3 * 32 * 8
        if optimized:
            assert comm["downlink"] == sum(down_sizes[round_id, name] for name in ("a", "b"))
            assert comm["downlink_active_fp32_state_estimate"] == 2 * (3 + 2) * 32 * 4
        else:
            assert "downlink" not in comm
    checkpoint = load_checkpoint(result["last_checkpoint"])
    assert set(checkpoint["adapter"]) == set(initial)
