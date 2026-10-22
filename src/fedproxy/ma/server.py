from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from fedproxy.federated.state import ParameterSpec, add_states, flatten_state, zeros_like_state
from fedproxy.models.lora import export_adapter
from fedproxy.utils.checkpoint import load_checkpoint, save_checkpoint
from fedproxy.utils.reproducibility import seed_everything

from .aggregation import aggregate
from .client import run_client_job
from .codec import decode_packet
from .scheduler import audit_residual_delivery, plan_round
from .schema import BlockRegistry


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _round_config(cfg: dict, previous: dict) -> dict:
    output = copy.deepcopy(cfg)
    cc = output["ma"]["compression"]
    if cc["retention_mode"] == "fixed":
        return output
    adaptive = cc["adaptive"]
    initial = previous.get("initial_loss")
    last = previous.get("loss_ema")
    progress = max(0.0, initial - last) if initial is not None and last is not None else 0.0
    for factor in ("a", "b"):
        low = float(adaptive[f"keep_min_{factor}"])
        high = float(adaptive[f"keep_max_{factor}"])
        rate = low + (high - low) * math.exp(-float(adaptive[f"gamma_{factor}"]) * progress)
        cc[f"keep_{factor}"] = rate
    return output


def train_ma(cfg: dict, resume: str | None = None, dry_run: bool = False) -> dict:
    from fedproxy.workflows import _load_train_model, audit_training_prompts
    from fedproxy.config import load_config

    run_dir = Path(cfg["run"]["output_dir"])
    proxy_path = run_dir / "compression" / "proxy"
    manifest_path = run_dir / "data_manifest.json"
    if not proxy_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run prepare-data and compress with this MA configuration before train")
    source = cfg["run"].get("reuse_artifacts_from")
    if source and (run_dir / "compression").is_symlink():
        source_config = Path(source) / "resolved_config.train.yaml"
        if not source_config.exists():
            source_config = Path(source) / "resolved_config.compress.yaml"
        if not source_config.exists():
            raise FileNotFoundError("Reused proxy source lacks a resolved configuration")
        original = load_config(source_config)
        for section in ("model", "compression", "data"):
            if cfg[section] != original[section]:
                raise ValueError(f"Reused proxy/partition has different {section} settings")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cfg["data"].get("truncation_audit", True):
        audit_training_prompts(cfg, manifest, tokenizer_path=proxy_path if resume else None)
    seed_everything(int(cfg["run"]["seed"]))
    execution = cfg["federated"]["client_execution"]
    visible = torch.cuda.device_count()
    if execution == "parallel":
        requested = int(cfg["federated"]["max_parallel_clients"])
        parallelism = min(len(manifest["clients"]), visible if requested == 0 else requested)
        if visible < 1 or parallelism < 1 or parallelism > visible:
            raise ValueError("MA parallel execution requires enough visible GPUs")
    else:
        parallelism = 1
    initial_model = _load_train_model(proxy_path, cfg, torch.device("cpu"))
    initial = export_adapter(initial_model)
    del initial_model
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(initial))
    client_ids = list(manifest["clients"])
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    invariants = {
        "config_hash": config_hash, "partition_hash": manifest["partition_hash"],
        "source_model": cfg["model"]["name_or_path"], "ma_protocol_version": 1,
        "compression_manifest_hash": hashlib.sha256(
            (run_dir / "compression" / "compression_manifest.json").read_bytes()
        ).hexdigest(),
    }
    rounds = int(cfg["federated"]["rounds"])
    schedule_audit = audit_residual_delivery(registry, client_ids, rounds, cfg)
    if dry_run:
        preview = plan_round(registry, client_ids, 0, cfg)
        return {
            "dry_run": True, "schema_hash": registry.spec.schema_hash,
            "adapter_parameters": registry.spec.total_numel,
            "proxy_blocks": len(registry.blocks), "parallel_clients": parallelism,
            "residual_schedule_audit": schedule_audit,
            "round_1_schedule": {
                key: {"train": sorted(item.train_blocks), "upload": sorted(item.upload_blocks),
                      "anchor": sorted(item.anchor_blocks)} for key, item in preview.items()
            },
        }
    _write_json(run_dir / "metrics" / "schedule_audit.json", schedule_audit)
    checkpoint_root = run_dir / "checkpoints"
    if resume:
        loaded = load_checkpoint(resume, expected_state=initial, expected_metadata=invariants)
        base = loaded["adapter"]
        conflict = loaded["conflict"]
        previous_meta = loaded["metadata"]
        start = int(previous_meta["completed_round"])
    else:
        if checkpoint_root.exists() and any(checkpoint_root.glob("round_*")):
            raise FileExistsError("Existing checkpoints require --resume")
        base = initial
        conflict = zeros_like_state(base)
        previous_meta = {}
        start = 0
    if start > rounds:
        raise ValueError("Resume checkpoint exceeds configured rounds")
    history = []
    for round_id in range(start, rounds):
        started = time.perf_counter()
        round_cfg = _round_config(cfg, previous_meta)
        plans = plan_round(registry, client_ids, round_id, round_cfg)
        if round_cfg["ma"]["aggregation"]["q0"] == "auto_anchor_numel":
            anchor = plans[client_ids[0]].anchor_blocks
            round_cfg["ma"]["aggregation"]["q0"] = sum(
                entry.numel for entry in registry.spec.entries
                if registry.tensor_block[entry.name] in anchor
            )
        _write_json(run_dir / "schedules" / f"round_{round_id + 1:04d}.json", {
            client: {"train": sorted(plan.train_blocks), "upload": sorted(plan.upload_blocks),
                     "anchor": sorted(plan.anchor_blocks)} for client, plan in plans.items()
        })
        base_vector = flatten_state(base, registry.spec).contiguous()
        conflict_vector = flatten_state(conflict, registry.spec).contiguous()
        base_hash = hashlib.sha256(base_vector.numpy().tobytes()).hexdigest()
        if execution == "parallel":
            base_vector.share_memory_()
            conflict_vector.share_memory_()
        jobs = []
        for client in client_ids:
            prior = previous_meta.get("client_states", {}).get(client)
            if round_id and not prior:
                raise ValueError(f"Missing committed residual state for {client}")
            client_cfg = copy.deepcopy(round_cfg)
            if client_cfg["ma"]["compression"]["retention_mode"] == "adaptive":
                adaptive = client_cfg["ma"]["compression"]["adaptive"]
                correction = float(adaptive.get("zeta", 0.0)) * float(
                    previous_meta.get("client_heterogeneity", {}).get(client, 0.0)
                )
                for factor in ("a", "b"):
                    comp = client_cfg["ma"]["compression"]
                    comp[f"keep_{factor}"] = max(
                        float(adaptive[f"keep_min_{factor}"]),
                        min(float(adaptive[f"keep_max_{factor}"]), comp[f"keep_{factor}"] - correction),
                    )
            jobs.append({
                "cfg": client_cfg, "client_id": client, "round_id": round_id,
                "registry": registry, "plan": plans[client], "proxy_path": str(proxy_path),
                "manifest": manifest, "base_vector": base_vector, "conflict_vector": conflict_vector,
                "base_hash": base_hash, "run_hash": config_hash,
                "prior_state": prior,
                "stage_dir": str(run_dir / "clients" / client / f"round_{round_id + 1:04d}"),
            })
        packets = []
        for start_index in range(0, len(jobs), parallelism):
            wave = jobs[start_index:start_index + parallelism]
            if execution == "parallel":
                for device_index, job in enumerate(wave):
                    job["device_index"] = device_index
                with ProcessPoolExecutor(
                    max_workers=len(wave), mp_context=multiprocessing.get_context("spawn")
                ) as executor:
                    packets.extend(executor.map(run_client_job, wave))
            else:
                packets.extend(run_client_job(job) for job in wave)
        if [packet["client_id"] for packet in packets] != client_ids:
            raise ValueError("Client packet order/identity differs from scheduled clients")
        observations = []
        packet_ids = {}
        for packet in packets:
            client = packet["client_id"]
            wire = packet["wire"]
            packet_ids[client] = hashlib.sha256(wire).hexdigest()
            observations.append(decode_packet(
                wire, registry, plans[client], expected_base_hash=base_hash,
                expected_run_hash=config_hash,
            ))
        merged, next_conflict, heterogeneity, analysis_metrics = aggregate(
            observations, registry, conflict, round_cfg
        )
        server_lr = float(cfg["ma"]["aggregation"]["server_lr"])
        next_adapter = add_states(base, {name: value * server_lr for name, value in merged.items()})
        round_loss = sum(float(obs.metrics["task_loss"]) for obs in observations) / len(observations)
        initial_loss = previous_meta.get("initial_loss", round_loss)
        prior_ema = previous_meta.get("loss_ema", round_loss)
        smoothing = float(cfg["ma"]["compression"]["adaptive"]["loss_ema_decay"])
        loss_ema = smoothing * prior_ema + (1 - smoothing) * round_loss
        up_bytes = sum(obs.wire_bytes for obs in observations)
        # The current transport sends a complete FP32 P and G to every worker.
        down_bytes = len(observations) * 4 * registry.spec.total_numel * 2
        metadata = {
            **invariants, "completed_round": round_id + 1,
            "client_states": {
                client: str(run_dir / "clients" / client / f"round_{round_id + 1:04d}" / "residual.pt")
                for client in client_ids
            },
            "packet_ids": packet_ids, "client_heterogeneity": heterogeneity,
            "client_residual_state_bytes": {
                item["client_id"]: item["residual_state_bytes"] for item in packets
            },
            "client_metrics": {obs.client_id: obs.metrics for obs in observations},
            "initial_loss": initial_loss, "loss_ema": loss_ema,
            "retention_rates": {factor: round_cfg["ma"]["compression"][f"keep_{factor}"] for factor in ("a", "b")},
            "analysis": analysis_metrics,
            "communication_bytes": {"uplink": up_bytes, "downlink_fp32_state_estimate": down_bytes},
            "round_precommit_seconds": time.perf_counter() - started,
        }
        path = checkpoint_root / f"round_{round_id + 1:04d}"
        save_checkpoint(path, next_adapter, next_conflict, metadata)
        _write_json(run_dir / "metrics" / f"round_{round_id + 1:04d}.json", {
            "completed_round": round_id + 1,
            "round_wall_seconds": time.perf_counter() - started,
        })
        history.append(metadata)
        previous_meta = metadata
        base, conflict = next_adapter, next_conflict
        print(f"[ma] round={round_id + 1}/{rounds} checkpoint={path} upload_bytes={up_bytes}", flush=True)
    all_metadata = []
    for number in range(1, rounds + 1):
        path = checkpoint_root / f"round_{number:04d}" / "metadata.json"
        if path.exists():
            all_metadata.append(json.loads(path.read_text(encoding="utf-8")))
    _write_json(run_dir / "metrics" / "aggregation.json", all_metadata)
    return {
        "completed_rounds": rounds, "new_rounds": len(history),
        "last_checkpoint": str(checkpoint_root / f"round_{rounds:04d}"),
    }
