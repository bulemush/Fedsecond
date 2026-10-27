from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from fedproxy.federated.client import train_client
from fedproxy.federated.state import TensorState, unflatten_state
from fedproxy.models.backbone import load_tokenizer
from fedproxy.utils.reproducibility import derived_seed, seed_everything

from .codec import encode_packet
from .downlink import decode_downlink
from .schema import BlockRegistry, ClientPlan


def _selected_indices(values: torch.Tensor, k: int, probes: int, seed: int) -> torch.Tensor:
    size = values.numel()
    if k >= size:
        return torch.arange(size, dtype=torch.int64)
    rng = np.random.default_rng(seed)
    probe = rng.choice(size, min(probes, k), replace=False).astype(np.int64)
    available = np.ones(size, dtype=np.bool_)
    available[probe] = False
    candidates = np.nonzero(available)[0]
    magnitudes = values.abs().detach().cpu().numpy()
    order = np.lexsort((candidates, -magnitudes[candidates]))
    selected = np.sort(np.r_[probe, candidates[order[:k - len(probe)]]])
    return torch.from_numpy(selected.copy()).long()


def build_update(
    local: TensorState, base: TensorState, previous: dict[int, dict[str, torch.Tensor]],
    registry: BlockRegistry, plan: ClientPlan, cfg: dict,
) -> tuple[dict, dict[int, dict[str, torch.Tensor]], dict]:
    cc = cfg["ma"]["compression"]
    rc = cfg["ma"]["residual"]
    residual_enabled = bool(rc["enabled"])
    max_age = int(rc["max_age"])
    mu = float(rc["decay_mu"])
    expired = {
        int(source): values for source, values in previous.items()
        if residual_enabled and plan.round_id - int(source) > max_age
    }
    expired_l1 = sum(float(value.abs().sum()) for tensors in expired.values() for value in tensors.values())
    retained = {
        int(source): {name: tensor.detach().cpu().float().clone() for name, tensor in values.items()}
        for source, values in previous.items() if residual_enabled and plan.round_id - int(source) <= max_age
    }
    delta = {}
    for name in registry.names_in(plan.train_blocks):
        value = local[name].detach().cpu().float() - base[name].detach().cpu().float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite client update: {name}")
        delta[name] = value
    for name in set(local) - set(delta):
        if not torch.equal(local[name].detach().cpu().float(), base[name].detach().cpu().float()):
            raise ValueError(f"Inactive LoRA parameter changed: {name}")
    retained[plan.round_id] = delta
    records = {}
    selected_values = 0
    for entry in registry.spec.entries:
        name = entry.name
        if registry.tensor_block[name] not in plan.upload_blocks:
            continue
        candidate = torch.zeros(entry.numel, dtype=torch.float32)
        raw_mass = torch.zeros(entry.numel, dtype=torch.float32)
        decayed_mass = torch.zeros(entry.numel, dtype=torch.float32)
        for source, tensors in retained.items():
            if name not in tensors:
                continue
            raw = tensors[name].reshape(-1)
            scale = math.exp(-mu * (plan.round_id - source))
            candidate.add_(raw, alpha=scale)
            raw_mass.add_(raw.abs())
            decayed_mass.add_(raw.abs(), alpha=scale)
        if not torch.isfinite(candidate).all():
            raise ValueError(f"Non-finite residual candidate: {name}")
        anchor = registry.tensor_block[name] in plan.anchor_blocks
        if anchor:
            indices = torch.arange(entry.numel, dtype=torch.int64)
        else:
            factor = registry.tensor_factor[name].lower()
            keep = float(cc[f"keep_{factor}"])
            k = min(entry.numel, max(1, math.ceil(keep * entry.numel)))
            probes = math.floor(k * float(cc.get("probe_fraction_of_k", 0.0)))
            indices = _selected_indices(
                candidate, k, probes, derived_seed(int(cfg["run"]["seed"]), plan.round_id, name)
            )
        mass = float(raw_mass[indices].sum())
        freshness = float(decayed_mass[indices].sum()) / mass if mass > 0 else 1.0
        records[name] = (indices, candidate[indices].contiguous(), min(1.0, max(0.0, freshness)))
        selected_values += indices.numel()
        for source in retained:
            if name in retained[source]:
                retained[source][name].reshape(-1)[indices] = 0
    if not residual_enabled:
        retained = {}
    else:
        retained = {
            source: {name: value for name, value in tensors.items() if torch.count_nonzero(value).item()}
            for source, tensors in retained.items()
        }
        retained = {source: tensors for source, tensors in retained.items() if tensors}
    retained_l1 = sum(float(value.abs().sum()) for tensors in retained.values() for value in tensors.values())
    return records, retained, {
        "transmitted_values": selected_values, "residual_sources": len(retained),
        "residual_expired_sources": len(expired), "residual_expired_l1": expired_l1,
        "residual_retained_l1": retained_l1,
    }


def run_client_job(payload: dict) -> dict:
    """One isolated logical client job. A completed packet is replayed verbatim."""
    from fedproxy.workflows import _load_client_examples, _load_train_model, _make_client_loader

    cfg = payload["cfg"]
    client_id = payload["client_id"]
    round_id = int(payload["round_id"])
    registry: BlockRegistry = payload["registry"]
    plan: ClientPlan = payload["plan"]
    stage = Path(payload["stage_dir"])
    packet_path = stage / "update.bin"
    state_path = stage / "residual.pt"
    if packet_path.exists():
        if not state_path.exists():
            raise ValueError("Staged packet has no matching residual state")
        wire = packet_path.read_bytes()
        return {"client_id": client_id, "wire": wire, "replayed": True,
                "residual_state_bytes": state_path.stat().st_size}
    seed_everything(derived_seed(int(cfg["run"]["seed"]), round_id, client_id))
    device_index = payload.get("device_index")
    if device_index is not None:
        torch.cuda.set_device(int(device_index))
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "downlink_wire" in payload:
        base, conflict = decode_downlink(
            payload["downlink_wire"], registry, plan,
            expected_base_hash=payload["base_hash"], expected_run_hash=payload["run_hash"],
        )
    else:  # Legacy configurations/checkpoints retain the original transport.
        base = unflatten_state(payload["base_vector"], registry.spec)
        conflict = unflatten_state(payload["conflict_vector"], registry.spec)
    prior_path = payload.get("prior_state")
    previous = torch.load(prior_path, map_location="cpu", weights_only=True) if prior_path else {}
    proxy_path = payload["proxy_path"]
    tokenizer = load_tokenizer({**cfg["model"], "name_or_path": proxy_path, "revision": None})
    examples = _load_client_examples(cfg, payload["manifest"], client_id)
    loader = _make_client_loader(cfg, examples, tokenizer, seed=derived_seed(int(cfg["run"]["seed"]), round_id, client_id))
    def model_factory():
        return _load_train_model(proxy_path, cfg, device)
    active = set(registry.names_in(plan.train_blocks))
    multiplier = (
        registry.spec.total_numel / sum(base[name].numel() for name in active)
        if cfg["ma"]["pcr"]["mode"] == "active_mean_scaled" else 1.0
    )
    started = time.perf_counter()
    result = train_client(
        client_id, model_factory, base, conflict, loader, cfg,
        round_id=round_id, device_label=str(device), active_names=active, pcr_multiplier=multiplier,
    )
    records, next_state, compression = build_update(
        result.adapter_state, base, previous, registry, plan, cfg
    )
    result.metrics.update(compression)
    result.metrics["client_through_sparsification_seconds"] = time.perf_counter() - started
    wire = encode_packet(
        client_id, round_id, registry, plan, records, result.num_examples, result.metrics,
        index_codec=cfg["ma"]["compression"]["index_codec"],
        base_hash=payload["base_hash"], run_hash=payload["run_hash"],
        elide_zero=cfg["ma"].get("communication", {}).get("elide_zero_states", False),
        elide_constant=cfg["ma"].get("communication", {}).get("elide_constant_states", False),
    )
    stage.mkdir(parents=True, exist_ok=True)
    temporary_state = state_path.with_suffix(".pt.tmp")
    torch.save(next_state, temporary_state)
    os.replace(temporary_state, state_path)
    temporary_packet = packet_path.with_suffix(".bin.tmp")
    temporary_packet.write_bytes(wire)
    os.replace(temporary_packet, packet_path)
    del result, base, conflict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"client_id": client_id, "wire": wire, "replayed": False,
            "residual_state_bytes": state_path.stat().st_size}
