"""A serialized full adapter plus only the scheduled PCR conflict tensors."""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import torch

from fedproxy.federated.state import TensorState

from .schema import BlockRegistry, ClientPlan
from .values import decode_values, encode_values


def _plan_header(plan: ClientPlan) -> dict:
    return {"train": sorted(plan.train_blocks), "upload": sorted(plan.upload_blocks),
            "anchor": sorted(plan.anchor_blocks)}


def encode_downlink(base: TensorState, conflict: TensorState, registry: BlockRegistry,
                    plan: ClientPlan, *, base_hash: str, run_hash: str,
                    elide_zero: bool = True, elide_constant: bool = True) -> bytes:
    plan.validate(registry)
    active = set(registry.names_in(plan.train_blocks))
    expected = {entry.name for entry in registry.spec.entries}
    if set(base) != expected or not active <= set(conflict):
        raise ValueError("Downlink states do not match schema")
    body = bytearray()
    records = []
    base_digest = hashlib.sha256()
    for kind, state in (("P", base), ("G", conflict)):
        for entry in registry.spec.entries:
            if kind == "G" and entry.name not in active:
                continue
            tensor = state[entry.name]
            if tuple(tensor.shape) != entry.shape or tensor.dtype != torch.float32:
                raise ValueError("Downlink requires schema-shaped FP32 tensors")
            values = np.asarray(tensor.detach().cpu().numpy(), dtype="<f4").reshape(-1)
            if kind == "P":
                base_digest.update(values.tobytes())
            codec, raw = encode_values(values, elide_zero=elide_zero, elide_constant=elide_constant)
            records.append({"kind": kind, "name": entry.name, "codec": codec, "bytes": len(raw)})
            body.extend(raw)
    if base_digest.hexdigest() != base_hash:
        raise ValueError("Downlink global base mismatch")
    header = {
        "version": 1, "schema_hash": registry.spec.schema_hash,
        "client_id": plan.client_id, "round_id": plan.round_id, "plan": _plan_header(plan),
        "base_hash": base_hash, "run_hash": run_hash, "records": records,
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("<I", len(raw_header)) + raw_header + body


def decode_downlink(wire: bytes, registry: BlockRegistry, plan: ClientPlan, *,
                    expected_base_hash: str, expected_run_hash: str) -> tuple[TensorState, TensorState]:
    plan.validate(registry)
    if len(wire) < 4:
        raise ValueError("Truncated downlink")
    length = struct.unpack_from("<I", wire)[0]
    if length > 10_000_000 or len(wire) < 4 + length:
        raise ValueError("Invalid downlink header length")
    header = json.loads(wire[4:4 + length])
    body = wire[4 + length:]
    if header.get("version") != 1 or header.get("schema_hash") != registry.spec.schema_hash:
        raise ValueError("Downlink schema mismatch")
    if (header.get("client_id") != plan.client_id or header.get("round_id") != plan.round_id
            or header.get("plan") != _plan_header(plan)):
        raise ValueError("Downlink client/round/plan mismatch")
    if header.get("base_hash") != expected_base_hash or header.get("run_hash") != expected_run_hash:
        raise ValueError("Downlink base/run mismatch")
    if hashlib.sha256(body).hexdigest() != header.get("body_sha256"):
        raise ValueError("Downlink checksum mismatch")
    entries = {entry.name: entry for entry in registry.spec.entries}
    active = set(registry.names_in(plan.train_blocks))
    states: dict[str, TensorState] = {"P": {}, "G": {}}
    offset = 0
    for record in header["records"]:
        kind, name, length = record["kind"], record["name"], int(record["bytes"])
        if (kind not in states or name not in entries or name in states[kind]
                or (kind == "G" and name not in active)):
            raise ValueError("Unexpected or duplicate downlink tensor")
        if length < 0 or offset + length > len(body):
            raise ValueError("Invalid downlink record length")
        entry = entries[name]
        values = decode_values(body[offset:offset + length], entry.numel, record["codec"])
        states[kind][name] = torch.from_numpy(values).reshape(entry.shape)
        offset += length
    if offset != len(body) or set(states["P"]) != set(entries) or set(states["G"]) != active:
        raise ValueError("Downlink has missing tensors or trailing bytes")
    digest = hashlib.sha256()
    for entry in registry.spec.entries:
        digest.update(states["P"][entry.name].numpy().tobytes())
    if digest.hexdigest() != expected_base_hash:
        raise ValueError("Decoded downlink base hash mismatch")
    return states["P"], states["G"]
