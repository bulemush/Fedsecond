from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .schema import BlockRegistry, ClientPlan
from .values import decode_values, encode_values


@dataclass
class Observation:
    client_id: str
    num_examples: int
    records: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    metrics: dict
    wire_bytes: int


def _varint(value: int) -> bytes:
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _read_varints(raw: bytes, count: int) -> np.ndarray:
    result = []
    offset = 0
    for _ in range(count):
        shift = value = 0
        while True:
            if offset >= len(raw) or shift > 35:
                raise ValueError("Truncated or oversized varint")
            byte = raw[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
        result.append(value)
    if offset != len(raw):
        raise ValueError("Trailing varint bytes")
    return np.asarray(result, dtype=np.int64)


def _rice_encode(gaps: list[int], m: int = 4) -> tuple[bytes, int]:
    """Golomb coding with power-of-two divisor (Rice form)."""
    output = bytearray()
    current = used = bits = 0
    def push(bit: int) -> None:
        nonlocal current, used, bits
        current = (current << 1) | bit
        used += 1
        bits += 1
        if used == 8:
            output.append(current)
            current = used = 0
    width = m.bit_length() - 1
    for gap in gaps:
        if gap < 1:
            raise ValueError("Golomb gaps must be positive")
        q, remainder = divmod(gap, m)
        for _ in range(q):
            push(1)
        push(0)
        for shift in range(width - 1, -1, -1):
            push((remainder >> shift) & 1)
    if used:
        output.append(current << (8 - used))
    return bytes(output), bits


def _rice_decode(raw: bytes, count: int, bit_length: int, m: int = 4) -> np.ndarray:
    if bit_length < 0 or bit_length > len(raw) * 8 or len(raw) != (bit_length + 7) // 8:
        raise ValueError("Invalid Golomb bit length")
    offset = 0
    width = m.bit_length() - 1
    def read() -> int:
        nonlocal offset
        if offset >= bit_length:
            raise ValueError("Truncated Golomb code")
        result = (raw[offset // 8] >> (7 - offset % 8)) & 1
        offset += 1
        return result
    gaps = []
    for _ in range(count):
        quotient = 0
        while read():
            quotient += 1
            if quotient > 2_147_483_647:
                raise ValueError("Oversized Golomb quotient")
        remainder = 0
        for _ in range(width):
            remainder = (remainder << 1) | read()
        gap = quotient * m + remainder
        if gap < 1:
            raise ValueError("Invalid zero Golomb gap")
        gaps.append(gap)
    if offset != bit_length or (bit_length % 8 and raw[-1] & ((1 << (8 - bit_length % 8)) - 1)):
        raise ValueError("Non-canonical Golomb padding or trailing data")
    return np.asarray(gaps, dtype=np.int64)


def encode_packet(
    client_id: str,
    round_id: int,
    registry: BlockRegistry,
    plan: ClientPlan,
    records: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    num_examples: int,
    metrics: dict,
    *,
    index_codec: str = "varint",
    base_hash: str = "",
    run_hash: str = "",
    elide_zero: bool = False,
    elide_constant: bool = False,
) -> bytes:
    if index_codec not in {"int32", "varint", "golomb", "bitmap"}:
        raise ValueError("Unsupported index codec")
    version = 2 if index_codec == "bitmap" or elide_zero or elide_constant else 1
    if client_id != plan.client_id or round_id != plan.round_id:
        raise ValueError("Packet does not match its round plan")
    body = bytearray()
    description = []
    entries = {entry.name: entry for entry in registry.spec.entries}
    for name in sorted(records):
        if name not in entries or registry.tensor_block[name] not in plan.upload_blocks:
            raise ValueError(f"Unscheduled tensor: {name}")
        indices, values, freshness = records[name]
        idx = np.asarray(indices.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
        val = np.asarray(values.detach().cpu().numpy(), dtype="<f4").reshape(-1)
        if len(idx) != len(val) or not np.isfinite(val).all() or not 0 <= freshness <= 1:
            raise ValueError("Invalid sparse tensor values")
        if len(idx) and (idx[0] < 0 or idx[-1] >= entries[name].numel or np.any(idx[1:] <= idx[:-1])):
            raise ValueError("Sparse indices must be unique, sorted and in bounds")
        dense = len(idx) == entries[name].numel and np.array_equal(idx, np.arange(len(idx)))
        if registry.tensor_block[name] in plan.anchor_blocks and not dense:
            raise ValueError("Anchor tensors must be densely observed")
        if dense:
            encoded_idx = b""
            codec = "dense"
            bit_length = 0
        elif index_codec == "int32":
            encoded_idx = idx.astype("<i4").tobytes()
            codec = "int32"
            bit_length = 8 * len(encoded_idx)
        elif index_codec == "bitmap":
            mask = np.zeros(entries[name].numel, dtype=np.uint8)
            mask[idx] = 1
            encoded_idx = np.packbits(mask, bitorder="little").tobytes()
            codec = "bitmap"
            bit_length = entries[name].numel
        elif index_codec == "golomb":
            gaps = np.diff(np.r_[-1, idx]).tolist()
            encoded_idx, bit_length = _rice_encode([int(gap) for gap in gaps])
            codec = "golomb"
        else:
            gaps = np.diff(np.r_[-1, idx]).tolist()
            encoded_idx = b"".join(_varint(int(gap)) for gap in gaps)
            codec = "varint"
            bit_length = 8 * len(encoded_idx)
            if len(encoded_idx) >= entries[name].numel * 4 and dense:
                raise AssertionError("Dense selection failed")
        value_codec, encoded_val = encode_values(val, elide_zero=elide_zero, elide_constant=elide_constant)
        description.append({
            "name": name, "count": len(idx), "codec": codec, "index_bytes": len(encoded_idx),
            "value_bytes": len(encoded_val), "freshness": float(freshness),
            "bit_length": bit_length,
        })
        if version == 2:
            description[-1]["value_codec"] = value_codec
        body.extend(encoded_idx)
        body.extend(encoded_val)
    header = {
        "version": version, "schema_hash": registry.spec.schema_hash, "client_id": client_id,
        "round_id": round_id, "num_examples": int(num_examples), "metrics": metrics,
        "base_hash": base_hash, "run_hash": run_hash,
        "records": description, "body_sha256": hashlib.sha256(body).hexdigest(),
        "plan": {"train": sorted(plan.train_blocks), "upload": sorted(plan.upload_blocks),
                 "anchor": sorted(plan.anchor_blocks)},
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("<I", len(encoded_header)) + encoded_header + body


def decode_packet(
    wire: bytes, registry: BlockRegistry, plan: ClientPlan,
    *, expected_base_hash: str | None = None, expected_run_hash: str | None = None,
) -> Observation:
    if len(wire) < 4:
        raise ValueError("Truncated update packet")
    header_size = struct.unpack_from("<I", wire)[0]
    if header_size > 10_000_000 or len(wire) < 4 + header_size:
        raise ValueError("Invalid packet header length")
    header = json.loads(wire[4:4 + header_size])
    body = wire[4 + header_size:]
    version = header.get("version")
    if version not in {1, 2} or header.get("schema_hash") != registry.spec.schema_hash:
        raise ValueError("Packet schema mismatch")
    if header.get("client_id") != plan.client_id or header.get("round_id") != plan.round_id:
        raise ValueError("Packet round/client mismatch")
    if expected_base_hash is not None and header.get("base_hash") != expected_base_hash:
        raise ValueError("Packet global base mismatch")
    if expected_run_hash is not None and header.get("run_hash") != expected_run_hash:
        raise ValueError("Packet run mismatch")
    if header.get("plan") != {"train": sorted(plan.train_blocks), "upload": sorted(plan.upload_blocks),
                               "anchor": sorted(plan.anchor_blocks)}:
        raise ValueError("Packet training plan mismatch")
    if hashlib.sha256(body).hexdigest() != header.get("body_sha256"):
        raise ValueError("Packet checksum mismatch")
    entries = {entry.name: entry for entry in registry.spec.entries}
    offset = 0
    records = {}
    for item in header["records"]:
        name = item["name"]
        if name not in entries or name in records or registry.tensor_block[name] not in plan.upload_blocks:
            raise ValueError("Unknown, duplicate or unscheduled tensor")
        count = int(item["count"])
        index_bytes = int(item["index_bytes"])
        value_bytes = int(item["value_bytes"])
        if count < 0 or count > entries[name].numel or index_bytes < 0 or value_bytes < 0:
            raise ValueError("Invalid tensor record length")
        end = offset + index_bytes + value_bytes
        if end > len(body):
            raise ValueError("Truncated tensor record")
        raw_idx = body[offset:offset + index_bytes]
        raw_val = body[offset + index_bytes:end]
        offset = end
        codec = item["codec"]
        if codec == "dense" and index_bytes == 0 and count == entries[name].numel:
            idx = np.arange(count, dtype=np.int64)
        elif codec == "int32" and index_bytes == 4 * count:
            idx = np.frombuffer(raw_idx, dtype="<i4").astype(np.int64)
        elif codec == "bitmap" and version == 2:
            size = entries[name].numel
            if index_bytes != (size + 7) // 8 or int(item["bit_length"]) != size:
                raise ValueError("Invalid bitmap length")
            mask = np.unpackbits(np.frombuffer(raw_idx, dtype=np.uint8), bitorder="little")
            if np.any(mask[size:]):
                raise ValueError("Non-canonical bitmap padding")
            idx = np.flatnonzero(mask[:size]).astype(np.int64)
            if len(idx) != count:
                raise ValueError("Bitmap count mismatch")
        elif codec == "varint":
            idx = np.cumsum(_read_varints(raw_idx, count)) - 1
        elif codec == "golomb":
            idx = np.cumsum(_rice_decode(raw_idx, count, int(item["bit_length"]))) - 1
        else:
            raise ValueError("Invalid index encoding")
        if len(idx) and (idx[0] < 0 or idx[-1] >= entries[name].numel or np.any(idx[1:] <= idx[:-1])):
            raise ValueError("Invalid decoded indices")
        if registry.tensor_block[name] in plan.anchor_blocks and codec != "dense":
            raise ValueError("Anchor tensor missing dense observations")
        val = decode_values(raw_val, count, item.get("value_codec", "fp32") if version == 2 else "fp32")
        freshness = float(item["freshness"])
        if not 0 <= freshness <= 1:
            raise ValueError("Invalid freshness")
        records[name] = (torch.from_numpy(idx.copy()), torch.from_numpy(val), freshness)
    if offset != len(body) or int(header["num_examples"]) < 1:
        raise ValueError("Trailing bytes or invalid sample count")
    if set(records) != set(registry.names_in(plan.upload_blocks)):
        raise ValueError("Packet omits a scheduled upload tensor")
    return Observation(plan.client_id, int(header["num_examples"]), records, header["metrics"], len(wire))
