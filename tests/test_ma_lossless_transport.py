import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from fedproxy.federated.state import ParameterSpec, flatten_state
from fedproxy.ma.codec import decode_packet, encode_packet
from fedproxy.ma.downlink import decode_downlink, encode_downlink
from fedproxy.ma.schema import BlockRegistry, ClientPlan
from fedproxy.ma.values import decode_values, encode_values


def _fixture(size=10):
    base = {f"model.layers.{block}.q_proj.lora_A.weight": torch.arange(size).float()
            for block in range(3)}
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(base))
    plan = ClientPlan("a", 0, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    conflict = {name: torch.zeros_like(value) for name, value in base.items()}
    return base, conflict, registry, plan


def _header(wire):
    length = struct.unpack_from("<I", wire)[0]
    return json.loads(wire[4:4 + length]), wire[4 + length:]


def _repack(header, body):
    header["body_sha256"] = hashlib.sha256(body).hexdigest()
    raw = json.dumps(header).encode()
    return struct.pack("<I", len(raw)) + raw + body


@pytest.mark.parametrize("values,codec", [
    ([0.0] * 10, "zero"), ([-0.0] * 10, "constant"), ([0.25] * 10, "constant"),
    ([0.0, -0.0, 1e-40, -3.0], "fp32"), ([], "fp32"),
])
def test_value_elision_preserves_exact_bits(values, codec):
    original = np.asarray(values, dtype="<f4")
    mode, raw = encode_values(original, elide_zero=True, elide_constant=True)
    assert mode == codec
    assert decode_values(raw, len(original), mode).tobytes() == original.tobytes()


def test_bitmap_preserves_observed_zeros_and_saves_dense_support_indices():
    _, _, registry, plan = _fixture(1024)
    names = registry.names_in(plan.upload_blocks)
    records = {names[0]: (torch.arange(1024), torch.zeros(1024), 1.0),
               names[1]: (torch.arange(820), torch.arange(820).float(), 0.75)}
    records[names[1]][1][0] = -0.0
    old = encode_packet("a", 0, registry, plan, records, 3, {}, index_codec="varint")
    new = encode_packet("a", 0, registry, plan, records, 3, {}, index_codec="bitmap",
                        elide_zero=True, elide_constant=True)
    assert len(new) < len(old)
    header, _ = _header(new)
    assert header["version"] == 2
    assert header["records"][0]["value_codec"] == "zero"
    assert header["records"][1]["index_bytes"] == 128
    for wire in (old, new):
        decoded = decode_packet(wire, registry, plan)
        for name, (idx, val, freshness) in records.items():
            actual = decoded.records[name]
            assert torch.equal(idx, actual[0])
            assert val.numpy().tobytes() == actual[1].numpy().tobytes()
            assert actual[2] == freshness


@pytest.mark.parametrize("damage", ["padding", "count", "index_length", "negative_length", "version"])
def test_bitmap_rejects_malformed_support(damage):
    _, _, registry, plan = _fixture()
    names = registry.names_in(plan.upload_blocks)
    records = {names[0]: (torch.arange(10), torch.ones(10), 1.0),
               names[1]: (torch.tensor([0, 2, 9]), torch.zeros(3), 1.0)}
    wire = encode_packet("a", 0, registry, plan, records, 1, {}, index_codec="bitmap")
    header, body = _header(wire)
    if damage == "padding":
        body = bytearray(body)
        body[41] |= 128  # After dense values, bit 15 is outside the 10-coordinate tensor.
    elif damage == "count":
        header["records"][1]["count"] = 4
    elif damage == "index_length":
        header["records"][1]["index_bytes"] = 1
    elif damage == "negative_length":
        header["records"][1]["value_bytes"] = -1
    else:
        header["version"] = 1  # A legacy decoder must never accept bitmap extensions.
    with pytest.raises(ValueError):
        decode_packet(_repack(header, body), registry, plan)


def test_downlink_omits_inactive_conflict_and_restores_zero_and_signed_constants():
    base, conflict, registry, plan = _fixture(1024)
    names = list(base)
    base[names[0]].fill_(-0.0)
    conflict[names[1]].fill_(0.125)
    conflict[names[2]].fill_(float("nan"))  # Inactive G must never be read or sent.
    digest = hashlib.sha256(flatten_state(base, registry.spec).numpy().tobytes()).hexdigest()
    wire = encode_downlink(base, conflict, registry, plan, base_hash=digest, run_hash="run")
    raw = encode_downlink(base, conflict, registry, plan, base_hash=digest, run_hash="run",
                          elide_zero=False, elide_constant=False)
    assert len(wire) < len(raw)
    header, _ = _header(wire)
    assert [r["name"] for r in header["records"] if r["kind"] == "G"] == names[:2]
    received, active = decode_downlink(wire, registry, plan, expected_base_hash=digest,
                                       expected_run_hash="run")
    assert set(received) == set(base)
    assert set(active) == set(names[:2])
    for name in received:
        assert received[name].numpy().tobytes() == base[name].numpy().tobytes()
    for name in active:
        assert active[name].numpy().tobytes() == conflict[name].numpy().tobytes()


@pytest.mark.parametrize("damage", ["client", "round", "run", "base", "duplicate", "missing", "inactive", "corrupt"])
def test_downlink_rejects_wrong_context_or_payload(damage):
    base, conflict, registry, plan = _fixture()
    digest = hashlib.sha256(flatten_state(base, registry.spec).numpy().tobytes()).hexdigest()
    wire = encode_downlink(base, conflict, registry, plan, base_hash=digest, run_hash="run")
    header, body = _header(wire)
    if damage in {"client", "round", "run", "base"}:
        field = {"client": "client_id", "round": "round_id", "run": "run_hash", "base": "base_hash"}[damage]
        header[field] = "wrong"
    elif damage == "duplicate":
        header["records"].append(header["records"][-1])
    elif damage == "missing":
        header["records"].pop()
    elif damage == "inactive":
        header["records"][-1]["name"] = list(base)[-1]
    else:
        body = bytes([body[0] ^ 1]) + body[1:]
        # Recompute body checksum too: the decoded P hash must still detect corruption.
    with pytest.raises(ValueError):
        decode_downlink(_repack(header, body), registry, plan, expected_base_hash=digest,
                        expected_run_hash="run")
