import pytest
import torch

from fedproxy.federated.state import ParameterSpec
from fedproxy.ma.aggregation import aggregate
from fedproxy.ma.client import build_update, run_client_job
from fedproxy.ma.codec import decode_packet, encode_packet
from fedproxy.ma.scheduler import audit_residual_delivery, plan_round
from fedproxy.ma.schema import BlockRegistry, ClientPlan


def _registry():
    return BlockRegistry.from_spec(ParameterSpec.from_state({
        "model.layers.0.q_proj.lora_A.weight": torch.zeros(4),
        "model.layers.1.q_proj.lora_B.weight": torch.zeros(4),
    }))


def _config():
    return {
        "run": {"seed": 42},
        "ma": {
            "schedule": {"anchor_blocks": 1, "train_blocks": 2, "upload_blocks": 2,
                         "q_block": 0, "coverage_window": 1, "seed": 42},
            "compression": {"keep_a": 0.5, "keep_b": 0.5, "probe_fraction_of_k": 0.0},
            "residual": {"enabled": True, "max_age": 4, "decay_mu": 0.1},
            "aggregation": {"beta": 1.0, "rho": 1.1, "q_min": 1, "q0": 1.0,
                            "nu": 1.0, "n_ref": 2.0, "xi": 0.9,
                            "epsilon_update": 1e-12},
        },
    }


def test_scheduler_keeps_anchor_inside_upload_inside_train():
    registry = _registry()
    plans = plan_round(registry, ["a", "b"], 0, _config())
    assert plans["a"].anchor_blocks == plans["b"].anchor_blocks
    for plan in plans.values():
        assert plan.anchor_blocks <= plan.upload_blocks <= plan.train_blocks


@pytest.mark.parametrize("train_budget,upload_budget", [(12, 10), (8, 5)])
def test_scheduler_revisits_each_clients_trained_layers_before_residual_ttl(
    train_budget, upload_budget,
):
    # These are the Conservative and Balanced budgets on a 16-block proxy.
    state = {f"model.layers.{block}.q_proj.lora_A.weight": torch.zeros(1)
             for block in range(16)}
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(state))
    cfg = _config()
    cfg["ma"]["schedule"].update({"train_blocks": train_budget,
                                   "upload_blocks": upload_budget, "q_block": 2})
    clients = [f"client_{index}" for index in range(8)]
    rounds = [plan_round(registry, clients, round_id, cfg) for round_id in range(10)]
    audit = audit_residual_delivery(registry, clients, 10, cfg)
    assert audit["max_observed_wait_rounds"] <= 4
    for plans in rounds:
        non_anchor = set(registry.blocks) - set(next(iter(plans.values())).anchor_blocks)
        for block in non_anchor:
            assert sum(block in plan.upload_blocks for plan in plans.values()) >= 2
    for client in clients:
        for round_id in range(6):
            pending = rounds[round_id][client].train_blocks - rounds[round_id][client].upload_blocks
            delivered = set().union(*(rounds[later][client].upload_blocks
                                      for later in range(round_id + 1, round_id + 5)))
            assert pending <= delivered


def test_codec_round_trip_preserves_observed_zero_and_rejects_corruption():
    registry = _registry()
    plan = ClientPlan("a", 0, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    records = {}
    for name in registry.tensor_block:
        if registry.tensor_block[name] == 0:
            records[name] = (torch.arange(4), torch.tensor([0.0, 0.3, 0.0, -0.2]), 1.0)
        else:
            records[name] = (torch.tensor([0, 2]), torch.tensor([0.0, 0.4]), 0.8)
    for codec in ("int32", "varint", "golomb", "bitmap"):
        wire = encode_packet("a", 0, registry, plan, records, 3, {"task_loss": 1.0}, index_codec=codec)
        decoded = decode_packet(wire, registry, plan)
        assert decoded.wire_bytes == len(wire)
        for name, (idx, values, freshness) in records.items():
            actual = decoded.records[name]
            torch.testing.assert_close(actual[0], idx)
            torch.testing.assert_close(actual[1], values)
            assert actual[2] == pytest.approx(freshness)
        with pytest.raises(ValueError, match="checksum"):
            decode_packet(wire[:-1] + bytes([wire[-1] ^ 1]), registry, plan)


def test_masked_aggregation_uses_only_actual_contributors():
    registry = _registry()
    plans = [ClientPlan(str(i), 0, frozenset({0, 1}), frozenset({0, 1}), frozenset({0})) for i in range(3)]
    name0 = next(name for name, block in registry.tensor_block.items() if block == 0)
    name1 = next(name for name, block in registry.tensor_block.items() if block == 1)
    observations = []
    for i, sign in enumerate((0.3, 0.2, -0.1)):
        records = {name0: (torch.arange(4), torch.tensor([sign, 0.0, 0.0, 0.0]), 1.0)}
        if i == 0:
            records[name1] = (torch.tensor([1]), torch.tensor([0.6]), 1.0)
        else:
            records[name1] = (torch.tensor([0]), torch.tensor([0.0]), 1.0)
        wire = encode_packet(str(i), 0, registry, plans[i], records, 1, {"task_loss": 1.0})
        observations.append(decode_packet(wire, registry, plans[i]))
    previous = {name: torch.zeros(4) for name in registry.tensor_block}
    update, conflict, _, metrics = aggregate(observations, registry, previous, _config())
    # The observed zero shared by clients 1 and 2 changes their overlap
    # reliability, so clients 0 and 1 need not receive equal global weights.
    weights = metrics["client_weights"]
    positive_only = (0.3 * weights["0"] + 0.2 * weights["1"]) / (
        weights["0"] + weights["1"]
    )
    assert weights["0"] > weights["1"]
    assert update[name0][0].item() == pytest.approx(positive_only)
    assert conflict[name0][0].item() > 0
    assert update[name1][1].item() == pytest.approx(0.6)
    assert conflict[name1][1].item() == 0
    assert metrics["valid_similarity_pairs"] == 3


def test_residual_clears_sent_source_coordinates_only():
    registry = _registry()
    plan = ClientPlan("a", 1, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    base = {name: torch.zeros(4) for name in registry.tensor_block}
    local = {name: torch.zeros(4) for name in registry.tensor_block}
    name1 = next(name for name, block in registry.tensor_block.items() if block == 1)
    local[name1][0] = 0.5
    old = {0: {name1: torch.tensor([0.0, 0.0, 0.3, 0.1])}}
    records, remaining, _ = build_update(local, base, old, registry, plan, _config())
    sent = set(records[name1][0].tolist())
    assert 0 in sent and 2 in sent
    for source in remaining:
        assert torch.count_nonzero(remaining[source][name1][list(sent)]) == 0
    assert remaining[0][name1][3].item() == pytest.approx(0.1)


def test_staged_packet_is_replayed_without_retraining(tmp_path):
    registry = _registry()
    plan = ClientPlan("a", 0, frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    records = {
        name: (torch.arange(4), torch.ones(4), 1.0) for name in registry.tensor_block
    }
    wire = encode_packet("a", 0, registry, plan, records, 1, {"task_loss": 1.0})
    stage = tmp_path / "client_round"
    stage.mkdir()
    torch.save({}, stage / "residual.pt")
    (stage / "update.bin").write_bytes(wire)
    replayed = run_client_job({"cfg": {}, "client_id": "a", "round_id": 0,
                               "registry": registry, "plan": plan, "stage_dir": str(stage)})
    assert replayed["replayed"]
    assert replayed["wire"] == wire
