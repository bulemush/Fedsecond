"""Offline end-to-end MA path using the same PEFT training helper as real runs."""

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from fedproxy.config import load_config
from fedproxy.federated.client import train_client
from fedproxy.federated.state import ParameterSpec, add_states, zeros_like_state
from fedproxy.ma.aggregation import aggregate
from fedproxy.ma.client import build_update
from fedproxy.ma.codec import decode_packet, encode_packet
from fedproxy.ma.scheduler import plan_round
from fedproxy.ma.schema import BlockRegistry
from fedproxy.models.lora import export_adapter, inject_lora, load_adapter, merge_lora_into_proxy
from fedproxy.smoke import _BatchList, _batch


def test_tiny_ma_trains_partial_lora_transports_aggregates_and_merges():
    torch.manual_seed(42)
    cfg = load_config("configs/smoke.yaml")
    cfg["ma"] = {
        "schedule": {"anchor_blocks": 1, "train_blocks": 2, "upload_blocks": 2,
                     "q_block": 0, "coverage_window": 1, "seed": 42},
        "pcr": {"mode": "masked_sum"},
        "compression": {"keep_a": 0.5, "keep_b": 0.5, "probe_fraction_of_k": 0.0},
        "residual": {"enabled": True, "max_age": 4, "decay_mu": 0.1},
        "aggregation": {"beta": 1.0, "rho": 1.1, "q_min": 1, "q0": 1.0,
                        "nu": 1.0, "n_ref": 2.0, "xi": 0.9, "epsilon_update": 1e-12},
    }
    model_config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    backbone = LlamaForCausalLM(model_config)
    backbone_state = copy.deepcopy(backbone.state_dict())

    def make_model():
        instance = LlamaForCausalLM(model_config)
        instance.load_state_dict(backbone_state)
        return inject_lora(instance, cfg["lora"])

    global_adapter = export_adapter(make_model())
    registry = BlockRegistry.from_spec(ParameterSpec.from_state(global_adapter))
    assert len(registry.blocks) == 4
    conflict = zeros_like_state(global_adapter)
    client_ids = ["client_0", "client_1"]
    residuals = {client: {} for client in client_ids}
    for round_id in range(2):
        plans = plan_round(registry, client_ids, round_id, cfg)
        observations = []
        for i, client_id in enumerate(client_ids):
            plan = plans[client_id]
            loader = _BatchList([_batch(100 + i, 64)])
            result = train_client(
                client_id, make_model, global_adapter, conflict, loader, cfg,
                round_id=round_id, active_names=set(registry.names_in(plan.train_blocks)),
            )
            for name in set(global_adapter) - set(registry.names_in(plan.train_blocks)):
                torch.testing.assert_close(result.adapter_state[name], global_adapter[name])
            records, residuals[client_id], _ = build_update(
                result.adapter_state, global_adapter, residuals[client_id], registry, plan, cfg
            )
            packet = encode_packet(client_id, round_id, registry, plan, records, result.num_examples,
                                   result.metrics, index_codec="varint")
            observations.append(decode_packet(packet, registry, plan))
        merged, conflict, _, _ = aggregate(observations, registry, conflict, cfg)
        global_adapter = add_states(global_adapter, merged)
    tuned = make_model()
    load_adapter(tuned, global_adapter)
    merged_model = merge_lora_into_proxy(tuned)
    with torch.no_grad():
        logits = merged_model(input_ids=_batch(999, 64, batch_size=1)["input_ids"]).logits
    assert torch.isfinite(logits).all()
