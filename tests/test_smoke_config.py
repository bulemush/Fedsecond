import copy

from transformers import LlamaConfig, LlamaForCausalLM

from fedproxy.compression.prune import build_proxy


def test_full_config_snapshot_survives_proxy_pruning():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    original = LlamaForCausalLM(config)
    full_config = copy.deepcopy(original.config)
    reference = copy.deepcopy(original.state_dict())
    proxy, _ = build_proxy(original, [0, 2])
    assert proxy.config.num_hidden_layers == 2
    assert full_config.num_hidden_layers == 4
    rebuilt = LlamaForCausalLM(full_config)
    rebuilt.load_state_dict(reference, strict=True)
