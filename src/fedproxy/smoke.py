from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from fedproxy.compression.bi import score_blocks
from fedproxy.compression.prune import build_proxy, select_layers
from fedproxy.federated.client import train_client
from fedproxy.federated.server import run_federated
from fedproxy.fusion.plug_in import replace_mapped_layers
from fedproxy.models.lora import export_adapter, inject_lora, load_adapter, merge_lora_into_proxy
from fedproxy.utils.reproducibility import seed_everything


class _BatchList(list):
    @property
    def dataset(self):
        return range(sum(batch["input_ids"].shape[0] for batch in self))


def _batch(seed: int, vocab_size: int, batch_size: int = 2, length: int = 10):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab_size, (batch_size, length), generator=generator)
    attention = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, : length // 2] = -100
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def run_smoke(cfg: dict) -> dict:
    from transformers import LlamaConfig, LlamaForCausalLM

    seed_everything(int(cfg["run"]["seed"]))
    tiny = cfg["smoke_model"]
    model_config = LlamaConfig(
        vocab_size=int(tiny["vocab_size"]),
        hidden_size=int(tiny["hidden_size"]),
        intermediate_size=int(tiny["intermediate_size"]),
        num_hidden_layers=int(tiny["num_hidden_layers"]),
        num_attention_heads=int(tiny["num_attention_heads"]),
        num_key_value_heads=int(tiny["num_key_value_heads"]),
        max_position_embeddings=64,
    )
    original = LlamaForCausalLM(model_config)
    full_model_config = copy.deepcopy(original.config)
    original_reference = copy.deepcopy(original.state_dict())
    calibration = [_batch(10, model_config.vocab_size), _batch(11, model_config.vocab_size)]
    scores = score_blocks(original, calibration)
    kept = select_layers(scores, float(cfg["compression"]["remove_ratio"]))
    # Guarantee that the smoke also exercises a non-contiguous mapping when possible.
    if len(scores) == 4 and len(kept) == 2:
        kept = [0, 2]
    proxy, layer_map = build_proxy(original, kept)
    proxy_config = copy.deepcopy(proxy.config)
    proxy_backbone_state = copy.deepcopy(proxy.state_dict())

    def make_model():
        backbone = LlamaForCausalLM(proxy_config)
        backbone.load_state_dict(proxy_backbone_state, strict=True)
        return inject_lora(backbone, cfg["lora"])

    initialized = make_model()
    initial_adapter = export_adapter(initialized)
    client_loaders = {
        "client_0": _BatchList([_batch(100, model_config.vocab_size)]),
        "client_1": _BatchList([_batch(200, model_config.vocab_size)]),
    }

    def train_one(client_id, base, conflict, round_id):
        return train_client(
            client_id,
            make_model,
            base,
            conflict,
            client_loaders[client_id],
            cfg,
            round_id=round_id,
            device_label="cpu",
        )

    run_dir = Path(cfg["run"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    final_adapter, conflict, history = run_federated(
        initial_adapter,
        list(client_loaders),
        train_one,
        cfg,
        checkpoint_root=run_dir / "checkpoints",
    )
    tuned_proxy = make_model()
    load_adapter(tuned_proxy, final_adapter)
    merged_proxy = merge_lora_into_proxy(tuned_proxy)
    # build_proxy mutates the proxy model's config.num_hidden_layers. Rebuild the
    # full model from the independent pre-pruning config, not the proxy config.
    fresh_original = LlamaForCausalLM(full_model_config)
    fresh_original.load_state_dict(original_reference, strict=True)
    fused = replace_mapped_layers(fresh_original, merged_proxy, layer_map)
    fused.eval()
    probe = _batch(999, model_config.vocab_size, batch_size=1)
    with torch.no_grad():
        logits = fused(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("Smoke produced non-finite logits")
    export = run_dir / "exports" / "llm_fused"
    fused.save_pretrained(export, safe_serialization=True)
    reloaded = LlamaForCausalLM.from_pretrained(export, local_files_only=True)
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]).logits
    torch.testing.assert_close(logits, reloaded_logits, atol=1e-6, rtol=1e-5)
    result = {
        "synthetic": True,
        "scores": scores,
        "kept_indices": kept,
        "layer_map": layer_map,
        "rounds": len(history),
        "finite_logits": True,
        "conflict_mean": sum(float(value.mean()) for value in conflict.values()) / len(conflict),
    }
    (run_dir / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
