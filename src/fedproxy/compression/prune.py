from __future__ import annotations

import math

import torch

from fedproxy.models.backbone import get_decoder_layers, reset_runtime_layer_indices, set_decoder_layers


def select_layers(scores: list[float], remove_ratio: float) -> list[int]:
    if not scores:
        raise ValueError("No block scores")
    if not 0 <= remove_ratio < 1:
        raise ValueError("remove_ratio must be in [0, 1)")
    keep_count = max(1, len(scores) - math.floor(len(scores) * remove_ratio))
    ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))[:keep_count]
    return sorted(ranked)


def build_proxy(model, kept_indices: list[int]):
    layers = get_decoder_layers(model)
    if kept_indices != sorted(set(kept_indices)):
        raise ValueError("kept_indices must be unique and strictly increasing")
    if not kept_indices or kept_indices[-1] >= len(layers):
        raise ValueError("kept_indices outside decoder range")
    selected = torch.nn.ModuleList([layers[index] for index in kept_indices])
    set_decoder_layers(model, selected)
    model.config.num_hidden_layers = len(selected)
    reset_runtime_layer_indices(model)
    return model, {proxy_index: original_index for proxy_index, original_index in enumerate(kept_indices)}


def validate_layer_map(layer_map: dict, proxy_layers: int, original_layers: int) -> dict[int, int]:
    normalized = {int(key): int(value) for key, value in layer_map.items()}
    if sorted(normalized) != list(range(proxy_layers)):
        raise ValueError("Proxy indices in layer_map must be contiguous from zero")
    originals = [normalized[i] for i in range(proxy_layers)]
    if originals != sorted(set(originals)) or any(i < 0 or i >= original_layers for i in originals):
        raise ValueError("Original indices must be unique, increasing, and in range")
    return normalized

