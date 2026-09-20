from __future__ import annotations

import json
from pathlib import Path

import torch

from fedproxy.compression.prune import validate_layer_map
from fedproxy.models.backbone import get_decoder_layers, reset_runtime_layer_indices


def replace_mapped_layers(original_model, merged_proxy, layer_map: dict):
    original_layers = get_decoder_layers(original_model)
    proxy_layers = get_decoder_layers(merged_proxy)
    mapping = validate_layer_map(layer_map, len(proxy_layers), len(original_layers))
    with torch.no_grad():
        for proxy_index, original_index in mapping.items():
            source = proxy_layers[proxy_index].state_dict()
            target = original_layers[original_index]
            incompatible = target.load_state_dict(source, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"Layer state mismatch at proxy layer {proxy_index}")
    reset_runtime_layer_indices(original_model)
    return original_model


def fuse_and_save(original_model, merged_proxy, layer_map: dict, output_dir: str | Path, manifest: dict, tokenizer=None, overwrite=False):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty fusion output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fused = replace_mapped_layers(original_model, merged_proxy, layer_map)
    fused.save_pretrained(output, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)
    with (output / "fusion_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest | {"layer_map": {str(k): v for k, v in layer_map.items()}}, handle, indent=2)
    return fused

