from __future__ import annotations

from typing import Any


def get_decoder_layers(model: Any):
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("layers",),
    )
    for path in candidates:
        value = model
        try:
            for part in path:
                value = getattr(value, part)
            return value
        except AttributeError:
            continue
    raise TypeError(f"Unsupported decoder model type: {type(model).__name__}")


def set_decoder_layers(model: Any, layers) -> None:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        model.model.decoder.layers = layers
    elif hasattr(model, "layers"):
        model.layers = layers
    else:
        raise TypeError(f"Unsupported decoder model type: {type(model).__name__}")


def reset_runtime_layer_indices(model: Any) -> None:
    for index, layer in enumerate(get_decoder_layers(model)):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index


def load_causal_lm(cfg: dict, *, device_map=None):
    from transformers import AutoModelForCausalLM

    dtype_name = cfg.get("dtype", "auto")
    kwargs = {
        "revision": cfg.get("revision"),
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        "device_map": device_map,
        "local_files_only": bool(cfg.get("local_files_only", False)),
    }
    import torch

    kwargs["torch_dtype"] = resolve_dtype(dtype_name) if dtype_name == "auto" else getattr(torch, dtype_name)
    return AutoModelForCausalLM.from_pretrained(cfg["name_or_path"], **kwargs)


def resolve_dtype(dtype_name: str):
    import torch

    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    name = cfg.get("tokenizer_name_or_path") or cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=cfg.get("revision"),
        use_fast=True,
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
