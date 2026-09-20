from __future__ import annotations

import re

import torch

from fedproxy.federated.state import TensorState, canonical_state


_ADAPTER_SEGMENT = re.compile(r"\.default(?=\.|$)")


def normalize_adapter_name(name: str) -> str:
    return _ADAPTER_SEGMENT.sub("", name)


def inject_lora(model, cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        bias=cfg.get("bias", "none"),
        target_modules=list(cfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_cfg)


def export_adapter(model) -> TensorState:
    from peft import get_peft_model_state_dict

    raw = get_peft_model_state_dict(model)
    output: TensorState = {}
    for name, tensor in raw.items():
        canonical = normalize_adapter_name(name)
        if canonical in output:
            raise ValueError(f"Adapter name normalization collision: {canonical}")
        if "lora_A" in canonical or "lora_B" in canonical:
            output[canonical] = tensor.detach().cpu().float().clone()
    if not output:
        raise ValueError("No LoRA A/B parameters found")
    return canonical_state(output)


def _runtime_key_map(model) -> dict[str, str]:
    from peft import get_peft_model_state_dict

    return {normalize_adapter_name(key): key for key in get_peft_model_state_dict(model)}


def load_adapter(model, state: TensorState) -> None:
    from peft import set_peft_model_state_dict

    runtime = _runtime_key_map(model)
    missing = set(state) - set(runtime)
    if missing:
        raise ValueError(f"Adapter parameters absent from model: {sorted(missing)[:5]}")
    payload = {runtime[name]: tensor.to(next(model.parameters()).device) for name, tensor in state.items()}
    result = set_peft_model_state_dict(model, payload)
    if getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def trainable_adapter_parameters(model) -> dict[str, torch.nn.Parameter]:
    result = {}
    for runtime_name, parameter in model.named_parameters():
        canonical = normalize_adapter_name(runtime_name)
        if parameter.requires_grad and ("lora_A" in canonical or "lora_B" in canonical):
            result[canonical] = parameter
    return result


def merge_lora_into_proxy(model):
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Expected a PEFT model")
    return model.merge_and_unload(safe_merge=True)

