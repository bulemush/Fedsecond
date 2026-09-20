from __future__ import annotations

import math
import time
from itertools import islice
from collections.abc import Callable

import torch

from fedproxy.models.lora import export_adapter, load_adapter, trainable_adapter_parameters

from .regularization import fedprox_loss, pcr_loss
from .state import ClientResult, TensorState


def train_client(
    client_id: str,
    model_factory: Callable[[], object],
    global_adapter: TensorState,
    conflict_scores: TensorState,
    train_loader,
    config: dict,
) -> ClientResult:
    started = time.perf_counter()
    model = model_factory()
    load_adapter(model, global_adapter)
    anchor = {name: value.detach().cpu().float().clone() for name, value in global_adapter.items()}
    parameters = trainable_adapter_parameters(model)
    if set(parameters) != set(anchor):
        raise ValueError("Trainable LoRA parameters do not match the global adapter schema")
    training = config["training"]
    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["local_epochs"])
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    amp_enabled = device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}
    scaler_enabled = amp_enabled and model_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:  # Compatibility with older supported PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer_steps = 0
    task_total = raw_pcr_total = pcr_total = total_total = 0.0
    batches = 0
    supervised_tokens = 0
    for _epoch in range(epochs):
        iterator = iter(train_loader)
        while True:
            group = list(islice(iterator, accumulation))
            if not group:
                break
            # CausalLM shifts labels internally, so count supervised labels after index zero.
            token_counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in group]
            total_tokens = sum(token_counts)
            supervised_tokens += total_tokens
            if total_tokens == 0:
                raise ValueError("Gradient-accumulation group has no supervised target token")
            optimizer.zero_grad(set_to_none=True)
            for batch, token_count in zip(group, token_counts, strict=True):
                with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=amp_enabled):
                    outputs = model(**{key: value.to(device) for key, value in batch.items()})
                    task = outputs.loss.float()
                    raw_regularizer = torch.zeros((), device=device)
                    if config["pcr"].get("enabled", False):
                        raw_regularizer = pcr_loss(parameters, anchor, conflict_scores)
                        regularizer = raw_regularizer * float(config["pcr"]["lambda_reg"])
                    elif config.get("fedprox", {}).get("enabled", False):
                        regularizer = fedprox_loss(parameters, anchor, float(config["fedprox"]["mu"]))
                        raw_regularizer = regularizer
                    else:
                        regularizer = raw_regularizer
                    scaled = task * (token_count / total_tokens) + regularizer / len(group)
                scaler.scale(scaled).backward()
                batches += 1
                task_total += float(task.detach())
                raw_pcr_total += float(raw_regularizer.detach())
                pcr_total += float(regularizer.detach())
                total_total += float((task + regularizer).detach())
            max_grad_norm = training.get("max_grad_norm")
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters.values(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
    state = export_adapter(model)
    norm_sq = sum((state[name] - anchor[name]).square().sum() for name in state)
    denominator = max(1, batches)
    metrics = {
        "task_loss": task_total / denominator,
        "raw_pcr_loss": raw_pcr_total / denominator,
        "weighted_pcr_loss": pcr_total / denominator,
        "total_loss": total_total / denominator,
        "update_norm": float(torch.sqrt(norm_sq)),
        "optimizer_steps": float(optimizer_steps),
        "effective_target_tokens": float(supervised_tokens),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
    }
    num_examples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else batches * int(training["micro_batch_size"])
    del model
    return ClientResult(client_id, state, int(num_examples), metrics)


def estimate_optimizer_steps(num_examples: int, cfg: dict) -> int:
    micro = int(cfg["micro_batch_size"])
    accumulation = int(cfg["gradient_accumulation_steps"])
    epochs = int(cfg["local_epochs"])
    return math.ceil(num_examples / micro / accumulation) * epochs
