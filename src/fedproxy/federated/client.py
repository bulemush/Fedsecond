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
    *,
    round_id: int | None = None,
    device_label: str | None = None,
) -> ClientResult:
    started = time.perf_counter()
    training = config["training"]
    progress_enabled = bool(training.get("progress", True))
    total_rounds = int(config["federated"]["rounds"])
    round_number = 1 if round_id is None else round_id + 1
    device_label = device_label or "auto"
    prefix = (
        f"[train] round={round_number}/{total_rounds} "
        f"client={client_id} device={device_label}"
    )
    if progress_enabled:
        print(f"{prefix} status=loading_model", flush=True)
    model = model_factory()
    load_adapter(model, global_adapter)
    anchor = {name: value.detach().cpu().float().clone() for name, value in global_adapter.items()}
    parameters = trainable_adapter_parameters(model)
    if set(parameters) != set(anchor):
        raise ValueError("Trainable LoRA parameters do not match the global adapter schema")
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
    num_examples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    total_optimizer_steps = estimate_optimizer_steps(num_examples, training)
    progress_interval = float(training.get("progress_log_interval_seconds", 30))
    training_started = time.perf_counter()
    last_progress = training_started
    if progress_enabled:
        print(
            f"{prefix} status=started examples={num_examples} "
            f"epochs={epochs} optimizer_steps={total_optimizer_steps}",
            flush=True,
        )
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
            now = time.perf_counter()
            should_log = (
                optimizer_steps == 1
                or optimizer_steps == total_optimizer_steps
                or progress_interval == 0
                or now - last_progress >= progress_interval
            )
            if progress_enabled and should_log:
                elapsed = now - training_started
                rate = optimizer_steps / max(elapsed, 1e-12)
                remaining = max(0, total_optimizer_steps - optimizer_steps)
                eta_seconds = remaining / max(rate, 1e-12)
                percent = 100.0 * optimizer_steps / max(1, total_optimizer_steps)
                mean_task_loss = task_total / max(1, batches)
                mean_total_loss = total_total / max(1, batches)
                print(
                    f"{prefix} status=running epoch={_epoch + 1}/{epochs} "
                    f"step={optimizer_steps}/{total_optimizer_steps} "
                    f"progress={percent:.1f}% task_loss={mean_task_loss:.6f} "
                    f"total_loss={mean_total_loss:.6f} "
                    f"elapsed={elapsed:.0f}s eta={eta_seconds:.0f}s",
                    flush=True,
                )
                last_progress = now
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
    if not num_examples:
        num_examples = batches * int(training["micro_batch_size"])
    if progress_enabled:
        print(
            f"{prefix} status=completed steps={optimizer_steps} "
            f"task_loss={metrics['task_loss']:.6f} "
            f"update_norm={metrics['update_norm']:.6f} "
            f"elapsed={metrics['elapsed_seconds']:.0f}s",
            flush=True,
        )
    del model
    return ClientResult(client_id, state, int(num_examples), metrics)


def estimate_optimizer_steps(num_examples: int, cfg: dict) -> int:
    micro = int(cfg["micro_batch_size"])
    accumulation = int(cfg["gradient_accumulation_steps"])
    epochs = int(cfg["local_epochs"])
    return math.ceil(num_examples / micro / accumulation) * epochs
