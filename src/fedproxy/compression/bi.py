from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from fedproxy.models.backbone import get_decoder_layers


def _hidden(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Unsupported layer output type: {type(value).__name__}")
    return value


@contextmanager
def _preserve_training_mode(model):
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(was_training)


def score_blocks(model, calibration_loader) -> list[float]:
    layers = get_decoder_layers(model)
    sums = torch.zeros(len(layers), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.float64)
    current_mask: torch.Tensor | None = None
    handles = []

    def make_hook(index):
        def hook(_module, inputs, output):
            nonlocal current_mask
            incoming = _hidden(inputs[0]).detach().float()
            outgoing = _hidden(output).detach().float()
            cos = F.cosine_similarity(incoming, outgoing, dim=-1).double()
            mask = torch.ones_like(cos, dtype=torch.bool) if current_mask is None else current_mask.to(cos.device).bool()
            sums[index] += cos[mask].sum().cpu()
            counts[index] += mask.sum().cpu()

        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    try:
        with _preserve_training_mode(model), torch.no_grad():
            for batch in calibration_loader:
                current_mask = batch.get("attention_mask")
                device = next(model.parameters()).device
                model(**{key: value.to(device) for key, value in batch.items()})
    finally:
        for handle in handles:
            handle.remove()
    if (counts == 0).any():
        raise ValueError("Calibration loader produced no valid token for at least one block")
    return (1.0 - sums / counts).tolist()

