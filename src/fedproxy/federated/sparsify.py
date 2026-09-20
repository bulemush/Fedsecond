from __future__ import annotations

import math

import torch

from .state import ParameterSpec, TensorState, flatten_state, unflatten_state


def stable_global_topk(state: TensorState, retention: float, spec: ParameterSpec | None = None) -> TensorState:
    if not 0 <= retention <= 1:
        raise ValueError("retention must be in [0, 1]")
    spec = spec or ParameterSpec.from_state(state)
    vector = flatten_state(state, spec)
    keep = math.floor(retention * vector.numel())
    if keep == 0:
        return unflatten_state(torch.zeros_like(vector), spec)
    if keep == vector.numel():
        return unflatten_state(vector, spec)
    # Stable sort preserves canonical flat-index order for equal magnitudes.
    order = torch.argsort(vector.abs(), descending=True, stable=True)
    mask = torch.zeros(vector.numel(), dtype=torch.bool)
    mask[order[:keep]] = True
    return unflatten_state(torch.where(mask, vector, 0.0), spec)


def stable_tensor_topk(state: TensorState, retention: float) -> TensorState:
    output: TensorState = {}
    for name in sorted(state):
        one_spec = ParameterSpec.from_state({name: state[name]})
        output[name] = stable_global_topk({name: state[name]}, retention, one_spec)[name]
    return output
