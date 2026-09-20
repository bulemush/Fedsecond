from __future__ import annotations

import torch

from .sparsify import stable_global_topk, stable_tensor_topk
from .state import AnalysisResult, ParameterSpec, TensorState, flatten_state, unflatten_state, validate_state


def h_ties(
    deltas: list[TensorState],
    analysis: AnalysisResult,
    *,
    rho: float = 1.1,
    epsilon: float = 1e-12,
    sparsify_scope: str = "global_adapter",
) -> TensorState:
    if sparsify_scope not in {"global_adapter", "tensor"}:
        raise ValueError(f"Unknown sparsify scope: {sparsify_scope}")
    if len(deltas) == 1:
        return {name: value.detach().cpu().float().clone() for name, value in deltas[0].items()}
    spec = ParameterSpec.from_state(deltas[0])
    for state in deltas:
        validate_state(state, spec)
    sparse = []
    for state, retention in zip(deltas, analysis.retention_rates.tolist(), strict=True):
        fn = stable_global_topk if sparsify_scope == "global_adapter" else stable_tensor_topk
        sparse.append(fn(state, float(retention)))
    vectors = torch.stack([flatten_state(state, spec) for state in sparse])
    weights = analysis.weights.float().reshape(-1, 1)
    scaled = vectors * weights
    positive = scaled > 0
    negative = scaled < 0
    p_mag = torch.where(positive, scaled.abs(), 0.0).sum(dim=0)
    n_mag = torch.where(negative, scaled.abs(), 0.0).sum(dim=0)
    positive_dominant = p_mag / (n_mag + epsilon) >= rho
    negative_dominant = (~positive_dominant) & (n_mag / (p_mag + epsilon) >= rho)
    p_den = torch.where(positive, weights.expand_as(scaled), 0.0).sum(dim=0)
    n_den = torch.where(negative, weights.expand_as(scaled), 0.0).sum(dim=0)
    p_value = torch.where(p_den > 0, torch.where(positive, scaled, 0.0).sum(dim=0) / p_den.clamp_min(epsilon), 0.0)
    n_value = torch.where(n_den > 0, torch.where(negative, scaled, 0.0).sum(dim=0) / n_den.clamp_min(epsilon), 0.0)
    merged = torch.where(positive_dominant, p_value, torch.where(negative_dominant, n_value, 0.0))
    return unflatten_state(merged, spec)


def h_ties_chunked(
    deltas: list[TensorState],
    analysis: AnalysisResult,
    *,
    rho: float = 1.1,
    epsilon: float = 1e-12,
    sparsify_scope: str = "global_adapter",
    chunk_size: int = 1_048_576,
) -> TensorState:
    if sparsify_scope not in {"global_adapter", "tensor"}:
        raise ValueError(f"Unknown sparsify scope: {sparsify_scope}")
    if len(deltas) == 1:
        return {name: value.detach().cpu().float().clone() for name, value in deltas[0].items()}
    spec = ParameterSpec.from_state(deltas[0])
    for state in deltas:
        validate_state(state, spec)
    sparse = []
    for state, retention in zip(deltas, analysis.retention_rates.tolist(), strict=True):
        fn = stable_global_topk if sparsify_scope == "global_adapter" else stable_tensor_topk
        sparse.append(fn(state, float(retention)))
    weights = analysis.weights.float().reshape(-1, 1)
    result: TensorState = {}
    for entry in spec.entries:
        target = torch.empty(entry.shape, dtype=torch.float32)
        target_flat = target.reshape(-1)
        sources = [state[entry.name].reshape(-1) for state in sparse]
        for start in range(0, entry.numel, chunk_size):
            end = min(start + chunk_size, entry.numel)
            vectors = torch.stack([source[start:end] for source in sources])
            scaled = vectors * weights
            positive, negative = scaled > 0, scaled < 0
            p_mag = torch.where(positive, scaled.abs(), 0.0).sum(dim=0)
            n_mag = torch.where(negative, scaled.abs(), 0.0).sum(dim=0)
            p_dom = p_mag / (n_mag + epsilon) >= rho
            n_dom = (~p_dom) & (n_mag / (p_mag + epsilon) >= rho)
            p_den = torch.where(positive, weights.expand_as(scaled), 0.0).sum(dim=0)
            n_den = torch.where(negative, weights.expand_as(scaled), 0.0).sum(dim=0)
            p_value = torch.where(p_den > 0, torch.where(positive, scaled, 0.0).sum(dim=0) / p_den.clamp_min(epsilon), 0.0)
            n_value = torch.where(n_den > 0, torch.where(negative, scaled, 0.0).sum(dim=0) / n_den.clamp_min(epsilon), 0.0)
            target_flat[start:end] = torch.where(p_dom, p_value, torch.where(n_dom, n_value, 0.0))
        result[entry.name] = target
    return result


def fedavg(deltas: list[TensorState], num_examples: list[int] | None = None, uniform: bool = False) -> TensorState:
    spec = ParameterSpec.from_state(deltas[0])
    matrix = torch.stack([flatten_state(state, spec) for state in deltas])
    if uniform or num_examples is None:
        weights = torch.full((len(deltas),), 1 / len(deltas))
    else:
        weights = torch.tensor(num_examples, dtype=torch.float32)
        weights /= weights.sum()
    return unflatten_state((matrix * weights[:, None]).sum(dim=0), spec)


def ties(deltas: list[TensorState], retention: float = 1.0) -> TensorState:
    spec = ParameterSpec.from_state(deltas[0])
    matrix = torch.stack([flatten_state(stable_global_topk(state, retention, spec), spec) for state in deltas])
    sign = matrix.sum(dim=0).sign()
    selected = (matrix.sign() == sign) & (matrix != 0) & (sign != 0)
    count = selected.sum(dim=0)
    merged = torch.where(count > 0, torch.where(selected, matrix, 0.0).sum(dim=0) / count.clamp_min(1), 0.0)
    return unflatten_state(merged, spec)
