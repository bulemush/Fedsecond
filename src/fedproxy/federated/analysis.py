from __future__ import annotations

import torch

from .state import AnalysisResult, ParameterSpec, TensorState, flatten_state, unflatten_state, validate_state


def analyze_updates(
    deltas: list[TensorState], *, r0: float = 1.0, delta: float = 0.2, epsilon: float = 1e-12
) -> AnalysisResult:
    if not deltas:
        raise ValueError("At least one client update is required")
    spec = ParameterSpec.from_state(deltas[0])
    for state in deltas:
        validate_state(state, spec)
    matrix = torch.stack([flatten_state(state, spec) for state in deltas]).float()
    k = matrix.shape[0]
    if k == 1:
        zero_conflict = unflatten_state(torch.zeros(spec.total_numel), spec)
        return AnalysisResult(
            similarity=torch.ones((1, 1)),
            heterogeneity=torch.zeros(1),
            h_norm=torch.zeros(1),
            weights=torch.ones(1),
            retention_rates=torch.ones(1),
            next_conflict=zero_conflict,
        )
    norms = matrix.norm(dim=1)
    denom = norms[:, None] * norms[None, :]
    similarity = torch.where(denom > epsilon, matrix @ matrix.T / denom.clamp_min(epsilon), 0.0)
    diag = torch.where(norms > epsilon, torch.ones_like(norms), torch.zeros_like(norms))
    similarity.fill_diagonal_(0.0)
    similarity += torch.diag(diag)
    offdiag = similarity - torch.diag_embed(torch.diagonal(similarity))
    heterogeneity = 1.0 - offdiag.clamp_min(0).sum(dim=1) / (k - 1)
    consensus = offdiag.abs().sum(dim=1)
    weights = torch.softmax(consensus, dim=0)
    span = heterogeneity.max() - heterogeneity.min()
    h_norm = torch.zeros_like(heterogeneity) if span <= epsilon else (heterogeneity - heterogeneity.min()) / span
    retention_rates = torch.clamp(r0 - delta * h_norm, 0.0, 1.0)
    conflict_vector = 1.0 - matrix.sign().sum(dim=0).abs() / k
    return AnalysisResult(similarity, heterogeneity, h_norm, weights, retention_rates, unflatten_state(conflict_vector, spec))


def analyze_updates_chunked(
    deltas: list[TensorState],
    *,
    r0: float = 1.0,
    delta: float = 0.2,
    epsilon: float = 1e-12,
    chunk_size: int = 1_048_576,
) -> AnalysisResult:
    if not deltas:
        raise ValueError("At least one client update is required")
    if len(deltas) == 1:
        return analyze_updates(deltas, r0=r0, delta=delta, epsilon=epsilon)
    spec = ParameterSpec.from_state(deltas[0])
    for state in deltas:
        validate_state(state, spec)
    k = len(deltas)
    gram = torch.zeros((k, k), dtype=torch.float64)
    conflict: TensorState = {}
    for entry in spec.entries:
        target = torch.empty(entry.shape, dtype=torch.float32)
        target_flat = target.reshape(-1)
        sources = [state[entry.name].detach().cpu().float().reshape(-1) for state in deltas]
        for start in range(0, entry.numel, chunk_size):
            end = min(start + chunk_size, entry.numel)
            matrix = torch.stack([source[start:end] for source in sources]).double()
            gram += matrix @ matrix.T
            target_flat[start:end] = 1.0 - matrix.sign().sum(dim=0).abs().float() / k
        conflict[entry.name] = target
    norms = torch.diagonal(gram).clamp_min(0).sqrt()
    denominator = norms[:, None] * norms[None, :]
    similarity = torch.where(denominator > epsilon, gram / denominator.clamp_min(epsilon), 0.0).float()
    similarity.fill_diagonal_(0.0)
    similarity += torch.diag(torch.where(norms > epsilon, 1.0, 0.0).float())
    offdiag = similarity - torch.diag_embed(torch.diagonal(similarity))
    heterogeneity = 1.0 - offdiag.clamp_min(0).sum(dim=1) / (k - 1)
    weights = torch.softmax(offdiag.abs().sum(dim=1), dim=0)
    span = heterogeneity.max() - heterogeneity.min()
    h_norm = torch.zeros_like(heterogeneity) if span <= epsilon else (heterogeneity - heterogeneity.min()) / span
    retention_rates = torch.clamp(r0 - delta * h_norm, 0.0, 1.0)
    return AnalysisResult(similarity, heterogeneity, h_norm, weights, retention_rates, conflict)
