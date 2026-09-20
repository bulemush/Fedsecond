from __future__ import annotations

import torch


def pcr_loss(local_state, anchor_state, conflict_state) -> torch.Tensor:
    if set(local_state) != set(anchor_state) or set(local_state) != set(conflict_state):
        raise ValueError("PCR states have different parameter names")
    total = None
    for name in sorted(local_state):
        parameter = local_state[name]
        anchor = anchor_state[name].to(parameter.device)
        conflict = conflict_state[name].to(parameter.device)
        term = (conflict.float() * (parameter.float() - anchor.float()).square()).sum()
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)


def fedprox_loss(local_state, anchor_state, mu: float) -> torch.Tensor:
    total = None
    for name in sorted(local_state):
        parameter = local_state[name]
        term = (parameter.float() - anchor_state[name].to(parameter.device).float()).square().sum()
        total = term if total is None else total + term
    return (mu / 2.0) * (total if total is not None else torch.tensor(0.0))

