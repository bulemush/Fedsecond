from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch

TensorState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass(frozen=True)
class ParameterSpec:
    entries: tuple[ParameterEntry, ...]
    total_numel: int
    schema_hash: str
    space: Literal["lora_ab"] = "lora_ab"

    @classmethod
    def from_state(cls, state: TensorState) -> "ParameterSpec":
        offset = 0
        entries = []
        for name in sorted(state):
            tensor = state[name]
            entries.append(ParameterEntry(name, tuple(tensor.shape), tensor.numel(), offset))
            offset += tensor.numel()
        payload = [(e.name, e.shape, e.numel, e.offset) for e in entries]
        digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        return cls(tuple(entries), offset, digest)


@dataclass
class ClientResult:
    client_id: str
    adapter_state: TensorState
    num_examples: int
    metrics: dict[str, float]


@dataclass
class AnalysisResult:
    similarity: torch.Tensor
    heterogeneity: torch.Tensor
    h_norm: torch.Tensor
    weights: torch.Tensor
    retention_rates: torch.Tensor
    next_conflict: TensorState


def canonical_state(state: TensorState) -> TensorState:
    return {name: state[name].detach().cpu().float().contiguous().clone() for name in sorted(state)}


def validate_state(state: TensorState, spec: ParameterSpec, *, finite: bool = True) -> None:
    if set(state) != {entry.name for entry in spec.entries}:
        raise ValueError("State parameter names do not match schema")
    for entry in spec.entries:
        value = state[entry.name]
        if tuple(value.shape) != entry.shape:
            raise ValueError(f"Shape mismatch for {entry.name}: {tuple(value.shape)} != {entry.shape}")
        if not value.is_floating_point():
            raise TypeError(f"Non-floating tensor in adapter state: {entry.name}")
        if finite and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite value in {entry.name}")


def flatten_state(state: TensorState, spec: ParameterSpec | None = None) -> torch.Tensor:
    spec = spec or ParameterSpec.from_state(state)
    validate_state(state, spec)
    if not spec.entries:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([state[e.name].detach().cpu().float().reshape(-1) for e in spec.entries])


def unflatten_state(vector: torch.Tensor, spec: ParameterSpec) -> TensorState:
    if vector.numel() != spec.total_numel:
        raise ValueError(f"Vector length {vector.numel()} != schema length {spec.total_numel}")
    flat = vector.detach().cpu().float().reshape(-1)
    return {
        e.name: flat[e.offset : e.offset + e.numel].reshape(e.shape).clone()
        for e in spec.entries
    }


def add_states(left: TensorState, right: TensorState) -> TensorState:
    spec = ParameterSpec.from_state(left)
    validate_state(right, spec)
    return {name: left[name].detach().cpu().float() + right[name].detach().cpu().float() for name in left}


def subtract_states(left: TensorState, right: TensorState) -> TensorState:
    spec = ParameterSpec.from_state(left)
    validate_state(right, spec)
    return {name: left[name].detach().cpu().float() - right[name].detach().cpu().float() for name in left}


def zeros_like_state(state: TensorState) -> TensorState:
    return {name: torch.zeros_like(value, device="cpu", dtype=torch.float32) for name, value in state.items()}
