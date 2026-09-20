from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from safetensors.torch import load_file, save_file

from fedproxy.federated.state import ParameterSpec, TensorState, canonical_state, validate_state


def save_checkpoint(path: str | Path, adapter: TensorState, conflict: TensorState, metadata: dict) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    spec = ParameterSpec.from_state(adapter)
    validate_state(conflict, spec)
    save_file(canonical_state(adapter), str(temp / "adapter.safetensors"))
    save_file(canonical_state(conflict), str(temp / "conflict.safetensors"))
    serializable = metadata | {
        "schema_hash": spec.schema_hash,
        "parameter_space": spec.space,
        "complete": True,
    }
    with (temp / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, target)


def load_checkpoint(
    path: str | Path,
    expected_state: TensorState | None = None,
    expected_metadata: dict | None = None,
) -> dict:
    path = Path(path)
    with (path / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("complete"):
        raise ValueError(f"Incomplete checkpoint: {path}")
    adapter = load_file(str(path / "adapter.safetensors"), device="cpu")
    conflict = load_file(str(path / "conflict.safetensors"), device="cpu")
    spec = ParameterSpec.from_state(adapter)
    if metadata.get("schema_hash") != spec.schema_hash:
        raise ValueError("Checkpoint schema hash mismatch")
    validate_state(conflict, spec)
    if expected_state is not None and ParameterSpec.from_state(expected_state).schema_hash != spec.schema_hash:
        raise ValueError("Checkpoint is incompatible with the current adapter")
    for key, value in (expected_metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata mismatch for {key}")
    return {"adapter": canonical_state(adapter), "conflict": canonical_state(conflict), "metadata": metadata}
