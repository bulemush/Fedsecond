"""Explicit round extensions preserve the original config/packet identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def target_rounds(cfg: dict) -> int:
    initial = int(cfg["federated"]["rounds"])
    path = Path(cfg["run"]["output_dir"]) / "round_extension.json"
    if not path.exists():
        return initial
    value = json.loads(path.read_text(encoding="utf-8"))
    target = value.get("target_rounds")
    if (value.get("version") != 1 or value.get("config_hash") != config_digest(cfg)
            or value.get("initial_rounds") != initial
            or type(target) is not int or target <= initial):
        raise ValueError(f"Invalid round extension or changed original configuration: {path}")
    return target
