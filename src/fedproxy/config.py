from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


REPRODUCTION_DECISIONS = {
    "D01": "LoRA A/B parameter space",
    "D02": "remove_ratio means requested block removal ratio; record parameter ratio separately",
    "D03": "keep=max(1,L-floor(L*remove_ratio))",
    "D04": "no forced first/last decoder block",
    "D05": "q/k/v/o and gate/up/down LoRA targets",
    "D06": "PCR follows equations 7 and 9",
    "D07": "absolute cosine in equation 6",
    "D08": "all-zero coordinate conflict equals one",
    "D09": "constant heterogeneity normalizes to zero",
    "D10": "global adapter top-k",
    "D11": "floor top-k with canonical-index ties",
    "D12": "rounds must be explicit for real training",
    "D13": "ten local epochs per communication round",
    "D14": "reset AdamW every client and round",
    "D15": "unified_v1 response-only template",
    "D16": "fixed subsample before partition",
    "D17": "512 Alpaca calibration examples, seed 42",
    "D18": "constant LR and zero weight decay",
    "D19": "PCR sum reduction",
    "D20": "FP32 analysis with epsilon 1e-12",
    "D21": "batch size means effective client batch",
    "D22": "full participation",
    "D23": "versioned zero-shot lm-eval protocol",
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    inherited = data.pop("extends", None)
    if inherited:
        parent = (path.parent / inherited).resolve()
        data = _merge(load_config(parent), data)
    elif base_path:
        data = _merge(load_config(base_path), data)
    validate_config(data)
    return data


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value: {item}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ConfigError(f"Unknown override path: {dotted}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ConfigError(f"Unknown override key: {dotted}")
        cursor[parts[-1]] = value
    validate_config(out)
    return out


def validate_config(cfg: dict[str, Any], command: str | None = None) -> None:
    required = {"run", "model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties"}
    missing = required - set(cfg)
    if missing:
        raise ConfigError(f"Missing sections: {sorted(missing)}")
    ratio = float(cfg["compression"]["remove_ratio"])
    if not 0 <= ratio < 1:
        raise ConfigError("compression.remove_ratio must be in [0, 1)")
    if int(cfg["data"]["num_clients"]) < 1:
        raise ConfigError("data.num_clients must be positive")
    if not str(cfg["data"].get("local_dir", "")).strip():
        raise ConfigError("data.local_dir must be a non-empty path")
    if cfg["data"].get("scenario") not in {"homogeneous", "heterogeneous"}:
        raise ConfigError("data.scenario must be homogeneous or heterogeneous")
    if int(cfg["lora"]["r"]) < 1 or int(cfg["lora"]["alpha"]) < 1:
        raise ConfigError("LoRA rank and alpha must be positive")
    micro = int(cfg["training"]["micro_batch_size"])
    accumulation = int(cfg["training"]["gradient_accumulation_steps"])
    effective = int(cfg["training"]["effective_batch_size"])
    if micro < 1 or accumulation < 1 or effective != micro * accumulation:
        raise ConfigError("effective_batch_size must equal micro_batch_size * gradient_accumulation_steps")
    rounds = cfg["federated"].get("rounds")
    if rounds is not None and int(rounds) < 1:
        raise ConfigError("federated.rounds must be positive")
    if cfg["federated"].get("participation") != "full":
        raise ConfigError("v1 implements full participation only")
    execution = cfg["federated"].get("client_execution")
    if execution not in {"sequential", "parallel"}:
        raise ConfigError("federated.client_execution must be sequential or parallel")
    parallel_clients = int(cfg["federated"].get("max_parallel_clients", 1))
    if execution == "sequential" and parallel_clients != 1:
        raise ConfigError("Sequential execution requires max_parallel_clients=1")
    if execution == "parallel" and parallel_clients < 0:
        raise ConfigError("Parallel max_parallel_clients must be zero (auto) or positive")
    if cfg["compression"].get("device_map") not in {None, "auto", "balanced"}:
        raise ConfigError("compression.device_map must be null, auto, or balanced")
    if cfg["federated"].get("parameter_space") != "lora_ab":
        raise ConfigError("v1 implements LoRA A/B parameter space only")
    if cfg["pcr"].get("reduction") != "sum":
        raise ConfigError("paper-mode PCR reduction must be sum")
    if cfg["h_ties"].get("sparsify_scope") not in {"global_adapter", "tensor"}:
        raise ConfigError("Invalid sparsify scope")
    if float(cfg["h_ties"]["rho"]) < 1:
        raise ConfigError("h_ties.rho must be at least one")
    if not 0 <= float(cfg["h_ties"]["r0"]) <= 1 or float(cfg["h_ties"]["delta"]) < 0:
        raise ConfigError("Invalid H-TIES retention parameters")
    if command == "train" and not cfg["run"].get("synthetic") and cfg["federated"].get("rounds") is None:
        raise ConfigError("Real training requires an explicit federated.rounds")


def save_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
