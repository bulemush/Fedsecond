from __future__ import annotations

import csv
import gc
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from fedproxy.compression.bi import score_blocks
from fedproxy.compression.prune import build_proxy, select_layers
from fedproxy.data.partition import fixed_subsample, heterogeneous_partition, iid_partition, partition_hash
from fedproxy.data.prompts import format_alpaca
from fedproxy.data.registry import dataset_storage_path, load_registered
from fedproxy.evaluation.harness import evaluate_with_lm_eval
from fedproxy.evaluation.report import compare_runs, summarize
from fedproxy.federated.client import estimate_optimizer_steps, train_client
from fedproxy.federated.server import run_federated
from fedproxy.fusion.plug_in import fuse_and_save
from fedproxy.models.backbone import load_causal_lm, load_tokenizer, resolve_dtype
from fedproxy.models.lora import export_adapter, inject_lora, load_adapter, merge_lora_into_proxy
from fedproxy.utils.checkpoint import load_checkpoint
from fedproxy.utils.reproducibility import derived_seed, seed_everything


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_data(cfg: dict) -> dict:
    tasks = list(cfg["data"]["tasks"])
    seed = int(cfg["data"]["partition_seed"])
    limit = int(cfg["data"]["max_train_samples_per_task"])
    selected, fingerprints, local_paths = {}, {}, {}
    data_dir = cfg["data"].get("local_dir", "data")
    for task in tasks:
        dataset = load_registered(
            task,
            revision=cfg["data"].get("revisions", {}).get(task),
            data_dir=data_dir,
        )
        selected[task] = fixed_subsample(list(range(len(dataset))), limit, seed)
        fingerprints[task] = getattr(dataset, "_fingerprint", None)
        local_paths[task] = str(
            dataset_storage_path(
                data_dir,
                task,
                "train",
                cfg["data"].get("revisions", {}).get(task),
            )
        )
    if cfg["data"]["scenario"] == "heterogeneous":
        if len(tasks) != int(cfg["data"]["num_clients"]):
            raise ValueError("Heterogeneous v1 assigns exactly one task to every client")
        clients = heterogeneous_partition(tasks, selected)
    else:
        if len(tasks) != 1:
            raise ValueError("A homogeneous run must select exactly one task")
        clients = iid_partition(selected[tasks[0]], int(cfg["data"]["num_clients"]), seed)
    manifest = {
        "scenario": cfg["data"]["scenario"],
        "seed": seed,
        "dataset_fingerprints": fingerprints,
        "dataset_local_paths": local_paths,
        "clients": clients,
    }
    manifest["partition_hash"] = partition_hash(manifest)
    path = Path(cfg["run"]["output_dir"]) / "data_manifest.json"
    _write_json(path, manifest)
    return manifest


def compress(cfg: dict) -> dict:
    from torch.utils.data import DataLoader

    device_map = cfg["compression"].get("device_map")
    model = load_causal_lm(cfg["model"], device_map=device_map)
    if device_map is None:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(cfg["model"])
    source_config_fingerprint = hashlib.sha256(
        json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()
    compression = cfg["compression"]
    dataset = load_registered(
        "alpaca",
        revision=compression.get("calibration_revision"),
        data_dir=cfg["data"].get("local_dir", "data"),
    )
    indices = fixed_subsample(list(range(len(dataset))), int(compression["calibration_samples"]), int(compression["calibration_seed"]))
    records = [format_alpaca(dataset[i]["instruction"], dataset[i].get("input", ""), dataset[i]["output"]) for i in indices]

    def collate(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=int(compression["calibration_max_length"]), return_tensors="pt")

    loader = DataLoader(records, batch_size=int(compression.get("calibration_batch_size", 1)), collate_fn=collate)
    scores = score_blocks(model, loader)
    kept = select_layers(scores, float(compression["remove_ratio"]))
    original_layers = len(scores)
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    proxy, layer_map = build_proxy(model, kept)
    proxy_parameters = sum(parameter.numel() for parameter in proxy.parameters())
    output = Path(cfg["run"]["output_dir"]) / "compression" / "proxy"
    proxy.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    proxy_artifact_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    tokenizer_fingerprint = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "source_model": cfg["model"]["name_or_path"],
        "source_revision": getattr(model.config, "_commit_hash", None) or cfg["model"].get("revision"),
        "source_config_sha256": source_config_fingerprint,
        "calibration_dataset": "alpaca",
        "calibration_fingerprint": getattr(dataset, "_fingerprint", None),
        "calibration_local_path": str(
            dataset_storage_path(
                cfg["data"].get("local_dir", "data"),
                "alpaca",
                "train",
                compression.get("calibration_revision"),
            )
        ),
        "calibration_indices": indices,
        "calibration_template": "Instruction/Input/Response including output",
        "tokenizer_sha256": tokenizer_fingerprint,
        "block_influence": scores,
        "kept_indices": kept,
        "layer_map": {str(k): v for k, v in layer_map.items()},
        "requested_removed_block_ratio": float(compression["remove_ratio"]),
        "actual_removed_block_ratio": 1 - len(kept) / original_layers,
        "actual_removed_parameter_ratio": 1 - proxy_parameters / original_parameters,
        "original_layers": original_layers,
        "proxy_layers": len(kept),
        "original_parameters": original_parameters,
        "proxy_parameters": proxy_parameters,
        "proxy_artifact_bytes": proxy_artifact_bytes,
    }
    _write_json(output.parent / "compression_manifest.json", manifest)
    return manifest


class _ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _client_task_and_indices(cfg: dict, manifest: dict, client_id: str):
    assignment = manifest["clients"][client_id]
    if isinstance(assignment, dict):
        return assignment["task"], assignment["indices"]
    return cfg["data"]["tasks"][0], assignment


def _load_client_examples(cfg: dict, manifest: dict, client_id: str):
    from fedproxy.data.tasks import convert_row

    task, indices = _client_task_and_indices(cfg, manifest, client_id)
    dataset = load_registered(
        task,
        revision=cfg["data"].get("revisions", {}).get(task),
        data_dir=cfg["data"].get("local_dir", "data"),
    )
    return [convert_row(task, dataset[index], index, dataset) for index in indices]


def _make_client_loader(cfg: dict, examples, tokenizer, *, seed: int | None = None):
    from torch.utils.data import DataLoader

    from fedproxy.data.collator import ResponseOnlyCollator

    collator = ResponseOnlyCollator(
        tokenizer,
        max_input_length=int(cfg["data"]["max_input_length"]),
        max_target_length=int(cfg["data"]["max_target_length"]),
    )
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(
        _ExampleDataset(examples),
        batch_size=int(cfg["training"]["micro_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )


def _load_train_model(proxy_path: str | Path, cfg: dict, device: torch.device):
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        proxy_path,
        local_files_only=True,
        torch_dtype=resolve_dtype(cfg["model"].get("dtype", "auto")),
    )
    model = inject_lora(backbone, cfg["lora"])
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model.to(device)


def _parallel_client_worker(payload: dict):
    cfg = payload["cfg"]
    client_id = payload["client_id"]
    device_index = int(payload["device_index"])
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel client training requires CUDA")
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    local_seed = derived_seed(int(cfg["run"]["seed"]), payload["round_id"], client_id)
    seed_everything(local_seed)
    tokenizer = load_tokenizer(
        {**cfg["model"], "name_or_path": payload["proxy_path"], "revision": None}
    )
    examples = _load_client_examples(cfg, payload["manifest"], client_id)
    loader = _make_client_loader(cfg, examples, tokenizer, seed=local_seed)

    def model_factory():
        return _load_train_model(payload["proxy_path"], cfg, device)

    result = train_client(
        client_id,
        model_factory,
        payload["round_base"],
        payload["conflict"],
        loader,
        cfg,
    )
    result.metrics["visible_device_index"] = float(device_index)
    torch.cuda.empty_cache()
    return result


def _build_client_loaders(cfg: dict, manifest: dict, tokenizer):
    loaders = {}
    for client_id in manifest["clients"]:
        examples = _load_client_examples(cfg, manifest, client_id)
        loaders[client_id] = _make_client_loader(cfg, examples, tokenizer)
    return loaders


def train(cfg: dict, resume: str | None = None, dry_run: bool = False) -> dict:
    from torch.utils.data import DataLoader

    seed_everything(int(cfg["run"]["seed"]))
    run_dir = Path(cfg["run"]["output_dir"])
    proxy_path = run_dir / "compression" / "proxy"
    manifest_path = run_dir / "data_manifest.json"
    if not proxy_path.exists():
        raise FileNotFoundError("Run compress before train")
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare-data before train")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    invariants = {
        "config_hash": config_hash,
        "partition_hash": data_manifest["partition_hash"],
        "source_model": cfg["model"]["name_or_path"],
        "source_revision": cfg["model"].get("revision"),
        "seed_strategy": "sha256(base_seed, round_id, client_id)",
    }
    execution = cfg["federated"].get("client_execution", "sequential")
    client_ids = list(data_manifest["clients"])
    visible_gpus = torch.cuda.device_count()
    if execution == "parallel":
        if visible_gpus < 1:
            raise RuntimeError("Parallel client execution requested but no CUDA device is visible")
        requested = int(cfg["federated"].get("max_parallel_clients", 0))
        parallelism = min(len(client_ids), visible_gpus if requested == 0 else requested)
        if parallelism < 1 or parallelism > visible_gpus:
            raise ValueError("max_parallel_clients exceeds the number of visible CUDA devices")
        loaders = None
    else:
        parallelism = 1
        tokenizer = load_tokenizer(
            {**cfg["model"], "name_or_path": str(proxy_path), "revision": None}
        )
        loaders = _build_client_loaders(cfg, data_manifest, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_factory():
        return _load_train_model(proxy_path, cfg, device)

    client_sizes = {
        client_id: len(_client_task_and_indices(cfg, data_manifest, client_id)[1])
        for client_id in client_ids
    }
    budget = {
        client_id: {
            "examples": size,
            "optimizer_steps_per_round": estimate_optimizer_steps(size, cfg["training"]),
        }
        for client_id, size in client_sizes.items()
    }
    training_budget = {
        "rounds": int(cfg["federated"]["rounds"]),
        "local_epochs": int(cfg["training"]["local_epochs"]),
        "clients": budget,
        "device": str(device),
        "resolved_dtype": str(resolve_dtype(cfg["model"].get("dtype", "auto"))),
        "client_execution": execution,
        "parallel_clients": parallelism,
        "visible_cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(visible_gpus)
        ],
        "selected_physical_gpu_ids": os.environ.get("FEDPROXY_GPU_IDS"),
    }
    _write_json(run_dir / "training_budget.json", training_budget)
    print(json.dumps({"training_budget": training_budget}, indent=2))
    if dry_run:
        return {"dry_run": True, "training_budget": training_budget}

    # In parallel mode the parent only needs the initial LoRA tensors.  Build
    # them on CPU so it does not retain a CUDA model/context before spawning
    # one isolated worker per client GPU.
    initialization_device = torch.device("cpu") if execution == "parallel" else device
    initialized = _load_train_model(proxy_path, cfg, initialization_device)
    initial_adapter = export_adapter(initialized)
    del initialized
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_one = None
    train_many = None
    if execution == "sequential":
        assert loaders is not None

        def train_one(client_id, base, conflict, round_id):
            local_seed = derived_seed(int(cfg["run"]["seed"]), round_id, client_id)
            seed_everything(local_seed)
            template = loaders[client_id]
            loader = DataLoader(
                template.dataset,
                batch_size=template.batch_size,
                shuffle=True,
                collate_fn=template.collate_fn,
                generator=torch.Generator().manual_seed(local_seed),
            )
            return train_client(client_id, model_factory, base, conflict, loader, cfg)
    else:
        context = multiprocessing.get_context("spawn")

        def train_many(scheduled, base, conflict, round_id):
            shared_base = {name: tensor.share_memory_() for name, tensor in base.items()}
            shared_conflict = {name: tensor.share_memory_() for name, tensor in conflict.items()}
            results = []
            for start in range(0, len(scheduled), parallelism):
                wave = scheduled[start : start + parallelism]
                payloads = [
                    {
                        "cfg": cfg,
                        "client_id": client_id,
                        "device_index": offset,
                        "round_id": round_id,
                        "proxy_path": str(proxy_path),
                        "manifest": data_manifest,
                        "round_base": shared_base,
                        "conflict": shared_conflict,
                    }
                    for offset, client_id in enumerate(wave)
                ]
                with ProcessPoolExecutor(
                    max_workers=len(wave), mp_context=context
                ) as executor:
                    results.extend(executor.map(_parallel_client_worker, payloads))
            return results

    adapter, conflict, history = run_federated(
        initial_adapter,
        client_ids,
        train_one,
        cfg,
        checkpoint_root=run_dir / "checkpoints",
        resume=resume,
        invariants=invariants,
        train_many=train_many,
    )
    metrics_dir = run_dir / "metrics"
    aggregation_path = metrics_dir / "aggregation.json"
    previous = json.loads(aggregation_path.read_text(encoding="utf-8")) if resume and aggregation_path.exists() else []
    _write_json(aggregation_path, previous + history)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and (metrics_dir / "train.jsonl").exists() else "w"
    with (metrics_dir / "train.jsonl").open(mode, encoding="utf-8") as handle:
        for round_metadata in history:
            for client_id, metrics in round_metadata["client_metrics"].items():
                handle.write(json.dumps({"round": round_metadata["completed_round"], "client_id": client_id, **metrics}) + "\n")
    return {
        "completed_rounds": int(cfg["federated"]["rounds"]),
        "new_rounds": len(history),
        "last_checkpoint": str(run_dir / "checkpoints" / f"round_{int(cfg['federated']['rounds']):04d}"),
    }


def fuse(cfg: dict, checkpoint: str, overwrite: bool = False) -> dict:
    from transformers import AutoModelForCausalLM

    run_dir = Path(cfg["run"]["output_dir"])
    proxy_path = run_dir / "compression" / "proxy"
    compression_manifest = json.loads((run_dir / "compression" / "compression_manifest.json").read_text(encoding="utf-8"))
    proxy = AutoModelForCausalLM.from_pretrained(proxy_path, local_files_only=True)
    proxy = inject_lora(proxy, cfg["lora"])
    state = load_checkpoint(checkpoint)
    load_adapter(proxy, state["adapter"])
    merged_proxy = merge_lora_into_proxy(proxy)
    original = load_causal_lm(cfg["model"], device_map=None)
    tokenizer = load_tokenizer(cfg["model"])
    adapter_hash = hashlib.sha256(
        b"".join(state["adapter"][name].contiguous().numpy().tobytes() for name in sorted(state["adapter"]))
    ).hexdigest()
    proxy_output = run_dir / "exports" / "proxy_tuned"
    if proxy_output.exists() and any(proxy_output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty proxy export: {proxy_output}")
    proxy_output.mkdir(parents=True, exist_ok=True)
    merged_proxy.save_pretrained(proxy_output, safe_serialization=True)
    tokenizer.save_pretrained(proxy_output)
    output = run_dir / "exports" / "llm_fused"
    fuse_and_save(
        original,
        merged_proxy,
        compression_manifest["layer_map"],
        output,
        {
            "proxy_path": str(proxy_path),
            "checkpoint": str(checkpoint),
            "adapter_sha256": adapter_hash,
            "source_model": cfg["model"]["name_or_path"],
            "source_revision": cfg["model"].get("revision"),
        },
        tokenizer=tokenizer,
        overwrite=overwrite,
    )
    return {"output": str(output), "proxy_output": str(proxy_output), "adapter_sha256": adapter_hash}


def evaluate(cfg: dict, model_kind: str) -> dict:
    run = Path(cfg["run"]["output_dir"])
    if model_kind == "original":
        model_path = cfg["model"]["name_or_path"]
    else:
        model_path = run / ("exports/llm_fused" if model_kind == "fused" else "exports/proxy_tuned")
    tasks = list(cfg["evaluation"]["harness_tasks"])
    raw_path = run / "evaluation" / "raw" / f"{model_kind}_lm_eval.json"
    raw = evaluate_with_lm_eval(
        str(model_path), tasks, raw_path,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        num_fewshot=int(cfg["evaluation"]["num_fewshot"]),
    )
    aliases = {"openbookqa": "obqa"}
    metrics = {}
    for task in tasks:
        task_result = raw["results"][task]
        candidates = [key for key in task_result if key == "acc" or key.startswith("acc,")]
        if not candidates:
            raise ValueError(f"No accuracy metric found for lm-eval task {task}")
        metrics[aliases.get(task, task)] = float(task_result[sorted(candidates)[0]])
    summary = summarize(metrics)
    protocol = {
        "tasks": tasks,
        "num_fewshot": int(cfg["evaluation"]["num_fewshot"]),
        "batch_size": int(cfg["evaluation"]["batch_size"]),
        "primary_metric": cfg["evaluation"]["primary_metric"],
    }
    summary.update({
        "protocol_fingerprint": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "protocol": protocol,
        "model_kind": model_kind,
        "synthetic": bool(cfg["run"].get("synthetic", False)),
        "diagnostic": bool(cfg["run"].get("diagnostic", False)),
    })
    _write_json(run / "evaluation" / f"{model_kind}_summary.json", summary)
    if model_kind == "fused":
        _write_json(run / "evaluation" / "summary.json", summary)
    return summary


def compare(runs_root: str, output: str) -> None:
    rows = compare_runs(runs_root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
