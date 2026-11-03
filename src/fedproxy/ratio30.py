"""Matched inputs and independent FedProxy/MA single-task IID runs at ratio30."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from fedproxy.config import load_config, validate_config
from fedproxy.data.partition import partition_hash
from fedproxy.ma import homogeneous as suite
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, resolve_parallelism
from fedproxy.ma.round_budget import config_digest
from fedproxy.reuse import check_reuse_compatibility


PROJECT = suite.PROJECT
TASKS = suite.TASKS
ROOT = Path("runs/homogeneous_ratio30")
FOLDERS = {"fedproxy": "FedProxy", "ma": "MA"}


def task_config(method: str, task: str, root: Path = ROOT) -> dict:
    directory = "ma" if method == "ma" else "experiments"
    cfg = load_config(PROJECT / "configs" / directory / "homogeneous_ratio30" / f"{task}.yaml")
    cfg["run"]["output_dir"] = str(root / FOLDERS[method] / suite.run_name(task))
    if method == "ma":
        cfg["run"]["reuse_artifacts_from"] = str(root / "FedProxy" / suite.run_name(task))
    else:
        cfg["run"]["reuse_compression_from"] = str(root / "shared")
    return cfg


def shared_config(root: Path = ROOT) -> dict:
    cfg = task_config("fedproxy", "obqa", root)
    cfg["run"].update(name="llama2_homogeneous_ratio30_shared", output_dir=str(root / "shared"),
                      reuse_compression_from=None)
    return cfg


def check_contract(method: str, task: str, cfg: dict, *, heterogeneous: bool = False) -> None:
    validate_config(cfg, "train")
    required = {
        "run": {"seed": 42, "synthetic": False, "diagnostic": True, "adapter_initialization_seed": 42},
        "compression": {"remove_ratio": 0.30, "keep_count_rule": "floor_removed",
                        "calibration_dataset": "alpaca", "calibration_samples": 512,
                        "calibration_seed": 42, "calibration_max_length": 512, "calibration_batch_size": 1},
        "data": {"scenario": "homogeneous", "tasks": [task], "num_clients": 4,
                 "max_train_samples_per_task": 3000, "max_input_length": 320,
                 "max_target_length": 128, "partition_seed": 42},
        "training": {"local_epochs": 5, "micro_batch_size": 4, "gradient_accumulation_steps": 4,
                     "effective_batch_size": 16, "optimizer": "adamw", "learning_rate": 5.0e-5,
                     "weight_decay": 0.0, "scheduler": "constant", "warmup_steps": 0,
                     "gradient_checkpointing": True, "reset_optimizer_each_round": True},
        "federated": {"rounds": 10, "participation": "full", "aggregation": "h_ties",
                      "client_execution": "parallel", "max_parallel_clients": 0},
        "pcr": {"enabled": True, "lambda_reg": 1.0e-5, "reduction": "sum"},
        "lora": {"r": 32, "alpha": 64, "dropout": 0.1},
        "evaluation": {"harness_tasks": ["openbookqa" if task == "obqa" else task],
                       "num_fewshot": 0, "batch_size": 1, "primary_metric": "acc"},
    }
    if heterogeneous:
        required["data"].update(scenario="heterogeneous", tasks=list(TASKS), num_clients=8)
        required["evaluation"]["harness_tasks"] = ["openbookqa" if name == "obqa" else name for name in TASKS]
    errors = [f"{section}.{key} must be {value!r}" for section, fields in required.items()
              for key, value in fields.items() if cfg[section].get(key) != value]
    expected_ma = copy.deepcopy(load_config(PROJECT / "configs/ma/base.yaml")["ma"])
    expected_ma["schedule"].update(train_blocks=17, upload_blocks=14)
    if method == "ma" and cfg.get("ma") != expected_ma:
        errors.append("MA must match base.yaml except train_blocks=17, upload_blocks=14 (scheme B)")
    if method == "fedproxy" and cfg.get("ma", {}).get("enabled"):
        errors.append("FedProxy must use the original aggregation path with MA disabled")
    if errors:
        raise ValueError(f"Invalid ratio30 {method}/{task}:\n" + "\n".join(errors))


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except (OSError, ValueError):
        return False


def _check_link(path: Path, expected: Path) -> None:
    if (path.exists() or path.is_symlink()) and not _same_file(path, expected):
        raise ValueError(f"Existing artifact does not reference the selected input: {path} -> {expected}")


def _create_link(path: Path, target: Path) -> None:
    path.symlink_to(target.resolve(strict=True), target_is_directory=target.is_dir())


def _link(path: Path, target: Path) -> None:
    _check_link(path, target)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _create_link(path, target)


def check_partition(task: str, path: Path) -> dict:
    manifest = suite._json(path)
    clients = manifest.get("clients", {})
    if (manifest.get("scenario") != "homogeneous" or manifest.get("seed") != 42
            or set(manifest.get("dataset_fingerprints", {})) != {task}
            or set(clients) != {f"client_{i}" for i in range(4)}
            or not all(isinstance(values, list) for values in clients.values())):
        raise ValueError(f"Expected one task and four IID clients: {path}")
    indices = [index for values in clients.values() for index in values]
    sizes = [len(values) for values in clients.values()]
    if (not indices or len(indices) > 3000 or min(sizes) < 1 or max(sizes) - min(sizes) > 1
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError(f"Invalid sample cap or IID partition: {path}")
    if partition_hash({key: value for key, value in manifest.items() if key != "partition_hash"}) != manifest.get("partition_hash"):
        raise ValueError(f"Partition fingerprint does not match contents: {path}")
    return manifest


def check_shared(cfg: dict) -> dict:
    root = Path(cfg["run"]["output_dir"])
    compression = suite._json(root / "compression/compression_manifest.json")
    suite._weights_present(root / "compression/proxy")
    proxy_cfg = suite._json(root / "compression/proxy/config.json")
    kept = compression.get("kept_indices", [])
    expected_map = {str(index): block for index, block in enumerate(kept)}
    if (compression.get("original_layers") != 32 or compression.get("proxy_layers") != 23
            or proxy_cfg.get("num_hidden_layers") != 23
            or compression.get("requested_removed_block_ratio") != 0.30
            or len(kept) != 23 or kept != sorted(set(kept))
            or any(type(block) is not int or not 0 <= block < 32 for block in kept)
            or compression.get("layer_map") != expected_map
            or compression.get("actual_removed_block_ratio") != 1 - 23 / 32
            or not _same_file(Path(compression.get("source_model", "")), Path(cfg["model"]["name_or_path"]))):
        raise ValueError(f"Expected a new ratio30 23/32-layer proxy and matching layer map: {root}")
    if suite._yaml(root / "resolved_config.compress.yaml") != cfg:
        raise ValueError(f"Shared compression configuration changed: {root}")
    return compression


def _check_snapshots(cfg: dict, *, shared: bool = False) -> None:
    run = Path(cfg["run"]["output_dir"])
    names = ["execution_config.yaml", "resolved_config.train.yaml", "resolved_config.prepare-data.yaml"]
    if shared:
        names.append("resolved_config.compress.yaml")
    for name in names:
        path = run / name
        if path.is_file() and suite._yaml(path) != cfg:
            raise ValueError(f"Existing configuration changed; use another output root: {path}")
    if (run / "round_extension.json").exists():
        raise ValueError(f"This suite requires 10 rounds; a round extension exists: {run}")


def _check_run(cfg: dict, method: str, task: str, *, partition_validator=None) -> None:
    run = Path(cfg["run"]["output_dir"])
    _check_snapshots(cfg)
    if (run / "data_manifest.json").is_file():
        if partition_validator is None:
            check_partition(task, run / "data_manifest.json")
        else:
            partition_validator(run / "data_manifest.json")
    finished = suite.completed_rounds(run, cfg, ma=method == "ma")
    if finished:
        if not (run / "resolved_config.train.yaml").is_file():
            raise FileNotFoundError(f"Checkpoints require their original training snapshot: {run}")
        manifest = suite._json(run / "data_manifest.json")
        for number in range(1, finished + 1):
            checkpoint = run / "checkpoints" / f"round_{number:04d}"
            metadata = suite._json(checkpoint / "metadata.json")
            if (metadata.get("config_hash") != config_digest(cfg)
                    or metadata.get("partition_hash") != manifest["partition_hash"]
                    or metadata.get("source_model") != cfg["model"]["name_or_path"]):
                raise ValueError(f"Checkpoint config/data/model changed: {checkpoint}")
            if not (checkpoint / "conflict.safetensors").is_file():
                raise FileNotFoundError(f"Missing committed conflict state: {checkpoint}")


def preflight(root: Path, tasks: tuple, methods: tuple) -> tuple[dict, dict]:
    shared = shared_config(root)
    _check_snapshots(shared, shared=True)
    suite._weights_present(Path(shared["model"]["name_or_path"]))
    if suite._json(Path(shared["model"]["name_or_path"]) / "config.json").get("num_hidden_layers") != 32:
        raise ValueError("The approved ratio30 configuration requires the 32-layer LLaMA2 model")
    manifest_path = root / "shared/compression/compression_manifest.json"
    if manifest_path.is_file():
        check_shared(shared)
    configs = {}
    for task in tasks:
        pair = {method: task_config(method, task, root) for method in ("fedproxy", "ma")}
        for method, cfg in pair.items():
            check_contract(method, task, cfg)
        sections = ("model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties", "evaluation")
        if any(pair["fedproxy"][key] != pair["ma"][key] for key in sections):
            raise ValueError(f"FedProxy/MA input and training settings differ: {task}")
        for method in set(methods) | {"fedproxy"}:
            cfg = pair[method]
            _check_run(cfg, method, task)
            run = Path(cfg["run"]["output_dir"])
            _check_link(run / "compression", root / "shared/compression")
            if method == "ma":
                _check_link(run / "data_manifest.json", Path(pair["fedproxy"]["run"]["output_dir"]) / "data_manifest.json")
            configs[method, task] = cfg
    trained = any(list((Path(cfg["run"]["output_dir"]) / "checkpoints").glob("round_*")) for cfg in configs.values())
    if trained and not (root / "experiment_manifest.json").is_file():
        raise FileNotFoundError(f"Trained artifacts require the original input provenance: {root}/experiment_manifest.json")
    if not manifest_path.is_file() and (root / "shared/compression").exists():
        if trained or not (root / "shared/execution_config.yaml").is_file():
            raise ValueError("Unbound/incomplete shared compression; preserve it and use a new root or restore its manifest")
    return shared, configs


def check_provenance(root: Path, configs: dict, *, partition_validator=None) -> None:
    path = root / "experiment_manifest.json"
    if not path.is_file():
        return
    recorded = suite._json(path)
    compression = root / "shared/compression/compression_manifest.json"
    if (recorded.get("requested_remove_ratio") != 0.30 or recorded.get("proxy_layers") != 23
            or not compression.is_file()
            or recorded.get("compression_manifest_sha256") != hashlib.sha256(compression.read_bytes()).hexdigest()):
        raise ValueError(f"Shared compression provenance changed: {path}")
    for (method, task), cfg in configs.items():
        key = f"{method}/{task}"
        previous = recorded.get("config_hashes", {}).get(key)
        if previous is not None and previous != config_digest(cfg):
            raise ValueError(f"Recorded configuration changed for {key}: {path}")
        if previous is None and any((Path(cfg["run"]["output_dir"]) / "checkpoints").glob("round_*")):
            raise ValueError(f"Trained run lacks its original provenance for {key}: {path}")
        partition = Path(cfg["run"]["output_dir"]) / "data_manifest.json"
        expected = recorded.get("partitions", {}).get(task)
        if not partition.exists() and previous is None:
            continue  # A newly selected method will link to the already bound partition.
        if expected is not None:
            if not partition.is_file():
                raise ValueError(f"Recorded partition changed or is missing for {task}: {path}")
            manifest = check_partition(task, partition) if partition_validator is None else partition_validator(partition)
            if manifest["partition_hash"] != expected:
                raise ValueError(f"Recorded partition changed or is missing for {task}: {path}")


def ensure_shared(cfg: dict) -> None:
    run = Path(cfg["run"]["output_dir"])
    frozen = suite._freeze_config(cfg)
    if not (run / "compression/compression_manifest.json").is_file():
        suite._stage("compress", frozen, os.environ["FEDPROXY_GPU_IDS"])
    check_shared(cfg)


def ensure_inputs(task: str, configs: dict, shared: dict, gpu_ids: str, methods: tuple) -> None:
    base = configs["fedproxy", task]
    run = Path(base["run"]["output_dir"])
    frozen = suite._freeze_config(base)
    _link(run / "compression", Path(shared["run"]["output_dir"]) / "compression")
    if not (run / "data_manifest.json").is_file():
        suite._stage("prepare", frozen, gpu_ids)
    check_partition(task, run / "data_manifest.json")
    if not any((run / filename).is_file() for filename in
               ("resolved_config.train.yaml", "resolved_config.prepare-data.yaml")):
        raise FileNotFoundError(f"Partition requires its original preparation snapshot: {run}")
    if "ma" in methods:
        cfg = configs["ma", task]
        suite._freeze_config(cfg)
        check_reuse_compatibility(cfg, run)
        ma_run = Path(cfg["run"]["output_dir"])
        _link(ma_run / "compression", run / "compression")
        _link(ma_run / "data_manifest.json", run / "data_manifest.json")


def record_inputs(root: Path, configs: dict, *, partition_validator=None) -> None:
    path = root / "experiment_manifest.json"
    value = {"requested_remove_ratio": 0.30, "proxy_layers": 23,
             "compression_manifest_sha256": hashlib.sha256(
                 (root / "shared/compression/compression_manifest.json").read_bytes()).hexdigest(),
             "partitions": {}, "config_hashes": {}}
    for (method, task), cfg in configs.items():
        partition_path = Path(cfg["run"]["output_dir"]) / "data_manifest.json"
        manifest = check_partition(task, partition_path) if partition_validator is None else partition_validator(partition_path)
        value["partitions"][task] = manifest["partition_hash"]
        value["config_hashes"][f"{method}/{task}"] = config_digest(cfg)
    if path.is_file():
        previous = suite._json(path)
        for key in ("requested_remove_ratio", "proxy_layers", "compression_manifest_sha256"):
            if value[key] != previous.get(key):
                raise ValueError(f"Shared input provenance changed in {key}: {path}")
        for field in ("partitions", "config_hashes"):
            for key, old in previous[field].items():
                if key in value[field] and value[field][key] != old:
                    raise ValueError(f"Input provenance changed for {field}.{key}: {path}")
                value[field][key] = old
    temporary = path.with_suffix(".json.tmp")
    suite._write(temporary, value)
    os.replace(temporary, path)


def summarize(root: Path, method: str) -> dict:
    folder = root / FOLDERS[method]
    check_shared(shared_config(root))
    configs = {}
    for task in TASKS:
        cfg = suite._yaml(folder / suite.run_name(task) / "execution_config.yaml")
        check_contract(method, task, cfg)
        _check_run(cfg, method, task)
        _check_link(Path(cfg["run"]["output_dir"]) / "compression", root / "shared/compression")
        configs[method, task] = cfg
    if not (root / "experiment_manifest.json").is_file():
        raise FileNotFoundError(f"Missing original input provenance: {root}/experiment_manifest.json")
    check_provenance(root, configs)
    return suite.write_suite_summary(folder, ma=method == "ma")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ratio30 IID FedProxy and MA scheme-B suite")
    parser.add_argument("action", choices=("validate", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--method", choices=("both", "fedproxy", "ma"), default="both")
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--gpu-ids", default="0,1")
    args = parser.parse_args(argv)
    methods = tuple(FOLDERS) if args.method == "both" else (args.method,)
    tasks = TASKS if args.task == "all" else (args.task,)
    os.chdir(PROJECT)
    if args.action == "summarize":
        for method in methods:
            summarize(args.root, method)
        return
    if not re.fullmatch(r"[0-9]+,[0-9]+", args.gpu_ids) or len(set(args.gpu_ids.split(","))) != 2:
        parser.error("Two distinct GPU IDs are required, e.g. --gpu-ids 2,3")
    if args.action == "validate":
        _, configs = preflight(args.root, tasks, methods)
        check_provenance(args.root, configs)
        print(f"[ratio30] validated methods={','.join(methods)} tasks={len(tasks)} proxy_layers=23 MA=17/14/1 q_block=2", flush=True)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_ids, FEDPROXY_GPU_IDS=args.gpu_ids,
                      PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    os.environ[REQUIRED_PARALLEL_ENV] = "2"
    os.environ["PYTHONPATH"] = str(PROJECT / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    with suite._suite_lock(args.root):
        shared, configs = preflight(args.root, tasks, methods)
        check_provenance(args.root, configs)
        import torch

        resolve_parallelism(shared["federated"], 4, torch.cuda.device_count())
        for cfg in configs.values():
            suite._freeze_config(cfg)
        ensure_shared(shared)
        for task in tasks:
            ensure_inputs(task, configs, shared, args.gpu_ids, methods)
        record_inputs(args.root, configs)
        for method in methods:
            for task in tasks:
                cfg = configs[method, task]
                print(f"[ratio30] method={method} task={task} GPUs={args.gpu_ids} output={cfg['run']['output_dir']}", flush=True)
                suite.execute_task(task, cfg, args.gpu_ids, ma=method == "ma", inputs_ready=True)
            if args.task == "all":
                summarize(args.root, method)
        print("[ratio30] selected experiments completed", flush=True)


if __name__ == "__main__":
    main()
