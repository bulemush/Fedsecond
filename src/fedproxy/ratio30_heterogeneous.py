"""Full pipeline for two global models trained on eight task-level non-IID clients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

from fedproxy import ratio30 as common
from fedproxy.config import load_config
from fedproxy.data.partition import partition_hash
from fedproxy.evaluation.report import summarize as summarize_scores
from fedproxy.ma import homogeneous as suite
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, resolve_parallelism
from fedproxy.ma.report import summarize_run
from fedproxy.ma.scheduler import audit_residual_delivery
from fedproxy.reuse import check_reuse_compatibility


ROOT = Path("runs/heterogeneous_ratio30")
TASKS = common.TASKS
KEY = "heterogeneous"
HARNESS_TASKS = ["openbookqa" if task == "obqa" else task for task in TASKS]


def method_config(method: str, root: Path = ROOT) -> dict:
    filename = "configs/ma/heterogeneous_ratio30.yaml" if method == "ma" else "configs/experiments/heterogeneous_ratio30.yaml"
    cfg = load_config(common.PROJECT / filename)
    cfg["run"]["output_dir"] = str(root / common.FOLDERS[method])
    if method == "ma":
        cfg["run"]["reuse_artifacts_from"] = str(root / "FedProxy")
    else:
        cfg["run"]["reuse_compression_from"] = str(root / "shared")
    return cfg


def shared_config(root: Path = ROOT) -> dict:
    cfg = load_config(common.PROJECT / "configs/experiments/heterogeneous_ratio30_shared.yaml")
    cfg["run"]["output_dir"] = str(root / "shared")
    return cfg


def check_partition(path: Path) -> dict:
    manifest = suite._json(path)
    clients = manifest.get("clients", {})
    if (manifest.get("scenario") != "heterogeneous" or manifest.get("seed") != 42
            or set(manifest.get("dataset_fingerprints", {})) != set(TASKS)
            or set(clients) != {f"client_{i}" for i in range(8)}):
        raise ValueError(f"Expected eight task-level non-IID clients: {path}")
    for index, task in enumerate(TASKS):
        assignment = clients[f"client_{index}"]
        if not isinstance(assignment, dict) or assignment.get("task") != task:
            raise ValueError(f"Incorrect task assignment for client_{index}: {path}")
        indices = assignment.get("indices")
        if (not isinstance(indices, list) or not 1 <= len(indices) <= 3000
                or any(type(item) is not int or item < 0 for item in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"Invalid or overlapping sample indices for {task}: {path}")
    # Index 0 in two different tasks identifies different examples and is valid.
    if partition_hash({key: value for key, value in manifest.items() if key != "partition_hash"}) != manifest.get("partition_hash"):
        raise ValueError(f"Partition fingerprint does not match contents: {path}")
    return manifest


def preflight(root: Path, methods: tuple) -> tuple[dict, dict]:
    shared = shared_config(root)
    common._check_snapshots(shared, shared=True)
    suite._weights_present(Path(shared["model"]["name_or_path"]))
    if suite._json(Path(shared["model"]["name_or_path"]) / "config.json").get("num_hidden_layers") != 32:
        raise ValueError("The ratio30 profile requires the 32-layer LLaMA2 model")
    manifest = root / "shared/compression/compression_manifest.json"
    if manifest.is_file():
        common.check_shared(shared)
    pair = {method: method_config(method, root) for method in common.FOLDERS}
    for method, cfg in pair.items():
        common.check_contract(method, KEY, cfg, heterogeneous=True)
    sections = ("model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties", "evaluation")
    if any(pair["fedproxy"][key] != pair["ma"][key] for key in sections):
        raise ValueError("FedProxy/MA input and training settings differ")
    if any(shared[key] != pair["fedproxy"][key] for key in sections):
        raise ValueError("Shared proxy configuration differs from the method configurations")
    audit_residual_delivery(SimpleNamespace(blocks=tuple(range(23))),
                            [f"client_{index}" for index in range(8)], 10, pair["ma"])
    configs = {}
    for method in set(methods) | {"fedproxy"}:
        cfg = pair[method]
        common._check_run(cfg, method, KEY, partition_validator=check_partition)
        run = Path(cfg["run"]["output_dir"])
        common._check_link(run / "compression", root / "shared/compression")
        if method == "ma":
            common._check_link(run / "data_manifest.json", root / "FedProxy/data_manifest.json")
        configs[method, KEY] = cfg
    trained = any(any((Path(cfg["run"]["output_dir"]) / "checkpoints").glob("round_*")) for cfg in configs.values())
    if trained and not (root / "experiment_manifest.json").is_file():
        raise FileNotFoundError(f"Trained artifacts require their original input provenance: {root}/experiment_manifest.json")
    if not manifest.is_file() and (root / "shared/compression").exists():
        if trained or not (root / "shared/execution_config.yaml").is_file():
            raise ValueError("Unbound/incomplete shared compression; restore its manifest or choose a new root")
    common.check_provenance(root, configs, partition_validator=check_partition)
    return shared, configs


def ensure_inputs(shared: dict, configs: dict, methods: tuple, gpu_ids: str) -> None:
    common.ensure_shared(shared)
    base = configs["fedproxy", KEY]
    run = Path(base["run"]["output_dir"])
    frozen = suite._freeze_config(base)
    common._link(run / "compression", Path(shared["run"]["output_dir"]) / "compression")
    if not (run / "data_manifest.json").is_file():
        suite._stage("prepare", frozen, gpu_ids)
    check_partition(run / "data_manifest.json")
    if not any((run / name).is_file() for name in ("resolved_config.train.yaml", "resolved_config.prepare-data.yaml")):
        raise FileNotFoundError(f"Partition requires its original preparation snapshot: {run}")
    if "ma" in methods:
        cfg = configs["ma", KEY]
        ma_run = Path(cfg["run"]["output_dir"])
        suite._freeze_config(cfg)
        check_reuse_compatibility(cfg, run)
        common._link(ma_run / "compression", run / "compression")
        common._link(ma_run / "data_manifest.json", run / "data_manifest.json")


def verified_result(cfg: dict, method: str) -> dict:
    run = Path(cfg["run"]["output_dir"])
    common.check_contract(method, KEY, cfg, heterogeneous=True)
    if suite._yaml(run / "resolved_config.train.yaml") != cfg:
        raise ValueError(f"Saved training configuration differs: {run}")
    common._check_run(cfg, method, KEY, partition_validator=check_partition)
    if suite.completed_rounds(run, cfg, ma=method == "ma") != 10:
        raise ValueError(f"Not all 10 rounds are complete: {run}")
    evaluation = suite._json(run / "evaluation/fused_summary.json")
    scores = evaluation.get("tasks", {})
    if (evaluation.get("model_kind") != "fused" or evaluation.get("synthetic")
            or not isinstance(scores, dict) or set(scores) != set(TASKS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                   for value in scores.values())):
        raise ValueError(f"Expected real fused evaluation on all eight tasks: {run}")
    expected = summarize_scores(scores)
    for key in ("qa_avg", "glue_avg", "all_avg"):
        value = evaluation.get(key)
        if type(value) not in (int, float) or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Invalid {key} in eight-task evaluation: {run}")
    protocol = evaluation.get("protocol", {})
    if (not isinstance(protocol, dict) or protocol.get("tasks") != HARNESS_TASKS or protocol.get("num_fewshot") != 0
            or protocol.get("batch_size") != 1 or protocol.get("primary_metric") != "acc"
            or protocol.get("dtype") not in ("float16", "bfloat16", "float32")
            or evaluation.get("protocol_fingerprint") != hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()):
        raise ValueError(f"Evaluation protocol differs from the approved eight-task protocol: {run}")
    suite._weights_present(run / "exports/llm_fused")
    result = summarize_run(run)
    if not result["evaluation_checkpoint_verified"]:
        raise ValueError(f"Fused evaluation is not bound to its final checkpoint: {run}")
    return {**result, "scenario": KEY, "num_clients": 8, "proxy_layers": 23,
            "requested_remove_ratio": 0.30, "local_epochs": 5, "max_input_length": 320,
            "max_target_length": 128, "effective_batch_size": 16}


def write_summary(cfg: dict, method: str) -> dict:
    result = verified_result(cfg, method)
    run = Path(cfg["run"]["output_dir"])
    suite._write(run / "result.json", result)
    suite._write(run / "summary.json", result)
    with (run / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "accuracy", "accuracy_percent"))
        writer.writeheader()
        rows = {**result["tasks"], **{key: result[key] for key in ("qa_avg", "glue_avg", "all_avg")}}
        for key, value in rows.items():
            writer.writerow({"metric": key, "accuracy": value, "accuracy_percent": 100 * value})
    print(json.dumps({"method": method, **{key: result[key] for key in ("qa_avg", "glue_avg", "all_avg")}}, ensure_ascii=False), flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ratio30 task-level non-IID FedProxy and MA full pipeline")
    parser.add_argument("action", choices=("validate", "run", "summarize"))
    parser.add_argument("--method", choices=("both", "fedproxy", "ma"), default="both")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    args = parser.parse_args(argv)
    methods = tuple(common.FOLDERS) if args.method == "both" else (args.method,)
    os.chdir(common.PROJECT)
    if args.action == "summarize":
        _, configs = preflight(args.root, methods)
        if not (args.root / "experiment_manifest.json").is_file():
            raise FileNotFoundError(f"Missing original input provenance: {args.root}/experiment_manifest.json")
        for method in methods:
            write_summary(configs[method, KEY], method)
        return
    if not re.fullmatch(r"[0-9]+,[0-9]+", args.gpu_ids) or len(set(args.gpu_ids.split(","))) != 2:
        parser.error("Two distinct GPU IDs are required, e.g. --gpu-ids 2,3")
    if args.action == "validate":
        preflight(args.root, methods)
        print(f"[ratio30-heterogeneous] validated methods={','.join(methods)} clients=8 tasks=8 rounds=10 epochs=5 MA=17/14/1", flush=True)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_ids, FEDPROXY_GPU_IDS=args.gpu_ids,
                      PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    os.environ[REQUIRED_PARALLEL_ENV] = "2"
    os.environ["PYTHONPATH"] = str(common.PROJECT / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    with suite._suite_lock(args.root, label="ratio30 heterogeneous"):
        shared, configs = preflight(args.root, methods)
        import torch

        resolve_parallelism(shared["federated"], 8, torch.cuda.device_count())
        for cfg in configs.values():
            suite._freeze_config(cfg)
        ensure_inputs(shared, configs, methods, args.gpu_ids)
        common.record_inputs(args.root, configs, partition_validator=check_partition)
        for method in methods:
            cfg = configs[method, KEY]
            print(f"[ratio30-heterogeneous] method={method} GPUs={args.gpu_ids} clients=8 waves=4 output={cfg['run']['output_dir']}", flush=True)
            suite.execute_task(KEY, cfg, args.gpu_ids, ma=method == "ma", inputs_ready=True,
                               result_reader=lambda cfg=cfg, method=method: verified_result(cfg, method))
            write_summary(cfg, method)
        print("[ratio30-heterogeneous] selected methods completed", flush=True)


if __name__ == "__main__":
    main()
