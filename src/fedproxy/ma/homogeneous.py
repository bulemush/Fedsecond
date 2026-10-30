"""Resumable single-task IID MA training and evaluation using existing dependencies."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

from fedproxy.config import load_config, save_resolved_config, validate_config
from fedproxy.data.partition import partition_hash
from fedproxy.evaluation.report import QA_TASKS, GLUE_TASKS, summarize
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, device_assignment, resolve_parallelism
from fedproxy.ma.report import summarize_run
from fedproxy.reuse import check_reuse_compatibility, normalized_reuse_sections


PROJECT = Path(__file__).resolve().parents[3]
TASKS = QA_TASKS + GLUE_TASKS
DEFAULT_ROOT = Path("runs/MA_homogeneous")
_SHARED = ("lora", "training", "federated", "pcr", "h_ties", "evaluation")


def run_name(task: str) -> str:
    return f"llama2_homogeneous_{task}_len320_mb4_seed42"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _differences(actual, expected, prefix="") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [item for key in sorted(actual.keys() | expected.keys())
                for item in _differences(actual.get(key), expected.get(key), f"{prefix}.{key}".strip("."))]
    return [] if actual == expected else [f"{prefix}: MA={actual!r}, baseline={expected!r}"]


def _shared(cfg: dict) -> dict:
    result = {**normalized_reuse_sections(cfg), **{key: cfg[key] for key in _SHARED},
              "run.seed": cfg["run"]["seed"]}
    # Reusing the initial proxy/partition is independent of the source run's
    # local epochs. The MA budget is enforced separately by _check_contract.
    result["training"] = {key: value for key, value in cfg["training"].items() if key != "local_epochs"}
    return result


def task_config(task: str, root: Path = DEFAULT_ROOT, baseline_root: Path | None = None) -> dict:
    cfg = load_config(PROJECT / "configs" / "ma" / "homogeneous" / f"{task}.yaml")
    cfg["run"]["output_dir"] = str(root / run_name(task))
    if baseline_root is not None:
        cfg["run"]["reuse_artifacts_from"] = str(baseline_root / run_name(task))
    return cfg


def _check_contract(task: str, cfg: dict) -> None:
    validate_config(cfg, "train")
    wanted = {
        "data": {"scenario": "homogeneous", "tasks": [task], "num_clients": 4,
                 "max_train_samples_per_task": 3000, "max_input_length": 320, "max_target_length": 128,
                 "partition_seed": 42},
        "training": {"local_epochs": 10, "micro_batch_size": 4,
                     "gradient_accumulation_steps": 4, "effective_batch_size": 16},
        "federated": {"rounds": 10, "participation": "full",
                      "client_execution": "parallel", "max_parallel_clients": 0},
        "evaluation": {"harness_tasks": ["openbookqa" if task == "obqa" else task], "num_fewshot": 0},
        "run": {"seed": 42},
    }
    errors = [f"{section}.{key} must be {value!r}" for section, fields in wanted.items()
              for key, value in fields.items() if cfg[section].get(key) != value]
    if cfg.get("ma") != load_config(PROJECT / "configs/ma/base.yaml")["ma"]:
        errors.append("ma must exactly match configs/ma/base.yaml")
    if errors:
        raise ValueError(f"Invalid homogeneous experiment {task}:\n" + "\n".join(errors))


def _weights_present(directory: Path) -> None:
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Model config.json not found: {directory}")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        if (directory / filename).is_file() and (directory / filename).stat().st_size:
            return
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / filename
        if index.is_file():
            shards = set(_json(index).get("weight_map", {}).values())
            if shards and all((directory / shard).is_file() and (directory / shard).stat().st_size
                              for shard in shards):
                return
    raise FileNotFoundError(f"Model weights or referenced shards are missing: {directory}")


def completed_rounds(run: Path, cfg: dict, ma: bool) -> int:
    paths = sorted((run / "checkpoints").glob("round_*"))
    if len(paths) > int(cfg["federated"]["rounds"]):
        raise ValueError(f"Too many checkpoints: {run}")
    invariants = {}
    client_ids = set(_json(run / "data_manifest.json")["clients"]) if paths else set()
    if paths and ma:
        invariants = {
            "partition_hash": _json(run / "data_manifest.json")["partition_hash"],
            "compression_manifest_hash": hashlib.sha256(
                (run / "compression/compression_manifest.json").read_bytes()).hexdigest(),
        }
    for number, checkpoint in enumerate(paths, 1):
        meta = _json(checkpoint / "metadata.json")
        if checkpoint.name != f"round_{number:04d}" or not meta.get("complete") or meta.get("completed_round") != number:
            raise ValueError(f"Incomplete/non-contiguous checkpoint: {checkpoint}")
        if (meta.get("ma_protocol_version") == 1) != ma:
            raise ValueError(f"Unexpected checkpoint method: {checkpoint}")
        if set(meta.get("client_metrics", {})) != client_ids or len(client_ids) != int(cfg["data"]["num_clients"]):
            raise ValueError(f"Checkpoint participants differ from the configured partition: {checkpoint}")
        if not (checkpoint / "adapter.safetensors").is_file():
            raise FileNotFoundError(f"Missing adapter: {checkpoint}")
        if ma:
            digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            if meta.get("config_hash") != digest:
                raise ValueError(f"Checkpoint configuration differs from the requested run: {checkpoint}")
            if not (checkpoint / "conflict.safetensors").is_file():
                raise FileNotFoundError(f"Missing conflict state: {checkpoint}")
            for key, value in invariants.items():
                if meta.get(key) != value:
                    raise ValueError(f"Checkpoint {key} differs from the reused artifacts: {checkpoint}")
    if ma and paths and len(paths) < int(cfg["federated"]["rounds"]):
        states = _json(paths[-1] / "metadata.json").get("client_states", {})
        if set(states) != client_ids or not all(
            Path(states[client]).is_file() and Path(states[client]).resolve() == (
                run / "clients" / client / paths[-1].name / "residual.pt").resolve()
            for client in client_ids
        ):
            raise FileNotFoundError(f"Committed client residual states required for resume are missing: {paths[-1]}")
    return len(paths)


def task_result(task: str, run: Path, ma: bool) -> dict:
    cfg = _yaml(run / "resolved_config.train.yaml")
    if cfg["data"]["scenario"] != "homogeneous" or cfg["data"]["tasks"] != [task]:
        raise ValueError(f"Wrong single-task configuration: {run}")
    if cfg["run"].get("synthetic", False):
        raise ValueError(f"Synthetic run cannot be reported as an experiment result: {run}")
    if completed_rounds(run, cfg, ma) != int(cfg["federated"]["rounds"]):
        raise ValueError(f"Not all configured rounds are complete: {run}")
    evaluation = _json(run / "evaluation/fused_summary.json")
    score = evaluation.get("task_accuracy")
    if (evaluation.get("model_kind") != "fused" or evaluation.get("synthetic")
            or evaluation.get("task") != task or set(evaluation.get("tasks", {})) != {task}
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
            or evaluation["tasks"][task] != score):
        raise ValueError(f"Invalid single-task fused evaluation: {run}")
    _weights_present(run / "exports/llm_fused")
    result = summarize_run(run)  # Checks final-checkpoint identity, eval binding and cost provenance.
    return {**result, "task": task, "task_accuracy": score,
            "local_epochs": cfg["training"]["local_epochs"],
            "max_input_length": cfg["data"]["max_input_length"]}


def preflight(task: str, config: dict) -> dict:
    """Return a frozen effective config; read-only, including old-path verification."""
    cfg = copy.deepcopy(config)
    _check_contract(task, cfg)
    source = Path(cfg["run"]["reuse_artifacts_from"])
    run = Path(cfg["run"]["output_dir"])
    if source.resolve() == run.resolve():
        raise ValueError("The MA output cannot be its own baseline")
    original = _yaml(source / "resolved_config.train.yaml")
    if original.get("ma", {}).get("enabled", False):
        raise ValueError(f"Expected an original FedProxy baseline: {source}")
    # Only the already declared old->new model mapping may be auto-selected.
    known_move = load_config(PROJECT / "configs/ma/base.yaml")["run"].get("reuse_model_relocation")
    if cfg["run"].get("reuse_model_relocation") is None and known_move:
        old = original["model"]
        if (old["name_or_path"] == known_move["original"]
                and (old.get("tokenizer_name_or_path") or old["name_or_path"]) == known_move["original"]
                and cfg["model"]["name_or_path"] == known_move["current"]
                and (cfg["model"].get("tokenizer_name_or_path") or cfg["model"]["name_or_path"]) == known_move["current"]):
            cfg["run"]["reuse_model_relocation"] = copy.deepcopy(known_move)
    errors = _differences(_shared(cfg), _shared(original))
    if errors:
        raise ValueError(f"Baseline settings differ for {task} ({source}):\n" + "\n".join(errors))
    check_reuse_compatibility(cfg, source)
    _weights_present(Path(cfg["model"]["name_or_path"]))
    _weights_present(source / "compression/proxy")
    compression = _json(source / "compression/compression_manifest.json")
    if compression["proxy_layers"] != 16 or _json(source / "compression/proxy/config.json").get("num_hidden_layers") != 16:
        raise ValueError(f"This approved MA configuration expects 16 proxy layers: {source}")
    manifest = _json(source / "data_manifest.json")
    clients = manifest["clients"]
    if (manifest["scenario"] != "homogeneous" or manifest.get("seed") != 42
            or set(manifest.get("dataset_fingerprints", {})) != {task}
            or len(clients) != 4 or not all(isinstance(v, list) for v in clients.values())):
        raise ValueError(f"Expected a four-client IID partition: {source}")
    indices = [index for values in clients.values() for index in values]
    sizes = [len(v) for v in clients.values()]
    if (not indices or len(indices) > 3000 or any(not isinstance(i, int) or i < 0 for i in indices)
            or len(set(indices)) != len(indices) or min(sizes) < 1 or max(sizes) - min(sizes) > 1):
        raise ValueError(f"Invalid sample cap or overlapping/unbalanced IID partition: {source}")
    if partition_hash({k: v for k, v in manifest.items() if k != "partition_hash"}) != manifest["partition_hash"]:
        raise ValueError(f"Partition fingerprint does not match its contents: {source}")
    for filename in ("execution_config.yaml", "resolved_config.train.yaml"):
        path = run / filename
        if path.is_file() and _yaml(path) != cfg:
            raise ValueError(f"Existing MA settings changed; use a new output directory: {path}")
    for name in ("compression", "data_manifest.json"):
        path = run / name
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != (source / name).resolve():
                raise ValueError(f"Existing MA artifact does not link to the selected baseline: {path}")
    completed_rounds(run, cfg, ma=True)
    return cfg


def verified_task_result(task: str, cfg: dict) -> dict:
    """Validate only this MA run, without reading any FedProxy evaluation."""
    run = Path(cfg["run"]["output_dir"])
    saved = _yaml(run / "resolved_config.train.yaml")
    if saved != cfg:
        raise ValueError(f"Saved MA configuration differs from the selected experiment: {run}")
    return task_result(task, run, ma=True)


def collect_suite(root: Path = DEFAULT_ROOT) -> dict:
    results = []
    ma_settings = None
    for task in TASKS:
        cfg = _yaml(root / run_name(task) / "execution_config.yaml")
        if ma_settings is not None and cfg["ma"] != ma_settings:
            raise ValueError("Cannot combine tasks trained with different MA configurations")
        ma_settings = cfg["ma"]
        results.append(verified_task_result(task, cfg))
    scores = summarize({item["task"]: item["task_accuracy"] for item in results})
    return {"method": "fedproxy_ma", "scenario": "homogeneous", "model_kind": "fused",
            "independent_models": 8, **scores, "task_results": results,
            "inference": "Single-seed descriptive results from eight independently trained MA models."}


def write_suite_summary(root: Path) -> dict:
    result = collect_suite(root)
    _write(root / "summary.json", result)
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("task", "accuracy", "accuracy_percent", "rounds", "local_epochs",
                  "uplink_bytes", "downlink_bytes", "total_communication_bytes",
                  "sum_client_train_seconds", "ma_round_wall_seconds")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result["task_results"]:
            writer.writerow({"accuracy": item["task_accuracy"], "accuracy_percent": 100 * item["task_accuracy"],
                             **{key: item[key] for key in fields if key not in {"accuracy", "accuracy_percent"}}})
    print(json.dumps({key: result[key] for key in ("qa_avg", "glue_avg", "all_avg")}, ensure_ascii=False), flush=True)
    return result


@contextmanager
def _suite_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".suite.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another MA homogeneous suite is running under {root}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stage(stage: str, config_path: Path, gpu_ids: str) -> None:
    print(f"[ma-homogeneous] stage={stage} config={config_path}", flush=True)
    subprocess.run(["bash", str(PROJECT / "scripts/run_experiment.sh"), str(config_path), gpu_ids, stage],
                   check=True, cwd=PROJECT)


def _freeze_config(cfg: dict) -> Path:
    run = Path(cfg["run"]["output_dir"])
    run.mkdir(parents=True, exist_ok=True)
    frozen = run / "execution_config.yaml"
    if frozen.exists() and _yaml(frozen) != cfg:
        raise ValueError(f"Refusing to change the frozen experiment configuration: {frozen}")
    if not frozen.exists():
        save_resolved_config(cfg, frozen)
    return frozen


def execute_task(task: str, cfg: dict, gpu_ids: str) -> dict:
    run = Path(cfg["run"]["output_dir"])
    frozen = _freeze_config(cfg)
    if (run / "evaluation/fused_summary.json").is_file():
        try:
            result = verified_task_result(task, cfg)
        except (ValueError, FileNotFoundError, KeyError) as error:
            print(f"[ma-homogeneous] Completing/refreshing final fusion and evaluation: {error}", flush=True)
        else:
            print(f"[ma-homogeneous] Already complete and verified: {task}", flush=True)
            _write(run / "result.json", result)
            return result
    _stage("reuse-audit", frozen, gpu_ids)
    finished = completed_rounds(run, cfg, ma=True)
    if finished < int(cfg["federated"]["rounds"]):
        _stage("dry-run", frozen, gpu_ids)
        _stage("resume" if finished else "train", frozen, gpu_ids)
    final = run / "checkpoints" / f"round_{int(cfg['federated']['rounds']):04d}"
    fusion_path = run / "exports/llm_fused/fusion_manifest.json"
    valid_fusion = False
    if fusion_path.is_file():
        try:
            valid_fusion = Path(_json(fusion_path)["checkpoint"]).resolve() == final.resolve()
            _weights_present(run / "exports/llm_fused")
        except (ValueError, FileNotFoundError, KeyError):
            valid_fusion = False
    if not valid_fusion:
        print(f"[ma-homogeneous] stage=fuse task={task}", flush=True)
        command = [sys.executable, "-m", "fedproxy.cli", "fuse", "--config", str(frozen), "--checkpoint", str(final)]
        if (run / "exports").exists():
            command.append("--overwrite")
        subprocess.run(command, check=True, cwd=PROJECT)
    _stage("evaluate-fused", frozen, gpu_ids)
    result = verified_task_result(task, cfg)
    _write(run / "result.json", result)
    print(f"[ma-homogeneous] evaluated task={task} accuracy={result['task_accuracy']:.6f}", flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate eight four-client IID MA models")
    parser.add_argument("action", choices=("validate", "run", "summarize"))
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--gpu-ids", default="0,1")
    args = parser.parse_args(argv)
    os.chdir(PROJECT)
    if args.action == "summarize":
        write_suite_summary(args.root)
        return
    tasks = TASKS if args.task == "all" else (args.task,)
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpu_ids) or len(set(args.gpu_ids.split(","))) != len(args.gpu_ids.split(",")):
        parser.error("--gpu-ids must contain distinct comma-separated physical GPU IDs")
    if args.action == "run":
        if len(args.gpu_ids.split(",")) != 2:
            parser.error("The MA homogeneous suite requires exactly two GPU IDs, e.g. --gpu-ids 0,1")
        # Set visibility before preflight can import any optional CUDA dependencies.
        # This runtime constraint does not change frozen configs or checkpoint hashes.
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_ids, FEDPROXY_GPU_IDS=args.gpu_ids,
                          PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        os.environ[REQUIRED_PARALLEL_ENV] = "2"
    # Complete every selected preflight before writing task outputs.
    configs = {task: preflight(task, task_config(task, args.root, args.baseline_root)) for task in tasks}
    for task, cfg in configs.items():
        print(f"[ma-homogeneous] validated task={task} epochs={cfg['training']['local_epochs']} clients=4 output={cfg['run']['output_dir']}", flush=True)
    if args.action == "validate":
        return
    if any(cfg["ma"] != next(iter(configs.values()))["ma"] for cfg in configs.values()):
        raise ValueError("All selected tasks must use identical MA configurations")
    import torch

    for cfg in configs.values():
        resolve_parallelism(cfg["federated"], cfg["data"]["num_clients"], torch.cuda.device_count())
    print(f"[ma-homogeneous] parallel_clients=2 {device_assignment(0)}, {device_assignment(1)}", flush=True)
    os.environ["PYTHONPATH"] = str(PROJECT / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    with _suite_lock(args.root):
        for cfg in configs.values():
            _freeze_config(cfg)
        for task, cfg in configs.items():
            execute_task(task, cfg, args.gpu_ids)
        if all((args.root / run_name(task) / "evaluation/fused_summary.json").is_file() for task in TASKS):
            write_suite_summary(args.root)


if __name__ == "__main__":
    main()
