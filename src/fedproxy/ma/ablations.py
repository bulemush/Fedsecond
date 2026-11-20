"""Four single-seed MA ablations derived from a completed ratio30 run."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

from fedproxy.config import load_config, validate_config
from fedproxy.data.partition import partition_hash
from fedproxy.evaluation.report import GLUE_TASKS, QA_TASKS, summarize
from fedproxy.ma import homogeneous as suite
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, resolve_parallelism
from fedproxy.ma.report import summarize_run
from fedproxy.ma.scheduler import plan_round
from fedproxy.ma.schema import BlockRegistry


GROUPS = {
    "A1": {"ma.aggregation.similarity_scope": "union",
           "ma.aggregation.merge_rule": "observed_mean"},
    "A2": {"ma.residual.enabled": False},
    "A3": {"pcr.lambda_reg": 0.0},
    "A4": {"ma.compression.selection": "random"},
}
LABELS = {
    "MA": "完整 MA", "A1": "共同观测与主导符号融合",
    "A2": "时间戳残差", "A3": "可靠冲突反馈与 PCR", "A4": "幅值 Top-k",
}
TASKS = QA_TASKS + GLUE_TASKS
DEFAULT_REFERENCE = Path("runs/heterogeneous_ratio30/MA")
DEFAULT_ROOT = Path("runs/heterogeneous_ratio30/ablations")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def schedule_records(registry: BlockRegistry, clients: list[str], cfg: dict) -> list[dict]:
    return [
        {client: {"train": sorted(plan.train_blocks), "upload": sorted(plan.upload_blocks),
                  "anchor": sorted(plan.anchor_blocks)}
         for client, plan in plan_round(registry, clients, number, cfg).items()}
        for number in range(int(cfg["federated"]["rounds"]))
    ]


def verify_schedules(run: Path, expected: list[dict], *, completed: int) -> None:
    for number in range(1, completed + 1):
        path = run / "schedules" / f"round_{number:04d}.json"
        if suite._json(path) != expected[number - 1]:
            raise ValueError(f"Training/upload/anchor schedule differs from reference: {path}")


def verified_scores(run: Path, cfg: dict) -> dict:
    if suite._yaml(run / "resolved_config.train.yaml") != cfg:
        raise ValueError(f"Training snapshot differs from the frozen configuration: {run}")
    if suite.completed_rounds(run, cfg, ma=True) != 10:
        raise ValueError(f"Expected ten completed MA rounds: {run}")
    result = summarize_run(run)
    evaluation = suite._json(run / "evaluation/fused_summary.json")
    scores = evaluation.get("tasks", {})
    if (evaluation.get("model_kind") != "fused" or evaluation.get("synthetic")
            or set(scores) != set(TASKS)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in scores.values())):
        raise ValueError(f"Expected real final fused scores on all eight tasks: {run}")
    for name, value in summarize(scores).items():
        if name != "tasks" and not math.isclose(evaluation.get(name, -1), value, abs_tol=1e-12):
            raise ValueError(f"Incorrect {name} in fused summary: {run}")
    protocol = evaluation.get("protocol", {})
    expected_tasks = ["openbookqa" if task == "obqa" else task for task in TASKS]
    if (protocol.get("tasks") != expected_tasks or protocol.get("num_fewshot") != 0
            or protocol.get("batch_size") != 1 or protocol.get("primary_metric") != "acc"
            or evaluation.get("protocol_fingerprint") != hashlib.sha256(
                json.dumps(protocol, sort_keys=True).encode()).hexdigest()):
        raise ValueError(f"Unexpected eight-task evaluation protocol: {run}")
    return result


def reference_context(reference: Path) -> dict:
    """Read-only validation of the existing complete MA and all ten layer plans."""
    reference = reference.resolve()
    snapshot = reference / "resolved_config.train.yaml"
    cfg = load_config(snapshot)
    expected = {
        "run": {"seed": 42}, "compression": {"remove_ratio": 0.30},
        "data": {"scenario": "heterogeneous", "num_clients": 8, "tasks": list(TASKS),
                 "max_train_samples_per_task": 3000, "max_input_length": 320},
        "training": {"local_epochs": 5, "effective_batch_size": 16,
                     "micro_batch_size": 4, "gradient_accumulation_steps": 4},
        "federated": {"rounds": 10, "participation": "full"},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if cfg[section].get(key) != value:
                raise ValueError(f"Reference requires {section}.{key}={value!r}")
    ma = cfg.get("ma", {})
    if (not ma.get("enabled") or cfg["run"].get("synthetic") or cfg.get("ma_ablation")
            or ma["schedule"]["train_blocks"] != 17 or ma["schedule"]["upload_blocks"] != 14
            or ma["schedule"]["anchor_blocks"] != 1
            or ma["compression"]["retention_mode"] != "fixed"
            or ma["compression"]["keep_a"] != 0.8 or ma["compression"]["keep_b"] != 0.8
            or ma["compression"]["probe_fraction_of_k"] != 0
            or ma["compression"].get("selection", "magnitude") != "magnitude"
            or ma["aggregation"].get("similarity_scope", "intersection") != "intersection"
            or ma["aggregation"].get("merge_rule", "dominant_sign") != "dominant_sign"
            or not ma["residual"]["enabled"] or not cfg["pcr"]["enabled"]
            or float(cfg["pcr"]["lambda_reg"]) <= 0):
        raise ValueError("Reference must be the complete ratio30 scheme B with PCR and residuals")
    if (reference / "round_extension.json").exists():
        raise ValueError("This ablation suite compares the fixed tenth round")
    if Path(cfg["run"]["output_dir"]).resolve() != reference:
        raise ValueError("Reference path differs from its training snapshot output_dir")
    manifest = suite._json(reference / "data_manifest.json")
    clients = list(manifest["clients"])
    if (manifest.get("scenario") != "heterogeneous" or len(clients) != 8
            or {x["task"] for x in manifest["clients"].values()} != set(TASKS)
            or partition_hash({k: v for k, v in manifest.items() if k != "partition_hash"})
            != manifest.get("partition_hash")):
        raise ValueError("Reference partition content/hash does not match eight task clients")
    compression = suite._json(reference / "compression/compression_manifest.json")
    if compression.get("proxy_layers") != 23:
        raise ValueError("ratio30 reference must use its original 23-layer proxy")
    registry = BlockRegistry(None, tuple(range(23)), {}, {})
    schedules = schedule_records(registry, clients, cfg)
    verify_schedules(reference, schedules, completed=10)
    result = verified_scores(reference, cfg)
    provenance = {
        "reference_run": str(reference), "reference_config_sha256": _sha(snapshot),
        "reference_evaluation_sha256": _sha(reference / "evaluation/fused_summary.json"),
        "reference_fusion_sha256": _sha(reference / "exports/llm_fused/fusion_manifest.json"),
        "partition_sha256": _sha(reference / "data_manifest.json"),
        "compression_sha256": _sha(reference / "compression/compression_manifest.json"),
        "schedules_sha256": hashlib.sha256(
            json.dumps(schedules, sort_keys=True).encode()).hexdigest(),
    }
    return {"config": cfg, "schedules": schedules, "provenance": provenance, "result": result}


def variant_config(context: dict, group: str, output: Path) -> dict:
    reference = Path(context["provenance"]["reference_run"])
    if output.resolve() == reference or reference in output.resolve().parents:
        raise ValueError("Ablation output must be outside the completed reference run")
    cfg = copy.deepcopy(context["config"])
    cfg["run"].update(name=f"ratio30_ma_{group.lower()}_seed42", output_dir=str(output.resolve()),
                      reuse_artifacts_from=context["provenance"]["reference_run"])
    cfg["ma_ablation"] = {"group": group, **context["provenance"]}
    for dotted, value in GROUPS[group].items():
        _set(cfg, dotted, value)
    validate_config(cfg, "train")
    return cfg


def verify_variant(cfg: dict, context: dict) -> None:
    group = cfg["ma_ablation"]["group"]
    expected = variant_config(context, group, Path(cfg["run"]["output_dir"]))
    if cfg != expected:
        differences = suite._differences(cfg, expected)
        raise ValueError("Unexpected changes outside the declared ablation:\n" + "\n".join(differences))


def verify_initialization(cfg: dict, registry: BlockRegistry, initial_hash: str) -> dict:
    """Check retained first-round packet headers without loading their tensor bodies."""
    reference = Path(cfg["ma_ablation"]["reference_run"])
    matches = 0
    for path in sorted((reference / "clients").glob("*/round_0001/update.bin")):
        with path.open("rb") as handle:
            length = handle.read(4)
            if len(length) != 4:
                raise ValueError(f"Truncated reference packet: {path}")
            size = struct.unpack("<I", length)[0]
            if not 0 < size <= 10_000_000:
                raise ValueError(f"Invalid reference header length: {path}")
            header = json.loads(handle.read(size))
        if (header.get("round_id") != 0 or header.get("schema_hash") != registry.spec.schema_hash
                or header.get("base_hash") != initial_hash):
            raise ValueError(f"Initial adapter differs from the completed MA: {path}")
        matches += 1
    return {"initial_adapter_sha256": initial_hash, "reference_packet_headers_checked": matches,
            "evidence": "matching_initial_hash" if matches else "shared_proxy_and_initialization_seed"}


def check_training_inputs(cfg: dict, registry: BlockRegistry, initial_hash: str) -> dict:
    context = reference_context(Path(cfg["ma_ablation"]["reference_run"]))
    verify_variant(cfg, context)
    run = Path(cfg["run"]["output_dir"])
    if (run / "round_extension.json").exists():
        raise ValueError("Ablations use exactly ten rounds")
    for relative, key in (("data_manifest.json", "partition_sha256"),
                          ("compression/compression_manifest.json", "compression_sha256")):
        if _sha(run / relative) != context["provenance"][key]:
            raise ValueError(f"Ablation inputs differ from reference: {relative}")
    clients = list(suite._json(run / "data_manifest.json")["clients"])
    if schedule_records(registry, clients, cfg) != context["schedules"]:
        raise ValueError("Generated ablation layer schedules differ from the completed MA")
    return verify_initialization(cfg, registry, initial_hash)


def variant_result(cfg: dict, context: dict) -> dict:
    verify_variant(cfg, context)
    run = Path(cfg["run"]["output_dir"])
    verify_schedules(run, context["schedules"], completed=10)
    result = verified_scores(run, cfg)
    base = context["result"]
    for key in ("seed", "partition_hash", "compression_manifest_hash", "protocol_fingerprint",
                "harness_detail_hash", "downlink_accounting"):
        if result[key] != base[key]:
            raise ValueError(f"Ablation/reference {key} mismatch: {run}")
    verified_fields = set()
    reference = Path(context["provenance"]["reference_run"])
    for number in range(1, 11):
        relative = f"checkpoints/round_{number:04d}/metadata.json"
        original_metrics = suite._json(reference / relative)["client_metrics"]
        actual_metrics = suite._json(run / relative)["client_metrics"]
        for client, original in original_metrics.items():
            for field in ("optimizer_steps", "effective_target_tokens",
                          "trainable_adapter_parameters", "transmitted_values"):
                if field in original:
                    if actual_metrics[client].get(field) != original[field]:
                        raise ValueError(f"Round {number} {client} {field} budget mismatch")
                    verified_fields.add(field)
    task_losses = {task: 100 * (base["tasks"][task] - result["tasks"][task]) for task in TASKS}
    return {**result, "group": cfg["ma_ablation"]["group"],
            "component": LABELS[cfg["ma_ablation"]["group"]],
            "contribution_pp": 100 * (base["all_avg"] - result["all_avg"]),
            "task_contribution_pp": task_losses,
            "largest_task_decline": max(task_losses, key=task_losses.get),
            "budget_fields_verified": sorted(verified_fields),
            "initialization": suite._json(run / "metrics/ablation_initialization.json")}


def write_report(context: dict, root: Path) -> dict:
    variants = []
    pending = []
    invalid = {}
    for group in GROUPS:
        run = root / group
        if not (run / "evaluation/fused_summary.json").is_file():
            pending.append(group)
            continue
        try:
            cfg = suite._yaml(run / "execution_config.yaml")
            variants.append(variant_result(cfg, context))
        except (ValueError, FileNotFoundError, KeyError) as error:
            pending.append(group)
            invalid[group] = str(error)
    baseline = {**context["result"], "group": "MA", "component": LABELS["MA"],
                "contribution_pp": 0.0, "task_contribution_pp": dict.fromkeys(TASKS, 0.0)}
    report = {"reference": context["provenance"], "rows": [baseline, *variants],
              "pending": pending, "invalid": invalid,
              "ranking": [r["group"] for r in sorted(
                  variants, key=lambda row: row["contribution_pp"], reverse=True)],
              "inference": "Descriptive comparison conditional on seed 42; no across-seed inference.",
              "attribution": "A1 and A3 each measure a joint mechanism.",
              "cost_note": context["result"]["byte_accounting_note"]}
    suite._write(root / "summary.json", report)
    fields = ["group", "component", "all_percent", "qa_percent", "glue_percent", "contribution_pp",
              "uplink_bytes", "downlink_bytes", "downlink_accounting", "sum_client_train_seconds",
              "ma_round_wall_seconds", *[f"{task}_percent" for task in TASKS],
              *[f"{task}_loss_pp" for task in TASKS]]
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            values = {key: row[key] for key in fields if key in row}
            values.update({f"{name}_percent": 100 * row[f"{name}_avg"]
                           for name in ("all", "qa", "glue")})
            values.update({f"{task}_percent": 100 * row["tasks"][task] for task in TASKS})
            values.update({f"{task}_loss_pp": row["task_contribution_pp"][task] for task in TASKS})
            writer.writerow(values)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "summarize"))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--gpu-ids", default="0,1")
    args = parser.parse_args(argv)
    os.chdir(suite.PROJECT)
    if args.action == "run":
        if not re.fullmatch(r"[0-9]+,[0-9]+", args.gpu_ids) or len(set(args.gpu_ids.split(","))) != 2:
            parser.error("Use two distinct GPU IDs, e.g. --gpu-ids 0,1")
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_ids, FEDPROXY_GPU_IDS=args.gpu_ids,
                          PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        os.environ[REQUIRED_PARALLEL_ENV] = "2"
    reference, root = args.reference.resolve(), args.root.resolve()
    if root == reference or root in reference.parents:
        raise ValueError("Ablation root must have independent output directories")
    context = reference_context(reference)
    if args.action == "summarize":
        write_report(context, root)
        return
    groups = tuple(GROUPS) if args.group == "all" else (args.group,)
    configs = {group: variant_config(context, group, root / group) for group in groups}
    for group, cfg in configs.items():
        run = Path(cfg["run"]["output_dir"])
        for name in ("compression", "data_manifest.json"):
            path = run / name
            if (path.exists() or path.is_symlink()) and not path.samefile(reference / name):
                raise ValueError(f"Existing ablation input does not reference the completed MA: {path}")
        for name in ("execution_config.yaml", "resolved_config.train.yaml"):
            if (run / name).exists() and suite._yaml(run / name) != cfg:
                raise ValueError(f"Existing ablation configuration differs: {run / name}")
        if (run / "checkpoints").exists():
            completed = suite.completed_rounds(run, cfg, ma=True)
            verify_schedules(run, context["schedules"], completed=completed)
        print(f"[ma-ablation] validated {group} intervention={GROUPS[group]} output={run}", flush=True)
    if args.action == "validate":
        return
    import torch
    from fedproxy.ratio30 import _link

    for cfg in configs.values():
        resolve_parallelism(cfg["federated"], 8, torch.cuda.device_count())
    with suite._suite_lock(root, label="MA ablation"):
        for group, cfg in configs.items():
            run = Path(cfg["run"]["output_dir"])
            suite._freeze_config(cfg)
            _link(run / "compression", reference / "compression")
            _link(run / "data_manifest.json", reference / "data_manifest.json")
            suite.execute_task(group, cfg, args.gpu_ids, inputs_ready=True,
                               result_reader=lambda cfg=cfg: variant_result(cfg, context))
            write_report(context, root)


if __name__ == "__main__":
    main()
