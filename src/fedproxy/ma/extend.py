"""Extend completed homogeneous MA runs without changing their frozen configs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from fedproxy.ma import homogeneous as suite
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, resolve_parallelism
from fedproxy.ma.round_budget import config_digest, target_rounds


def preflight(task: str, root: Path, requested: int) -> dict:
    run = root / suite.run_name(task)
    cfg = suite._yaml(run / "execution_config.yaml")
    if cfg != suite._yaml(run / "resolved_config.train.yaml"):
        raise ValueError(f"Frozen/training configuration mismatch: {run}")
    if (Path(cfg["run"]["output_dir"]).resolve() != run.resolve()
            or cfg["data"]["scenario"] != "homogeneous" or cfg["data"]["tasks"] != [task]
            or cfg["data"]["num_clients"] != 4 or not cfg.get("ma", {}).get("enabled")
            or cfg["run"].get("synthetic")):
        raise ValueError(f"Expected an existing four-client homogeneous MA run: {run}")
    current = target_rounds(cfg)
    if requested < current or requested <= int(cfg["federated"]["rounds"]):
        raise ValueError(f"Target rounds must extend the original budget and cannot shrink {current}: {run}")
    finished = suite.completed_rounds(run, cfg, ma=True, require_residual=requested > current)
    if requested > current and finished != current:
        raise ValueError(f"Complete the current {current}-round budget before extending: {run} ({finished} done)")
    if requested > finished:
        suite._weights_present(run / "compression/proxy")
        suite._weights_present(Path(cfg["model"]["name_or_path"]))
    return cfg


def _copy_once(source: Path, destination: Path) -> None:
    if source.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def register_extension(cfg: dict, requested: int) -> None:
    run = Path(cfg["run"]["output_dir"])
    current = target_rounds(cfg)
    if requested == current:
        return
    # Save small score/provenance artifacts, not another full LLM weight copy.
    archive = run / "round_results" / f"round_{current:04d}"
    for name in ("evaluation/fused_summary.json", "evaluation/summary.json",
                 "evaluation/raw/fused_lm_eval.json", "exports/llm_fused/fusion_manifest.json", "result.json"):
        _copy_once(run / name, archive / name)
    extension = {"version": 1, "config_hash": config_digest(cfg),
                 "initial_rounds": int(cfg["federated"]["rounds"]), "target_rounds": requested}
    temporary = run / "round_extension.json.tmp"
    suite._write(temporary, extension)
    os.replace(temporary, run / "round_extension.json")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Continue existing homogeneous MA checkpoints to a larger round budget")
    parser.add_argument("--rounds", type=int, required=True, help="Total target rounds, e.g. 20 (not 20 extra rounds)")
    parser.add_argument("--root", type=Path, default=suite.DEFAULT_ROOT)
    parser.add_argument("--task", choices=("all", *suite.TASKS), default="all")
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"([0-9]+),([0-9]+)", args.gpu_ids) or len(set(args.gpu_ids.split(","))) != 2:
        parser.error("--gpu-ids requires two distinct GPU IDs, e.g. 2,3")
    os.chdir(suite.PROJECT)
    tasks = suite.TASKS if args.task == "all" else (args.task,)

    def validate():
        configs = {task: preflight(task, args.root, args.rounds) for task in tasks}
        if any(cfg["ma"] != next(iter(configs.values()))["ma"] for cfg in configs.values()):
            raise ValueError("Selected tasks have different MA settings")
        for task, cfg in configs.items():
            print(f"[ma-extend] validated task={task} target_rounds={args.rounds} "
                  f"local_epochs={cfg['training']['local_epochs']} output={cfg['run']['output_dir']}", flush=True)
        return configs

    if args.validate_only:
        validate()
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_ids, FEDPROXY_GPU_IDS=args.gpu_ids,
                      PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    os.environ[REQUIRED_PARALLEL_ENV] = "2"
    os.environ["PYTHONPATH"] = str(suite.PROJECT / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    with suite._suite_lock(args.root):
        configs = validate()  # Check every selected task before changing any budget.
        import torch

        for cfg in configs.values():
            resolve_parallelism(cfg["federated"], cfg["data"]["num_clients"], torch.cuda.device_count())
        for cfg in configs.values():
            current = target_rounds(cfg)
            if current < args.rounds:
                summary_path = args.root / "summary.json"
                if summary_path.is_file():
                    summary = suite._json(summary_path)
                    recorded = {item["rounds"] for item in summary.get("task_results", [])}
                    if recorded == {current}:
                        for name in ("summary.json", "results.csv"):
                            _copy_once(args.root / name, args.root / "round_results" / f"round_{current:04d}" / name)
        for cfg in configs.values():
            register_extension(cfg, args.rounds)
        for task, cfg in configs.items():
            suite.execute_task(task, cfg, args.gpu_ids)
        if args.task == "all":
            suite.write_suite_summary(args.root)
        else:
            # Do not turn a successful single-task extension into a failure
            # merely because another task still has an older/missing result.
            try:
                suite.collect_suite(args.root)
            except (ValueError, FileNotFoundError, KeyError):
                print("[ma-extend] Eight-task summary waits for all tasks to finish at the same round budget", flush=True)
            else:
                suite.write_suite_summary(args.root)
        print(f"[ma-extend] selected tasks finished at round {args.rounds}", flush=True)


if __name__ == "__main__":
    main()
