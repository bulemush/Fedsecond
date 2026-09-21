from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedproxy.config import REPRODUCTION_DECISIONS, apply_overrides, load_config, save_resolved_config, validate_config
from fedproxy.smoke import run_smoke
from fedproxy.utils.reproducibility import write_environment
from fedproxy.workflows import compare, compress, evaluate, fuse, prepare_data, train


def _config(args, command):
    cfg = apply_overrides(load_config(args.config), args.set or [])
    if getattr(args, "rounds", None) is not None:
        cfg["federated"]["rounds"] = args.rounds
    validate_config(cfg, command)
    run_dir = Path(cfg["run"]["output_dir"])
    # Keep an immutable command-level snapshot. Later fuse/evaluate commands
    # must not erase the training snapshot that contains the explicit rounds.
    save_resolved_config(cfg, run_dir / f"resolved_config.{command}.yaml")
    primary = run_dir / "resolved_config.yaml"
    if command == "train" or not primary.exists():
        save_resolved_config(cfg, primary)
    write_environment(run_dir / "environment.json")
    (run_dir / "reproduction_decisions.json").write_text(
        json.dumps(REPRODUCTION_DECISIONS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cfg


def build_parser():
    parser = argparse.ArgumentParser(description="FedProxy reproduction CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "compress", "smoke"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--set", action="append", default=[])
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--rounds", type=int)
    train.add_argument("--resume")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--set", action="append", default=[])
    fuse = sub.add_parser("fuse")
    fuse.add_argument("--config", required=True)
    fuse.add_argument("--checkpoint", required=True)
    fuse.add_argument("--overwrite", action="store_true")
    fuse.add_argument("--set", action="append", default=[])
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--model-kind", choices=("original", "proxy", "fused"), required=True)
    evaluation.add_argument("--set", action="append", default=[])
    comparison = sub.add_parser("compare")
    comparison.add_argument("--runs-root", required=True)
    comparison.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        compare(args.runs_root, args.output)
        return
    cfg = _config(args, args.command)
    if args.command == "prepare-data":
        result = prepare_data(cfg)
    elif args.command == "compress":
        result = compress(cfg)
    elif args.command == "smoke":
        result = run_smoke(cfg)
    elif args.command == "train":
        result = train(cfg, resume=args.resume, dry_run=args.dry_run)
    elif args.command == "fuse":
        result = fuse(cfg, args.checkpoint, overwrite=args.overwrite)
    elif args.command == "evaluate":
        result = evaluate(cfg, args.model_kind)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
