from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DatasetSource:
    path: str
    name: str | None
    train_split: str = "train"
    eval_split: str = "validation"


REGISTRY = {
    "alpaca": DatasetSource("tatsu-lab/alpaca", None),
    "obqa": DatasetSource("allenai/openbookqa", "main", eval_split="validation"),
    "arc_easy": DatasetSource("allenai/ai2_arc", "ARC-Easy", eval_split="validation"),
    "arc_challenge": DatasetSource("allenai/ai2_arc", "ARC-Challenge", eval_split="validation"),
    "commonsense_qa": DatasetSource("tau/commonsense_qa", None, eval_split="validation"),
    "sst2": DatasetSource("nyu-mll/glue", "sst2", eval_split="validation"),
    "mrpc": DatasetSource("nyu-mll/glue", "mrpc", eval_split="validation"),
    "rte": DatasetSource("nyu-mll/glue", "rte", eval_split="validation"),
    "mnli": DatasetSource("nyu-mll/glue", "mnli", eval_split="validation_matched"),
}


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not component:
        raise ValueError(f"Invalid empty dataset path component derived from {value!r}")
    return component


def dataset_storage_path(
    data_dir: str | Path, name: str, split: str, revision: str | None = None
) -> Path:
    revision_key = _safe_component(revision) if revision else "default"
    return (
        Path(data_dir).expanduser().resolve()
        / "datasets"
        / _safe_component(name)
        / revision_key
        / _safe_component(split)
    )


def _saved_dataset_markers(path: Path) -> bool:
    return (path / "state.json").is_file() or (path / "dataset_dict.json").is_file()


def _select_saved_split(dataset: Any, split: str, path: Path):
    if hasattr(dataset, "keys") and split in dataset:
        return dataset[split]
    if hasattr(dataset, "keys") and split not in dataset:
        raise KeyError(
            f"Local DatasetDict at {path} does not contain split {split!r}; "
            f"available splits: {sorted(dataset.keys())}"
        )
    return dataset


def _manual_local_candidates(root: Path, name: str, split: str):
    alias = _safe_component(name)
    split_key = _safe_component(split)
    directories = [root / alias, root / "datasets" / alias]
    for directory in directories:
        yield "saved", directory / split_key
        yield "saved", directory
        for suffix in ("jsonl", "json", "csv", "parquet"):
            yield "raw", directory / f"{split}.{suffix}"
    if split == "train":
        for suffix in ("jsonl", "json", "csv", "parquet"):
            yield "raw", root / f"{alias}.{suffix}"


def _load_manual_local(
    root: Path,
    name: str,
    split: str,
    *,
    load_dataset: Callable,
    load_from_disk: Callable,
    cache_dir: Path,
):
    seen: set[Path] = set()
    for kind, candidate in _manual_local_candidates(root, name, split):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if kind == "saved" and candidate.is_dir() and _saved_dataset_markers(candidate):
            try:
                dataset = _select_saved_split(load_from_disk(str(candidate)), split, candidate)
            except Exception as error:
                raise RuntimeError(f"Invalid local Hugging Face dataset at {candidate}") from error
            return dataset, candidate, "local_save_to_disk"
        if kind == "raw" and candidate.is_file():
            suffix = candidate.suffix.lower()
            builder = "json" if suffix in {".json", ".jsonl"} else suffix.lstrip(".")
            try:
                dataset = load_dataset(
                    builder,
                    data_files={split: str(candidate)},
                    split=split,
                    cache_dir=str(cache_dir),
                )
            except Exception as error:
                raise RuntimeError(f"Failed to load local dataset file {candidate}") from error
            return dataset, candidate, f"local_{builder}"
    return None


def _persist_dataset(dataset, target: Path, metadata: dict[str, Any], *, load_from_disk):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temp_dir:
        staging = Path(temp_dir) / "dataset"
        dataset.save_to_disk(str(staging))
        (staging / "_fedproxy_source.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            os.replace(staging, target)
        except OSError:
            # Another process may have completed the same materialization concurrently.
            if not target.exists():
                raise
    return load_from_disk(str(target))


def load_registered(
    name: str,
    *,
    split: str | None = None,
    revision: str | None = None,
    data_dir: str | Path = "data",
):
    from datasets import load_dataset, load_from_disk

    if name not in REGISTRY:
        raise KeyError(f"Unknown task: {name}")
    source = REGISTRY[name]
    split = split or source.train_split
    target = dataset_storage_path(data_dir, name, split, revision)
    if target.exists():
        try:
            dataset = load_from_disk(str(target))
        except Exception as error:
            raise RuntimeError(
                f"Local dataset cache is incomplete or invalid: {target}. "
                "Move the directory aside and retry to download a clean copy."
            ) from error
        print(f"[fedproxy:data] loaded local dataset {name}/{split} from {target}")
        return dataset

    root = Path(data_dir).expanduser().resolve()
    hf_cache = root / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    local = _load_manual_local(
        root,
        name,
        split,
        load_dataset=load_dataset,
        load_from_disk=load_from_disk,
        cache_dir=hf_cache,
    )
    if local is not None:
        dataset, local_path, source_kind = local
        print(f"[fedproxy:data] found local dataset {name}/{split} at {local_path}")
        metadata = {
            "alias": name,
            "source_kind": source_kind,
            "local_source": str(local_path),
            "split": split,
            "requested_revision": revision,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        }
        materialized = _persist_dataset(
            dataset, target, metadata, load_from_disk=load_from_disk
        )
        print(f"[fedproxy:data] materialized reusable dataset {name}/{split} at {target}")
        return materialized

    print(
        f"[fedproxy:data] local dataset missing; downloading "
        f"{source.path}/{source.name or '-'} split={split} into {root}"
    )
    try:
        dataset = load_dataset(
            source.path,
            source.name,
            split=split,
            revision=revision,
            cache_dir=str(hf_cache),
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to obtain dataset {name}/{split}. No saved copy exists at {target}, "
            f"and Hugging Face download failed."
        ) from error

    metadata = {
        "alias": name,
        "source_kind": "huggingface",
        "source_path": source.path,
        "source_name": source.name,
        "split": split,
        "requested_revision": revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
    }
    persisted = _persist_dataset(dataset, target, metadata, load_from_disk=load_from_disk)
    print(f"[fedproxy:data] saved reusable dataset {name}/{split} to {target}")
    return persisted
