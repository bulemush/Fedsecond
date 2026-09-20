from __future__ import annotations

from dataclasses import dataclass


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


def load_registered(name: str, *, split: str | None = None, revision: str | None = None):
    from datasets import load_dataset

    if name not in REGISTRY:
        raise KeyError(f"Unknown task: {name}")
    source = REGISTRY[name]
    split = split or source.train_split
    return load_dataset(source.path, source.name, split=split, revision=revision)

