"""Combine eight independently trained homogeneous fused-model evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from fedproxy.evaluation.report import GLUE_TASKS, QA_TASKS


def collect(root: Path) -> dict:
    metrics: dict[str, float] = {}
    sources: dict[str, str] = {}
    reference_protocol: dict | None = None
    for task in QA_TASKS + GLUE_TASKS:
        path = root / f"llama2_homogeneous_{task}_len320_mb4_seed42" / "evaluation" / "fused_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("model_kind") != "fused" or summary.get("synthetic"):
            raise ValueError(f"Expected real fused-model evaluation: {path}")
        if summary.get("task") != task or set(summary.get("tasks", {})) != {task}:
            raise ValueError(f"Expected only {task} in {path}")
        protocol = dict(summary["protocol"])
        protocol.pop("tasks", None)
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"Evaluation protocol mismatch: {path}")
        metrics[task] = float(summary["task_accuracy"])
        if metrics[task] != float(summary["tasks"][task]):
            raise ValueError(f"Inconsistent task accuracy: {path}")
        sources[task] = str(path)
    qa = sum(metrics[task] for task in QA_TASKS) / len(QA_TASKS)
    glue = sum(metrics[task] for task in GLUE_TASKS) / len(GLUE_TASKS)
    return {
        "scenario": "homogeneous",
        "model_kind": "fused",
        "tasks": metrics,
        "qa_avg": qa,
        "glue_avg": glue,
        "all_avg": (qa + glue) / 2,
        "shared_protocol_fields": reference_protocol,
        "sources": sources,
    }


if __name__ == "__main__":
    result = collect(Path("runs"))
    output = Path("runs/llama2_homogeneous_suite_len320_mb4_seed42/summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
