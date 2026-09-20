from __future__ import annotations

import json
from pathlib import Path


QA_TASKS = ("obqa", "arc_easy", "arc_challenge", "commonsense_qa")
GLUE_TASKS = ("sst2", "mrpc", "rte", "mnli")


def summarize(metrics: dict[str, float]) -> dict[str, object]:
    missing = set(QA_TASKS + GLUE_TASKS) - set(metrics)
    if missing:
        raise ValueError(f"Missing task metrics: {sorted(missing)}")
    qa = sum(metrics[name] for name in QA_TASKS) / len(QA_TASKS)
    glue = sum(metrics[name] for name in GLUE_TASKS) / len(GLUE_TASKS)
    return {"tasks": metrics, "qa_avg": qa, "glue_avg": glue, "all_avg": (qa + glue) / 2}


def compare_runs(runs_root: str | Path) -> list[dict]:
    rows = []
    for path in Path(runs_root).glob("*/evaluation/summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("synthetic") or data.get("diagnostic"):
            continue
        rows.append({"run": path.parents[1].name, **data})
    fingerprints = {row.get("protocol_fingerprint") for row in rows}
    if len(fingerprints) > 1:
        raise ValueError("Cannot combine runs with different evaluation protocol fingerprints")
    return rows

