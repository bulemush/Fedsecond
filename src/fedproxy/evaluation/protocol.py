"""Versioned evaluation recipes, independent of training/checkpoint configuration."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fedproxy.data.prompts import MNLI_LABELS, MNLI_PROMPT_TEMPLATE


def task_overrides(tasks: list[str]) -> dict:
    if "mnli" not in tasks:
        return {}
    return {
        "mnli": {
            "version": "mnli_train_template_v1",
            "prompt_template": MNLI_PROMPT_TEMPLATE,
            "choices": list(MNLI_LABELS),
            "doc_to_target": "label",
            "target_delimiter": " ",
            "validation_split": "validation_matched",
            "output_type": "multiple_choice",
            "primary_metric": "acc",
            "apply_chat_template": False,
        }
    }


def protocol_fingerprint(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def validate_mnli_evaluation(summary: dict) -> None:
    """Reject stale MNLI results while retaining unchanged scores for other tasks."""
    if "mnli" not in summary.get("tasks", {}):
        return
    protocol = summary.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("task_overrides") != task_overrides(["mnli"])
        or summary.get("protocol_fingerprint") != protocol_fingerprint(protocol)
    ):
        raise ValueError(
            "Stale or incompatible MNLI evaluation protocol. Re-evaluate the existing "
            "fused model with mnli_train_template_v1; retraining is not required."
        )


def archive_previous_mnli_evaluation(
    run: str | Path, model_kind: str, tasks: list[str]
) -> Path | None:
    """Copy old-protocol scores and raw output before their canonical paths are reused."""
    if "mnli" not in tasks:
        return None
    evaluation = Path(run) / "evaluation"
    summary_path = evaluation / f"{model_kind}_summary.json"
    paths = [summary_path, evaluation / "raw" / f"{model_kind}_lm_eval.json"]
    if model_kind == "fused":
        paths.append(evaluation / "summary.json")
    if not any(path.is_file() for path in paths):
        return None
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_mnli_evaluation(previous)
            if "mnli" in previous.get("tasks", {}):
                return None
        except (ValueError, TypeError):
            pass
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = evaluation / "archive" / f"{model_kind}_{stamp}"
    for source in paths:
        if source.is_file():
            target = archive / source.relative_to(evaluation)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return archive
