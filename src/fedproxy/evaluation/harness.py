from __future__ import annotations

import json
from pathlib import Path

from fedproxy.data.prompts import MNLI_LABELS, mnli_prompt

from .protocol import task_overrides


def mnli_doc_to_text(doc: dict) -> str:
    return mnli_prompt(doc["premise"], doc["hypothesis"])


def _aligned_tasks(tasks: list[str]) -> list:
    if "mnli" not in tasks:
        return list(tasks)
    from lm_eval.tasks import TaskManager

    # Load task objects, rather than reading version-dependent dict/Entry indexes.
    task = TaskManager().load_task_or_group(["mnli"])["mnli"]
    if task.config.validation_split != "validation_matched" or task.config.test_split:
        raise ValueError("MNLI evaluation requires validation_matched and no test split")
    labels = tuple(task.validation_docs().features["label"].names)
    if labels != MNLI_LABELS:
        raise ValueError(f"Unexpected MNLI label schema: {labels!r}")
    task.set_config("doc_to_text", mnli_doc_to_text)
    task.set_config("doc_to_choice", list(MNLI_LABELS))
    task.set_config("doc_to_target", "label")
    task.set_config("target_delimiter", " ")
    metadata = dict(task.config.metadata or {})
    metadata["fedproxy_template"] = task_overrides(["mnli"])["mnli"]["version"]
    task.set_config("metadata", metadata)
    return [task if name == "mnli" else name for name in tasks]


def evaluate_with_lm_eval(
    model_path: str,
    tasks: list[str],
    output_path: str | Path,
    batch_size: int = 1,
    num_fewshot: int = 0,
    *,
    dtype: str = "auto",
):
    from lm_eval import evaluator

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={dtype}",
        tasks=_aligned_tasks(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        apply_chat_template=False,
        cache_requests=False,
    )
    if task_overrides(tasks):
        results["fedproxy_task_overrides"] = task_overrides(tasks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results
