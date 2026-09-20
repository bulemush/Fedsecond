from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import qa_prompt


@dataclass(frozen=True)
class TaskExample:
    sample_id: str
    task_name: str
    prompt: str
    response: str


def _label_name(dataset, row: dict[str, Any], field: str = "label") -> str:
    feature = dataset.features[field]
    if not hasattr(feature, "names"):
        raise ValueError(f"Dataset feature {field} has no declared label names")
    value = int(row[field])
    if value < 0:
        raise ValueError("Unlabeled sample cannot be used for supervised training")
    return str(feature.names[value]).lower()


def convert_row(task: str, row: dict[str, Any], index: int, dataset=None) -> TaskExample:
    sample_id = str(row.get("idx", row.get("id", index)))
    if task in {"obqa", "arc_easy", "arc_challenge", "commonsense_qa"}:
        choices = row["choices"]
        labels, texts = list(choices["label"]), list(choices["text"])
        answer = str(row["answerKey"])
        if answer not in labels:
            raise ValueError(f"Correct answer {answer!r} absent from choices")
        question_field = "question_stem" if task == "obqa" else "question"
        if question_field not in row:
            raise ValueError(f"Missing {question_field!r} in {task} sample {sample_id}")
        return TaskExample(sample_id, task, qa_prompt(row[question_field], labels, texts), answer)
    if dataset is None:
        raise ValueError("GLUE conversion requires the dataset feature schema")
    label = _label_name(dataset, row)
    if task == "sst2":
        mapping = {"negative": "negative", "positive": "positive"}
        prompt = f"Sentence: {row['sentence']}\nSentiment (negative or positive):"
    elif task == "mrpc":
        mapping = {"not_equivalent": "no", "equivalent": "yes"}
        prompt = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\nEquivalent (no or yes):"
    elif task == "rte":
        mapping = {"entailment": "yes", "not_entailment": "no"}
        prompt = f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}\nEntailment (yes or no):"
    elif task == "mnli":
        mapping = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}
        prompt = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nRelation (entailment, neutral, or contradiction):"
    else:
        raise KeyError(task)
    if label not in mapping:
        raise ValueError(f"Unexpected declared label {label!r} for {task}")
    return TaskExample(sample_id, task, prompt, mapping[label])
