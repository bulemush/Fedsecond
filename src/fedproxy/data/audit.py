from __future__ import annotations

import re
from collections.abc import Iterable

from .tasks import TaskExample
from .truncation import encode_prompt, required_cue


def _decoded_prompt(prepared, tokenizer) -> str:
    # Audit the exact token sequence passed to the model, not the intermediate
    # structured prompt (which may tokenize/decode differently).
    return tokenizer.decode(
        prepared.input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def audit_prompts(
    examples: Iterable[TaskExample],
    tokenizer,
    max_input_length: int,
    *,
    strategy: str = "prefix",
    truncation_side: str = "right",
) -> dict:
    count = truncated = removed_tokens = 0
    cue_missing = hypothesis_missing = choice_label_missing = 0
    original_lengths = []
    examples_truncated = []
    for example in examples:
        prepared = encode_prompt(
            example, tokenizer, max_input_length,
            strategy=strategy, truncation_side=truncation_side,
        )
        count += 1
        original_lengths.append(prepared.original_tokens)
        if prepared.truncated:
            truncated += 1
            removed_tokens += prepared.original_tokens - len(prepared.input_ids)
            if len(examples_truncated) < 5:
                examples_truncated.append({
                    "sample_id": example.sample_id,
                    "original_tokens": prepared.original_tokens,
                    "retained_tokens": len(prepared.input_ids),
                })
        rendered = _decoded_prompt(prepared, tokenizer)
        cue = required_cue(example.task_name)
        if cue and not rendered.rstrip().endswith(cue):
            cue_missing += 1
        if example.task_name in {"rte", "mnli"}:
            if "\nHypothesis: " not in rendered or not rendered.split(
                "\nHypothesis: ", 1
            )[1].split("\n", 1)[0].strip():
                hypothesis_missing += 1
        if example.task_name in {"obqa", "arc_easy", "arc_challenge", "commonsense_qa"}:
            original_choices = example.prompt.split("\nChoices:\n", 1)[1].rsplit(
                "\nAnswer (letter):", 1
            )[0]
            labels = re.findall(r"(?m)^([A-Za-z0-9]+)\. ", original_choices)
            if any(not re.search(rf"(?m)^{re.escape(label)}\.\s", rendered) for label in labels):
                choice_label_missing += 1
    if count == 0:
        raise ValueError("Cannot audit an empty task dataset")
    original_lengths.sort()
    return {
        "samples": count,
        "truncated": truncated,
        "truncation_rate": truncated / count,
        "mean_original_tokens": sum(original_lengths) / count,
        "p95_original_tokens": original_lengths[min(count - 1, int(0.95 * count))],
        "max_original_tokens": original_lengths[-1],
        "mean_removed_tokens_if_truncated": removed_tokens / max(truncated, 1),
        "cue_missing": cue_missing,
        "cue_missing_rate": cue_missing / count,
        "hypothesis_missing": hypothesis_missing,
        "choice_label_missing": choice_label_missing,
        "truncated_example_ids": examples_truncated,
    }
