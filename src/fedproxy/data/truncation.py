from __future__ import annotations

from dataclasses import dataclass
import re

from .tasks import TaskExample


@dataclass(frozen=True)
class PromptEncoding:
    input_ids: list[int]
    original_tokens: int
    rendered_prompt: str | None

    @property
    def truncated(self) -> bool:
        return self.original_tokens > len(self.input_ids)


def required_cue(task: str) -> str | None:
    return {
        "obqa": "Answer (letter):",
        "arc_easy": "Answer (letter):",
        "arc_challenge": "Answer (letter):",
        "commonsense_qa": "Answer (letter):",
        "sst2": "Sentiment (negative or positive):",
        "mrpc": "Equivalent (no or yes):",
        "rte": "Entailment (yes or no):",
        "mnli": "Relation (entailment, neutral, or contradiction):",
    }.get(task)


def _variable_parts(example: TaskExample) -> list[tuple[str, str, float, int]]:
    """Return fixed/variable prompt pieces; variable pieces have a positive weight."""
    prompt = example.prompt
    task = example.task_name
    if task in {"obqa", "arc_easy", "arc_challenge", "commonsense_qa"}:
        if not prompt.startswith("Question: "):
            raise ValueError(f"Unexpected QA prompt for {task}")
        question, rest = prompt[len("Question: "):].split("\nChoices:\n", 1)
        choices, ending = rest.rsplit("\nAnswer (letter):", 1)
        if ending:
            raise ValueError(f"Unexpected QA answer suffix for {task}")
        parts = [("fixed", "Question: ", 0.0, 0), ("variable", question, 1.3, 8),
                 ("fixed", "\nChoices:\n", 0.0, 0)]
        matches = list(re.finditer(r"(?m)^([A-Za-z0-9]+)\. ", choices))
        if not matches or matches[0].start() != 0:
            raise ValueError(f"Unparseable choices in {task} prompt")
        for index, match in enumerate(matches):
            label = match.group(1)
            end = matches[index + 1].start() - 1 if index + 1 < len(matches) else len(choices)
            choice = choices[match.end():end]
            if index:
                parts.append(("fixed", "\n", 0.0, 0))
            parts.extend([("fixed", f"{label}. ", 0.0, 0),
                          ("variable", choice, 1.0, 3)])
        parts.append(("fixed", "\nAnswer (letter):", 0.0, 0))
        return parts
    if task in {"rte", "mnli"}:
        cue = required_cue(task)
        assert cue is not None
        if not prompt.startswith("Premise: "):
            raise ValueError(f"Unexpected NLI prompt for {task}")
        premise, rest = prompt[len("Premise: "):].split("\nHypothesis: ", 1)
        hypothesis, ending = rest.rsplit(f"\n{cue}", 1)
        if ending:
            raise ValueError(f"Unexpected NLI answer suffix for {task}")
        return [("fixed", "Premise: ", 0.0, 0),
                ("variable", premise, 1.0, 8),
                ("fixed", "\nHypothesis: ", 0.0, 0),
                ("variable", hypothesis, 1.3, 8),
                ("fixed", f"\n{cue}", 0.0, 0)]
    if task == "mrpc":
        if not prompt.startswith("Sentence 1: "):
            raise ValueError("Unexpected MRPC prompt")
        first, rest = prompt[len("Sentence 1: "):].split("\nSentence 2: ", 1)
        second, ending = rest.rsplit("\nEquivalent (no or yes):", 1)
        if ending:
            raise ValueError("Unexpected MRPC answer suffix")
        return [("fixed", "Sentence 1: ", 0.0, 0),
                ("variable", first, 1.0, 8),
                ("fixed", "\nSentence 2: ", 0.0, 0),
                ("variable", second, 1.0, 8),
                ("fixed", "\nEquivalent (no or yes):", 0.0, 0)]
    if task == "sst2":
        if not prompt.startswith("Sentence: "):
            raise ValueError("Unexpected SST-2 prompt")
        sentence, ending = prompt[len("Sentence: "):].rsplit(
            "\nSentiment (negative or positive):", 1
        )
        if ending:
            raise ValueError("Unexpected SST-2 answer suffix")
        return [("fixed", "Sentence: ", 0.0, 0),
                ("variable", sentence, 1.0, 8),
                ("fixed", "\nSentiment (negative or positive):", 0.0, 0)]
    raise ValueError(f"Structured truncation is unsupported for task {task}")


def _decode(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def encode_prompt(
    example: TaskExample,
    tokenizer,
    max_input_length: int,
    *,
    strategy: str = "prefix",
    truncation_side: str = "right",
) -> PromptEncoding:
    if max_input_length < 1:
        raise ValueError("max_input_length must be positive")
    original = tokenizer.encode(example.prompt, add_special_tokens=True)
    if len(original) <= max_input_length:
        return PromptEncoding(original, len(original), example.prompt)
    if strategy == "prefix":
        kept = (original[:max_input_length] if truncation_side == "right"
                else original[-max_input_length:])
        return PromptEncoding(kept, len(original), None)
    if strategy != "structured":
        raise ValueError(f"Unknown prompt truncation strategy: {strategy}")

    parts = _variable_parts(example)
    variable_tokens = [tokenizer.encode(text, add_special_tokens=False)
                       for kind, text, _, _ in parts if kind == "variable"]
    minimums = [min(len(tokens), minimum)
                for tokens, (_, _, _, minimum) in zip(
                    variable_tokens, (part for part in parts if part[0] == "variable"), strict=True
                )]

    def render(scale: float) -> tuple[str, list[int]]:
        output = []
        variable_index = 0
        for kind, text, weight, _minimum in parts:
            if kind == "fixed":
                output.append(text)
                continue
            tokens = variable_tokens[variable_index]
            minimum = minimums[variable_index]
            quota = min(len(tokens), minimum + int((len(tokens) - minimum) * weight * scale))
            output.append(_decode(tokenizer, tokens[:quota]))
            variable_index += 1
        rendered = "".join(output)
        return rendered, tokenizer.encode(rendered, add_special_tokens=True)

    best_text, best_ids = render(0.0)
    if len(best_ids) > max_input_length:
        raise ValueError(
            f"Cannot preserve minimum task structure for {example.task_name} "
            f"within {max_input_length} tokens"
        )
    low, high = 0.0, 1.0
    for _ in range(16):
        midpoint = (low + high) / 2
        candidate_text, candidate_ids = render(midpoint)
        if len(candidate_ids) <= max_input_length:
            low = midpoint
            best_text, best_ids = candidate_text, candidate_ids
        else:
            high = midpoint
    return PromptEncoding(best_ids, len(original), best_text)
