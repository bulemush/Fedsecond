from __future__ import annotations


PROMPT_VERSION = "unified_v1"
MNLI_LABELS = ("entailment", "neutral", "contradiction")
MNLI_RELATION_CUE = "Relation (entailment, neutral, or contradiction):"
MNLI_PROMPT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}\n" + MNLI_RELATION_CUE


def mnli_prompt(premise: str, hypothesis: str) -> str:
    return MNLI_PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)


def format_alpaca(instruction: str, input_text: str, output: str | None = None) -> str:
    pieces = [f"Instruction: {instruction.strip()}"]
    if input_text.strip():
        pieces.append(f"Input: {input_text.strip()}")
    pieces.append("Response:" + (f" {output.strip()}" if output is not None else ""))
    return "\n".join(pieces)


def qa_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))
    return f"Question: {question}\nChoices:\n{choices}\nAnswer (letter):"

