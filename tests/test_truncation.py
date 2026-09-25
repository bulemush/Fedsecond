import pytest

from fedproxy.data.tasks import TaskExample
from fedproxy.data.audit import audit_prompts
from fedproxy.data.truncation import encode_prompt


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        return ([1] if add_special_tokens else []) + [ord(char) + 3 for char in text]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(token - 3) for token in tokens if token > 2)


class ChunkTokenizer:
    """Lossless four-character tokens, approximating a subword token budget."""

    def __init__(self):
        self.vocabulary = {}
        self.inverse = {}

    def encode(self, text, add_special_tokens):
        pieces = [text[index:index + 4] for index in range(0, len(text), 4)]
        for piece in pieces:
            if piece not in self.vocabulary:
                token_id = len(self.vocabulary) + 3
                self.vocabulary[piece] = token_id
                self.inverse[token_id] = piece
        return ([1] if add_special_tokens else []) + [self.vocabulary[piece] for piece in pieces]

    def decode(self, tokens, **_kwargs):
        return "".join(self.inverse[token] for token in tokens if token > 2)


def test_structured_qa_truncation_keeps_all_choice_labels_and_answer_cue():
    prompt = (
        "Question: " + "Where does the long question lead? " * 5
        + "\nChoices:\nA. " + "first option " * 8
        + "\nB. " + "second option " * 8
        + "\nC. " + "third option " * 8
        + "\nD. " + "fourth option " * 8
        + "\nAnswer (letter):"
    )
    example = TaskExample("1", "arc_easy", prompt, "B")
    prepared = encode_prompt(example, CharacterTokenizer(), 140, strategy="structured")
    assert prepared.truncated
    assert len(prepared.input_ids) <= 140
    assert prepared.rendered_prompt.endswith("Answer (letter):")
    assert all(f"\n{label}. " in prepared.rendered_prompt for label in "ABCD")


def test_structured_mnli_truncation_keeps_hypothesis_and_relation_cue():
    prompt = (
        "Premise: " + "A lengthy premise. " * 12
        + "\nHypothesis: " + "A lengthy hypothesis. " * 8
        + "\nRelation (entailment, neutral, or contradiction):"
    )
    example = TaskExample("2", "mnli", prompt, "neutral")
    prepared = encode_prompt(example, CharacterTokenizer(), 140, strategy="structured")
    assert prepared.truncated
    assert len(prepared.input_ids) <= 140
    assert "\nHypothesis: " in prepared.rendered_prompt
    assert prepared.rendered_prompt.endswith("Relation (entailment, neutral, or contradiction):")
    assert len(prepared.rendered_prompt.split("\nHypothesis: ", 1)[1].split("\nRelation", 1)[0]) >= 8


def test_prefix_strategy_matches_previous_right_truncation():
    example = TaskExample("1", "synthetic", "abcdef", "x")
    tokenizer = CharacterTokenizer()
    prepared = encode_prompt(example, tokenizer, 4, strategy="prefix")
    assert prepared.input_ids == tokenizer.encode(example.prompt, add_special_tokens=True)[:4]


def test_audit_reports_cue_loss_before_and_preservation_after_structured_truncation():
    prompt = (
        "Premise: " + "A long premise. " * 12
        + "\nHypothesis: " + "A long hypothesis. " * 10
        + "\nRelation (entailment, neutral, or contradiction):"
    )
    examples = [TaskExample("1", "mnli", prompt, "neutral")]
    tokenizer = CharacterTokenizer()
    prefix = audit_prompts(examples, tokenizer, 140)
    structured = audit_prompts(examples, tokenizer, 140, strategy="structured")
    assert prefix["truncated"] == structured["truncated"] == 1
    assert prefix["cue_missing"] == 1
    assert structured["cue_missing"] == 0
    assert structured["hypothesis_missing"] == 0


def test_structured_64_budget_preserves_multiline_qa_choices():
    prompt = (
        "Question: " + "Which statement is correct? " * 12
        + "\nChoices:\nA. " + "An answer that\ncontinues on the next line. " * 2
        + "\nB. " + "Another lengthy option. " * 5
        + "\nC. " + "The third lengthy option. " * 5
        + "\nD. " + "A fourth lengthy option. " * 5
        + "\nAnswer (letter):"
    )
    example = TaskExample("qa", "arc_easy", prompt, "A")
    report = audit_prompts([example], ChunkTokenizer(), 64, strategy="structured")
    assert report["truncated"] == 1
    assert report["cue_missing"] == 0
    assert report["choice_label_missing"] == 0


@pytest.mark.parametrize(
    "task,prompt,cue",
    [
        ("rte", "Premise: " + "This is a very long premise. " * 20
         + "\nHypothesis: " + "A competing hypothesis. " * 12
         + "\nEntailment (yes or no):", "Entailment (yes or no):"),
        ("mrpc", "Sentence 1: " + "A long first sentence. " * 20
         + "\nSentence 2: " + "A long second sentence. " * 12
         + "\nEquivalent (no or yes):", "Equivalent (no or yes):"),
    ],
)
def test_structured_64_budget_preserves_pairwise_task_fields(task, prompt, cue):
    example = TaskExample("pair", task, prompt, "yes")
    prepared = encode_prompt(example, ChunkTokenizer(), 64, strategy="structured")
    assert prepared.truncated
    assert len(prepared.input_ids) <= 64
    assert prepared.rendered_prompt.endswith(cue)
    second_field = "\nHypothesis: " if task == "rte" else "\nSentence 2: "
    assert second_field in prepared.rendered_prompt


def test_impossible_structured_budget_fails_instead_of_silently_losing_fields():
    prompt = "Premise: one\nHypothesis: two\nRelation (entailment, neutral, or contradiction):"
    example = TaskExample("small", "mnli", prompt, "neutral")
    with pytest.raises(ValueError, match="Cannot preserve minimum task structure"):
        encode_prompt(example, CharacterTokenizer(), 8, strategy="structured")


def test_left_prefix_truncation_retains_tail():
    example = TaskExample("left", "synthetic", "abcdef", "x")
    tokenizer = CharacterTokenizer()
    prepared = encode_prompt(example, tokenizer, 4, truncation_side="left")
    assert prepared.input_ids == tokenizer.encode(example.prompt, add_special_tokens=True)[-4:]
