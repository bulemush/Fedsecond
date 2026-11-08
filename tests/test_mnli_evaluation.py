import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fedproxy.data.prompts import MNLI_LABELS, mnli_prompt
from fedproxy.data.tasks import convert_row
from fedproxy.evaluation.harness import evaluate_with_lm_eval
from fedproxy.evaluation.protocol import (
    archive_previous_mnli_evaluation,
    protocol_fingerprint,
    task_overrides,
    validate_mnli_evaluation,
)


def _backend(monkeypatch, labels=MNLI_LABELS):
    calls = []
    task = SimpleNamespace(config=SimpleNamespace(
        task="mnli", validation_split="validation_matched", test_split=None,
        metadata={"version": 1}, doc_to_choice=["True", "Neither", "False"],
    ))
    task.set_config = lambda key, value: setattr(task.config, key, value)
    feature = SimpleNamespace(names=list(labels))
    task.validation_docs = lambda: SimpleNamespace(features={"label": feature})

    class Manager:
        @property
        def task_index(self):
            pytest.fail("Do not depend on version-specific dict/Entry indexes")

        def load_task_or_group(self, names):
            assert names == ["mnli"]
            return {"mnli": task}

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"mnli": {"acc,none": 0.85}}}

    package = ModuleType("lm_eval")
    package.__path__ = []
    package.evaluator = SimpleNamespace(simple_evaluate=evaluate)
    tasks = ModuleType("lm_eval.tasks")
    tasks.TaskManager = Manager
    monkeypatch.setitem(sys.modules, "lm_eval", package)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks)
    return task, calls


@pytest.mark.parametrize("label", range(3))
def test_evaluation_uses_exact_training_prompt_and_schema_mapping(tmp_path, monkeypatch, label):
    task, calls = _backend(monkeypatch)
    row = {"premise": "  A premise.\n第二行 ", "hypothesis": " A hypothesis without a period ",
           "label": label}
    dataset = SimpleNamespace(features={"label": SimpleNamespace(names=list(MNLI_LABELS))})
    trained = convert_row("mnli", row, 0, dataset)
    output = tmp_path / "evaluation.json"
    result = evaluate_with_lm_eval("existing-model", ["rte", "mnli"], output, dtype="bfloat16")
    assert task.config.doc_to_text(row) == trained.prompt
    assert task.config.doc_to_choice[label] == trained.response
    assert task.config.doc_to_target == "label"
    assert task.config.target_delimiter == " "
    assert calls[0]["tasks"] == ["rte", task]
    assert calls[0]["num_fewshot"] == 0
    assert calls[0]["apply_chat_template"] is False
    assert calls[0]["cache_requests"] is False
    assert result["fedproxy_task_overrides"] == task_overrides(["mnli"])
    assert json.loads(output.read_text())["results"]["mnli"]["acc,none"] == 0.85


def test_other_tasks_keep_the_native_evaluation_path(tmp_path, monkeypatch):
    _, calls = _backend(monkeypatch)
    # A non-MNLI evaluation must not load task objects at all.
    monkeypatch.delitem(sys.modules, "lm_eval.tasks")
    result = evaluate_with_lm_eval("model", ["rte"], tmp_path / "rte.json")
    assert calls[0]["tasks"] == ["rte"]
    assert "fedproxy_task_overrides" not in result


def test_unexpected_label_order_fails_instead_of_swapping_classes(tmp_path, monkeypatch):
    _, calls = _backend(monkeypatch, ("neutral", "entailment", "contradiction"))
    with pytest.raises(ValueError, match="label schema"):
        evaluate_with_lm_eval("model", ["mnli"], tmp_path / "wrong.json")
    assert not calls
    assert not (tmp_path / "wrong.json").exists()


def _summary():
    protocol = {
        "tasks": ["mnli"], "num_fewshot": 0, "batch_size": 1,
        "primary_metric": "acc", "dtype": "bfloat16",
        "task_overrides": task_overrides(["mnli"]),
    }
    return {
        "tasks": {"mnli": 0.85}, "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
    }


@pytest.mark.parametrize("damage", ["legacy", "prompt", "choices", "fingerprint"])
def test_stale_or_changed_protocol_cannot_be_treated_as_current(damage):
    summary = _summary()
    validate_mnli_evaluation(summary)
    if damage == "legacy":
        del summary["protocol"]["task_overrides"]
    elif damage == "fingerprint":
        summary["protocol_fingerprint"] = "stale"
    else:
        summary["protocol"]["task_overrides"]["mnli"][damage if damage == "choices" else "prompt_template"] = "wrong"
    if damage != "fingerprint":
        summary["protocol_fingerprint"] = protocol_fingerprint(summary["protocol"])
    with pytest.raises(ValueError, match="Re-evaluate.*retraining is not required"):
        validate_mnli_evaluation(summary)


def test_unaffected_task_results_remain_valid_without_new_metadata():
    validate_mnli_evaluation({"tasks": {"rte": 0.8}})


def test_old_outputs_are_copied_without_touching_training_artifacts(tmp_path):
    evaluation = tmp_path / "evaluation"
    (evaluation / "raw").mkdir(parents=True)
    summary = {"tasks": {"mnli": 0.6443}, "protocol_fingerprint": "legacy"}
    for name in ("fused_summary.json", "summary.json"):
        (evaluation / name).write_text(json.dumps(summary))
    (evaluation / "raw/fused_lm_eval.json").write_text('{"old": true}')
    checkpoint = tmp_path / "checkpoints/adapter.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"existing trained weights")
    archive = archive_previous_mnli_evaluation(tmp_path, "fused", ["mnli"])
    assert archive is not None
    assert json.loads((archive / "fused_summary.json").read_text()) == summary
    assert (archive / "raw/fused_lm_eval.json").read_text() == '{"old": true}'
    assert (evaluation / "fused_summary.json").is_file()
    assert checkpoint.read_bytes() == b"existing trained weights"


def test_current_protocol_is_not_archived_again(tmp_path):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "fused_summary.json").write_text(json.dumps(_summary()))
    assert archive_previous_mnli_evaluation(tmp_path, "fused", ["mnli"]) is None
    assert not (evaluation / "archive").exists()


def test_training_prompt_text_is_unchanged():
    assert mnli_prompt("P", "H") == (
        "Premise: P\nHypothesis: H\nRelation (entailment, neutral, or contradiction):"
    )
