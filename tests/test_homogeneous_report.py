import importlib.util
import json
from pathlib import Path

import pytest

from fedproxy.evaluation.report import GLUE_TASKS, QA_TASKS, summarize_homogeneous_task


def _suite_module():
    path = Path(__file__).parents[1] / "scripts" / "summarize_homogeneous_suite.py"
    spec = importlib.util.spec_from_file_location("summarize_homogeneous_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_task_summary_requires_exact_task():
    assert summarize_homogeneous_task({"obqa": 0.42}, "obqa")["task_accuracy"] == 0.42
    with pytest.raises(ValueError, match="only obqa"):
        summarize_homogeneous_task({"obqa": 0.42, "rte": 0.7}, "obqa")


def test_suite_collects_eight_fused_models(monkeypatch):
    contents = {}
    for index, task in enumerate(QA_TASKS + GLUE_TASKS):
        path = Path("runs") / f"llama2_homogeneous_{task}_len320_mb4_seed42" / "evaluation" / "fused_summary.json"
        accuracy = (index + 1) / 10
        contents[str(path)] = json.dumps({
            "model_kind": "fused", "synthetic": False, "task": task,
            "tasks": {task: accuracy}, "task_accuracy": accuracy,
            "protocol": {"tasks": ["openbookqa" if task == "obqa" else task], "num_fewshot": 0, "batch_size": 1},
        })
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: contents[str(self)])
    result = _suite_module().collect(Path("runs"))
    assert len(result["sources"]) == 8
    assert result["qa_avg"] == pytest.approx(0.25)
    assert result["glue_avg"] == pytest.approx(0.65)
    assert result["all_avg"] == pytest.approx(0.45)
