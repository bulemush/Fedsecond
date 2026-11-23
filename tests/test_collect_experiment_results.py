import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fedproxy.evaluation.protocol import protocol_fingerprint, task_overrides
from fedproxy.evaluation.report import GLUE_TASKS, QA_TASKS, summarize


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_experiment_results.py"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value).encode()
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(tmp_path):
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--runs-dir", "runs", "--output-dir", "report"],
        cwd=tmp_path, capture_output=True, encoding="utf-8", env=environment,
    )


def test_collects_nested_runs_and_keeps_baselines_separate(tmp_path):
    root = tmp_path / "runs"
    run = root / "heterogeneous_ratio30" / "MA"
    manifest = run / "exports/llm_fused/fusion_manifest.json"
    digest = write_json(manifest, {"checkpoint": "/server/run/checkpoints/round_0002"})
    write_json(run / "checkpoints/round_0002/metadata.json", {"completed_round": 2})
    write_json(run / "training_budget.json", {"rounds": 2})
    write_json(run / "metrics/aggregation.json", [
        {"completed_round": 1, "ma_protocol_version": 1, "communication_bytes": {
            "uplink": 35, "downlink_fp32_state_estimate": 100,
        }},
        {"completed_round": 2, "ma_protocol_version": 1, "communication_bytes": {
            "uplink": 40, "downlink": 60, "downlink_fp32_state_estimate": 100,
        }},
    ])
    scores = {**dict.fromkeys(QA_TASKS, 0.5), **dict.fromkeys(GLUE_TASKS, 0.75)}
    protocol = {"task_overrides": task_overrides(["mnli"])}
    fused = {**summarize(scores), "model_kind": "fused", "fusion_manifest_sha256": digest,
             "protocol": protocol, "protocol_fingerprint": protocol_fingerprint(protocol)}
    primary = run / "evaluation/fused_summary.json"
    write_json(primary, fused)
    write_json(run / "evaluation/summary.json", fused)
    write_json(run / "evaluation/original_summary.json", {**fused, "model_kind": "original"})
    single = {"tasks": {"rte": 0.625}, "task": "rte", "task_accuracy": 0.625, "diagnostic": True}
    write_json(root / "homogeneous/FedProxy/rte/evaluation/summary.json", single)
    write_json(run / "round_results/round_0001/evaluation/fused_summary.json", fused)
    write_json(run / "evaluation/archive/fused_before/evaluation/fused_summary.json", fused)

    process = run_script(tmp_path)

    assert process.returncode == 0, process.stderr
    report = tmp_path / "report"
    rows = read_rows(report / "evaluation_results.csv")
    assert len(rows) == 3
    ma, original, homogeneous = rows
    assert (ma["run"], ma["method"], ma["checkpoint_binding"]) == (
        "heterogeneous_ratio30/MA", "fedproxy_ma", "verified",
    )
    assert ma["evaluated_round"] == "2"
    assert [float(ma[key]) for key in ("qa_pct", "glue_pct", "all_pct", "mnli_pct")] == [
        50, 75, 62.5, 75,
    ]
    assert (ma["uplink_bytes"], ma["downlink_bytes"], ma["total_communication_bytes"]) == (
        "75", "160", "235",
    )
    assert ma["downlink_accounting"] == "mixed_serialized_and_legacy_estimate"
    assert original["model_kind"] == "original"
    assert original["total_communication_bytes"] == original["evaluated_round"] == ""
    assert homogeneous["task_accuracy_pct"] == "62.5"
    assert homogeneous["diagnostic"] == "True"
    assert homogeneous["all_pct"] == ""
    assert homogeneous["checkpoint_binding"] == "legacy_unbound"
    assert "62.50%" in (report / "experiment_summary.md").read_text(encoding="utf-8")
    assert json.loads(primary.read_bytes()) == fused


def test_excludes_incomplete_invalid_and_synthetic_runs(tmp_path):
    root = tmp_path / "runs"
    single = {"tasks": {"rte": 0.8}, "model_kind": "fused"}
    write_json(root / "pending/training_budget.json", {"rounds": 20})
    write_json(root / "synthetic/evaluation/fused_summary.json", {**single, "synthetic": True})
    write_json(root / "old_mnli/evaluation/fused_summary.json", {**single, "tasks": {"mnli": 0.4}})
    write_json(root / "malformed/evaluation/fused_summary.json", [])
    write_json(root / "bad_score/evaluation/fused_summary.json", {**single, "tasks": {"rte": float("nan")}})
    for name in ("resuming", "stale", "metadata_only"):
        run = root / name
        digest = hashlib.sha256(b"copied evaluation only").hexdigest()
        if name != "metadata_only":
            actual = write_json(run / "exports/llm_fused/fusion_manifest.json", {
                "checkpoint": r"C:\remote\checkpoints\round_0010",
            })
            if name == "resuming":
                digest = actual
                write_json(run / "training_budget.json", {"rounds": 10})
                write_json(run / "round_extension.json", {"target_rounds": 20})
        write_json(run / "evaluation/fused_summary.json", {**single, "fusion_manifest_sha256": digest})

    process = run_script(tmp_path)

    assert process.returncode == 0, process.stderr
    rows = read_rows(tmp_path / "report/evaluation_results.csv")
    assert [(row["run"], row["checkpoint_binding"]) for row in rows] == [
        ("metadata_only", "manifest_missing"),
    ]
    statuses = {row["run"]: row for row in read_rows(tmp_path / "report/run_status.csv")}
    assert {run: row["status"] for run, row in statuses.items()} == {
        "pending": "pending", "synthetic": "synthetic", "old_mnli": "invalid",
        "malformed": "invalid", "bad_score": "invalid", "resuming": "pending",
        "stale": "invalid", "metadata_only": "evaluated",
    }
    assert statuses["resuming"]["evaluated_round"] == "10"
    assert statuses["resuming"]["target_rounds"] == "20"
    assert "MNLI" in statuses["old_mnli"]["detail"]


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_results_directory(tmp_path, exists):
    if exists:
        (tmp_path / "runs").mkdir()

    process = run_script(tmp_path)

    if exists:
        assert process.returncode == 0
        assert "未找到可汇总" in process.stdout
        assert read_rows(tmp_path / "report/evaluation_results.csv") == []
    else:
        assert process.returncode == 2
        assert "实验目录不存在" in process.stderr
        assert not (tmp_path / "report").exists()
