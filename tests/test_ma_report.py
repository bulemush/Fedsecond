import json
import hashlib
from pathlib import Path

import pytest
import yaml

from fedproxy.ma.report import compare, paired_noninferiority


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _run(path: Path, ma: bool, protocol: str = "same", seed: int = 42, score: float | None = None):
    config = {name: {} for name in (
        "model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties"
    )}
    config["run"] = {"seed": seed}
    path.mkdir()
    (path / "resolved_config.train.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    fusion = path / "exports/llm_fused/fusion_manifest.json"
    _write(fusion, {"checkpoint": str(path / "checkpoints/round_0001"),
                    "adapter_sha256": "adapter"})
    _write(path / "evaluation/fused_summary.json", {
        "protocol_fingerprint": protocol, "all_avg": score if score is not None else (0.60 if ma else 0.61),
        "qa_avg": 0.5, "glue_avg": 0.7, "tasks": {"obqa": 0.5},
        "fusion_manifest_sha256": hashlib.sha256(fusion.read_bytes()).hexdigest(),
    })
    _write(path / "evaluation/raw/fused_lm_eval.json", {"versions": {"obqa": 1}})
    _write(path / "data_manifest.json", {"partition_hash": "p"})
    (path / "compression").mkdir()
    (path / "compression/compression_manifest.json").write_text("{}", encoding="utf-8")
    communication = (
        {"uplink": 25, "downlink_fp32_state_estimate": 50} if ma else
        {"client_upload_total": 100, "server_down_total_including_conflict": 200}
    )
    metadata = {"communication_bytes": communication,
                "client_metrics": {"a": {"elapsed_seconds": 10}}}
    if ma:
        metadata.update({"ma_protocol_version": 1, "round_precommit_seconds": 20,
                         "analysis": {"observed_coordinate_count": 10,
                                      "multi_contributor_coordinate_count": 5}})
        _write(path / "metrics/round_0001.json", {"round_wall_seconds": 22})
    _write(path / "checkpoints/round_0001/metadata.json", metadata)


def test_report_compares_only_matching_runs(tmp_path):
    baseline, candidate = tmp_path / "base", tmp_path / "ma"
    _run(baseline, False)
    _run(candidate, True)
    result = compare(baseline, [candidate])
    assert result["candidates"][0]["all_delta_pp"] == pytest.approx(-1.0)
    assert result["candidates"][0]["uplink_ratio"] == pytest.approx(0.25)
    assert result["candidates"][0]["ma_round_wall_seconds"] == 22


def test_report_uses_same_reuse_normalization(tmp_path):
    baseline, candidate = tmp_path / "base", tmp_path / "ma"
    _run(baseline, False)
    _run(candidate, True)
    for run, alias in ((baseline, "model"), (candidate, "model/../model")):
        model = tmp_path / "model"
        model.mkdir(exist_ok=True)
        path = run / "resolved_config.train.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["model"].update({"name_or_path": str(tmp_path / alias),
                                "tokenizer_name_or_path": None if run == baseline else str(model)})
        config["model"]["local_files_only"] = run == baseline
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert compare(baseline, [candidate])["candidates"][0]["all_delta_pp"] == pytest.approx(-1.0)


def test_report_rejects_different_evaluation_protocol(tmp_path):
    baseline, candidate = tmp_path / "base", tmp_path / "ma"
    _run(baseline, False)
    _run(candidate, True, protocol="different")
    with pytest.raises(ValueError, match="Evaluation protocol mismatch"):
        compare(baseline, [candidate])


def test_report_rejects_evaluation_from_another_checkpoint(tmp_path):
    run = tmp_path / "ma"
    _run(run, True)
    fusion = run / "exports/llm_fused/fusion_manifest.json"
    _write(fusion, {"checkpoint": str(run / "checkpoints/round_0002")})
    with pytest.raises(ValueError, match="final costed checkpoint"):
        compare(run, [])


def test_report_rejects_stale_ma_evaluation_after_refusion(tmp_path):
    run = tmp_path / "ma"
    _run(run, True)
    fusion = run / "exports/llm_fused/fusion_manifest.json"
    _write(fusion, {"checkpoint": str(run / "checkpoints/round_0001"),
                    "adapter_sha256": "different"})
    with pytest.raises(ValueError, match="predates the current fusion export"):
        compare(run, [])


def test_paired_report_uses_independent_matched_seeds(tmp_path):
    baselines, candidates = [], []
    for seed in (42, 43, 44):
        baseline, candidate = tmp_path / f"base_{seed}", tmp_path / f"ma_{seed}"
        _run(baseline, False, seed=seed, score=0.61)
        _run(candidate, True, seed=seed, score=0.605)
        baselines.append(baseline)
        candidates.append(candidate)
    result = paired_noninferiority(baselines, candidates)
    assert result["mean_all_delta_pp"] == pytest.approx(-0.5)
    assert result["all_noninferior"]
    with pytest.raises(ValueError, match="same unique training seed"):
        paired_noninferiority(baselines, candidates[::-1])
