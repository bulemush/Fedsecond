from pathlib import Path

import pytest

from fedproxy.config import load_config, save_resolved_config
from fedproxy.reuse import check_reuse_compatibility, reuse_differences


PROJECT = Path(__file__).resolve().parents[1]


def _configs():
    target = load_config(PROJECT / "configs/ma/dense.yaml")
    source = load_config(PROJECT / "configs/experiments/llama2_full_len320_mb4.yaml")
    return target, source


def test_reuse_accepts_matching_baseline(tmp_path):
    target, source = _configs()
    snapshot = tmp_path / "resolved_config.train.yaml"
    save_resolved_config(source, snapshot)
    assert check_reuse_compatibility(target, tmp_path) == snapshot


def test_reuse_accepts_same_local_paths_and_runtime_flags(tmp_path):
    target, source = _configs()
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data"
    model_dir.mkdir()
    data_dir.mkdir()
    source["model"]["name_or_path"] = str(model_dir)
    source["model"]["tokenizer_name_or_path"] = None
    source["data"]["local_dir"] = str(data_dir)
    target["model"]["name_or_path"] = str(tmp_path / "." / "model")
    target["model"]["tokenizer_name_or_path"] = str(model_dir / ".." / "model")
    target["data"]["local_dir"] = str(data_dir / ".." / "data")
    target["model"]["local_files_only"] = False
    target["compression"]["device_map"] = "auto"
    target["data"]["truncation_audit"] = False
    assert reuse_differences(target, source) == []


def test_reuse_reports_exact_incompatible_fields_and_source(tmp_path):
    target, source = _configs()
    target["model"]["name_or_path"] = "different/model"
    target["data"]["max_input_length"] = 64
    target["data"]["revisions"] = {"obqa": "new"}
    snapshot = tmp_path / "resolved_config.train.yaml"
    save_resolved_config(source, snapshot)
    with pytest.raises(ValueError) as error:
        check_reuse_compatibility(target, tmp_path)
    message = str(error.value)
    assert str(snapshot) in message
    assert "model.name_or_path: MA='different/model'" in message
    assert "data.max_input_length: MA=64, baseline=320" in message
    assert "data.revisions.obqa: MA='new', baseline=<missing>" in message


def test_reuse_does_not_accept_different_existing_model_directories(tmp_path):
    target, source = _configs()
    for name in ("model_a", "model_b"):
        (tmp_path / name).mkdir()
    source["model"]["name_or_path"] = str(tmp_path / "model_a")
    target["model"]["name_or_path"] = str(tmp_path / "model_b")
    assert any(field == "model.name_or_path" for field, _, _ in reuse_differences(target, source))
    target["model"]["name_or_path"] = source["model"]["name_or_path"]
    target["model"]["tokenizer_name_or_path"] = str(tmp_path / "model_b")
    assert any(field == "model.tokenizer_name_or_path"
               for field, _, _ in reuse_differences(target, source))


def test_reuse_requires_baseline_snapshot(tmp_path):
    target, _ = _configs()
    with pytest.raises(FileNotFoundError, match="resolved_config.train.yaml"):
        check_reuse_compatibility(target, tmp_path)


def test_reuse_reads_compression_snapshot_when_train_snapshot_is_absent(tmp_path):
    target, source = _configs()
    snapshot = tmp_path / "resolved_config.compress.yaml"
    save_resolved_config(source, snapshot)
    assert check_reuse_compatibility(target, tmp_path) == snapshot
