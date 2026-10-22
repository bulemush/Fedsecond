import json
from pathlib import Path

import pytest

from fedproxy.config import load_config, save_resolved_config
from fedproxy.reuse import check_reuse_compatibility, reuse_differences


PROJECT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = "/home/async/data-disk/wgh/FedKG/federatedscope/llm/data/models/Llama-2-7b-hf"


def _configs():
    target = load_config(PROJECT / "configs/ma/dense.yaml")
    source = load_config(PROJECT / "configs/experiments/llama2_full_len320_mb4.yaml")
    source["model"]["name_or_path"] = BASELINE_MODEL
    source["model"]["tokenizer_name_or_path"] = BASELINE_MODEL
    source["data"]["max_train_samples_per_task"] = 3000
    return target, source


def _without_relocation(target, source):
    target["run"].pop("reuse_model_relocation")
    target["model"]["name_or_path"] = source["model"]["name_or_path"]
    target["model"]["tokenizer_name_or_path"] = source["model"]["tokenizer_name_or_path"]


def test_reuse_accepts_matching_baseline(tmp_path):
    target, source = _configs()
    _without_relocation(target, source)
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
    _without_relocation(target, source)
    snapshot = tmp_path / "resolved_config.compress.yaml"
    save_resolved_config(source, snapshot)
    assert check_reuse_compatibility(target, tmp_path) == snapshot


def test_reuse_accepts_declared_model_relocation_after_metadata_checks(tmp_path, monkeypatch):
    target, source = _configs()
    previous = str(tmp_path / "deleted_model")
    current = tmp_path / "new_model"
    current.mkdir()
    (current / "config.json").write_text(json.dumps({
        "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
    }), encoding="utf-8")
    target["model"]["name_or_path"] = str(current)
    target["model"]["tokenizer_name_or_path"] = str(current)
    target["run"]["reuse_model_relocation"]["original"] = previous
    target["run"]["reuse_model_relocation"]["current"] = str(current)
    source["model"]["name_or_path"] = previous
    source["model"]["tokenizer_name_or_path"] = previous
    source_run = tmp_path / "baseline"
    proxy = source_run / "compression" / "proxy"
    proxy.mkdir(parents=True)
    (proxy / "config.json").write_text(json.dumps({
        "model_type": "llama", "num_hidden_layers": 16, "hidden_size": 4096,
    }), encoding="utf-8")
    (proxy.parent / "compression_manifest.json").write_text(json.dumps({
        "source_model": previous, "original_layers": 32,
        "tokenizer_sha256": "matching-vocab",
    }), encoding="utf-8")
    snapshot = source_run / "resolved_config.train.yaml"
    save_resolved_config(source, snapshot)
    monkeypatch.setattr("fedproxy.reuse._tokenizer_vocab_sha256", lambda _: "matching-vocab")
    assert check_reuse_compatibility(target, source_run) == snapshot

    monkeypatch.setattr("fedproxy.reuse._tokenizer_vocab_sha256", lambda _: "wrong-vocab")
    with pytest.raises(ValueError, match="tokenizer vocabulary differs"):
        check_reuse_compatibility(target, source_run)
    (current / "config.json").write_text(json.dumps({
        "model_type": "llama", "num_hidden_layers": 31, "hidden_size": 4096,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="layer count differs"):
        check_reuse_compatibility(target, source_run)
    (current / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="lacks config.json"):
        check_reuse_compatibility(target, source_run)
