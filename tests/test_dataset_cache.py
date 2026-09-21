import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fedproxy.data.registry import dataset_storage_path, load_registered


class FakeDataset:
    _fingerprint = "fingerprint-123"

    def __init__(self, origin):
        self.origin = origin

    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True)
        (path / "state.json").write_text("{}", encoding="utf-8")


def test_dataset_storage_path_separates_revision_and_split(tmp_path):
    path = dataset_storage_path(tmp_path, "arc_easy", "validation", "refs/pr/7")
    assert path == tmp_path / "datasets" / "arc_easy" / "refs_pr_7" / "validation"


def test_download_once_then_load_local_copy(monkeypatch, tmp_path):
    downloads = []
    local_loads = []

    def fake_download(path, name, split, revision, cache_dir):
        downloads.append((path, name, split, revision, cache_dir))
        return FakeDataset("remote")

    def fake_local(path):
        local_loads.append(path)
        assert (Path(path) / "state.json").is_file()
        return FakeDataset("local")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_download, load_from_disk=fake_local),
    )

    first = load_registered("obqa", data_dir=tmp_path)
    second = load_registered("obqa", data_dir=tmp_path)
    target = dataset_storage_path(tmp_path, "obqa", "train")

    assert first.origin == "local"
    assert second.origin == "local"
    assert len(downloads) == 1
    assert len(local_loads) == 2
    assert downloads[0][4] == str(tmp_path / "huggingface")
    metadata = json.loads((target / "_fedproxy_source.json").read_text(encoding="utf-8"))
    assert metadata["source_path"] == "allenai/openbookqa"
    assert metadata["source_kind"] == "huggingface"
    assert metadata["dataset_fingerprint"] == "fingerprint-123"


def test_manual_save_to_disk_copy_preempts_huggingface(monkeypatch, tmp_path):
    manual = tmp_path / "obqa" / "train"
    manual.mkdir(parents=True)
    (manual / "state.json").write_text("{}", encoding="utf-8")
    downloads = []

    def fail_download(*args, **kwargs):
        downloads.append((args, kwargs))
        raise AssertionError("Hugging Face must not be used when a local dataset exists")

    def fake_local(path):
        assert (Path(path) / "state.json").is_file()
        return FakeDataset("local")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fail_download, load_from_disk=fake_local),
    )
    result = load_registered("obqa", data_dir=tmp_path)
    target = dataset_storage_path(tmp_path, "obqa", "train")

    assert result.origin == "local"
    assert downloads == []
    metadata = json.loads((target / "_fedproxy_source.json").read_text(encoding="utf-8"))
    assert metadata["source_kind"] == "local_save_to_disk"
    assert metadata["local_source"] == str(manual.resolve())


def test_local_jsonl_preempts_huggingface_and_is_materialized(monkeypatch, tmp_path):
    raw = tmp_path / "obqa" / "train.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"id":"one"}\n', encoding="utf-8")
    calls = []

    def fake_load_dataset(path, name=None, **kwargs):
        calls.append((path, name, kwargs))
        assert path == "json"
        assert kwargs["data_files"] == {"train": str(raw.resolve())}
        return FakeDataset("raw")

    def fake_local(path):
        assert (Path(path) / "state.json").is_file()
        return FakeDataset("local")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset, load_from_disk=fake_local),
    )
    result = load_registered("obqa", data_dir=tmp_path)
    target = dataset_storage_path(tmp_path, "obqa", "train")

    assert result.origin == "local"
    assert len(calls) == 1
    metadata = json.loads((target / "_fedproxy_source.json").read_text(encoding="utf-8"))
    assert metadata["source_kind"] == "local_json"
    assert metadata["local_source"] == str(raw.resolve())
