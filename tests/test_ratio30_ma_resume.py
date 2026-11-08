"""MA resume must not depend on retained FedProxy training checkpoints."""

import shutil
from pathlib import Path

import pytest

from fedproxy import ratio30 as ratio
from fedproxy.config import save_resolved_config
from fedproxy.ma import homogeneous as suite
from test_ma_homogeneous import _write
from test_ratio30 import isolate_environment, simulation  # noqa: F401


@pytest.fixture
def interrupted_ma_suite(simulation, monkeypatch):
    root = simulation.root
    ratio.main(["run", "--root", str(root), "--method", "fedproxy"])

    def interrupt(name, frozen, gpu_ids):
        cfg = suite._yaml(frozen)
        if name == "train" and cfg["ma"]["enabled"]:
            run = Path(cfg["run"]["output_dir"])
            save_resolved_config(cfg, run / "resolved_config.train.yaml")
            simulation.rounds(run, cfg, 1, 3)
            raise RuntimeError("interrupted MA after round 3")
        return simulation.stage(name, frozen, gpu_ids)

    with monkeypatch.context() as patch:
        patch.setattr(suite, "_stage", interrupt)
        with pytest.raises(RuntimeError, match="interrupted MA"):
            ratio.main(["run", "--root", str(root), "--method", "ma"])
    simulation.calls.clear()
    return simulation


def _prune_fedproxy(root):
    for task in ratio.TASKS:
        checkpoints = root / "FedProxy" / suite.run_name(task) / "checkpoints"
        for checkpoint in checkpoints.glob("round_*"):
            if checkpoint.name != "round_0010":
                shutil.rmtree(checkpoint)


def test_ma_all_resumes_after_fedproxy_checkpoint_cleanup(interrupted_ma_suite):
    simulation = interrupted_ma_suite
    root = simulation.root
    _prune_fedproxy(root)
    first_task = ratio.TASKS[0]
    ma = root / "MA" / suite.run_name(first_task)
    checkpoint = ma / "checkpoints/round_0003/metadata.json"
    residual = ma / "clients/client_0/round_0003/residual.pt"
    saved = checkpoint.read_bytes(), residual.read_bytes()

    ratio.main(["validate", "--root", str(root), "--method", "ma", "--gpu-ids", "2,3"])
    assert simulation.calls == []
    ratio.main(["run", "--root", str(root), "--method", "ma", "--gpu-ids", "2,3"])

    assert (ma, "resume") in simulation.calls
    assert (ma, "train") not in simulation.calls
    assert (checkpoint.read_bytes(), residual.read_bytes()) == saved
    assert len([call for call in simulation.calls if call[1] == "train"]) == 7
    assert all(run.parent.name == "MA" for run, _ in simulation.calls)
    assert (root / "MA/results.csv").is_file()
    for task in ratio.TASKS:
        checkpoints = root / "FedProxy" / suite.run_name(task) / "checkpoints"
        assert [path.name for path in checkpoints.glob("round_*")] == ["round_0010"]

    simulation.calls.clear()
    ratio.main(["run", "--root", str(root), "--method", "ma", "--gpu-ids", "2,3"])
    assert simulation.calls == []


def test_selected_fedproxy_still_rejects_pruned_checkpoints(interrupted_ma_suite):
    root = interrupted_ma_suite.root
    _prune_fedproxy(root)
    with pytest.raises(ValueError, match="Incomplete/non-contiguous checkpoint"):
        ratio.main(["validate", "--root", str(root), "--method", "fedproxy"])


@pytest.mark.parametrize("damage", ["missing_residual", "incomplete_checkpoint", "changed_partition"])
def test_ma_resume_still_validates_its_own_state(interrupted_ma_suite, damage):
    simulation = interrupted_ma_suite
    root = simulation.root
    _prune_fedproxy(root)
    task = ratio.TASKS[0]
    ma = root / "MA" / suite.run_name(task)
    if damage == "missing_residual":
        (ma / "clients/client_0/round_0003/residual.pt").unlink()
        error, message = FileNotFoundError, "residual states"
    elif damage == "incomplete_checkpoint":
        path = ma / "checkpoints/round_0003/metadata.json"
        value = suite._json(path)
        value["complete"] = False
        _write(path, value)
        error, message = ValueError, "Incomplete/non-contiguous checkpoint"
    else:
        path = root / "FedProxy" / suite.run_name(task) / "data_manifest.json"
        value = suite._json(path)
        value["clients"]["client_0"][0] = 100
        _write(path, value)
        error, message = ValueError, "Partition fingerprint"
    with pytest.raises(error, match=message):
        ratio.main(["run", "--root", str(root), "--method", "ma"])
    assert simulation.calls == []
