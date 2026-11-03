import copy

import pytest

from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, client_waves, device_assignment, resolve_parallelism


@pytest.fixture(autouse=True)
def runtime_environment(monkeypatch):
    monkeypatch.delenv(REQUIRED_PARALLEL_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")


def test_dual_gpu_launch_assigns_four_clients_in_two_waves_without_changing_config(monkeypatch):
    monkeypatch.setenv(REQUIRED_PARALLEL_ENV, "2")
    cfg = {"client_execution": "parallel", "max_parallel_clients": 0}
    original = copy.deepcopy(cfg)
    count = resolve_parallelism(cfg, num_clients=4, visible_gpus=2)
    waves = list(client_waves([{"client_id": f"client_{i}"} for i in range(4)], count, "parallel"))
    assert [[(job["client_id"], job["device_index"]) for job in wave] for wave in waves] == [
        [("client_0", 0), ("client_1", 1)], [("client_2", 0), ("client_3", 1)],
    ]
    assert device_assignment(0) == "cuda:0(visible_id=2)"
    assert device_assignment(1) == "cuda:1(visible_id=3)"
    assert cfg == original


@pytest.mark.parametrize("execution,maximum,visible", [
    ("parallel", 0, 1), ("parallel", 1, 2), ("sequential", 0, 2),
])
def test_required_two_clients_cannot_silently_run_one(monkeypatch, execution, maximum, visible):
    monkeypatch.setenv(REQUIRED_PARALLEL_ENV, "2")
    with pytest.raises(ValueError, match="requires 2 parallel clients"):
        resolve_parallelism({"client_execution": execution, "max_parallel_clients": maximum}, 4, visible)


def test_legacy_single_gpu_and_cpu_sequential_behavior_is_preserved():
    assert resolve_parallelism({"client_execution": "parallel", "max_parallel_clients": 0}, 4, 1) == 1
    assert resolve_parallelism({"client_execution": "sequential", "max_parallel_clients": 0}, 4, 0) == 1
    with pytest.raises(ValueError, match="enough visible GPUs"):
        resolve_parallelism({"client_execution": "parallel", "max_parallel_clients": 2}, 4, 1)


@pytest.mark.parametrize("value", ["two", "-1"])
def test_invalid_runtime_requirement_is_reported(monkeypatch, value):
    monkeypatch.setenv(REQUIRED_PARALLEL_ENV, value)
    with pytest.raises(ValueError, match="nonnegative integer"):
        resolve_parallelism({"client_execution": "parallel", "max_parallel_clients": 0}, 4, 2)
