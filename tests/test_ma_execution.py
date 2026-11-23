import copy
from concurrent.futures import Future

import pytest
import torch

from fedproxy.ma import execution
from fedproxy.ma.execution import REQUIRED_PARALLEL_ENV, device_assignment, resolve_parallelism


@pytest.fixture(autouse=True)
def runtime_environment(monkeypatch):
    monkeypatch.delenv(REQUIRED_PARALLEL_ENV, raising=False)
    monkeypatch.delenv(execution.MIN_FREE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")


def test_dual_gpu_launch_preserves_config(monkeypatch):
    monkeypatch.setenv(REQUIRED_PARALLEL_ENV, "2")
    cfg = {"client_execution": "parallel", "max_parallel_clients": 0}
    original = copy.deepcopy(cfg)
    assert resolve_parallelism(cfg, num_clients=4, visible_gpus=2) == 2
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


@pytest.fixture
def gpu_scheduler(monkeypatch):
    now = [0]
    inflight = {}
    launches = []
    pools = []
    durations = {}
    outcomes = {}
    free = lambda device: 24

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 1
            self.closed = False
            self.device = None
            pools.append(self)

        def submit(self, worker, payload):
            device = payload["device_index"]
            assert not self.closed and device not in inflight
            assert self.device in (None, device)
            self.device = device
            client = payload["client_id"]
            future = Future()
            end = now[0] + durations.get((client, device), 1)
            inflight[device] = (end, future, outcomes.pop((client, device), client))
            launches.append((now[0], client, device))
            return future

        def shutdown(self, wait):
            self.closed = True

    def wait(futures, *, timeout, return_when):
        assert return_when == execution.FIRST_COMPLETED
        first = min(end for end, future, _ in inflight.values() if future in futures)
        now[0] = first if timeout is None else min(first, now[0] + timeout)
        for device, (end, future, result) in list(inflight.items()):
            if future in futures and end <= now[0]:
                del inflight[device]
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)

    monkeypatch.setattr(execution, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(execution, "wait", wait)
    monkeypatch.setattr(execution.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(execution.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free(device) * execution.GIB, 24 * execution.GIB))

    def run(clients, *, free_memory=None, replayed=()):
        nonlocal free
        if free_memory is not None:
            free = lambda device: free_memory(now[0], device)
        rows = [{"client_id": client, "round_id": 0} for client in clients]
        return execution.run_gpu_jobs(
            rows, 2, lambda job: job, None,
            lambda job: job["client_id"] if job["client_id"] in replayed else None,
            14 * execution.GIB,
        )

    return run, launches, pools, durations, outcomes


def test_idle_gpu_dispatches_before_slow_client_finishes_and_preserves_order(gpu_scheduler):
    run, launches, _, durations, _ = gpu_scheduler
    durations["slow", 1] = 10
    assert run(["saved", "fast", "slow", "next"], replayed={"saved"}) == ["saved", "fast", "slow", "next"]
    assert launches == [(0, "fast", 0), (0, "slow", 1), (1, "next", 0)]


def test_oom_retries_on_healthy_gpu(gpu_scheduler):
    run, launches, pools, durations, outcomes = gpu_scheduler
    outcomes["oom", 0] = torch.cuda.OutOfMemoryError("CUDA out of memory")
    durations["slow", 1] = 15
    durations["oom", 1] = 30
    free = lambda now, device: 14 if device == 0 and 1 <= now < 41 else 24
    assert run(["oom", "slow", "third", "fourth", "fifth"], free_memory=free) == [
        "oom", "slow", "third", "fourth", "fifth",
    ]
    assert launches == [(0, "oom", 0), (0, "slow", 1), (15, "third", 1), (16, "fourth", 1),
                        (17, "fifth", 1), (18, "oom", 1)]
    assert [pool.device for pool in pools] == [0, 1]
    assert all(pool.closed for pool in pools)


def test_oom_gpu_rejoins_only_when_above_threshold(gpu_scheduler):
    run, launches, pools, durations, outcomes = gpu_scheduler
    outcomes["oom", 0] = torch.cuda.OutOfMemoryError("CUDA out of memory")
    durations["slow", 1] = 50
    free = lambda now, device: 14 if device == 0 and 1 <= now < 41 else 24
    assert run(["oom", "slow", "third", "fourth"], free_memory=free) == [
        "oom", "slow", "third", "fourth",
    ]
    assert launches == [(0, "oom", 0), (0, "slow", 1), (41, "third", 0),
                        (42, "fourth", 0), (43, "oom", 0)]
    assert [pool.device for pool in pools] == [0, 1, 0]
    assert all(pool.closed for pool in pools)


def test_low_memory_worker_defers_and_both_low_memory_cards_wait(gpu_scheduler):
    run, launches, _, _, outcomes = gpu_scheduler
    outcomes["client", 0] = execution.InsufficientGpuMemory("memory changed after dispatch")
    def free(now, device):
        if now < 20 or (device == 1 and now < 30) or (device == 0 and now >= 21):
            return 0
        return 24
    assert run(["client"], free_memory=free) == ["client"]
    assert launches == [(20, "client", 0), (31, "client", 1)]


def test_non_oom_errors_are_not_retried(gpu_scheduler):
    run, launches, _, _, outcomes = gpu_scheduler
    outcomes["bad", 0] = RuntimeError("bad training input")
    with pytest.raises(RuntimeError, match="bad training input"):
        run(["bad"])
    assert launches == [(0, "bad", 0)]
