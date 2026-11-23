"""Runtime GPU scheduling; kept outside experiment configs and checkpoint hashes."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context


REQUIRED_PARALLEL_ENV = "FEDPROXY_REQUIRED_PARALLEL_CLIENTS"
MIN_FREE_ENV = "FEDPROXY_MA_MIN_FREE_GIB"
GIB = 1024 ** 3
POLL_SECONDS = 10
OOM_COOLDOWN_SECONDS = 30


class InsufficientGpuMemory(RuntimeError):
    """The worker saw free memory drop below the limit before training."""


def resolve_parallelism(federated: dict, num_clients: int, visible_gpus: int) -> int:
    try:
        required = int(os.environ.get(REQUIRED_PARALLEL_ENV, "0"))
    except ValueError as error:
        raise ValueError(f"{REQUIRED_PARALLEL_ENV} must be a nonnegative integer") from error
    if required < 0:
        raise ValueError(f"{REQUIRED_PARALLEL_ENV} must be a nonnegative integer")
    if federated["client_execution"] == "parallel":
        requested = int(federated["max_parallel_clients"])
        parallelism = min(num_clients, visible_gpus if requested == 0 else requested)
        if visible_gpus < 1 or not 1 <= parallelism <= visible_gpus:
            raise ValueError(f"MA parallel execution requires enough visible GPUs: visible={visible_gpus}, "
                             f"requested={requested}, clients={num_clients}")
    else:
        parallelism = 1
    if required and (federated["client_execution"] != "parallel" or parallelism != required):
        raise ValueError(
            f"This launch requires {required} parallel clients, one per GPU; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
            f"torch.cuda.device_count()={visible_gpus}, resolved_parallel_clients={parallelism}. "
            "Check GPU visibility and use client_execution=parallel with max_parallel_clients=0."
        )
    return parallelism


def minimum_free_bytes() -> int:
    try:
        gib = float(os.environ.get(MIN_FREE_ENV, "14"))
    except ValueError as error:
        raise ValueError(f"{MIN_FREE_ENV} must be a positive number") from error
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError(f"{MIN_FREE_ENV} must be a positive number")
    return int(gib * GIB)


def device_assignment(index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical = visible[index] if index < len(visible) and visible[index] else "unspecified"
    return f"cuda:{index}(visible_id={physical})"


def run_gpu_jobs(jobs, parallelism, prepare, worker, replay, min_free_bytes):
    """Dispatch one client per device. Return results in the input order."""
    import torch

    pending = deque()
    results = {}
    for job in jobs:
        saved = replay(job)
        if saved is None:
            pending.append(job)
        else:
            prepare(job)  # Keep the logical round's downlink byte accounting on replay.
            results[job["client_id"]] = saved
    pools = {}
    active = {}
    check_after = dict.fromkeys(range(parallelism), 0.0)
    waiting = {}
    last_wait_log = -math.inf
    try:
        while pending or active:
            for device, (future, job) in list(active.items()):
                if not future.done():
                    continue
                del active[device]
                try:
                    results[job["client_id"]] = future.result()
                except InsufficientGpuMemory as error:
                    check_after[device] = time.monotonic() + POLL_SECONDS
                    waiting[device] = str(error)
                    pending.appendleft(job)
                    print(f"[ma-gpu] deferred {job['client_id']} on {device_assignment(device)}: {error}",
                          flush=True)
                except RuntimeError as error:
                    if (not isinstance(error, torch.cuda.OutOfMemoryError)
                            and "cuda out of memory" not in str(error).lower()
                            and "cuda error: out of memory" not in str(error).lower()):
                        raise
                    print(f"[ma-gpu] OOM {job['client_id']} on {device_assignment(device)}. "
                          f"Releasing worker and checking again after {OOM_COOLDOWN_SECONDS}s: {error}",
                          flush=True)
                    pools.pop(device).shutdown(wait=True)
                    check_after[device] = time.monotonic() + OOM_COOLDOWN_SECONDS
                    waiting[device] = "OOM cooldown"
                    saved = replay(job)
                    if saved is None:
                        pending.append(job)
                    else:
                        results[job["client_id"]] = saved

            for device in tuple(check_after):
                if not pending:
                    break
                now = time.monotonic()
                if device in active or now < check_after[device]:
                    continue
                free, total = torch.cuda.mem_get_info(device)
                if total <= min_free_bytes:
                    del check_after[device]
                    waiting[device] = f"capacity {total / GIB:.2f} GiB"
                    print(f"[ma-gpu] {device_assignment(device)} cannot meet "
                          f">{min_free_bytes / GIB:g} GiB with its total capacity", flush=True)
                    continue
                if free <= min_free_bytes:
                    waiting[device] = f"{free / GIB:.2f} GiB free"
                    check_after[device] = now + POLL_SECONDS
                    continue
                if device not in pools:
                    pools[device] = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
                job = pending.popleft()
                payload = {**prepare(job), "device_index": device, "min_free_bytes": min_free_bytes}
                print(f"[ma-gpu] dispatch round={job['round_id'] + 1} client={job['client_id']} "
                      f"device={device_assignment(device)} free_gib={free / GIB:.2f} "
                      f"rejoined={device in waiting}", flush=True)
                waiting.pop(device, None)
                active[device] = (pools[device].submit(worker, payload), job)
                del payload

            now = time.monotonic()
            if pending and not check_after:
                raise ValueError(f"No visible GPU can meet >{min_free_bytes / GIB:g} GiB. "
                                 f"Check the selected GPUs and {MIN_FREE_ENV}.")
            if pending and waiting and now - last_wait_log >= 60:
                devices = ", ".join(f"{device_assignment(i)} ({reason})" for i, reason in waiting.items())
                print(f"[ma-gpu] pending={len(pending)} active={len(active)}; "
                      f"waiting for >{min_free_bytes / GIB:g} GiB on {devices}", flush=True)
                last_wait_log = now
            next_check = min(
                (when for device, when in check_after.items() if device not in active),
                default=now + POLL_SECONDS,
            )
            timeout = max(0.0, min(POLL_SECONDS, next_check - now))
            if active:
                wait([future for future, _ in active.values()],
                     timeout=timeout if pending else None, return_when=FIRST_COMPLETED)
            elif pending:
                time.sleep(timeout)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    return [results[job["client_id"]] for job in jobs]
