"""Runtime GPU scheduling; kept outside experiment configs and checkpoint hashes."""

from __future__ import annotations

import os


REQUIRED_PARALLEL_ENV = "FEDPROXY_REQUIRED_PARALLEL_CLIENTS"


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


def client_waves(jobs: list[dict], parallelism: int, execution: str):
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    for start in range(0, len(jobs), parallelism):
        wave = jobs[start:start + parallelism]
        if execution == "parallel":
            for device_index, job in enumerate(wave):
                job["device_index"] = device_index
        yield wave


def device_assignment(index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical = visible[index] if index < len(visible) and visible[index] else "unspecified"
    return f"cuda:{index}(visible_id={physical})"
