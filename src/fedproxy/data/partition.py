from __future__ import annotations

import hashlib
import json
import random


def fixed_subsample(indices: list[int], limit: int | None, seed: int) -> list[int]:
    if limit is None or len(indices) <= limit:
        return list(indices)
    rng = random.Random(seed)
    return sorted(rng.sample(indices, limit))


def iid_partition(indices: list[int], num_clients: int, seed: int) -> dict[str, list[int]]:
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    output = {f"client_{i}": [] for i in range(num_clients)}
    for position, value in enumerate(shuffled):
        output[f"client_{position % num_clients}"].append(value)
    return output


def heterogeneous_partition(tasks: list[str], selected: dict[str, list[int]]) -> dict[str, dict]:
    return {
        f"client_{index}": {"task": task, "indices": list(selected[task])}
        for index, task in enumerate(tasks)
    }


def partition_hash(partition: dict) -> str:
    payload = json.dumps(partition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()

