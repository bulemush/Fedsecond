from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fedproxy.utils.checkpoint import load_checkpoint, save_checkpoint

from .aggregation import fedavg, h_ties_chunked, ties
from .analysis import analyze_updates_chunked
from .state import TensorState, add_states, canonical_state, subtract_states, zeros_like_state


def run_federated(
    initial_adapter: TensorState,
    client_ids: list[str],
    train_one: Callable[[str, TensorState, TensorState, int], object] | None,
    cfg: dict,
    *,
    checkpoint_root: str | Path,
    resume: str | Path | None = None,
    invariants: dict | None = None,
    train_many: Callable[[list[str], TensorState, TensorState, int], list[object]] | None = None,
):
    if (train_one is None) == (train_many is None):
        raise ValueError("Provide exactly one of train_one or train_many")
    global_adapter = canonical_state(initial_adapter)
    conflict = zeros_like_state(global_adapter)
    start_round = 0
    if resume:
        loaded = load_checkpoint(resume, expected_state=global_adapter, expected_metadata=invariants)
        global_adapter, conflict = loaded["adapter"], loaded["conflict"]
        start_round = int(loaded["metadata"]["completed_round"])
    rounds = int(cfg["federated"]["rounds"])
    history = []
    for round_id in range(start_round, rounds):
        round_base = canonical_state(global_adapter)
        if train_many is not None:
            results = train_many(client_ids, canonical_state(round_base), canonical_state(conflict), round_id)
        else:
            assert train_one is not None
            results = [
                train_one(client_id, canonical_state(round_base), canonical_state(conflict), round_id)
                for client_id in client_ids
            ]
        if [result.client_id for result in results] != client_ids:
            raise ValueError("Client results must preserve the scheduled client order")
        deltas = [subtract_states(result.adapter_state, round_base) for result in results]
        analysis = analyze_updates_chunked(
            deltas,
            r0=float(cfg["h_ties"]["r0"]),
            delta=float(cfg["h_ties"]["delta"]),
            epsilon=float(cfg["h_ties"]["epsilon"]),
            chunk_size=int(cfg["federated"]["chunk_size"]),
        )
        method = cfg["federated"]["aggregation"]
        if method == "h_ties":
            merged = h_ties_chunked(
                deltas,
                analysis,
                rho=float(cfg["h_ties"]["rho"]),
                epsilon=float(cfg["h_ties"]["epsilon"]),
                sparsify_scope=cfg["h_ties"]["sparsify_scope"],
                chunk_size=int(cfg["federated"]["chunk_size"]),
            )
        elif method == "fedavg":
            merged = fedavg(deltas, [r.num_examples for r in results], uniform=cfg["federated"].get("fedavg_weighting") == "uniform")
        elif method == "ties":
            merged = ties(deltas, float(cfg.get("ties", {}).get("retention", 1.0)))
        else:
            raise ValueError(f"Unknown aggregator: {method}")
        global_adapter = add_states(round_base, merged)
        conflict = analysis.next_conflict
        merged_numel = sum(value.numel() for value in merged.values())
        metadata = {
            "completed_round": round_id + 1,
            "next_round": round_id + 1,
            "client_ids": client_ids,
            "heterogeneity": analysis.heterogeneity.tolist(),
            "h_norm": analysis.h_norm.tolist(),
            "weights": analysis.weights.tolist(),
            "retention_rates": analysis.retention_rates.tolist(),
            "client_metrics": {result.client_id: result.metrics for result in results},
            "update_norms": [
                sum(value.square().sum() for value in delta.values()).sqrt().item() for delta in deltas
            ],
            "next_conflict_mean": sum(value.sum().item() for value in conflict.values())
            / sum(value.numel() for value in conflict.values()),
            "merged_zero_fraction": sum((value == 0).sum().item() for value in merged.values()) / merged_numel,
            "communication_bytes": {
                "adapter_per_direction": sum(value.numel() * value.element_size() for value in global_adapter.values()),
                "client_upload_total": len(client_ids) * sum(value.numel() * value.element_size() for value in global_adapter.values()),
                "server_down_total_including_conflict": 2 * len(client_ids) * sum(value.numel() * value.element_size() for value in global_adapter.values()),
            },
            **(invariants or {}),
        }
        save_checkpoint(Path(checkpoint_root) / f"round_{round_id + 1:04d}", global_adapter, conflict, metadata)
        history.append(metadata)
    return global_adapter, conflict, history
