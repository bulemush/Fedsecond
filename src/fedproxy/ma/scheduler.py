from __future__ import annotations

import random

from .schema import BlockRegistry, ClientPlan


def plan_round(
    registry: BlockRegistry, client_ids: list[str], round_id: int, cfg: dict
) -> dict[str, ClientPlan]:
    if not client_ids:
        raise ValueError("No participating clients")
    sc = cfg["ma"]["schedule"]
    blocks = list(registry.blocks)
    total = len(blocks)
    anchors_count = int(sc["anchor_blocks"])
    train_count = int(sc["train_blocks"])
    upload_count = int(sc["upload_blocks"])
    if not 1 <= anchors_count <= upload_count <= train_count <= total:
        raise ValueError("Invalid MA block budgets")
    if int(sc.get("coverage_window", 1)) != 1:
        raise ValueError("Multi-round coverage windows are not implemented")
    permutation = blocks[:]
    random.Random(int(sc.get("seed", cfg["run"]["seed"]))).shuffle(permutation)
    anchors = {permutation[(round_id * anchors_count + j) % total] for j in range(anchors_count)}
    degree = {block: 0 for block in permutation if block not in anchors}
    output = {}
    for order, client_id in enumerate(client_ids):
        # Advance each client's own window by its non-anchor upload budget.
        # Balancing only the per-round block degree can repeatedly select the
        # same layers for a client and expire its unsent residuals.
        stride = upload_count - anchors_count
        start = (order * total // len(client_ids) + round_id * stride) % total
        ordered = [
            permutation[(start + offset) % total] for offset in range(total)
            if permutation[(start + offset) % total] not in anchors
        ]
        chosen = set(ordered[:stride])
        for block in chosen:
            degree[block] += 1
        upload = anchors | chosen
        train = upload | set(ordered[stride:stride + train_count - upload_count])
        plan = ClientPlan(client_id, round_id, frozenset(train), frozenset(upload), frozenset(anchors))
        plan.validate(registry)
        output[client_id] = plan
    minimum = int(sc.get("q_block", 0))
    if minimum and degree and min(degree.values()) < minimum:
        raise ValueError("Per-round non-anchor overlap is not met by these budgets")
    return output


def audit_residual_delivery(
    registry: BlockRegistry, client_ids: list[str], rounds: int, cfg: dict
) -> dict[str, int]:
    """Check layer opportunities; coordinate Top-k delivery is a separate question."""
    plans = [plan_round(registry, client_ids, round_id, cfg) for round_id in range(rounds)]
    if not cfg["ma"]["residual"]["enabled"]:
        return {"checked_pending_layers": 0, "max_observed_wait_rounds": 0,
                "right_censored_pending_layers": 0}
    ttl = int(cfg["ma"]["residual"]["max_age"])
    checked = censored = longest = 0
    for source_round, by_client in enumerate(plans):
        for client in client_ids:
            pending = by_client[client].train_blocks - by_client[client].upload_blocks
            for block in pending:
                checked += 1
                next_upload = next((later for later in range(source_round + 1, rounds)
                                    if block in plans[later][client].upload_blocks), None)
                if next_upload is None:
                    censored += 1
                    if source_round + ttl < rounds:
                        raise ValueError(
                            f"Residual upload opportunity missing before TTL: "
                            f"client={client} source_round={source_round + 1} block={block}"
                        )
                else:
                    delay = next_upload - source_round
                    longest = max(longest, delay)
                    if delay > ttl:
                        raise ValueError(
                            f"Residual upload opportunity exceeds TTL: "
                            f"client={client} source_round={source_round + 1} "
                            f"block={block} wait={delay} ttl={ttl}"
                        )
    return {"checked_pending_layers": checked, "max_observed_wait_rounds": longest,
            "right_censored_pending_layers": censored}
