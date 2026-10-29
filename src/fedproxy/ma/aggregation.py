from __future__ import annotations

import math

import torch

from fedproxy.federated.state import TensorState

from .codec import Observation
from .schema import BlockRegistry


def _overlap(
    left: tuple[torch.Tensor, torch.Tensor, float], right: tuple[torch.Tensor, torch.Tensor, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    li, lv, _ = left
    ri, rv, _ = right
    if not li.numel() or not ri.numel():
        return lv[:0], rv[:0]
    positions = torch.searchsorted(ri, li)
    safe = positions.clamp(max=ri.numel() - 1)
    mask = (positions < ri.numel()) & (ri[safe] == li)
    return lv[mask], rv[safe[mask]]


def aggregate(
    observations: list[Observation], registry: BlockRegistry,
    previous_conflict: TensorState, cfg: dict,
) -> tuple[TensorState, TensorState, dict[str, float], dict]:
    if not observations:
        raise ValueError("No client observations")
    ac = cfg["ma"]["aggregation"]
    count = len(observations)
    beta = float(ac["beta"])
    rho = float(ac["rho"])
    eps = float(ac["epsilon_update"])
    q_min = int(ac["q_min"])
    q0 = float(ac["q0"])
    nu = float(ac["nu"])
    n_ref = float(ac["n_ref"])
    xi = float(ac["xi"])
    if n_ref <= 1:
        raise ValueError("n_ref must exceed one")
    evidence = [0.0] * count
    similarity_positive = [0.0] * count
    valid_pairs = 0
    for i in range(count):
        for j in range(i + 1, count):
            dot = norm_i = norm_j = 0.0
            common = 0
            fresh_sum_i = fresh_sum_j = 0.0
            for name in observations[i].records.keys() & observations[j].records.keys():
                li = observations[i].records[name]
                rj = observations[j].records[name]
                lv, rv = _overlap(li, rj)
                matched = lv.numel()
                if not matched:
                    continue
                common += matched
                dot += float(torch.dot(lv.double(), rv.double()))
                norm_i += float(torch.dot(lv.double(), lv.double()))
                norm_j += float(torch.dot(rv.double(), rv.double()))
                fresh_sum_i += matched * li[2]
                fresh_sum_j += matched * rj[2]
            if common < q_min or norm_i <= eps * eps or norm_j <= eps * eps:
                continue
            cosine = max(-1.0, min(1.0, dot / math.sqrt(norm_i * norm_j)))
            omega = common / (common + q0) * math.sqrt(fresh_sum_i / common * fresh_sum_j / common)
            evidence[i] += omega
            evidence[j] += omega
            similarity_positive[i] += omega * max(cosine, 0.0)
            similarity_positive[j] += omega * max(cosine, 0.0)
            valid_pairs += 1
    heterogeneity = {}
    logits = []
    for i, observation in enumerate(observations):
        chi = evidence[i] / (evidence[i] + nu) if evidence[i] else 0.0
        consensus = similarity_positive[i] / evidence[i] if evidence[i] else 0.0
        heterogeneity[observation.client_id] = chi * (1.0 - consensus)
        logits.append(math.log(observation.num_examples) + beta * chi * consensus)
    largest = max(logits)
    weights = torch.tensor([math.exp(value - largest) for value in logits], dtype=torch.float64)
    weights /= weights.sum()
    merged: TensorState = {}
    conflict: TensorState = {}
    observed_count = nonzero_count = multi_count = 0
    for entry in registry.spec.entries:
        name = entry.name
        size = entry.numel
        pos_mag = torch.zeros(size, dtype=torch.float64)
        neg_mag = torch.zeros(size, dtype=torch.float64)
        pos_weight = torch.zeros(size, dtype=torch.float64)
        neg_weight = torch.zeros(size, dtype=torch.float64)
        pos_fresh_weight = torch.zeros(size, dtype=torch.float64)
        neg_fresh_weight = torch.zeros(size, dtype=torch.float64)
        fresh_weight = torch.zeros(size, dtype=torch.float64)
        fresh_total = torch.zeros(size, dtype=torch.float64)
        weight_sq = torch.zeros(size, dtype=torch.float64)
        contributors = torch.zeros(size, dtype=torch.int32)
        seen = torch.zeros(size, dtype=torch.bool)
        for i, observation in enumerate(observations):
            record = observation.records.get(name)
            if record is None:
                continue
            idx, val, freshness = record
            seen[idx.long()] = True
            selected = val.abs() > eps
            idx = idx[selected].long()
            val = val[selected].double()
            if not idx.numel():
                continue
            w = float(weights[i])
            wp = torch.where(val > 0, torch.full_like(val, w), torch.zeros_like(val))
            wn = torch.where(val < 0, torch.full_like(val, w), torch.zeros_like(val))
            pos_mag.index_add_(0, idx, wp * val.abs())
            neg_mag.index_add_(0, idx, wn * val.abs())
            pos_weight.index_add_(0, idx, wp)
            neg_weight.index_add_(0, idx, wn)
            fw = torch.full_like(val, w * freshness)
            pos_fresh_weight.index_add_(0, idx, wp * freshness)
            neg_fresh_weight.index_add_(0, idx, wn * freshness)
            fresh_weight.index_add_(0, idx, fw)
            fresh_total.index_add_(0, idx, torch.full_like(val, w))
            weight_sq.index_add_(0, idx, fw.square())
            contributors.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
        dominant_pos = (pos_mag > eps) & (pos_mag >= rho * neg_mag)
        dominant_neg = (neg_mag > eps) & (neg_mag >= rho * pos_mag)
        update = torch.zeros(size, dtype=torch.float32)
        update[dominant_pos] = (pos_mag[dominant_pos] / pos_weight[dominant_pos]).float()
        update[dominant_neg] = (-neg_mag[dominant_neg] / neg_weight[dominant_neg]).float()
        merged[name] = update.reshape(entry.shape)
        wf = fresh_weight
        effective = torch.where(weight_sq > 0, wf.square() / weight_sq.clamp_min(1e-30), 0.0)
        reliability = (wf / fresh_total.clamp_min(1e-30)) * ((effective - 1) / (n_ref - 1)).clamp(0, 1)
        conflict_score = 1.0 - (pos_fresh_weight - neg_fresh_weight).abs() / (
            pos_fresh_weight + neg_fresh_weight
        ).clamp_min(1e-30)
        current = torch.where(
            contributors >= 2, (reliability * conflict_score).float(),
            float(xi) * previous_conflict[name].reshape(-1),
        )
        conflict[name] = current.reshape(entry.shape)
        observed_count += int(seen.sum())
        nonzero_count += int((contributors > 0).sum())
        multi_count += int((contributors >= 2).sum())
    metrics = {
        "valid_similarity_pairs": valid_pairs, "possible_similarity_pairs": count * (count - 1) // 2,
        "observed_coordinate_count": observed_count, "nonzero_coordinate_count": nonzero_count,
        "multi_contributor_coordinate_count": multi_count,
        "client_weights": {obs.client_id: float(weights[i]) for i, obs in enumerate(observations)},
        "client_heterogeneity": heterogeneity,
    }
    return merged, conflict, heterogeneity, metrics
