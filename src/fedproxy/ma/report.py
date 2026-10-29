from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from itertools import islice
from pathlib import Path

import yaml

from fedproxy.reuse import normalized_reuse_sections


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class HarnessMismatchError(ValueError):
    """A real task-definition difference that rerunning the candidate cannot fix."""


_FUNCTION_REPR = re.compile(r"<function ([\w.<>]+) at 0x[0-9a-fA-F]+>")


def _stable_task_config(value):
    # harness.py uses json.dumps(default=str). TaskConfig callables therefore
    # contain process-local addresses, including inside functools.partial reprs.
    # Remove only function addresses; retain names and every other task setting.
    if isinstance(value, dict):
        return {key: _stable_task_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_task_config(item) for item in value]
    if isinstance(value, str) and (_FUNCTION_REPR.fullmatch(value) or value.startswith("functools.partial(")):
        return _FUNCTION_REPR.sub(r"<function \1>", value)
    return value


def _harness_details(raw: dict) -> dict:
    return {"versions": raw.get("versions"),
            "configs": _stable_task_config(raw.get("configs")), "n-shot": raw.get("n-shot")}


def _detail_differences(baseline, candidate, prefix="harness"):
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(baseline.keys() | candidate.keys()):
            if key not in baseline or key not in candidate:
                yield f"{prefix}.{key}: missing in {'baseline' if key not in baseline else 'MA'}"
            else:
                yield from _detail_differences(baseline[key], candidate[key], f"{prefix}.{key}")
    elif isinstance(baseline, list) and isinstance(candidate, list) and len(baseline) == len(candidate):
        for index, (left, right) in enumerate(zip(baseline, candidate)):
            yield from _detail_differences(left, right, f"{prefix}[{index}]")
    elif baseline != candidate:
        yield f"{prefix}: baseline={repr(baseline)[:240]}, MA={repr(candidate)[:240]}"


def _harness_mismatch(baseline: str | Path, candidate: str | Path) -> HarnessMismatchError:
    relative = Path("evaluation/raw/fused_lm_eval.json")
    base_file, candidate_file = Path(baseline) / relative, Path(candidate) / relative
    differences = list(islice(_detail_differences(
        _harness_details(_load(base_file)), _harness_details(_load(candidate_file))
    ), 12))
    return HarnessMismatchError(
        f"harness_detail_hash mismatch: {candidate}\n"
        + "\n".join(differences)
        + f"\nRaw evaluations: {base_file} ; {candidate_file}\n"
        "Re-evaluate the existing fused models with the same lm-eval task definitions. "
        "Training checkpoints can be retained."
    )


def summarize_run(path: str | Path) -> dict:
    run = Path(path)
    resolved = yaml.safe_load((run / "resolved_config.train.yaml").read_text(encoding="utf-8"))
    shared_config = {
        **normalized_reuse_sections(resolved),
        **{name: resolved[name] for name in ("lora", "training", "federated", "pcr", "h_ties")},
    }
    shared_hash = hashlib.sha256(json.dumps(shared_config, sort_keys=True).encode()).hexdigest()
    evaluation = _load(run / "evaluation" / "fused_summary.json")
    raw_eval = _load(run / "evaluation" / "raw" / "fused_lm_eval.json")
    harness_detail = {key: raw_eval.get(key) for key in ("versions", "configs", "n-shot")}
    raw_harness_hash = hashlib.sha256(json.dumps(harness_detail, sort_keys=True, default=str).encode()).hexdigest()
    harness_hash = hashlib.sha256(json.dumps(_harness_details(raw_eval), sort_keys=True).encode()).hexdigest()
    partition_hash = _load(run / "data_manifest.json")["partition_hash"]
    compression_hash = hashlib.sha256(
        (run / "compression" / "compression_manifest.json").read_bytes()
    ).hexdigest()
    metadata = sorted((run / "checkpoints").glob("round_*/metadata.json"))
    if not metadata:
        raise ValueError(f"No completed checkpoints in {run}")
    rounds = [_load(item) for item in metadata]
    is_ma = rounds[0].get("ma_protocol_version") == 1
    if any((item.get("ma_protocol_version") == 1) != is_ma for item in rounds):
        raise ValueError("Mixed checkpoint methods")
    fusion_path = run / "exports" / "llm_fused" / "fusion_manifest.json"
    fusion = _load(fusion_path)
    selected = Path(fusion["checkpoint"])
    expected = run / "checkpoints" / f"round_{len(rounds):04d}"
    if selected.resolve() != expected.resolve():
        raise ValueError(f"Fused evaluation does not use the final costed checkpoint: {run}")
    manifest_hash = hashlib.sha256(fusion_path.read_bytes()).hexdigest()
    bound = evaluation.get("fusion_manifest_sha256") == manifest_hash
    if evaluation.get("fusion_manifest_sha256") is not None and not bound:
        raise ValueError(f"Fused evaluation predates the current fusion export: {run}")
    if is_ma and not bound:
        raise ValueError(f"MA fused evaluation is not bound to its checkpoint: {run}")
    if is_ma:
        up = sum(int(item["communication_bytes"]["uplink"]) for item in rounds)
        serialized_rounds = sum("downlink" in item["communication_bytes"] for item in rounds)
        down = sum(int(item["communication_bytes"]["downlink"] if "downlink" in item["communication_bytes"]
                       else item["communication_bytes"]["downlink_fp32_state_estimate"]) for item in rounds)
        down_reference = sum(int(item["communication_bytes"]["downlink_fp32_state_estimate"]) for item in rounds)
        down_accounting = ("serialized_packets" if serialized_rounds == len(rounds) else
                           "mixed_serialized_and_legacy_estimate" if serialized_rounds else "fp32_state_estimate")
        train_seconds = sum(
            float(client.get("elapsed_seconds", 0))
            for item in rounds for client in item["client_metrics"].values()
        )
        timing_paths = [run / "metrics" / f"round_{number:04d}.json"
                        for number in range(1, len(rounds) + 1)]
        wall_seconds = (
            sum(float(_load(path)["round_wall_seconds"]) for path in timing_paths)
            if all(path.exists() for path in timing_paths) else None
        )
        mean_multi = sum(
            item["analysis"]["multi_contributor_coordinate_count"] /
            max(1, item["analysis"]["observed_coordinate_count"])
            for item in rounds
        ) / len(rounds)
    else:
        up = sum(int(item["communication_bytes"]["client_upload_total"]) for item in rounds)
        down = sum(int(item["communication_bytes"]["server_down_total_including_conflict"]) for item in rounds)
        down_reference = down
        down_accounting = "raw_fp32_state"
        train_seconds = sum(
            float(client.get("elapsed_seconds", 0))
            for item in rounds for client in item["client_metrics"].values()
        )
        wall_seconds = None
        mean_multi = None
    return {
        "run": str(run), "method": "fedproxy_ma" if is_ma else "fedproxy",
        "seed": resolved["run"].get("seed"),
        "rounds": len(rounds), "protocol_fingerprint": evaluation["protocol_fingerprint"],
        "evaluation_checkpoint_verified": bound,
        "shared_training_config_hash": shared_hash,
        "partition_hash": partition_hash, "compression_manifest_hash": compression_hash,
        "harness_detail_hash": harness_hash,
        "harness_detail_raw_hash": raw_harness_hash,
        "harness_hash_normalization": "python_function_addresses_v1",
        "all_avg": evaluation.get("all_avg"), "qa_avg": evaluation.get("qa_avg"),
        "glue_avg": evaluation.get("glue_avg"), "tasks": evaluation["tasks"],
        "uplink_bytes": up, "downlink_bytes": down, "downlink_accounting": down_accounting,
        "downlink_full_fp32_reference_bytes": down_reference,
        "total_communication_bytes": up + down,
        # Compatibility aliases: consult downlink_accounting for the measurement basis.
        "downlink_state_bytes_estimate": down,
        "total_state_bytes_estimate": up + down,
        "sum_client_train_seconds": train_seconds, "ma_round_wall_seconds": wall_seconds,
        "mean_multi_contributor_observed_fraction": mean_multi,
        "byte_accounting_note": (
            "MA uplink is serialized packet length; optimized downlink counts serialized P+active G packets, "
            "including headers and exact zero/constant encoding. Legacy MA downlink uses the FP32 P+G estimate. "
            "FedProxy counts raw FP32 P/G tensors. Network framing and initialization are excluded."
        ),
    }


def compare(baseline: str | Path, candidates: list[str | Path]) -> dict:
    base = summarize_run(baseline)
    results = []
    for path in candidates:
        item = summarize_run(path)
        if item["protocol_fingerprint"] != base["protocol_fingerprint"]:
            raise ValueError(f"Evaluation protocol mismatch: {path}")
        if item["shared_training_config_hash"] != base["shared_training_config_hash"]:
            raise ValueError(f"Shared model/data/training configuration mismatch: {path}")
        for key in ("partition_hash", "compression_manifest_hash"):
            if item[key] != base[key]:
                raise ValueError(f"{key} mismatch: {path}")
        if item["harness_detail_hash"] != base["harness_detail_hash"]:
            raise _harness_mismatch(baseline, path)
        if item["rounds"] != base["rounds"]:
            raise ValueError(f"Training round mismatch: {path}")
        item["all_delta_pp"] = (
            None if item["all_avg"] is None or base["all_avg"] is None
            else 100 * (item["all_avg"] - base["all_avg"])
        )
        item["uplink_ratio"] = item["uplink_bytes"] / base["uplink_bytes"]
        item["downlink_ratio"] = item["downlink_bytes"] / base["downlink_bytes"]
        item["total_state_byte_ratio"] = item["total_state_bytes_estimate"] / base["total_state_bytes_estimate"]
        results.append(item)
    return {"baseline": base, "candidates": results,
            "inference": "Descriptive only; paired multi-seed non-inferiority is not established."}


_ONE_SIDED_T95 = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
                  7: 1.895, 8: 1.860, 9: 1.833}


def paired_noninferiority(
    baselines: list[str | Path], candidates: list[str | Path], margin_pp: float = 1.0
) -> dict:
    if len(baselines) != len(candidates) or not 3 <= len(baselines) <= 10:
        raise ValueError("Paired report requires 3–10 baseline/candidate seed pairs")
    pairs = []
    seeds = set()
    for baseline, candidate in zip(baselines, candidates, strict=True):
        row = compare(baseline, [candidate])
        base, item = row["baseline"], row["candidates"][0]
        if base["seed"] != item["seed"] or base["seed"] in seeds:
            raise ValueError("Each pair must have the same unique training seed")
        if base["all_avg"] is None or item["all_avg"] is None:
            raise ValueError("Paired report requires heterogeneous ALL scores")
        seeds.add(base["seed"])
        pairs.append({"seed": base["seed"], "delta_pp": item["all_delta_pp"],
                      "baseline": base, "candidate": item})
    first = pairs[0]["baseline"]
    for pair in pairs[1:]:
        for key in ("shared_training_config_hash", "partition_hash",
                    "compression_manifest_hash", "harness_detail_hash", "protocol_fingerprint"):
            if pair["baseline"][key] != first[key]:
                raise ValueError(f"Paired seeds differ in {key}")
    differences = [pair["delta_pp"] for pair in pairs]
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    lower = mean - _ONE_SIDED_T95[len(differences) - 1] * sem
    qa_delta = statistics.mean(100 * (pair["candidate"]["qa_avg"] - pair["baseline"]["qa_avg"])
                               for pair in pairs)
    glue_delta = statistics.mean(100 * (pair["candidate"]["glue_avg"] - pair["baseline"]["glue_avg"])
                                 for pair in pairs)
    tasks = sorted(first["tasks"])
    task_deltas = {
        task: statistics.mean(100 * (pair["candidate"]["tasks"][task] -
                                     pair["baseline"]["tasks"][task]) for pair in pairs)
        for task in tasks
    }
    return {
        "pairs": [{"seed": pair["seed"], "all_delta_pp": pair["delta_pp"]} for pair in pairs],
        "margin_pp": margin_pp, "mean_all_delta_pp": mean,
        "one_sided_95_lower_pp": lower, "all_noninferior": lower > -margin_pp,
        "qa_delta_pp": qa_delta, "glue_delta_pp": glue_delta, "task_delta_pp": task_deltas,
        "task_point_guardrails_pass": all(value >= -2.0 for value in task_deltas.values()),
        "group_point_guardrails_pass": qa_delta >= -1.0 and glue_delta >= -1.0,
        "mean_uplink_ratio": statistics.mean(pair["candidate"]["uplink_ratio"] for pair in pairs),
        "mean_total_state_byte_ratio": statistics.mean(
            pair["candidate"]["total_state_byte_ratio"] for pair in pairs
        ),
        "assumption": "One-sided paired t interval across independent training seeds; inspect distribution and widen seed count if uncertain.",
        "cost_note": "MA uses serialized packet bytes where available and legacy downlink estimates otherwise; the baseline counts raw FP32 states. These ratios do not prove GPU compute savings.",
    }
