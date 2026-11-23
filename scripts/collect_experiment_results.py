"""Collect saved FedProxy and MA evaluations without loading models or training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fedproxy.evaluation.protocol import protocol_fingerprint, validate_mnli_evaluation
from fedproxy.evaluation.report import GLUE_TASKS, QA_TASKS, summarize


TASKS = QA_TASKS + GLUE_TASKS
AVERAGES = ("qa_avg", "glue_avg", "all_avg")
RUN_DIRS = {"evaluation", "checkpoints"}
RUN_FILES = {"training_budget.json", "resolved_config.train.yaml", "round_extension.json"}
SKIP_DIRS = RUN_DIRS | {"round_results", "compression", "clients", "exports", "logs", "metrics"}
COST_FIELDS = ("uplink_bytes", "downlink_bytes", "total_communication_bytes", "downlink_accounting")
ROUND_FIELDS = ("target_rounds", "latest_checkpoint_round", "evaluated_round")
RESULT_FIELDS = (
    "run", "method", "scenario", "model_kind", "diagnostic", *ROUND_FIELDS,
    "checkpoint_binding", "task", "task_accuracy_pct", "qa_pct", "glue_pct", "all_pct",
    *(f"{task}_pct" for task in TASKS), *COST_FIELDS, "protocol_fingerprint", "source",
)
STATUS_FIELDS = ("run", "status", *ROUND_FIELDS, "source", "detail")


def read_object(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def find_runs(root: Path):
    def fail(error):
        raise error

    for directory, children, files in os.walk(root, onerror=fail):
        if RUN_DIRS.intersection(children) or RUN_FILES.intersection(files):
            yield Path(directory)
        children[:] = sorted(name for name in children if name not in SKIP_DIRS and not name.startswith("."))


def accuracy(value) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1


def read_evaluation(path: Path, kind: str) -> dict:
    evaluation = read_object(path)
    if evaluation.get("model_kind", kind) != kind:
        raise ValueError(f"Expected model_kind={kind}")
    for flag in ("synthetic", "diagnostic"):
        if flag in evaluation and type(evaluation[flag]) is not bool:
            raise ValueError(f"Invalid {flag} flag")
    if evaluation.get("synthetic"):
        return evaluation
    scores = evaluation.get("tasks")
    if (not isinstance(scores, dict) or not scores or not set(scores).issubset(TASKS)
            or not all(accuracy(value) for value in scores.values())):
        raise ValueError("Invalid task names or accuracy outside [0, 1]")
    if len(scores) == 1:
        task, score = next(iter(scores.items()))
        recorded = evaluation.get("task_accuracy", score)
        if evaluation.get("task", task) != task or not accuracy(recorded) or recorded != score:
            raise ValueError("Single-task summary is inconsistent")
    elif set(scores) == set(TASKS):
        expected = summarize(scores)
        for key in AVERAGES:
            value = evaluation.get(key, expected[key])
            if not accuracy(value) or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Inconsistent {key}")
    else:
        raise ValueError("Expected one task or all eight tasks")
    validate_mnli_evaluation(evaluation)
    if "protocol" in evaluation or "protocol_fingerprint" in evaluation:
        protocol = evaluation.get("protocol")
        if not isinstance(protocol, dict) or evaluation.get("protocol_fingerprint") != protocol_fingerprint(protocol):
            raise ValueError("Evaluation protocol fingerprint does not match")
    return evaluation


def round_number(value) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:^|[/\\])round_(\d+)[/\\]?$", value)
    number = int(match[1]) if match else 0
    return number or None


def round_info(run: Path) -> dict:
    rounds = []
    for metadata in (run / "checkpoints").glob("round_*/metadata.json"):
        number = round_number(metadata.parent.name)
        if number is not None and metadata.is_file():
            rounds.append(number)
    info = {"latest_checkpoint_round": max(rounds, default=None)}
    for filename, key in (("round_extension.json", "target_rounds"), ("training_budget.json", "rounds")):
        path = run / filename
        if path.is_file():
            target = read_object(path).get(key)
            if type(target) is not int or target <= 0:
                raise ValueError(f"Invalid target round: {path}")
            info["target_rounds"] = target
            break
    return info


def fusion_info(run: Path, evaluation: dict) -> tuple[str, int | None]:
    path = run / "exports" / "llm_fused" / "fusion_manifest.json"
    expected_hash = evaluation.get("fusion_manifest_sha256")
    if expected_hash is not None and (not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)):
        raise ValueError("Invalid fusion_manifest_sha256")
    if not path.is_file():
        return ("manifest_missing" if expected_hash else "legacy_unbound"), None
    contents = path.read_bytes()
    if expected_hash and expected_hash != hashlib.sha256(contents).hexdigest():
        raise ValueError("Fused evaluation does not match the current fusion manifest")
    manifest = json.loads(contents)
    number = round_number(manifest.get("checkpoint")) if isinstance(manifest, dict) else None
    if number is None:
        raise ValueError("Fusion manifest has no valid checkpoint round")
    return ("verified" if expected_hash else "legacy_unbound"), number


def communication(run: Path, evaluated_round: int | None) -> dict:
    path = run / "metrics" / "aggregation.json"
    if evaluated_round is None or not path.is_file():
        return {}
    history = json.loads(path.read_bytes())
    if not isinstance(history, list):
        return {}
    rounds = history[:evaluated_round]
    if len(rounds) != evaluated_round or any(
        not isinstance(row, dict) or row.get("completed_round") != number
        for number, row in enumerate(rounds, start=1)
    ):
        return {}
    is_ma = [row.get("ma_protocol_version") == 1 for row in rounds]
    if any(is_ma) != all(is_ma):
        return {}
    method = "fedproxy_ma" if all(is_ma) else "fedproxy"
    totals = {"method": method}
    uplink = downlink = serialized = 0
    for row in rounds:
        sizes = row.get("communication_bytes")
        if not isinstance(sizes, dict):
            return totals
        if method == "fedproxy_ma":
            up = sizes.get("uplink")
            down = sizes.get("downlink", sizes.get("downlink_fp32_state_estimate"))
            serialized += "downlink" in sizes
        else:
            up = sizes.get("client_upload_total")
            down = sizes.get("server_down_total_including_conflict")
        if any(type(value) is not int or value < 0 for value in (up, down)):
            return totals
        uplink += up
        downlink += down
    if method == "fedproxy":
        accounting = "raw_fp32_state"
    elif serialized == evaluated_round:
        accounting = "serialized_packets"
    elif serialized:
        accounting = "mixed_serialized_and_legacy_estimate"
    else:
        accounting = "fp32_state_estimate"
    return {**totals, "uplink_bytes": uplink, "downlink_bytes": downlink,
            "total_communication_bytes": uplink + downlink, "downlink_accounting": accounting}


def result_row(evaluation: dict, kind: str, source: str, info: dict) -> dict:
    scores = evaluation["tasks"]
    row = {**info, "model_kind": kind, "source": source,
           "diagnostic": evaluation.get("diagnostic", False),
           "protocol_fingerprint": evaluation.get("protocol_fingerprint")}
    row.update({f"{task}_pct": score * 100 for task, score in scores.items()})
    if len(scores) == 1:
        task, score = next(iter(scores.items()))
        return {**row, "scenario": "homogeneous", "task": task, "task_accuracy_pct": score * 100}
    averages = summarize(scores)
    row.update({f"{key.removesuffix('_avg')}_pct": averages[key] * 100 for key in AVERAGES})
    return {**row, "scenario": "heterogeneous"}


def collect_run(run: Path, root: Path) -> tuple[list[dict], dict]:
    shared = {"run": run.relative_to(root).as_posix()}
    fused = run / "evaluation" / "fused_summary.json"
    legacy = fused.with_name("summary.json")
    if not fused.is_file() and legacy.is_file():
        fused = legacy
    status = {**shared, "source": fused.relative_to(root).as_posix() if fused.is_file() else ""}
    try:
        shared.update(round_info(run))
        status.update(shared)
        if not status["source"]:
            return [], {**status, "status": "pending", "detail": "缺少融合评测"}
        evaluation = read_evaluation(fused, "fused")
        if evaluation.get("synthetic"):
            return [], {**status, "status": "synthetic", "detail": "不计入正式结果"}
        binding, evaluated_round = fusion_info(run, evaluation)
        status["evaluated_round"] = evaluated_round
        required = max(shared.get(key) or 0 for key in ROUND_FIELDS[:2])
        if evaluated_round is not None and evaluated_round < required:
            return [], {**status, "status": "pending", "detail": "融合评测未覆盖最新或目标轮次"}
    except (OSError, ValueError) as error:
        return [], {**status, "status": "invalid", "detail": str(error)}

    notes = []
    try:
        cost = communication(run, evaluated_round) if binding == "verified" else {}
    except (OSError, ValueError) as error:
        cost = {}
        notes.append(f"无法读取通信记录：{error}")
    if "method" in cost:
        shared["method"] = cost["method"]
    rows = [result_row(evaluation, "fused", status["source"], {
        **shared, **cost, "evaluated_round": evaluated_round, "checkpoint_binding": binding,
    })]
    for kind in ("original", "proxy"):
        path = fused.with_name(f"{kind}_summary.json")
        if path.is_file():
            try:
                baseline = read_evaluation(path, kind)
                if not baseline.get("synthetic"):
                    rows.append(result_row(baseline, kind, path.relative_to(root).as_posix(), shared))
            except (OSError, ValueError) as error:
                notes.append(f"{kind}：{error}")
    return rows, {**status, "status": "evaluated", "detail": "; ".join(notes)}


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, root: Path, rows: list[dict], counts: Counter) -> None:
    def percent(value) -> str:
        return "—" if value is None else f"{value:.2f}%"

    lines = ["# 实验结果汇总", "", f"扫描目录：{root}", "",
             f"已评测 {counts['evaluated']}，待完成 {counts['pending']}，"
             f"无效 {counts['invalid']}，合成 {counts['synthetic']}。", "",
             "只展示融合模型；各任务精度、额外基线及通信量见 `evaluation_results.csv`，"
             "各运行的状态及异常见 `run_status.csv`。",
             "不同运行或评测协议之间不做平均或排名；同质单任务不计算 QA/GLUE/ALL。", "",
             "| 运行 | 评测轮 | 单任务 | QA | GLUE | ALL | 诊断 | 绑定 |",
             "|---|---:|---|---:|---:|---:|---|---|"]
    for row in rows:
        if row["model_kind"] != "fused":
            continue
        run = row["run"].replace("|", r"\|").replace("\r", " ").replace("\n", " ")
        task = f"{row['task']}: {percent(row['task_accuracy_pct'])}" if "task" in row else "—"
        cells = (run, str(row.get("evaluated_round") or "—"), task,
                 *(percent(row.get(key)) for key in ("qa_pct", "glue_pct", "all_pct")),
                 "是" if row["diagnostic"] else "否", row["checkpoint_binding"])
        lines.append("| " + " | ".join(cells) + " |")
    if not counts["evaluated"]:
        lines.extend(("", "未找到可汇总的融合评测。"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总 FedProxy / MA 已保存的实验评测，无需 GPU 或第三方包。")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs", help="实验目录或套件根目录")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "reports" / "experiment_summary",
                        help="结果输出目录")
    args = parser.parse_args(argv)
    root = args.runs_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"实验目录不存在：{root}。请用 --runs-dir 指向服务器上的 runs 或本地复制目录。")
    rows, statuses = [], []
    try:
        for run in find_runs(root):
            results, status = collect_run(run, root)
            rows.extend(results)
            statuses.append(status)
        counts = Counter(status["status"] for status in statuses)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "evaluation_results.csv", RESULT_FIELDS, rows)
        write_csv(output / "run_status.csv", STATUS_FIELDS, statuses)
        write_markdown(output / "experiment_summary.md", root, rows, counts)
    except OSError as error:
        parser.exit(1, f"汇总失败：{error}\n")
    print(f"已评测 {counts['evaluated']} | 待完成 {counts['pending']} | 无效 {counts['invalid']}"
          f" | 合成 {counts['synthetic']} | 输出：{output}")
    if not counts["evaluated"]:
        print("未找到可汇总的融合评测；请检查输入目录和 run_status.csv。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
