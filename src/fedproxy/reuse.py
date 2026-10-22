"""Validate that a run can safely reuse a baseline proxy and data partition."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from fedproxy.config import load_config


REUSED_SECTIONS = ("model", "compression", "data")
# These flags control access, device placement, or an optional audit. They do
# not change the already saved proxy or the training examples.
RUNTIME_ONLY_FIELDS = {
    "model": frozenset({"local_files_only"}),
    "compression": frozenset({"device_map"}),
    "data": frozenset({"truncation_audit"}),
}
LOCAL_PATH_FIELDS = {
    "model": frozenset({"name_or_path", "tokenizer_name_or_path"}),
    "data": frozenset({"local_dir"}),
}
_MISSING = object()


def _canonical_local_path(value):
    if not isinstance(value, str) or not value:
        return value
    try:
        return str(Path(value).expanduser().resolve(strict=True))
    except (OSError, RuntimeError):
        # A Hub identifier or a path that no longer exists is not proof that
        # two model/data sources are identical.
        return value


def normalized_reuse_sections(cfg: dict) -> dict:
    """Normalize only differences that cannot change the reused artifacts."""
    result = {}
    for section in REUSED_SECTIONS:
        values = copy.deepcopy(cfg.get(section, {}))
        for field in RUNTIME_ONLY_FIELDS[section]:
            values.pop(field, None)
        if section == "model" and "name_or_path" in values:
            # load_tokenizer falls back to name_or_path when this field is null.
            values["tokenizer_name_or_path"] = (
                values.get("tokenizer_name_or_path") or values["name_or_path"]
            )
        for field in LOCAL_PATH_FIELDS.get(section, ()):
            if field in values:
                values[field] = _canonical_local_path(values[field])
        result[section] = values
    return result


def reuse_differences(target: dict, source: dict) -> list[tuple[str, object, object]]:
    expected = normalized_reuse_sections(source)
    actual = normalized_reuse_sections(target)
    differences = []

    def visit(path: str, target_value: object, source_value: object) -> None:
        if isinstance(target_value, dict) and isinstance(source_value, dict):
            for field in sorted(target_value.keys() | source_value.keys()):
                visit(
                    f"{path}.{field}",
                    target_value.get(field, _MISSING),
                    source_value.get(field, _MISSING),
                )
        elif target_value != source_value:
            differences.append((path, target_value, source_value))

    for section in REUSED_SECTIONS:
        visit(section, actual[section], expected[section])
    return differences


def _display(value: object) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def check_reuse_compatibility(target: dict, source_run: str | Path) -> Path:
    """Return the baseline snapshot, or raise with every incompatible field."""
    source_run = Path(source_run)
    source_config = next(
        (path for path in (
            source_run / "resolved_config.train.yaml",
            source_run / "resolved_config.compress.yaml",
        ) if path.is_file()),
        None,
    )
    if source_config is None:
        raise FileNotFoundError(
            f"Reused proxy source lacks resolved_config.train.yaml or "
            f"resolved_config.compress.yaml: {source_run}"
        )
    original = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise ValueError(f"Baseline resolved configuration is not a YAML mapping: {source_config}")
    differences = reuse_differences(target, original)
    if differences:
        details = "\n".join(
            f"  {field}: MA={_display(actual)}, baseline={_display(expected)}"
            for field, actual, expected in differences
        )
        raise ValueError(
            f"Reused proxy/partition configuration differs from {source_config}:\n"
            f"{details}\nAlign the MA configuration with this baseline, or set "
            f"run.reuse_artifacts_from to a matching baseline."
        )
    return source_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MA baseline artifact compatibility")
    parser.add_argument("--config", required=True, help="MA experiment configuration")
    parser.add_argument("--source", required=True, help="baseline run directory")
    args = parser.parse_args()
    try:
        source = check_reuse_compatibility(load_config(args.config), args.source)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Reuse configuration compatible with {source}")


if __name__ == "__main__":
    main()
