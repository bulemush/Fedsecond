"""Validate that a run can safely reuse a baseline proxy and data partition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    """Normalize runtime differences and an explicitly declared model move."""
    result = {}
    relocation = cfg.get("run", {}).get("reuse_model_relocation")
    for section in REUSED_SECTIONS:
        values = copy.deepcopy(cfg.get(section, {}))
        for field in RUNTIME_ONLY_FIELDS[section]:
            values.pop(field, None)
        if section == "model" and "name_or_path" in values:
            # load_tokenizer falls back to name_or_path when this field is null.
            values["tokenizer_name_or_path"] = (
                values.get("tokenizer_name_or_path") or values["name_or_path"]
            )
            if isinstance(relocation, dict):
                for field in ("name_or_path", "tokenizer_name_or_path"):
                    if values[field] == relocation.get("current"):
                        values[field] = relocation.get("original")
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


def _tokenizer_vocab_sha256(model_cfg: dict) -> str:
    from fedproxy.models.backbone import load_tokenizer

    tokenizer = load_tokenizer(model_cfg)
    return hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_declared_model_relocation(target: dict, original: dict, source_run: Path) -> None:
    relocation = target.get("run", {}).get("reuse_model_relocation")
    if relocation is None:
        return
    if not isinstance(relocation, dict):
        raise ValueError("run.reuse_model_relocation must contain original and current paths")
    previous, current = relocation.get("original"), relocation.get("current")
    if not all(isinstance(value, str) and Path(value).is_absolute() for value in (previous, current)):
        raise ValueError("run.reuse_model_relocation requires absolute original and current paths")
    if previous == current:
        raise ValueError("run.reuse_model_relocation paths must differ")
    for cfg, expected, label in ((target, current, "MA"), (original, previous, "baseline")):
        model = cfg.get("model", {})
        if model.get("name_or_path") != expected or (
            model.get("tokenizer_name_or_path") or model.get("name_or_path")
        ) != expected:
            raise ValueError(f"{label} model and tokenizer paths do not match the declared relocation")

    model_dir = Path(current)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Relocated model lacks config.json: {config_path}")
    manifest_path = source_run / "compression" / "compression_manifest.json"
    proxy_config_path = source_run / "compression" / "proxy" / "config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_model") != previous:
        raise ValueError("Declared original model path differs from the baseline compression manifest")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    proxy_config = json.loads(proxy_config_path.read_text(encoding="utf-8"))
    if model_config.get("num_hidden_layers") != manifest.get("original_layers"):
        raise ValueError("Relocated model layer count differs from the baseline source model")
    for field in ("model_type", "hidden_size", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "vocab_size"):
        if field in proxy_config and model_config.get(field) != proxy_config[field]:
            raise ValueError(f"Relocated model {field} differs from the baseline proxy")
    expected_tokenizer = manifest.get("tokenizer_sha256")
    if not expected_tokenizer:
        raise ValueError("Baseline compression manifest lacks tokenizer_sha256")
    if _tokenizer_vocab_sha256(target["model"]) != expected_tokenizer:
        raise ValueError("Relocated tokenizer vocabulary differs from the baseline proxy")


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
    _check_declared_model_relocation(target, original, source_run)
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
