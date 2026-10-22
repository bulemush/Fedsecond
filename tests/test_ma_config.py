from pathlib import Path

import pytest

from fedproxy.config import ConfigError, load_config, validate_config


def test_ma_config_inherits_fedproxy_training_environment():
    project = Path(__file__).resolve().parents[1]
    parent = load_config(project / "configs/experiments/llama2_full_len320_mb4.yaml")
    for variant in ("base", "balanced", "dense", "partial", "rotate", "sparse", "golomb", "probe", "adaptive"):
        ma = load_config(project / "configs/ma" / f"{variant}.yaml")
        for section in ("model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties"):
            assert ma[section] == parent[section]
        assert ma["ma"]["enabled"]


def test_ma_rejects_unimplemented_multi_round_coverage_window():
    project = Path(__file__).resolve().parents[1]
    cfg = load_config(project / "configs/ma/base.yaml")
    cfg["ma"]["schedule"]["coverage_window"] = 2
    with pytest.raises(ConfigError, match="not implemented"):
        validate_config(cfg, command="train")
