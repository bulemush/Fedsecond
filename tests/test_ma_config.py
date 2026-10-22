import copy
from pathlib import Path

import pytest

from fedproxy.config import ConfigError, load_config, validate_config


def test_ma_config_matches_reused_baseline_and_inherits_other_settings():
    project = Path(__file__).resolve().parents[1]
    parent = load_config(project / "configs/experiments/llama2_full_len320_mb4.yaml")
    baseline_model = "/home/async/data-disk/wgh/FedKG/federatedscope/llm/data/models/Llama-2-7b-hf"
    current_model = "/home/async/data-disk/wjf/wgh/Fedsecond/data/models"
    expected = copy.deepcopy(parent)
    expected["model"]["name_or_path"] = current_model
    expected["model"]["tokenizer_name_or_path"] = current_model
    expected["data"]["max_train_samples_per_task"] = 3000
    for variant in ("base", "balanced", "dense", "partial", "rotate", "sparse", "golomb", "probe", "adaptive"):
        ma = load_config(project / "configs/ma" / f"{variant}.yaml")
        for section in ("model", "compression", "data", "lora", "training", "federated", "pcr", "h_ties"):
            assert ma[section] == expected[section]
        assert ma["run"]["reuse_model_relocation"] == {
            "original": baseline_model, "current": current_model,
        }
        assert ma["ma"]["enabled"]


def test_ma_rejects_unimplemented_multi_round_coverage_window():
    project = Path(__file__).resolve().parents[1]
    cfg = load_config(project / "configs/ma/base.yaml")
    cfg["ma"]["schedule"]["coverage_window"] = 2
    with pytest.raises(ConfigError, match="not implemented"):
        validate_config(cfg, command="train")
