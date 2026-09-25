from pathlib import Path

import pytest

from fedproxy.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name,execution,parallelism",
    [
        ("llama2_medium.yaml", "sequential", 1),
        ("llama2_large_multigpu.yaml", "parallel", 0),
        ("llama2_large_len320.yaml", "parallel", 0),
        ("llama2_large_structured64.yaml", "parallel", 0),
        ("llama2_full_multigpu.yaml", "parallel", 0),
    ],
)
def test_experiment_configs_resolve(name, execution, parallelism):
    cfg = load_config(ROOT / "configs" / "experiments" / name)
    assert cfg["federated"]["client_execution"] == execution
    assert cfg["federated"]["max_parallel_clients"] == parallelism
    assert cfg["model"]["local_files_only"] is True


def test_truncation_ablations_are_isolated():
    long_cfg = load_config(ROOT / "configs" / "experiments" / "llama2_large_len320.yaml")
    structured = load_config(ROOT / "configs" / "experiments" / "llama2_large_structured64.yaml")
    assert long_cfg["run"]["output_dir"] != structured["run"]["output_dir"]
    assert long_cfg["run"]["output_dir"] != "runs/llama2_large_multigpu"
    assert long_cfg["data"]["max_input_length"] == 320
    assert long_cfg["run"]["reuse_artifacts_from"] == "runs/llama2_large_multigpu"
    assert structured["data"]["max_input_length"] == 64
    assert structured["data"]["prompt_truncation_strategy"] == "structured"


def test_sequential_execution_rejects_parallel_workers():
    cfg = load_config(ROOT / "configs" / "experiments" / "llama2_medium.yaml")
    cfg["federated"]["max_parallel_clients"] = 2
    with pytest.raises(ConfigError, match="Sequential execution"):
        validate_config(cfg)
