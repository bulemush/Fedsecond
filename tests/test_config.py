from pathlib import Path

import pytest

from fedproxy.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name,execution,parallelism",
    [
        ("llama2_medium.yaml", "sequential", 1),
        ("llama2_large_multigpu.yaml", "parallel", 0),
        ("llama2_full_multigpu.yaml", "parallel", 0),
    ],
)
def test_experiment_configs_resolve(name, execution, parallelism):
    cfg = load_config(ROOT / "configs" / "experiments" / name)
    assert cfg["federated"]["client_execution"] == execution
    assert cfg["federated"]["max_parallel_clients"] == parallelism
    assert cfg["model"]["local_files_only"] is True


def test_sequential_execution_rejects_parallel_workers():
    cfg = load_config(ROOT / "configs" / "experiments" / "llama2_medium.yaml")
    cfg["federated"]["max_parallel_clients"] = 2
    with pytest.raises(ConfigError, match="Sequential execution"):
        validate_config(cfg)
