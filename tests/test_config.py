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
        ("llama2_full_len320_mb4.yaml", "parallel", 0),
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


def test_full_len320_mb4_preserves_effective_batch_and_full_budget():
    cfg = load_config(ROOT / "configs" / "experiments" / "llama2_full_len320_mb4.yaml")
    assert cfg["run"]["output_dir"] == "runs/llama2_full_len320_mb4_seed42"
    assert cfg["run"]["diagnostic"] is True
    assert cfg["data"]["max_input_length"] == 320
    assert cfg["data"]["max_train_samples_per_task"] == 5000
    assert cfg["training"]["local_epochs"] == 10
    assert cfg["federated"]["rounds"] == 10
    assert cfg["training"]["micro_batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 4
    assert cfg["training"]["effective_batch_size"] == 16


def test_sequential_execution_rejects_parallel_workers():
    cfg = load_config(ROOT / "configs" / "experiments" / "llama2_medium.yaml")
    cfg["federated"]["max_parallel_clients"] = 2
    with pytest.raises(ConfigError, match="Sequential execution"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "task,harness_task",
    [
        ("obqa", "openbookqa"),
        ("arc_easy", "arc_easy"),
        ("arc_challenge", "arc_challenge"),
        ("commonsense_qa", "commonsense_qa"),
        ("sst2", "sst2"),
        ("mrpc", "mrpc"),
        ("rte", "rte"),
        ("mnli", "mnli"),
    ],
)
def test_homogeneous_suite_configs(task, harness_task):
    cfg = load_config(ROOT / "configs" / "experiments" / f"llama2_homogeneous_{task}_len320_mb4.yaml")
    source = load_config(ROOT / "configs" / "experiments" / "llama2_homogeneous_obqa_len320_mb4.yaml")
    assert cfg["run"]["output_dir"] == f"runs/llama2_homogeneous_{task}_len320_mb4_seed42"
    assert cfg["data"]["scenario"] == "homogeneous"
    assert cfg["data"]["tasks"] == [task]
    assert cfg["data"]["num_clients"] == 4
    assert cfg["data"]["max_train_samples_per_task"] == 3000
    assert cfg["data"]["max_input_length"] == 320
    assert cfg["training"]["local_epochs"] == 5
    assert cfg["training"]["micro_batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 4
    assert cfg["training"]["effective_batch_size"] == 16
    assert cfg["federated"]["rounds"] == 10
    assert cfg["evaluation"]["harness_tasks"] == [harness_task]
    assert cfg["model"] == source["model"]
    assert cfg["compression"] == source["compression"]
    assert cfg["data"]["local_dir"] == source["data"]["local_dir"]
    assert cfg["run"].get("reuse_compression_from") == (
        None if task == "obqa" else "runs/llama2_homogeneous_obqa_len320_mb4_seed42"
    )
