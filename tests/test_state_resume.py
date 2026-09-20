import torch

from fedproxy.federated.state import ParameterSpec, flatten_state, unflatten_state, validate_state
from fedproxy.utils.checkpoint import load_checkpoint, save_checkpoint


def test_flatten_roundtrip_and_canonical_order():
    state = {"z": torch.arange(6.0).reshape(2, 3), "a": torch.tensor([9.0])}
    spec = ParameterSpec.from_state(state)
    rebuilt = unflatten_state(flatten_state(state, spec), spec)
    assert list(rebuilt) == ["a", "z"]
    for name in state:
        torch.testing.assert_close(rebuilt[name], state[name])


def test_schema_shape_mismatch_fails():
    spec = ParameterSpec.from_state({"x": torch.zeros(2)})
    try:
        validate_state({"x": torch.zeros(3)}, spec)
    except ValueError as error:
        assert "Shape mismatch" in str(error)
    else:
        raise AssertionError("Expected shape mismatch")


def test_checkpoint_roundtrip(tmp_path):
    adapter = {"x": torch.tensor([1.0, 2.0])}
    conflict = {"x": torch.tensor([0.0, 1.0])}
    path = tmp_path / "round_0001"
    save_checkpoint(path, adapter, conflict, {"completed_round": 1})
    loaded = load_checkpoint(path, expected_state=adapter)
    torch.testing.assert_close(loaded["adapter"]["x"], adapter["x"])
    assert loaded["metadata"]["next_round"] if "next_round" in loaded["metadata"] else 1

