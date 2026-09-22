import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest
import torch

from fedproxy.federated.state import (
    ClientResult,
    ParameterSpec,
    flatten_state,
    unflatten_state,
)
from fedproxy.workflows import _pack_client_result, _unpack_client_result


def _packed_state_echo(payload):
    state = unflatten_state(payload["vector"], payload["spec"])
    return flatten_state(state, payload["spec"])


def test_parallel_client_result_uses_one_tensor_and_round_trips():
    adapter = {
        f"layer.{index}.lora_A": torch.tensor([float(index), float(index + 1)])
        for index in range(256)
    }
    spec = ParameterSpec.from_state(adapter)
    result = ClientResult("client_03", adapter, 17, {"loss": 0.25})

    packed = _pack_client_result(result, spec)

    assert set(packed) == {
        "client_id",
        "adapter_vector",
        "num_examples",
        "metrics",
        "schema_hash",
    }
    assert isinstance(packed["adapter_vector"], torch.Tensor)
    assert sum(isinstance(value, torch.Tensor) for value in packed.values()) == 1
    assert packed["adapter_vector"].numel() == spec.total_numel

    rebuilt = _unpack_client_result(packed, spec)
    assert rebuilt.client_id == result.client_id
    assert rebuilt.num_examples == result.num_examples
    assert rebuilt.metrics == result.metrics
    assert rebuilt.adapter_state.keys() == result.adapter_state.keys()
    for name in adapter:
        torch.testing.assert_close(rebuilt.adapter_state[name], adapter[name])


def test_parallel_client_result_rejects_schema_mismatch():
    source = {"a": torch.ones(2)}
    packed = _pack_client_result(
        ClientResult("client", source, 1, {}),
        ParameterSpec.from_state(source),
    )
    incompatible = ParameterSpec.from_state({"b": torch.ones(2)})

    with pytest.raises(ValueError, match="incompatible adapter schema"):
        _unpack_client_result(packed, incompatible)


def test_packed_state_crosses_spawn_process_boundary():
    state = {
        f"parameter.{index}": torch.tensor([float(index)])
        for index in range(1024)
    }
    spec = ParameterSpec.from_state(state)
    vector = flatten_state(state, spec).share_memory_()
    payload = {"vector": vector, "spec": spec}
    outputs = []

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        for output in executor.map(_packed_state_echo, [payload, payload]):
            outputs.append(output.clone())
        del output

    assert len(outputs) == 2
    for output in outputs:
        torch.testing.assert_close(output, vector)
