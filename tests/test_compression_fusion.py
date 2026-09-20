import copy

import torch

from fedproxy.compression.prune import build_proxy, select_layers
from fedproxy.fusion.plug_in import replace_mapped_layers


class Tiny(torch.nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(layers)])
        self.config = type("Config", (), {"num_hidden_layers": layers})()


def test_selection_uses_score_but_preserves_original_order():
    assert select_layers([0.1, 0.9, 0.2, 0.8], 0.5) == [1, 3]


def test_non_contiguous_mapping_changes_only_selected_layers():
    original = Tiny()
    baseline = copy.deepcopy(original.state_dict())
    proxy_source = Tiny()
    proxy, mapping = build_proxy(proxy_source, [0, 2, 5])
    with torch.no_grad():
        for layer in proxy.model.layers:
            layer.weight.add_(10)
    replace_mapped_layers(original, proxy, mapping)
    for index, layer in enumerate(original.model.layers):
        if index in {0, 2, 5}:
            assert not torch.equal(layer.weight, baseline[f"model.layers.{index}.weight"])
        else:
            torch.testing.assert_close(layer.weight, baseline[f"model.layers.{index}.weight"])

