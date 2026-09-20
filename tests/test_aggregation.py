import torch

from fedproxy.federated.aggregation import h_ties, h_ties_chunked, ties
from fedproxy.federated.sparsify import stable_global_topk
from fedproxy.federated.state import AnalysisResult


def analysis(weights, retention):
    k = len(weights)
    return AnalysisResult(
        similarity=torch.eye(k),
        heterogeneity=torch.zeros(k),
        h_norm=torch.zeros(k),
        weights=torch.tensor(weights),
        retention_rates=torch.tensor(retention),
        next_conflict={"x": torch.zeros(1)},
    )


def test_h_ties_uses_only_conforming_weights_in_denominator():
    deltas = [{"x": torch.tensor([2.0])}, {"x": torch.tensor([1.0])}, {"x": torch.tensor([-1.0])}]
    merged = h_ties(deltas, analysis([0.5, 0.3, 0.2], [1, 1, 1]), rho=1.1)
    torch.testing.assert_close(merged["x"], torch.tensor([1.625]))


def test_h_ties_returns_zero_without_dominant_sign():
    deltas = [{"x": torch.tensor([1.0])}, {"x": torch.tensor([-1.0])}]
    merged = h_ties(deltas, analysis([0.5, 0.5], [1, 1]), rho=1.1)
    torch.testing.assert_close(merged["x"], torch.zeros(1))


def test_sparse_zero_client_not_in_coordinate_denominator():
    deltas = [{"x": torch.tensor([2.0, 0.1])}, {"x": torch.tensor([1.0, 3.0])}]
    merged = h_ties(deltas, analysis([0.75, 0.25], [0.5, 0.5]), rho=1.0)
    torch.testing.assert_close(merged["x"], torch.tensor([2.0, 3.0]))


def test_stable_topk_boundaries_and_ties():
    source = {"x": torch.tensor([2.0, -2.0, 1.0])}
    torch.testing.assert_close(stable_global_topk(source, 0)["x"], torch.zeros(3))
    torch.testing.assert_close(stable_global_topk(source, 1)["x"], source["x"])
    torch.testing.assert_close(stable_global_topk(source, 2 / 3)["x"], torch.tensor([2.0, -2.0, 0.0]))


def test_ties_sum_sign_and_disjoint_mean():
    merged = ties([{"x": torch.tensor([2.0])}, {"x": torch.tensor([-1.0])}], retention=1.0)
    torch.testing.assert_close(merged["x"], torch.tensor([2.0]))


def test_chunked_h_ties_matches_reference():
    deltas = [{"x": torch.tensor([2.0, -1.0, 0.2])}, {"x": torch.tensor([1.0, -3.0, -0.2])}]
    details = analysis([0.6, 0.4], [1.0, 1.0])
    reference = h_ties(deltas, details)
    chunked = h_ties_chunked(deltas, details, chunk_size=1)
    torch.testing.assert_close(chunked["x"], reference["x"])
