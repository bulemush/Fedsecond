import torch

from fedproxy.federated.analysis import analyze_updates, analyze_updates_chunked


def state(values):
    return {"adapter": torch.tensor(values, dtype=torch.float32)}


def test_conflict_sign_cases():
    same_positive = analyze_updates([state([1]), state([2]), state([3])])
    same_negative = analyze_updates([state([-1]), state([-2]), state([-3])])
    balanced = analyze_updates([state([1]), state([-1])])
    all_zero = analyze_updates([state([0]), state([0])])
    assert same_positive.next_conflict["adapter"].item() == 0
    assert same_negative.next_conflict["adapter"].item() == 0
    assert balanced.next_conflict["adapter"].item() == 1
    assert all_zero.next_conflict["adapter"].item() == 1


def test_identical_updates_have_uniform_weights_and_zero_normalized_h():
    result = analyze_updates([state([1, 2]), state([1, 2]), state([1, 2])])
    torch.testing.assert_close(result.h_norm, torch.zeros(3))
    torch.testing.assert_close(result.weights, torch.full((3,), 1 / 3))


def test_opposite_updates_use_absolute_cosine_for_weights():
    result = analyze_updates([state([1, 0]), state([-1, 0]), state([0, 1])])
    assert result.weights[0] > result.weights[2]
    assert torch.isfinite(result.similarity).all()


def test_single_client_boundary_rule():
    result = analyze_updates([state([2, -1])])
    torch.testing.assert_close(result.weights, torch.ones(1))
    torch.testing.assert_close(result.next_conflict["adapter"], torch.zeros(2))


def test_chunked_analysis_matches_reference():
    updates = [state([1, -2, 0, 4]), state([-1, -2, 0, 1]), state([2, 3, 0, -1])]
    reference = analyze_updates(updates)
    chunked = analyze_updates_chunked(updates, chunk_size=2)
    torch.testing.assert_close(chunked.similarity, reference.similarity)
    torch.testing.assert_close(chunked.weights, reference.weights)
    torch.testing.assert_close(chunked.next_conflict["adapter"], reference.next_conflict["adapter"])
