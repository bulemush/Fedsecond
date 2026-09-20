import torch

from fedproxy.federated.server import run_federated
from fedproxy.federated.state import ClientResult, add_states


def config(rounds):
    return {
        "federated": {
            "rounds": rounds,
            "aggregation": "h_ties",
            "chunk_size": 2,
        },
        "h_ties": {
            "r0": 1.0,
            "delta": 0.2,
            "epsilon": 1e-12,
            "rho": 1.1,
            "sparsify_scope": "global_adapter",
        },
    }


def deterministic_client(records=None):
    def train_one(client_id, base, conflict, round_id):
        if records is not None:
            records.append((round_id, client_id, base["x"].clone(), conflict["x"].clone()))
        sign = 1.0 if client_id == "a" else -1.0
        delta = {"x": torch.tensor([sign * (round_id + 1), 0.0])}
        return ClientResult(client_id, add_states(base, delta), 1, {"loss": 0.0})

    return train_one


def test_clients_share_round_base_and_conflict_lags_one_round(tmp_path):
    records = []
    initial = {"x": torch.zeros(2)}
    run_federated(initial, ["a", "b"], deterministic_client(records), config(2), checkpoint_root=tmp_path / "checkpoints")
    round_zero = [record for record in records if record[0] == 0]
    torch.testing.assert_close(round_zero[0][2], round_zero[1][2])
    torch.testing.assert_close(round_zero[0][3], torch.zeros(2))
    round_one = [record for record in records if record[0] == 1]
    torch.testing.assert_close(round_one[0][3], torch.ones(2))


def test_resume_matches_continuous_round_boundaries(tmp_path):
    initial = {"x": torch.zeros(2)}
    continuous, conflict_a, _ = run_federated(
        initial, ["a", "b"], deterministic_client(), config(2), checkpoint_root=tmp_path / "continuous"
    )
    run_federated(initial, ["a", "b"], deterministic_client(), config(1), checkpoint_root=tmp_path / "resumed")
    resumed, conflict_b, _ = run_federated(
        initial,
        ["a", "b"],
        deterministic_client(),
        config(2),
        checkpoint_root=tmp_path / "resumed",
        resume=tmp_path / "resumed" / "round_0001",
    )
    torch.testing.assert_close(resumed["x"], continuous["x"])
    torch.testing.assert_close(conflict_b["x"], conflict_a["x"])

