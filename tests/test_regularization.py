import torch

from fedproxy.federated.regularization import pcr_loss


def test_pcr_value_and_gradient():
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = pcr_loss({"x": parameter}, {"x": torch.zeros(2)}, {"x": torch.tensor([0.0, 1.0])})
    assert loss.item() == 4.0
    loss.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([0.0, 4.0]))


def test_pcr_initial_step_is_zero():
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = pcr_loss({"x": parameter}, {"x": parameter.detach().clone()}, {"x": torch.ones(2)})
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(parameter.grad, torch.zeros(2))

