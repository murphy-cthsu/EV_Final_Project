import torch
import pytest

from motionprior.losses.rest_state import rest_state_l2


def test_zero_deformation_zero_loss():
    deformed = torch.zeros(10, 3)
    rest = torch.zeros(10, 3)
    loss = rest_state_l2(deformed, rest)
    assert loss.item() == pytest.approx(0.0)


def test_unit_displacement_returns_correct_norm():
    rest = torch.zeros(3, 3)
    deformed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss = rest_state_l2(deformed, rest)
    # Mean of ||(1,0,0)||^2 etc = 1.0
    assert loss.item() == pytest.approx(1.0)


def test_mismatched_shapes_raise():
    rest = torch.zeros(3, 3)
    deformed = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        rest_state_l2(deformed, rest)


def test_supports_per_point_weight():
    rest = torch.zeros(2, 3)
    deformed = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    weights = torch.tensor([1.0, 0.0])  # zero out second point
    loss = rest_state_l2(deformed, rest, weights=weights)
    # Only first point counts: ||(1,0,0)||^2 = 1; weighted mean = 1 / (1+0) = 1
    assert loss.item() == pytest.approx(1.0)


def test_is_differentiable():
    rest = torch.zeros(4, 3)
    deformed = torch.randn(4, 3, requires_grad=True)
    loss = rest_state_l2(deformed, rest)
    loss.backward()
    assert deformed.grad is not None
    assert deformed.grad.shape == (4, 3)
