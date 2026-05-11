import torch
import pytest

from motionprior.losses.gating import compute_gating_weights, AdaptiveAlpha


def test_gating_weights_decrease_with_energy():
    energies = torch.tensor([0.0, 0.5, 1.0, 2.0])
    w = compute_gating_weights(energies, alpha=1.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] < w[0]
    assert w[2] < w[1]
    assert w[3] < w[2]
    assert torch.all(w >= 0.0) and torch.all(w <= 1.0)


def test_gating_weights_zero_alpha_returns_ones():
    energies = torch.tensor([0.1, 5.0, 100.0])
    w = compute_gating_weights(energies, alpha=0.0)
    torch.testing.assert_close(w, torch.ones_like(w))


def test_gating_weights_rejects_negative_alpha():
    with pytest.raises(ValueError):
        compute_gating_weights(torch.tensor([0.0]), alpha=-1.0)


def test_gating_weights_rejects_negative_energy():
    with pytest.raises(ValueError):
        compute_gating_weights(torch.tensor([-0.1]), alpha=1.0)


def test_adaptive_alpha_initial_value_returns_alpha0():
    a = AdaptiveAlpha(alpha0=2.0, momentum=0.99)
    # First update with energy=1 should return alpha0 / 1 = 2.0
    assert a(torch.tensor(1.0)).item() == pytest.approx(2.0)


def test_adaptive_alpha_normalizes_by_ema():
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.5)
    # Step 1: warm-start ema = 4.0; alpha = 1 / 4 = 0.25
    out1 = a(torch.tensor(4.0))
    assert out1.item() == pytest.approx(0.25)
    # Step 2: ema = 0.5 * 4 + 0.5 * 2 = 3.0; alpha = 1 / 3
    out2 = a(torch.tensor(2.0))
    assert out2.item() == pytest.approx(1.0 / 3.0)


def test_adaptive_alpha_clamps_zero_ema():
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.5, eps=1e-6)
    out = a(torch.tensor(0.0))
    # Must not divide by zero
    assert torch.isfinite(out).all()


def test_gating_end_to_end_matches_formula():
    energies = torch.tensor([0.1, 0.2, 0.4])
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.0)  # no smoothing
    # mean energy = (0.1+0.2+0.4)/3 = 0.2333...; alpha = 1/0.2333 = 4.2857...
    alpha = a(energies.mean())
    w = compute_gating_weights(energies, alpha=alpha.item())
    expected = torch.exp(-alpha * energies)
    torch.testing.assert_close(w, expected)
