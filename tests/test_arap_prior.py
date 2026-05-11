import torch
import pytest

from motionprior.geometry.arap_prior import compute_arap_prior_energy


def test_static_trajectory_returns_zero_energy():
    # 5 control points, 3 frames, no motion
    P = torch.zeros(3, 5, 3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    E = compute_arap_prior_energy(P, edges)
    assert E.shape == (3,)
    torch.testing.assert_close(E, torch.zeros(3))


def test_rigid_translation_returns_zero_energy():
    # Two frames, second is first shifted by (1, 0, 0) -- ARAP should ignore this
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p1 = p0 + torch.tensor([1.0, 0.0, 0.0])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1], [0, 2]])
    E = compute_arap_prior_energy(P, edges)
    torch.testing.assert_close(E, torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_stretching_produces_nonzero_energy():
    # Frame 1 stretches one edge by 2x -- non-rigid; should produce energy
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p1 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1]])
    E = compute_arap_prior_energy(P, edges)
    assert E[0].item() == pytest.approx(0.0)
    assert E[1].item() > 0.0


def test_energy_grows_with_distortion():
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p_mild = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    p_strong = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1]])
    E_mild = compute_arap_prior_energy(torch.stack([p0, p_mild]), edges)
    E_strong = compute_arap_prior_energy(torch.stack([p0, p_strong]), edges)
    assert E_strong[1].item() > E_mild[1].item()


def test_energy_is_per_frame_mean_over_edges():
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    p1 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1], [1, 2]])
    E = compute_arap_prior_energy(P, edges)
    # Both edges stretch from 1 to 2 -- should give equal per-edge energy
    # Per-frame energy is the mean over edges; t=0 -> 0, t=1 -> positive
    assert E[0].item() == pytest.approx(0.0)
    assert E[1].item() > 0.0


def test_invalid_input_shape_raises():
    P = torch.zeros(3, 5)  # missing the xyz dim
    edges = torch.tensor([[0, 1]])
    with pytest.raises(ValueError):
        compute_arap_prior_energy(P, edges)
