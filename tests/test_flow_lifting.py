import torch
import pytest

from motionprior.geometry.flow_lifting import (
    backproject_pixels,
    lift_flow_to_3d,
    sample_control_points,
)


def _identity_intrinsics(focal: float = 100.0, cx: float = 32.0, cy: float = 32.0) -> torch.Tensor:
    K = torch.tensor([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ])
    return K


def test_backproject_identity_depth_returns_z_eq_one():
    # Pixels at principal point with depth 1 should project to (0, 0, 1).
    K = _identity_intrinsics()
    pixels = torch.tensor([[32.0, 32.0]])  # (y=32, x=32) in (y, x) -> hits cx,cy
    depths = torch.tensor([1.0])
    out = backproject_pixels(pixels, depths, K)
    assert out.shape == (1, 3)
    # At the principal point with z=1, we expect x=0, y=0, z=1
    torch.testing.assert_close(out[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)


def test_backproject_offset_pixel_has_offset_x():
    K = _identity_intrinsics(focal=100.0, cx=32.0, cy=32.0)
    # Pixel one unit right of center at depth 1 -> x = (33 - 32)/100 * 1 = 0.01
    pixels = torch.tensor([[32.0, 33.0]])  # (y, x)
    depths = torch.tensor([1.0])
    out = backproject_pixels(pixels, depths, K)
    torch.testing.assert_close(out[0], torch.tensor([0.01, 0.0, 1.0]), atol=1e-5, rtol=1e-5)


def test_backproject_batched_input():
    K = _identity_intrinsics()
    pixels = torch.tensor([[32.0, 32.0], [32.0, 32.0], [32.0, 33.0]])
    depths = torch.tensor([1.0, 2.0, 1.0])
    out = backproject_pixels(pixels, depths, K)
    assert out.shape == (3, 3)
    torch.testing.assert_close(out[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5, rtol=1e-5)


def test_backproject_rejects_bad_shape():
    K = _identity_intrinsics()
    pixels = torch.tensor([32.0, 32.0])  # (2,) not (N, 2)
    depths = torch.tensor([1.0])
    with pytest.raises(ValueError):
        backproject_pixels(pixels, depths, K)


def test_lift_flow_zero_flow_returns_constant_trajectory():
    # Single point, depth = 1 everywhere, zero flow across 3 frames.
    K = _identity_intrinsics()
    pixels0 = torch.tensor([[32.0, 32.0]])
    flow = torch.zeros(2, 1, 2)  # (T-1, N, 2): T-1=2 transitions, N=1 point
    depths = torch.ones(3, 1)    # (T, N): depth=1 at each frame
    traj = lift_flow_to_3d(pixels0, flow, depths, K)
    assert traj.shape == (3, 1, 3)
    # All three frames at (0, 0, 1)
    for t in range(3):
        torch.testing.assert_close(traj[t, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)


def test_lift_flow_horizontal_motion_at_constant_depth():
    K = _identity_intrinsics(focal=100.0)
    # Start at center; flow each frame is (dy=0, dx=1) pixels; depth=1.
    pixels0 = torch.tensor([[32.0, 32.0]])
    flow = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])  # (2, 1, 2)
    depths = torch.ones(3, 1)
    traj = lift_flow_to_3d(pixels0, flow, depths, K)
    # Frame 0: at center -> (0, 0, 1)
    # Frame 1: pixel (32, 33), depth=1 -> (0.01, 0, 1)
    # Frame 2: pixel (32, 34), depth=1 -> (0.02, 0, 1)
    torch.testing.assert_close(traj[0, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(traj[1, 0], torch.tensor([0.01, 0.0, 1.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(traj[2, 0], torch.tensor([0.02, 0.0, 1.0]), atol=1e-5, rtol=1e-5)


def test_lift_flow_rejects_mismatched_depth_length():
    K = _identity_intrinsics()
    pixels0 = torch.tensor([[32.0, 32.0]])
    flow = torch.zeros(2, 1, 2)
    depths = torch.ones(5, 1)  # T=5 but flow says T=3
    with pytest.raises(ValueError):
        lift_flow_to_3d(pixels0, flow, depths, K)


def test_sample_control_points_returns_expected_count():
    # A 32x32 grid mask, ask for 64 control points -- should return 64.
    mask = torch.ones(32, 32, dtype=torch.bool)
    points = sample_control_points(mask, num_points=64, seed=0)
    assert points.shape == (64, 2)
    # All points within mask bounds
    assert (points[:, 0] >= 0).all() and (points[:, 0] < 32).all()
    assert (points[:, 1] >= 0).all() and (points[:, 1] < 32).all()


def test_sample_control_points_respects_mask():
    # Top-left 4x4 only is True
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:4, :4] = True
    points = sample_control_points(mask, num_points=8, seed=0)
    assert points.shape == (8, 2)
    # All inside the masked region
    assert (points[:, 0] < 4).all()
    assert (points[:, 1] < 4).all()


def test_sample_control_points_too_few_available_raises():
    # Only 3 pixels masked but ask for 10
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[0, 2] = True
    with pytest.raises(ValueError):
        sample_control_points(mask, num_points=10, seed=0)


def test_sample_control_points_deterministic_with_seed():
    mask = torch.ones(32, 32, dtype=torch.bool)
    p1 = sample_control_points(mask, num_points=16, seed=42)
    p2 = sample_control_points(mask, num_points=16, seed=42)
    torch.testing.assert_close(p1, p2)
