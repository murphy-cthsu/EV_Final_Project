import math

import pytest
import torch

from motionprior.metrics.articulation import (
    part_principal_axes,
    inter_part_angle_trajectory,
    angular_consistency_score,
)


def test_principal_axis_of_line_along_x_returns_x_axis():
    # Points along x-axis. Principal axis must be +/-(1,0,0).
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    parts = torch.tensor([0, 0, 0])
    axes = part_principal_axes(pts, parts, num_parts=1)
    assert axes.shape == (1, 3)
    assert abs(abs(axes[0, 0].item()) - 1.0) < 1e-5
    assert abs(axes[0, 1].item()) < 1e-5
    assert abs(axes[0, 2].item()) < 1e-5


def test_principal_axes_two_parts():
    # Part 0 along x, part 1 along y.
    pts = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0],
    ])
    parts = torch.tensor([0, 0, 0, 1, 1, 1])
    axes = part_principal_axes(pts, parts, num_parts=2)
    assert axes.shape == (2, 3)
    # Part 0 -> x-axis
    assert abs(abs(axes[0, 0].item()) - 1.0) < 1e-5
    # Part 1 -> y-axis
    assert abs(abs(axes[1, 1].item()) - 1.0) < 1e-5


def test_principal_axes_ignores_static_label():
    pts = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # part 0
        [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # static, ignored
    ])
    parts = torch.tensor([0, 0, -1, -1])
    axes = part_principal_axes(pts, parts, num_parts=1, static_label=-1)
    assert axes.shape == (1, 3)
    assert abs(abs(axes[0, 0].item()) - 1.0) < 1e-5


def test_principal_axes_handles_empty_part_gracefully():
    # part 1 has zero members -> axis should be a zero vector
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    parts = torch.tensor([0, 0])
    axes = part_principal_axes(pts, parts, num_parts=2)
    assert axes.shape == (2, 3)
    assert torch.allclose(axes[1], torch.zeros(3))


def test_inter_part_angle_zero_when_parallel():
    # Both parts along x at every frame
    T, N = 3, 4
    pts = torch.zeros(T, N, 3)
    pts[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    pts[:, 1] = torch.tensor([1.0, 0.0, 0.0])
    pts[:, 2] = torch.tensor([5.0, 0.0, 0.0])
    pts[:, 3] = torch.tensor([6.0, 0.0, 0.0])
    parts = torch.tensor([0, 0, 1, 1])
    angles = inter_part_angle_trajectory(pts, parts, num_parts=2, pair=(0, 1))
    assert angles.shape == (T,)
    torch.testing.assert_close(angles, torch.zeros(T), atol=1e-5, rtol=1e-5)


def test_inter_part_angle_ninety_degrees_when_orthogonal():
    T = 2
    pts = torch.zeros(T, 4, 3)
    # Part 0 along x, part 1 along y at all t
    pts[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    pts[:, 1] = torch.tensor([1.0, 0.0, 0.0])
    pts[:, 2] = torch.tensor([0.0, 0.0, 0.0])
    pts[:, 3] = torch.tensor([0.0, 1.0, 0.0])
    parts = torch.tensor([0, 0, 1, 1])
    angles = inter_part_angle_trajectory(pts, parts, num_parts=2, pair=(0, 1))
    expected = torch.full((T,), math.pi / 2)
    torch.testing.assert_close(angles, expected, atol=1e-4, rtol=1e-4)


def test_angle_trajectory_under_rotation():
    # Part 0 fixed along x; part 1 rotates from +y to +x over 3 frames.
    T = 3
    pts = torch.zeros(T, 4, 3)
    parts = torch.tensor([0, 0, 1, 1])
    # part 0 fixed
    pts[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    pts[:, 1] = torch.tensor([1.0, 0.0, 0.0])
    # part 1 axis rotates from +y (t=0) to (1,1)/sqrt2 (t=1) to +x (t=2)
    pts[0, 2] = torch.tensor([0.0, 0.0, 0.0]); pts[0, 3] = torch.tensor([0.0, 1.0, 0.0])
    pts[1, 2] = torch.tensor([0.0, 0.0, 0.0]); pts[1, 3] = torch.tensor([1.0, 1.0, 0.0])
    pts[2, 2] = torch.tensor([0.0, 0.0, 0.0]); pts[2, 3] = torch.tensor([1.0, 0.0, 0.0])
    angles = inter_part_angle_trajectory(pts, parts, num_parts=2, pair=(0, 1))
    # Expected angles: 90, 45, 0 degrees
    expected = torch.tensor([math.pi / 2, math.pi / 4, 0.0])
    torch.testing.assert_close(angles, expected, atol=5e-3, rtol=1e-3)


def test_angular_consistency_score_smooth_monotonic_is_high():
    # Smooth monotonic decrease should produce high consistency score
    angles = torch.linspace(math.pi / 2, 0.0, 10)
    score = angular_consistency_score(angles)
    assert score > 0.95  # smooth & monotonic


def test_angular_consistency_score_jittery_is_low():
    # Random jittery angles should produce low score
    torch.manual_seed(42)
    angles = torch.rand(20) * math.pi
    score = angular_consistency_score(angles)
    assert score < 0.5


def test_angular_consistency_score_constant_returns_high():
    # Constant angles (no motion) — should be high (smooth)
    angles = torch.full((10,), math.pi / 4)
    score = angular_consistency_score(angles)
    assert score >= 0.99
