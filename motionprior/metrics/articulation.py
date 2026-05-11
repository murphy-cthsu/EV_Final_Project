"""Inter-part angular consistency metric.

For a scene with K segmented parts, we measure whether the deformation field
recovers piecewise-rigid joint behavior. A rigid joint between two parts
produces a smooth, monotonic angle trajectory between the parts' principal
axes over time; an elastic-bend (over-smoothed) deformation produces jittery
or non-monotonic angles.

The metric is the headline new evaluation axis for MotionPrior-4DGS.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def part_principal_axes(
    positions: Tensor,
    parts: Tensor,
    num_parts: int,
    static_label: int = -1,
) -> Tensor:
    """First principal axis of each part's point cloud at a single time.

    Args:
        positions: (N, 3) tensor of point positions.
        parts: (N,) int tensor of per-point part labels. `static_label` is excluded.
        num_parts: number of non-static parts (0..K-1).
        static_label: label value for points to exclude.

    Returns:
        (num_parts, 3) tensor. For empty parts, the axis is the zero vector.
        The sign of the axis is not constrained (an axis and its negation
        are equivalent for our purposes; downstream calls use |cos|).
    """
    out = torch.zeros(num_parts, 3, dtype=torch.float32, device=positions.device)
    for k in range(num_parts):
        mask = (parts == k) & (parts != static_label)
        pk = positions[mask]
        if pk.shape[0] < 2:
            continue
        centered = pk - pk.mean(dim=0, keepdim=True)
        # SVD on the centered matrix; first right-singular vector = first PC.
        try:
            _, _, V = torch.linalg.svd(centered, full_matrices=False)
            out[k] = V[0]
        except RuntimeError:
            # Singular matrix; leave as zeros.
            continue
    return out


def inter_part_angle_trajectory(
    positions: Tensor,
    parts: Tensor,
    num_parts: int,
    pair: tuple[int, int],
    static_label: int = -1,
) -> Tensor:
    """Angle between two parts' principal axes at each frame.

    Args:
        positions: (T, N, 3) tensor of point positions over T frames.
        parts: (N,) int part labels (assumed constant over time).
        num_parts: total number of parts.
        pair: (k, k') indices into [0, num_parts).
        static_label: see `part_principal_axes`.

    Returns:
        (T,) tensor of angles in radians, range [0, pi/2]. We use `|cos|` to
        make the angle direction-agnostic (a principal axis is undirected).
    """
    T = positions.shape[0]
    out = torch.zeros(T, dtype=torch.float32, device=positions.device)
    k_a, k_b = pair
    for t in range(T):
        axes = part_principal_axes(
            positions[t], parts, num_parts, static_label=static_label
        )
        a = axes[k_a]
        b = axes[k_b]
        # Guard against zero-vector axes (empty parts).
        na = a.norm()
        nb = b.norm()
        if na < 1e-8 or nb < 1e-8:
            continue
        cos_abs = (a @ b).abs() / (na * nb)
        cos_abs = cos_abs.clamp(min=0.0, max=1.0)
        out[t] = torch.arccos(cos_abs)
    return out


def angular_consistency_score(angle_trajectory: Tensor) -> Tensor:
    """Smoothness-of-trajectory score in [0, 1].

    The score rewards angle trajectories that are smooth (low second-order
    variation) regardless of whether they are constant, monotonic, or
    bidirectional. A rigid joint produces smooth angles; an elastic bend
    or hallucinated motion produces jittery angles.

    Formula:
        d1 = first differences of `angles`
        d2 = second differences (curvature proxy)
        score = exp(-||d2||_2 / (||d1||_2 + eps))

    Properties:
        - Constant trajectory  -> d1 = d2 = 0 -> exp(0) = 1 (perfect).
        - Linear (constant velocity) -> d2 = 0 -> score = 1.
        - Smooth curved -> d2 small relative to d1 -> score close to 1.
        - Jittery / non-monotonic -> d2 dominant -> score near 0.

    Args:
        angle_trajectory: (T,) tensor of angles.

    Returns:
        Scalar tensor in [0, 1].
    """
    if angle_trajectory.numel() < 3:
        # Not enough samples to compute curvature; treat as perfectly smooth.
        return torch.tensor(1.0)
    d1 = angle_trajectory[1:] - angle_trajectory[:-1]
    d2 = d1[1:] - d1[:-1]
    n1 = d1.norm()
    n2 = d2.norm()
    eps = 1e-8
    # If there is essentially no motion at all (n1 ~ 0), the trajectory is
    # constant -> perfectly smooth -> score 1.
    if n1 < eps:
        return torch.tensor(1.0)
    ratio = n2 / (n1 + eps)
    return torch.exp(-ratio).clamp(min=0.0, max=1.0)
