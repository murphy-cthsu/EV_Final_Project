"""Lift 2D optical-flow trajectories to 3D control-point trajectories.

Pipeline (offline, run once per scene before training):
  1. RAFT (or similar) produces per-frame-pair optical flow at full image resolution.
  2. Per-frame depth (DepthAnything V2 / V3 or from the static 3DGS render).
  3. Pick sparse control points on the dynamic-region mask (typically ~512 points).
  4. Track each control point across frames by accumulating flow.
  5. Back-project each (pixel, depth) pair into 3D via camera intrinsics.

The result is `positions: (T, N, 3)` feeding directly into
`motionprior.geometry.arap_prior.compute_arap_prior_energy`.

This module is the **CPU-testable** part. RAFT and DepthAnything calls are
elsewhere (they need GPU) and produce the input arrays this module consumes.
"""

from __future__ import annotations

import torch
from torch import Tensor


def backproject_pixels(
    pixels: Tensor,
    depths: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Pinhole back-projection: pixel + depth + K -> 3D point in camera frame.

    Args:
        pixels: (N, 2) tensor of (y, x) pixel coordinates (floats; sub-pixel OK).
        depths: (N,) tensor of depths (z values in camera frame).
        intrinsics: (3, 3) camera matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Returns:
        (N, 3) tensor of 3D points in camera frame (x, y, z).
    """
    if pixels.dim() != 2 or pixels.shape[-1] != 2:
        raise ValueError(
            f"pixels must have shape (N, 2); got {tuple(pixels.shape)}"
        )
    if depths.shape != (pixels.shape[0],):
        raise ValueError(
            f"depths shape {tuple(depths.shape)} must match pixels {tuple(pixels.shape)}"
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"intrinsics must be (3, 3); got {tuple(intrinsics.shape)}"
        )

    ys = pixels[:, 0]
    xs = pixels[:, 1]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    z = depths
    x3 = (xs - cx) / fx * z
    y3 = (ys - cy) / fy * z
    return torch.stack([x3, y3, z], dim=-1)


def lift_flow_to_3d(
    pixels0: Tensor,
    flow: Tensor,
    depths: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Accumulate optical-flow into 3D trajectories.

    Args:
        pixels0: (N, 2) initial pixel coordinates at frame 0, as (y, x).
        flow: (T-1, N, 2) per-step pixel displacements (dy, dx). Frame t+1
            pixel = pixels_t + flow[t].
        depths: (T, N) depths at each frame for each tracked point.
        intrinsics: (3, 3) camera matrix.

    Returns:
        (T, N, 3) 3D trajectories in camera frame.
    """
    if flow.dim() != 3 or flow.shape[-1] != 2:
        raise ValueError(
            f"flow must have shape (T-1, N, 2); got {tuple(flow.shape)}"
        )
    T_minus_1 = flow.shape[0]
    N = pixels0.shape[0]
    if flow.shape[1] != N:
        raise ValueError(
            f"flow N {flow.shape[1]} must match pixels0 N {N}"
        )
    if depths.shape != (T_minus_1 + 1, N):
        raise ValueError(
            f"depths shape {tuple(depths.shape)} must be (T={T_minus_1+1}, N={N})"
        )

    T = T_minus_1 + 1
    pixel_track = torch.zeros(T, N, 2, dtype=pixels0.dtype, device=pixels0.device)
    pixel_track[0] = pixels0
    for t in range(T - 1):
        pixel_track[t + 1] = pixel_track[t] + flow[t]

    trajectories = torch.zeros(T, N, 3, dtype=pixels0.dtype, device=pixels0.device)
    for t in range(T):
        trajectories[t] = backproject_pixels(pixel_track[t], depths[t], intrinsics)
    return trajectories


def sample_control_points(
    mask: Tensor,
    num_points: int,
    seed: int = 0,
) -> Tensor:
    """Sample control points uniformly from the True region of a mask.

    Args:
        mask: (H, W) bool tensor. True = candidate pixel.
        num_points: number of control points to sample.
        seed: RNG seed for determinism.

    Returns:
        (num_points, 2) long tensor of (y, x) pixel coordinates.
    """
    if mask.dim() != 2:
        raise ValueError(
            f"mask must be 2-D; got shape {tuple(mask.shape)}"
        )
    coords = mask.nonzero(as_tuple=False)  # (M, 2) where M = sum(mask)
    M = coords.shape[0]
    if M < num_points:
        raise ValueError(
            f"mask has only {M} True pixels; cannot sample {num_points} points"
        )
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(M, generator=g)[:num_points]
    return coords[perm]
