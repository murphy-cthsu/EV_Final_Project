"""Offline ARAP-prior energy precomputation.

Given sparse control-point trajectories (typically ~512 points obtained by
lifting RAFT optical flow into 3D via the static-3DGS depth map), compute
a per-frame energy score that captures how badly the video prior violates
local rigidity. The energy is independent of any deformation MLP -- it is
a fixed property of the video prior, computed once per scene.

For each edge (i, j) and frame t, the violation is the deviation between
the rotated rest-edge vector and the observed edge vector at frame t. We
use the closed-form best-fit rotation per edge neighborhood (single-edge
case: the best rotation aligns rest direction with observed direction, so
the residual reduces to a length difference). This is a tractable proxy
for full ARAP energy that is exact in the rigid limit and grows with
distortion.

Formally, for an edge with rest length L0 and frame-t length Lt:
    e_t = (Lt - L0)^2

The per-frame energy is the mean of e_t over all edges.

This proxy is invariant under rigid motion (rotation + translation): both
preserve edge lengths. It captures stretching, shearing, and joint-induced
length changes -- exactly the failure modes of video-prior supervision.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_arap_prior_energy(
    positions: Tensor,
    edges: Tensor,
    rest_frame: int = 0,
) -> Tensor:
    """Per-frame ARAP-prior energy.

    Args:
        positions: (T, N, 3) -- control-point positions over T frames.
        edges: (E, 2) -- index pairs into the N control points.
        rest_frame: which frame defines the rest configuration. Default 0.

    Returns:
        (T,) tensor; values are nonnegative, with positions[rest_frame] giving 0.
    """
    if positions.dim() != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape (T, N, 3); got {tuple(positions.shape)}"
        )
    if edges.dim() != 2 or edges.shape[-1] != 2:
        raise ValueError(
            f"edges must have shape (E, 2); got {tuple(edges.shape)}"
        )

    rest = positions[rest_frame]                      # (N, 3)
    rest_vec = rest[edges[:, 0]] - rest[edges[:, 1]]  # (E, 3)
    rest_len = rest_vec.norm(dim=-1)                  # (E,)

    obs_vec = positions[:, edges[:, 0]] - positions[:, edges[:, 1]]  # (T, E, 3)
    obs_len = obs_vec.norm(dim=-1)                                    # (T, E)

    per_edge = (obs_len - rest_len.unsqueeze(0)).pow(2)               # (T, E)
    return per_edge.mean(dim=-1)                                      # (T,)
