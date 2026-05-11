"""Lift per-Gaussian part labels to per-control-node part labels.

SC-GS represents the deformation field on a sparse set of *control nodes*
(typically ~512), wrapping dense ~50K Gaussians. Our SAM-2-based part
segmentation produces labels at the Gaussian level (via 2D mask -> depth
projection). The SC-GS ARAP adapter requires labels at the control-node level
because the K-NN edge graph is over control nodes.

The mapping is K-nearest-neighbour majority voting in 3D canonical space.
For each control node:
  1. Find its K nearest Gaussians by Euclidean distance
  2. Count part labels among those Gaussians
  3. Assign the most-common label; ties broken in favour of non-static

This module is pure tensor math; CPU-testable.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor


def nearest_neighbor_indices(
    queries: Tensor,
    sources: Tensor,
) -> Tensor:
    """For each query point, the index of its nearest source point.

    Args:
        queries: ``(Q, D)`` query points.
        sources: ``(S, D)`` source points.

    Returns:
        ``(Q,)`` long tensor of source indices.
    """
    # Pairwise distances (Q, S)
    dists = torch.cdist(queries, sources)
    return dists.argmin(dim=1)


def lift_gaussian_parts_to_nodes(
    gaussian_positions: Tensor,
    gaussian_parts: Tensor,
    node_positions: Tensor,
    knn: int = 5,
    static_label: int = -1,
    prefer_dynamic_on_tie: bool = True,
) -> Tensor:
    """Per-control-node part labels via K-NN majority voting.

    Args:
        gaussian_positions: ``(N_g, 3)`` Gaussian canonical positions.
        gaussian_parts: ``(N_g,)`` int part labels per Gaussian.
        node_positions: ``(N_n, 3)`` control-node canonical positions.
        knn: number of nearest Gaussians to consult per node.
        static_label: sentinel for static Gaussians.
        prefer_dynamic_on_tie: if True, ties between static and non-static
            labels resolve toward the non-static label (we'd rather have a
            joint that's slightly mis-assigned to a part than miss it
            entirely by labelling its node static).

    Returns:
        ``(N_n,)`` long tensor of part labels.
    """
    if gaussian_positions.dim() != 2 or gaussian_positions.shape[-1] != 3:
        raise ValueError(
            f"gaussian_positions must have shape (N_g, 3); got "
            f"{tuple(gaussian_positions.shape)}"
        )
    if gaussian_parts.shape != (gaussian_positions.shape[0],):
        raise ValueError(
            f"gaussian_parts length {gaussian_parts.shape[0]} does not match "
            f"gaussian_positions N_g = {gaussian_positions.shape[0]}"
        )
    if node_positions.dim() != 2 or node_positions.shape[-1] != 3:
        raise ValueError(
            f"node_positions must have shape (N_n, 3); got "
            f"{tuple(node_positions.shape)}"
        )

    n_g = gaussian_positions.shape[0]
    n_n = node_positions.shape[0]
    k_eff = min(knn, n_g)

    # (N_n, N_g) distances
    dists = torch.cdist(node_positions, gaussian_positions)
    # Top-k smallest (we use -dists + topk for stability across torch versions)
    _, nn_idx = dists.topk(k_eff, largest=False, dim=1)  # (N_n, k_eff)

    # Vote per node
    out = torch.full((n_n,), static_label, dtype=torch.long, device=node_positions.device)
    for i in range(n_n):
        labels = gaussian_parts[nn_idx[i]].tolist()
        counts = Counter(labels)
        # Find the most common; on tie, optionally prefer non-static
        max_count = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == max_count]
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            # Tie. If prefer_dynamic_on_tie, drop static_label from contenders.
            if prefer_dynamic_on_tie:
                non_static = [w for w in winners if w != static_label]
                if non_static:
                    # Pick lowest-numbered non-static for determinism
                    out[i] = min(non_static)
                else:
                    out[i] = static_label
            else:
                out[i] = min(winners)
    return out
