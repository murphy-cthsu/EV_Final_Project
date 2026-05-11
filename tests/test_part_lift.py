"""Tests for the Gaussian -> control-node part-label lifter.

SC-GS uses ~512 sparse control nodes that wrap dense ~50K Gaussians. Our
articulation pipeline labels parts at the *Gaussian* level (from SAM 2 masks
projected through the canonical 3DGS depth). The SC-GS ARAP adapter needs
*control-node*-level labels because it operates on the control-point K-NN
graph. This module bridges that gap.
"""

import torch
import pytest

from motionprior.segmentation.part_lift import (
    lift_gaussian_parts_to_nodes,
    nearest_neighbor_indices,
)


def test_nearest_neighbor_basic():
    # Sources at (0,0,0), (10,0,0). Queries at (1,0,0), (9,0,0).
    sources = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    queries = torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    idx = nearest_neighbor_indices(queries, sources)
    assert idx.tolist() == [0, 1]


def test_lift_majority_voting():
    # 6 Gaussians, parts: [0,0,0, 1,1,1]
    # 2 control nodes near the cluster centers
    gaussian_positions = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],  # cluster 0
        [10.0, 0.0, 0.0], [10.1, 0.0, 0.0], [10.0, 0.1, 0.0],  # cluster 1
    ])
    gaussian_parts = torch.tensor([0, 0, 0, 1, 1, 1])
    node_positions = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    node_parts = lift_gaussian_parts_to_nodes(
        gaussian_positions, gaussian_parts, node_positions
    )
    assert node_parts.shape == (2,)
    assert node_parts[0].item() == 0
    assert node_parts[1].item() == 1


def test_lift_mixed_cluster_picks_majority():
    # 4 Gaussians around one node; 3 are part 0, 1 is part 1 -> node gets 0
    gaussian_positions = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1],
    ])
    gaussian_parts = torch.tensor([0, 0, 0, 1])
    node_positions = torch.tensor([[0.05, 0.05, 0.0]])
    node_parts = lift_gaussian_parts_to_nodes(
        gaussian_positions, gaussian_parts, node_positions, knn=4
    )
    assert node_parts[0].item() == 0


def test_lift_static_label_does_not_dominate():
    # Mix of static (-1) and dynamic part 0; we shouldn't vote -1 unless
    # everything in the neighborhood is -1.
    gaussian_positions = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0],
    ])
    gaussian_parts = torch.tensor([-1, -1, 0, 0])
    node_positions = torch.tensor([[0.15, 0.0, 0.0]])
    # With knn=4 and equal votes, prefer non-static
    node_parts = lift_gaussian_parts_to_nodes(
        gaussian_positions, gaussian_parts, node_positions,
        knn=4, static_label=-1, prefer_dynamic_on_tie=True,
    )
    assert node_parts[0].item() == 0


def test_lift_all_static_neighborhood_keeps_static():
    gaussian_positions = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0],
    ])
    gaussian_parts = torch.tensor([-1, -1, -1])
    node_positions = torch.tensor([[0.1, 0.0, 0.0]])
    node_parts = lift_gaussian_parts_to_nodes(
        gaussian_positions, gaussian_parts, node_positions, knn=3,
    )
    assert node_parts[0].item() == -1


def test_lift_handles_knn_larger_than_population():
    gaussian_positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gaussian_parts = torch.tensor([0, 1])
    node_positions = torch.tensor([[0.3, 0.0, 0.0]])
    # Ask for 5 neighbors but only 2 Gaussians exist
    node_parts = lift_gaussian_parts_to_nodes(
        gaussian_positions, gaussian_parts, node_positions, knn=5,
    )
    # Closest is part 0; should still produce a sane result
    assert node_parts[0].item() == 0


def test_lift_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        lift_gaussian_parts_to_nodes(
            gaussian_positions=torch.zeros(5, 3),
            gaussian_parts=torch.zeros(4, dtype=torch.long),  # length mismatch
            node_positions=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError):
        lift_gaussian_parts_to_nodes(
            gaussian_positions=torch.zeros(5, 2),  # not (N, 3)
            gaussian_parts=torch.zeros(5, dtype=torch.long),
            node_positions=torch.zeros(2, 3),
        )


def test_lift_returns_correct_dtype():
    gp = torch.tensor([[0.0, 0.0, 0.0]])
    gparts = torch.tensor([7], dtype=torch.long)
    np_ = torch.tensor([[0.0, 0.0, 0.0]])
    out = lift_gaussian_parts_to_nodes(gp, gparts, np_)
    assert out.dtype == torch.long
