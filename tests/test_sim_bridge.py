"""Tests for the simulator-import bridge.

These verify the URDF / point-cloud emission from a (4DGS + part labels)
representation. All CPU-testable -- we don't actually run the simulator.
"""

from pathlib import Path

import pytest
import torch

from motionprior.integration.sim_bridge import (
    ArticulatedPart,
    PartTrajectory,
    extract_parts_from_4dgs,
    emit_urdf,
    emit_genesis_yaml,
)


def test_part_trajectory_centroid_and_axis():
    # 3 frames, 2 points in part:
    points = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ])
    pt = PartTrajectory(positions=points)
    assert pt.num_frames == 3
    assert pt.num_points == 2
    centroid_t0 = pt.centroid(0)
    torch.testing.assert_close(centroid_t0, torch.tensor([0.5, 0.0, 0.0]))
    # Principal axis along x at every frame
    axis_t0 = pt.principal_axis(0)
    assert abs(abs(axis_t0[0].item()) - 1.0) < 1e-5


def test_extract_parts_groups_by_label():
    # 4 frames, 5 Gaussians; parts [0, 0, 1, 1, -1]
    positions = torch.randn(4, 5, 3)
    parts = torch.tensor([0, 0, 1, 1, -1])
    out = extract_parts_from_4dgs(positions, parts, static_label=-1)
    assert isinstance(out, list)
    # Two non-static parts, so len == 2
    assert len(out) == 2
    assert out[0].part_id == 0
    assert out[0].trajectory.num_points == 2
    assert out[1].part_id == 1
    assert out[1].trajectory.num_points == 2


def test_extract_parts_skips_empty_part_id():
    # parts {0, 2} -- id 1 has no Gaussians
    positions = torch.randn(2, 4, 3)
    parts = torch.tensor([0, 0, 2, 2])
    out = extract_parts_from_4dgs(positions, parts, static_label=-1)
    # Result should only contain the two non-empty parts
    assert len(out) == 2
    assert {p.part_id for p in out} == {0, 2}


def test_emit_urdf_creates_valid_xml(tmp_path):
    positions = torch.zeros(2, 4, 3)
    positions[1] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5],
                                  [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    parts = torch.tensor([0, 0, 1, 1])
    parts_list = extract_parts_from_4dgs(positions, parts)
    out = tmp_path / "scene.urdf"
    emit_urdf(parts_list, out, robot_name="motionprior_scene")
    assert out.exists()
    content = out.read_text()
    # Sanity checks on URDF structure
    assert "<robot" in content and "</robot>" in content
    assert 'name="motionprior_scene"' in content
    # One link per part + a world base link
    assert content.count("<link") == 3  # base + 2 parts
    # One joint connecting each part to base
    assert content.count("<joint") == 2


def test_emit_urdf_skeleton_revolute_joints_have_axes(tmp_path):
    positions = torch.zeros(3, 6, 3)
    # Part 0 stays put
    positions[:, 0:3] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Part 1 rotates over time -- different axis sets
    positions[0, 3:6] = torch.tensor([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 0.0, 1.0]])
    positions[1, 3:6] = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.7, 0.7], [5.0, -0.7, 0.7]])
    positions[2, 3:6] = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 1.0], [5.0, -1.0, 0.0]])
    parts = torch.tensor([0, 0, 0, 1, 1, 1])
    parts_list = extract_parts_from_4dgs(positions, parts)

    out = tmp_path / "scene.urdf"
    emit_urdf(parts_list, out, joint_type="revolute")
    content = out.read_text()
    # Revolute joints must have an axis tag
    assert 'type="revolute"' in content
    assert "<axis" in content


def test_emit_genesis_yaml_round_trip(tmp_path):
    positions = torch.zeros(2, 4, 3)
    positions[1] = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5],
                                  [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    parts = torch.tensor([0, 0, 1, 1])
    parts_list = extract_parts_from_4dgs(positions, parts)

    out = tmp_path / "scene.yaml"
    emit_genesis_yaml(parts_list, out)
    assert out.exists()
    import yaml
    data = yaml.safe_load(out.read_text())
    assert "parts" in data
    assert len(data["parts"]) == 2
    assert data["parts"][0]["id"] == 0
    assert "centroid" in data["parts"][0]
    assert "principal_axis" in data["parts"][0]
    assert "num_points" in data["parts"][0]


def test_extract_parts_rejects_bad_input_shape():
    with pytest.raises(ValueError):
        extract_parts_from_4dgs(
            positions=torch.zeros(5, 3),  # 2D, missing time dim
            parts=torch.tensor([0, 0, 1, 1, 1]),
        )
    with pytest.raises(ValueError):
        extract_parts_from_4dgs(
            positions=torch.zeros(3, 5, 3),
            parts=torch.tensor([0, 0]),  # length mismatch
        )
