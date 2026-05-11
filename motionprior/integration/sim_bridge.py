"""Simulator-import bridge.

Converts a trained 4DGS (per-frame Gaussian positions) plus part labels into
formats a physics simulator can consume for downstream embodied-AI use:

* **URDF** -- universal, supported by PyBullet / MuJoCo / Isaac Sim / Genesis.
  Each non-static part becomes a rigid link; joints between parts are revolute
  by default (placeholder; W4 work will identify joint type per part-pair).
* **Genesis YAML** -- a leaner description for Genesis's scene-loading API
  with per-part centroid + principal axis + point cloud reference.

This module is the *output side* of the MotionPrior-4DGS pipeline -- the
representation an external VWM dynamics module or robot policy ingests. It
is the load-bearing demo for the "perception module for visual world models"
framing in the paper.

Pure tensor / file emission. CPU-testable. No physics simulation here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import torch
import yaml
from torch import Tensor


@dataclass
class PartTrajectory:
    """Trajectory of Gaussians belonging to one part over time.

    Attributes:
        positions: ``(T, N_k, 3)`` -- positions of this part's Gaussians.
    """

    positions: Tensor

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def num_points(self) -> int:
        return self.positions.shape[1]

    def centroid(self, frame: int) -> Tensor:
        """``(3,)`` -- mean position at frame ``frame``."""
        return self.positions[frame].mean(dim=0)

    def principal_axis(self, frame: int) -> Tensor:
        """``(3,)`` -- first principal component of the part at ``frame``.
        Zero vector if part has < 2 points."""
        pts = self.positions[frame]
        if pts.shape[0] < 2:
            return torch.zeros(3, dtype=pts.dtype)
        centered = pts - pts.mean(dim=0, keepdim=True)
        try:
            _, _, V = torch.linalg.svd(centered, full_matrices=False)
            return V[0]
        except RuntimeError:
            return torch.zeros(3, dtype=pts.dtype)


@dataclass
class ArticulatedPart:
    """A non-static part of the scene with its trajectory."""

    part_id: int
    trajectory: PartTrajectory
    color: tuple[float, float, float] = field(default=(0.8, 0.5, 0.2))


def extract_parts_from_4dgs(
    positions: Tensor,
    parts: Tensor,
    static_label: int = -1,
) -> list[ArticulatedPart]:
    """Group Gaussian trajectories by part label.

    Args:
        positions: ``(T, N, 3)`` Gaussian positions over time.
        parts: ``(N,)`` per-Gaussian part labels.
        static_label: sentinel for static Gaussians (excluded from output).

    Returns:
        One ``ArticulatedPart`` per unique non-static label, in ascending id order.
    """
    if positions.dim() != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape (T, N, 3); got {tuple(positions.shape)}"
        )
    if parts.shape[0] != positions.shape[1]:
        raise ValueError(
            f"parts length {parts.shape[0]} != positions N {positions.shape[1]}"
        )

    unique = sorted(set(int(p) for p in parts.tolist()))
    out: list[ArticulatedPart] = []
    for pid in unique:
        if pid == static_label:
            continue
        mask = parts == pid
        if mask.sum().item() == 0:
            continue
        part_positions = positions[:, mask]
        out.append(
            ArticulatedPart(
                part_id=pid,
                trajectory=PartTrajectory(positions=part_positions),
            )
        )
    return out


# ---------------------------------------------------------------------------- #
# URDF emission                                                                #
# ---------------------------------------------------------------------------- #


def emit_urdf(
    parts: Iterable[ArticulatedPart],
    out_path: Path | str,
    robot_name: str = "motionprior_scene",
    joint_type: str = "revolute",
    rest_frame: int = 0,
) -> Path:
    """Emit a URDF with one link per part + revolute joints to a `base_link`.

    The joint *origin* is set to the part centroid at ``rest_frame``. The joint
    axis (for revolute joints) is the part's first principal axis (an
    informative default; the agent's IK solver typically refines this).

    Args:
        parts: parts to include.
        out_path: where to write the .urdf file.
        robot_name: ``<robot name="...">`` attribute.
        joint_type: ``revolute`` (default), ``prismatic``, ``fixed``.
        rest_frame: which frame defines the joint origin + axis.

    Returns:
        The output path.
    """
    out_path = Path(out_path)
    parts = list(parts)

    robot = ET.Element("robot", attrib={"name": robot_name})

    # Base link (world frame anchor)
    base_link = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    base_visual = ET.SubElement(base_link, "visual")
    base_geom = ET.SubElement(base_visual, "geometry")
    ET.SubElement(base_geom, "sphere", attrib={"radius": "0.01"})

    for part in parts:
        link_name = f"part_{part.part_id}"
        link = ET.SubElement(robot, "link", attrib={"name": link_name})
        visual = ET.SubElement(link, "visual")
        geom = ET.SubElement(visual, "geometry")
        # Use a sphere bounding the centroid; W4 work can refine to a convex
        # hull mesh from the Gaussian cloud.
        ET.SubElement(geom, "sphere", attrib={"radius": "0.05"})
        material = ET.SubElement(visual, "material", attrib={"name": f"mat_{part.part_id}"})
        r, g, b = part.color
        ET.SubElement(material, "color", attrib={"rgba": f"{r} {g} {b} 1.0"})

        # Joint connecting this part to the base
        joint_name = f"joint_{part.part_id}"
        joint = ET.SubElement(robot, "joint", attrib={"name": joint_name, "type": joint_type})
        ET.SubElement(joint, "parent", attrib={"link": "base_link"})
        ET.SubElement(joint, "child", attrib={"link": link_name})
        centroid = part.trajectory.centroid(rest_frame)
        ET.SubElement(
            joint, "origin",
            attrib={
                "xyz": f"{centroid[0].item():.6f} {centroid[1].item():.6f} {centroid[2].item():.6f}",
                "rpy": "0 0 0",
            },
        )
        if joint_type in {"revolute", "continuous", "prismatic"}:
            axis = part.trajectory.principal_axis(rest_frame)
            ET.SubElement(
                joint, "axis",
                attrib={"xyz": f"{axis[0].item():.6f} {axis[1].item():.6f} {axis[2].item():.6f}"},
            )
            if joint_type == "revolute":
                ET.SubElement(
                    joint, "limit",
                    attrib={"lower": "-3.14159", "upper": "3.14159", "effort": "100", "velocity": "1.0"},
                )

    # Pretty-print
    _indent(robot)
    tree = ET.ElementTree(robot)
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def _indent(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty indentation for an ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ---------------------------------------------------------------------------- #
# Genesis YAML emission                                                        #
# ---------------------------------------------------------------------------- #


def emit_genesis_yaml(
    parts: Iterable[ArticulatedPart],
    out_path: Path | str,
    rest_frame: int = 0,
) -> Path:
    """Emit a lean YAML scene description for Genesis-style loading.

    Format:

        parts:
          - id: 0
            num_points: 24
            centroid: [x, y, z]
            principal_axis: [x, y, z]
          - id: 1
            ...

    Companion to URDF emission. Genesis's articulated-body loader accepts
    URDF; the YAML is for our own evaluation script's articulated-IK probe.
    """
    out_path = Path(out_path)
    parts = list(parts)
    data: dict = {"parts": []}
    for part in parts:
        centroid = part.trajectory.centroid(rest_frame)
        axis = part.trajectory.principal_axis(rest_frame)
        data["parts"].append(
            {
                "id": part.part_id,
                "num_points": part.trajectory.num_points,
                "centroid": [float(centroid[i].item()) for i in range(3)],
                "principal_axis": [float(axis[i].item()) for i in range(3)],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path
