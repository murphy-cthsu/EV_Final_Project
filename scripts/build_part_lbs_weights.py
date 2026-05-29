"""Stage C upgrade: per-Gaussian soft LBS weights instead of hard part ID.

For each canonical Gaussian, compute a soft weight w_arm in [0,1] (so
w_body = 1 - w_arm) by:
    1. Project Gaussian to each view at t=0.
    2. Compute signed distance to "arm" mask boundary (positive = inside arm,
       negative = inside body, magnitude = pixels to boundary).
    3. Apply soft sigmoid: w_arm_view = sigmoid(d / temperature).
    4. Average across views (weighted by in-frame count).

Output (overwrites Stage C hard IDs with soft weights):
    runs_aux/part_assignment/
        gaussian_arm_weights.npy           # (N_g,) float in [0, 1]
        gaussian_part_ids.npy (kept)       # hard IDs for backward compat / comparison

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_part_lbs_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
import torch  # noqa: E402

CANON = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PARTS_DIR = REPO_ROOT / "runs_aux" / "parts_motion"
SRC = REPO_ROOT / "data" / "custom" / "scene00_masked"
OUT = REPO_ROOT / "runs_aux" / "part_assignment"


def make_K(W, H, fov_x):
    import math
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    return np.array([[fx, 0, W/2.0], [0, fx, H/2.0], [0, 0, 1.0]], dtype=np.float64)


def c2w_b_to_w2c_cv(c2w_b):
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    return np.linalg.inv(c2w_b @ flip)


def project(xyz, K, w2c, W, H):
    h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    c = h @ w2c.T
    z = c[:, 2]
    in_front = z > 1e-3
    z_safe = np.where(in_front, z, 1.0)
    uv = (c[:, :2] / z_safe[:, None]) @ K[:2, :2].T + K[:2, 2]
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & in_front
    return uv, in_image


def signed_distance_to_arm(arm_mask: np.ndarray) -> np.ndarray:
    """Per-pixel: positive if inside arm, negative if outside, magnitude = px to boundary."""
    arm = arm_mask.astype(bool)
    not_arm = ~arm
    d_in = distance_transform_edt(arm)    # px from each arm pixel to nearest not-arm
    d_out = distance_transform_edt(not_arm)  # px from each not-arm pixel to nearest arm
    return d_in - d_out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[LBS] loading canonical {CANON.name}")
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    N = xyz.shape[0]
    print(f"[LBS] N={N}")

    data = json.loads((SRC / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    H = W = 576
    K = make_K(W, H, fov_x)

    # Load per-view frame-0 part masks + cameras
    view_meta = {}
    for f in data["frames"]:
        v = int(f["view_idx"])
        t = int(f["frame_idx"])
        if t != 0:
            continue
        view_meta[v] = {
            "c2w": np.asarray(f["transform_matrix"], dtype=np.float64),
            "masks": np.load(PARTS_DIR / f"view{v}_frame0_part_masks.npy"),  # (2, H, W)
        }

    # Per-view per-Gaussian signed distance to arm
    # temperature controls softness; in pixels. Smaller -> sharper boundary.
    TAU = 12.0

    sum_weights = np.zeros(N, dtype=np.float64)
    sum_inframe = np.zeros(N, dtype=np.int32)
    for v, m in view_meta.items():
        sd = signed_distance_to_arm(m["masks"][0])  # arm = channel 0
        w2c = c2w_b_to_w2c_cv(m["c2w"])
        uv, in_image = project(xyz, K, w2c, W, H)
        u_clip = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W - 1)
        v_clip = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H - 1)
        sd_g = sd[v_clip, u_clip]               # signed distance per Gaussian under this view
        w_v = 1.0 / (1.0 + np.exp(-sd_g / TAU))  # sigmoid -> arm-weight per view
        # Only count views where the Gaussian projects in-frame
        sum_weights[in_image] += w_v[in_image]
        sum_inframe += in_image.astype(np.int32)

    safe = sum_inframe > 0
    arm_weights = np.where(safe, sum_weights / np.maximum(sum_inframe, 1), 0.5)
    # For Gaussians never in any frame, fall back to 0 (treat as static body)
    arm_weights = np.where(sum_inframe == 0, 0.0, arm_weights)

    np.save(OUT / "gaussian_arm_weights.npy", arm_weights.astype(np.float32))
    # Diagnostics
    soft_pure_arm = (arm_weights > 0.9).sum()
    soft_pure_body = (arm_weights < 0.1).sum()
    boundary = ((arm_weights > 0.1) & (arm_weights < 0.9)).sum()
    arm_mean_weight = arm_weights.mean()
    print(f"[LBS] arm_weights distribution:")
    print(f"[LBS]   pure arm  (w>0.9): {soft_pure_arm:>6}  ({100*soft_pure_arm/N:.1f}%)")
    print(f"[LBS]   pure body (w<0.1): {soft_pure_body:>6}  ({100*soft_pure_body/N:.1f}%)")
    print(f"[LBS]   boundary [0.1, 0.9]: {boundary:>6}  ({100*boundary/N:.1f}%)")
    print(f"[LBS]   mean arm weight = {arm_mean_weight:.3f}  (compare to hard arm fraction 0.30)")
    print(f"[LBS] wrote {OUT / 'gaussian_arm_weights.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
