"""Decompose arm Gaussians into BUCKET (high motion amplitude) vs ARM-SHAFT
(medium motion amplitude) for lego_v2 setup. Generic motion-variance based,
no SAM, no lego-specific clicks.

Method:
  - For each arm Gaussian (via existing arm_weights > 0.5), project to each view
  - Compute per-(view, t) temporal std of pixel color (gray) in the projection
  - High temporal-std rank = bucket (moves most across t)
  - Low temporal-std rank = arm-shaft (moves less)

Output:
  runs_aux/part_assignment_lego_v2/bucket_mask.npy   bool (N,) — True if bucket
  runs_aux/part_assignment_lego_v2/sub_assignment.npy int (N,) 0=body, 1=arm-shaft, 2=bucket
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402


def project(xyz, c2w, fov_x, H, W):
    w2c = np.linalg.inv(c2w)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c = flip @ w2c
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return u, v, z > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canon", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--part_dir", default=REPO / "runs_aux/part_assignment_lego_v2")
    p.add_argument("--scene_dir", default=REPO / "data/custom/lego_v2")
    p.add_argument("--src_videos", default=Path("/mnt/HDD_1/cthsu/lego/sv4d2"))
    p.add_argument("--bucket_top_frac", type=float, default=0.35,
                   help="Top fraction of arm Gaussians (by motion amplitude) → bucket")
    args = p.parse_args()

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.canon), og_number_points=0); break
        except Exception:
            g = None
    xyz = g.get_xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    print(f"[bucket-decomp] canonical N={N}")

    arm_w = np.load(Path(args.part_dir) / "gaussian_arm_weights.npy")
    arm_mask = arm_w > 0.5
    print(f"[bucket-decomp] arm Gaussians (w>0.5): {int(arm_mask.sum())}")

    meta = json.loads((Path(args.scene_dir) / "transforms_train.json").read_text())
    meta_test = json.loads((Path(args.scene_dir) / "transforms_test.json").read_text())
    all_frames = meta["frames"] + meta_test["frames"]
    fov_x = meta["camera_angle_x"]
    H = W = 576
    T = max(int(f["frame_idx"]) for f in all_frames) + 1
    V = max(int(f["view_idx"]) for f in all_frames) + 1

    cams_by_view = {}
    for f in all_frames:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        cams_by_view[v] = np.asarray(f["transform_matrix"], dtype=np.float64)

    # Compute per-arm-Gaussian motion amplitude:
    # = sum over views of (temporal std of pixel color at projection across t)
    motion_amp = np.zeros(N)
    for v in range(V):
        # Load all T frames of view v
        r = imageio.get_reader(str(Path(args.src_videos) / f"000001_v00{v}.mp4"))
        frames = np.stack([r.get_data(t).mean(-1) for t in range(T)], axis=0)  # (T, H, W) gray
        # Project all arm Gaussians at canonical position
        u, vp, valid = project(xyz, cams_by_view[v], fov_x, H, W)
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(vp.astype(int), 0, H - 1)
        # Temporal std at each pixel
        pixel_std = frames.std(axis=0)  # (H, W)
        # Each Gaussian gets the std of its projection
        per_g = pixel_std[vi, ui] * valid.astype(np.float32)
        motion_amp += per_g
    motion_amp = motion_amp / V

    # Rank arm Gaussians by motion amplitude; top fraction = bucket
    arm_idx = np.where(arm_mask)[0]
    arm_motion = motion_amp[arm_mask]
    n_bucket = int(len(arm_idx) * args.bucket_top_frac)
    threshold = np.sort(arm_motion)[-n_bucket] if n_bucket > 0 else float("inf")
    bucket_mask_local = arm_motion >= threshold
    bucket_mask = np.zeros(N, dtype=bool)
    bucket_mask[arm_idx[bucket_mask_local]] = True

    # 3-part: 0=body, 1=arm-shaft, 2=bucket
    sub_assign = np.full(N, 0, dtype=np.int32)  # body default
    sub_assign[arm_mask] = 1  # arm-shaft
    sub_assign[bucket_mask] = 2

    n_body = int((sub_assign == 0).sum())
    n_shaft = int((sub_assign == 1).sum())
    n_bucket = int((sub_assign == 2).sum())
    print(f"[bucket-decomp] body={n_body}  arm-shaft={n_shaft}  bucket={n_bucket}  "
          f"(bucket motion threshold={threshold:.2f})")

    np.save(Path(args.part_dir) / "bucket_mask.npy", bucket_mask)
    np.save(Path(args.part_dir) / "sub_assignment.npy", sub_assign)
    print(f"[bucket-decomp] saved to {args.part_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
