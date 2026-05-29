"""Stages C + D: per-Gaussian part ID + 3D part centroid trajectory.

Inputs:
    - Canonical 3DGS .ply (from P1)
    - Per-view per-(time) part masks from runs_aux/parts_motion/
    - transforms_train.json (camera poses)

Outputs (runs_aux/part_assignment/):
    - gaussian_part_ids.npy           # (N_g,) int in {0=arm, 1=body, -1=unassigned}
    - part_centroid_3d.npy            # (T, 2, 3) float -- per-time per-part 3D centroid
    - part_centroid_confidence.npy    # (T, 2) float -- 1/(1+reproj_rmse)
    - vote_diagnostics.json           # per-Gaussian vote summary

Stage C method (per-Gaussian voting):
    1. Project each canonical Gaussian to each view at t=0 (camera intrinsics from FOV).
    2. Look up frame-0 part mask at projected pixel → per-view part vote.
    3. Majority vote across V views → per-Gaussian part ID.
    4. Gaussians with no foreground projection in any view → -1 (static background).

Stage D method (DLT triangulation):
    For each (t, part), build per-view rays from camera center through the
    per-(view, time) 2D centroid pixel. Solve a 3D point that minimizes the
    sum of squared point-to-line distances via SVD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

import torch  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402

CANON = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PARTS_DIR = REPO_ROOT / "runs_aux" / "parts_motion"
SRC = REPO_ROOT / "data" / "custom" / "scene00_masked"
OUT = REPO_ROOT / "runs_aux" / "part_assignment"


def make_K(W: int, H: int, fov_x: float) -> np.ndarray:
    fx = 0.5 * W / np.tan(0.5 * fov_x)
    return np.array([[fx, 0, W/2.0],
                     [0, fx, H/2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def c2w_blender_to_w2c_cv(c2w_b: np.ndarray) -> np.ndarray:
    """Blender c2w (cam down -Z) -> OpenCV w2c (cam down +Z, Y down, X right)."""
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_cv = c2w_b @ flip
    return np.linalg.inv(c2w_cv)


def project(xyz: np.ndarray, K: np.ndarray, w2c: np.ndarray, W: int, H: int):
    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    cam = homog @ w2c.T
    z = cam[:, 2]
    in_front = z > 1e-3
    z_safe = np.where(in_front, z, 1.0)
    uv = (cam[:, :2] / z_safe[:, None]) @ K[:2, :2].T + K[:2, 2]
    u = uv[:, 0]
    v = uv[:, 1]
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front
    return uv, in_image


def triangulate_dlt(rays_origin: np.ndarray, rays_dir: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form mid-point triangulation: 3D point that minimizes sum of
    squared distances to all input rays (each defined by origin + direction).

    Returns (point_3d, reprojection_rmse).
    """
    # Each ray defines an outer-product projector I - dd^T
    N = rays_origin.shape[0]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N):
        d = rays_dir[i] / (np.linalg.norm(rays_dir[i]) + 1e-12)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ rays_origin[i]
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
    # RMSE: distance from p to each ray
    dists = []
    for i in range(N):
        d = rays_dir[i] / (np.linalg.norm(rays_dir[i]) + 1e-12)
        delta = p - rays_origin[i]
        perp = delta - (delta @ d) * d
        dists.append(np.linalg.norm(perp))
    rmse = float(np.sqrt(np.mean(np.square(dists))))
    return p, rmse


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[stage_CD] loading canonical {CANON.name}")
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    N = xyz.shape[0]
    print(f"[stage_CD] canonical Gaussians: {N}")

    data = json.loads((SRC / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]

    by_view = {}
    for f in data["frames"]:
        by_view.setdefault(int(f["view_idx"]), []).append(f)
    V = len(by_view)
    n_frames = max(int(f["frame_idx"]) for f in data["frames"]) + 1
    print(f"[stage_CD] V={V}  T={n_frames}  fov_x={fov_x:.4f}")

    # ----- Stage C: per-Gaussian part voting -----
    H = W = 576
    K = make_K(W, H, fov_x)

    # Load per-view frame-0 part masks (P=2: arm, body)
    view_masks_f0 = {}
    view_c2w = {}
    for v in sorted(by_view.keys()):
        # use the spatial stddev mask (better-quality moving region at t=0)
        view_masks_f0[v] = np.load(PARTS_DIR / f"view{v}_frame0_part_masks.npy")  # (2, H, W) uint8
        # camera c2w for frame 0 of this view
        f0 = next(f for f in by_view[v] if int(f["frame_idx"]) == 0)
        view_c2w[v] = np.asarray(f0["transform_matrix"], dtype=np.float64)

    votes = np.zeros((N, 2), dtype=np.int32)
    inframe_count = np.zeros(N, dtype=np.int32)
    for v in sorted(by_view.keys()):
        w2c = c2w_blender_to_w2c_cv(view_c2w[v])
        uv, in_image = project(xyz, K, w2c, W, H)
        u_clip = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W - 1)
        v_clip = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H - 1)
        for part_idx in range(2):
            mask = view_masks_f0[v][part_idx].astype(bool)
            hits = in_image & mask[v_clip, u_clip]
            votes[hits, part_idx] += 1
        inframe_count += in_image.astype(np.int32)

    # Vote: pick part with most votes; tie → 1 (body, conservative default)
    arm_v = votes[:, 0]
    body_v = votes[:, 1]
    part_id = np.where(arm_v > body_v, 0,
              np.where(body_v > arm_v, 1, -1))
    # Gaussians with no in-frame projection in any view → -1
    part_id = np.where(inframe_count == 0, -1, part_id)
    n_arm = int((part_id == 0).sum())
    n_body = int((part_id == 1).sum())
    n_unk = int((part_id == -1).sum())
    print(f"[stage_C] votes: arm={n_arm}  body={n_body}  unassigned={n_unk}  total={N}")
    print(f"[stage_C] arm fraction: {n_arm / max(n_arm + n_body, 1):.3f}")
    np.save(OUT / "gaussian_part_ids.npy", part_id)

    # ----- Stage D: 3D per-(time, part) centroid trajectory -----
    centroid_3d = np.full((n_frames, 2, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((n_frames, 2), dtype=np.float64)

    # Precompute camera rays for each view's pixel centers
    # Per-time per-part: build rays through (view, time, part) centroid pixel
    per_view_centroids = {}
    for v in sorted(by_view.keys()):
        per_view_centroids[v] = np.load(PARTS_DIR / f"view{v}_centroids.npy")  # (T, 2, 2) (y, x)

    for t in range(n_frames):
        for part_idx in range(2):
            ray_o = []
            ray_d = []
            for v in sorted(by_view.keys()):
                cy, cx = per_view_centroids[v][t, part_idx]
                if np.isnan(cy):
                    continue
                # back-project pixel (cx, cy) to a 3D ray
                # K^-1 * [cx, cy, 1] → cam-space dir
                pix = np.array([cx, cy, 1.0])
                cam_dir = np.linalg.solve(K, pix)
                # c2w for this view (Blender), convert to OpenCV
                c2w_b = view_c2w[v]
                flip = np.diag([1, -1, -1, 1]).astype(np.float64)
                c2w_cv = c2w_b @ flip
                R_w = c2w_cv[:3, :3]
                t_w = c2w_cv[:3, 3]
                d_world = R_w @ cam_dir
                ray_o.append(t_w)
                ray_d.append(d_world)
            if len(ray_o) < 2:
                # not enough views to triangulate
                continue
            ray_o = np.stack(ray_o, axis=0)
            ray_d = np.stack(ray_d, axis=0)
            p3d, rmse = triangulate_dlt(ray_o, ray_d)
            centroid_3d[t, part_idx] = p3d
            confidence[t, part_idx] = 1.0 / (1.0 + rmse)

    np.save(OUT / "part_centroid_3d.npy", centroid_3d)
    np.save(OUT / "part_centroid_confidence.npy", confidence)

    # Summary print
    arm_traj = centroid_3d[:, 0]
    arm_valid = ~np.isnan(arm_traj[:, 0])
    if arm_valid.sum() > 0:
        arm_range = arm_traj[arm_valid].max(0) - arm_traj[arm_valid].min(0)
        print(f"[stage_D] arm 3D centroid sweep: dx={arm_range[0]:.2f} dy={arm_range[1]:.2f} dz={arm_range[2]:.2f}")
        print(f"[stage_D] arm confidence mean={confidence[:, 0].mean():.3f}  min={confidence[:, 0].min():.3f}")
    body_traj = centroid_3d[:, 1]
    body_valid = ~np.isnan(body_traj[:, 0])
    if body_valid.sum() > 0:
        body_range = body_traj[body_valid].max(0) - body_traj[body_valid].min(0)
        print(f"[stage_D] body 3D centroid sweep: dx={body_range[0]:.2f} dy={body_range[1]:.2f} dz={body_range[2]:.2f}")
        print(f"[stage_D] body confidence mean={confidence[:, 1].mean():.3f}")

    # Diagnostics
    diag = {
        "n_gaussians": N,
        "n_arm": n_arm, "n_body": n_body, "n_unassigned": n_unk,
        "vote_arm_mean": float(arm_v.mean()), "vote_body_mean": float(body_v.mean()),
        "inframe_mean": float(inframe_count.mean()),
        "arm_centroid_sweep_xyz": arm_range.tolist() if arm_valid.any() else None,
        "body_centroid_sweep_xyz": body_range.tolist() if body_valid.any() else None,
        "arm_centroid_t0": centroid_3d[0, 0].tolist() if arm_valid[0] else None,
        "arm_centroid_t14": centroid_3d[14, 0].tolist() if arm_valid[14] else None,
        "body_confidence_mean": float(confidence[:, 1].mean()),
        "arm_confidence_mean": float(confidence[:, 0].mean()),
    }
    (OUT / "vote_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"[stage_CD] outputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
