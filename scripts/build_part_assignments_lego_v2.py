"""Build per-Gaussian part assignment + 3D arm trajectory + LBS weights for
the lego_v2 setup with the provided canonical 3DGS.

Inputs:
  - Canonical: outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply
  - Motion masks: runs_aux/parts_motion_lego_v2/view{v}_part_masks.npy
  - Cameras: data/custom/lego_v2/transforms_{train,test}.json (combined)

Outputs (mirrors old part_assignment/ layout for hier_v2 training compat):
  runs_aux/part_assignment_lego_v2/
    gaussian_arm_weights.npy        (N,) soft arm weight ∈ [0, 1]
    part_id.npy                     (N,) hard 0=arm, 1=body, 2=unassigned
    part_centroid_3d.npy            (T, 2, 3) [arm_3d, body_3d] per frame
    part_centroid_confidence.npy    (T, 2) per-part multi-view confidence
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402

CANON = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
MASK_DIR = REPO / "runs_aux/parts_motion_lego_v2"
OUT = REPO / "runs_aux/part_assignment_lego_v2"
SRC_META = REPO / "data/custom/lego_v2/transforms_train.json"
SRC_TEST = REPO / "data/custom/lego_v2/transforms_test.json"


def project_to_view(xyz, c2w, fov_x, H, W):
    """Returns (N, 2) pixel coords + (N,) z-depth (camera frame)."""
    w2c = np.linalg.inv(c2w)
    # Apply Blender convention: rotate 180° around X to go from c2w (Y-up, -Z fwd)
    # to OpenCV (Y-down, +Z fwd) — same as SC-GS internally
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c = flip @ w2c
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    cam = (w2c @ xyz_h.T).T[:, :3]  # (N, 3) in camera frame
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    fy = fx
    cx = W / 2
    cy = H / 2
    valid = z > 0
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
    return np.stack([u, v], axis=-1), z, valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Try fea_dim=8 (SV4D-trained canonical has features); fall back to 0 if needed
    try:
        g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
        g.load_ply(str(CANON), og_number_points=0)
    except (IndexError, RuntimeError, ValueError) as e:
        print(f"[parts-v2] fea_dim=8 failed ({e}); retry with fea_dim=0")
        g = GaussianModel(3, fea_dim=0, with_motion_mask=False)
        g.load_ply(str(CANON), og_number_points=0)
    xyz = g.get_xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    print(f"[parts-v2] canonical N={N}")

    # Combine train + test cameras
    meta_train = json.loads(SRC_META.read_text())
    meta_test = json.loads(SRC_TEST.read_text()) if SRC_TEST.exists() else {"frames": []}
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    H = W = 576
    T = max(int(f["frame_idx"]) for f in all_frames) + 1
    N_VIEWS = max(int(f["view_idx"]) for f in all_frames) + 1
    print(f"[parts-v2] views={N_VIEWS}, T={T}, total frames={len(all_frames)}")

    # Per-view motion mask (binary)
    motion_masks = {}    # v -> (H, W) moving region (cap. all 21 t's union)
    for v in range(N_VIEWS):
        m = np.load(MASK_DIR / f"view{v}_part_masks.npy")  # (2, H, W) [moving, static]
        motion_masks[v] = m[0].astype(bool)

    # === Per-Gaussian voting ===
    # Project each Gaussian's CANONICAL position to each view → check if in moving mask
    # Note: canonical pose != t=0 pose. The Gaussian may project to slightly different
    # location than the actual arm position. We accept this approximation.
    votes_moving = np.zeros(N, dtype=np.int32)
    votes_static = np.zeros(N, dtype=np.int32)
    votes_total = np.zeros(N, dtype=np.int32)
    # Get one camera per view (use t=0 transform)
    cams_by_view = {}
    for f in all_frames:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        cams_by_view[v] = np.asarray(f["transform_matrix"], dtype=np.float64)

    static_mask_per_view = {}
    for v in range(N_VIEWS):
        m = np.load(MASK_DIR / f"view{v}_part_masks.npy")
        static_mask_per_view[v] = m[1].astype(bool)

    for v in range(N_VIEWS):
        uv, z, valid = project_to_view(xyz, cams_by_view[v], fov_x, H, W)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        in_moving = motion_masks[v][vi, ui] & valid
        in_static = static_mask_per_view[v][vi, ui] & valid
        in_any = in_moving | in_static
        votes_moving += in_moving.astype(np.int32)
        votes_static += in_static.astype(np.int32)
        votes_total += in_any.astype(np.int32)

    print(f"[parts-v2] vote stats:")
    print(f"  Gaussians voted moving by >=3 views: {(votes_moving >= 3).sum()}")
    print(f"  Gaussians voted static by >=3 views: {(votes_static >= 3).sum()}")
    print(f"  no votes (off-screen?):              {(votes_total == 0).sum()}")

    # Hard part assignment
    part_id = np.full(N, 2, dtype=np.int32)  # 2 = unassigned
    part_id[(votes_moving > votes_static) & (votes_moving >= 2)] = 0   # arm
    part_id[(votes_static > votes_moving) & (votes_static >= 2)] = 1   # body
    n_arm = int((part_id == 0).sum())
    n_body = int((part_id == 1).sum())
    n_unassigned = int((part_id == 2).sum())
    print(f"[parts-v2] FINAL: arm={n_arm} ({n_arm/N*100:.1f}%)  "
          f"body={n_body} ({n_body/N*100:.1f}%)  unassigned={n_unassigned} ({n_unassigned/N*100:.1f}%)")

    # Soft arm weights (LBS): fraction of views that saw this Gaussian as moving
    valid_count = np.maximum(votes_total, 1)
    arm_weights = votes_moving / valid_count
    print(f"[parts-v2] LBS arm weights: mean={arm_weights.mean():.3f}  "
          f">0.5: {(arm_weights > 0.5).sum()}  boundary [0.1, 0.9]: {((arm_weights > 0.1) & (arm_weights < 0.9)).sum()}")

    np.save(OUT / "gaussian_arm_weights.npy", arm_weights.astype(np.float32))
    np.save(OUT / "part_id.npy", part_id)

    # === Per-frame 2D centroid → DLT triangulation ===
    # For each (view, t), compute centroid of motion mask region (using
    # per-time motion mask, which we don't have yet — only union).
    # Re-compute per-time motion mask from temporal-variance comparison.
    import imageio
    sv4d_dir = Path("/mnt/HDD_1/cthsu/lego/sv4d2")
    arm_centroid_2d = np.zeros((T, N_VIEWS, 2), dtype=np.float32)
    arm_conf_2d = np.zeros((T, N_VIEWS), dtype=np.float32)

    print(f"[parts-v2] computing per-(v, t) motion centroids…")
    for v in range(N_VIEWS):
        r = imageio.get_reader(str(sv4d_dir / f"000001_v00{v}.mp4"))
        frames = np.stack([r.get_data(t).mean(-1) for t in range(T)], axis=0)  # (T, H, W) gray
        # Background = ~constant high (white 255). Detect moving by frame-to-frame diff w/ canonical frame 0
        median_frame = np.median(frames, axis=0)  # (H, W) — temporal median ≈ background mean
        for t in range(T):
            diff = np.abs(frames[t] - median_frame)
            # Apply union motion mask to restrict to arm region
            diff_in_motion = diff * motion_masks[v].astype(np.float32)
            if diff_in_motion.sum() < 10:
                arm_centroid_2d[t, v] = [W / 2, H / 2]
                arm_conf_2d[t, v] = 0.0
                continue
            # Weighted centroid in motion region
            ys, xs = np.where(diff_in_motion > 0)
            ws = diff_in_motion[ys, xs]
            cx = np.sum(xs * ws) / np.sum(ws)
            cy = np.sum(ys * ws) / np.sum(ws)
            arm_centroid_2d[t, v] = [cx, cy]
            # Confidence = signal-to-noise ratio
            arm_conf_2d[t, v] = float(min(1.0, ws.sum() / 50000.0))

    # DLT triangulation per time
    centroid_3d = np.zeros((T, 2, 3), dtype=np.float32)  # [arm, body]
    conf_3d = np.zeros((T, 2), dtype=np.float32)

    # Build projection matrices per view
    Ks = []
    extrinsics = []
    for v in range(N_VIEWS):
        c2w = cams_by_view[v]
        w2c = np.linalg.inv(c2w)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = flip @ w2c
        fx = (W / 2) / np.tan(fov_x / 2)
        K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]])
        Ks.append(K)
        extrinsics.append(w2c[:3, :])

    for t in range(T):
        # Triangulate arm centroid
        A = []
        for v in range(N_VIEWS):
            if arm_conf_2d[t, v] < 0.05: continue
            P = Ks[v] @ extrinsics[v]  # (3, 4)
            u, vp = arm_centroid_2d[t, v]
            A.append(u * P[2] - P[0])
            A.append(vp * P[2] - P[1])
        if len(A) < 4:  # need >= 2 views (4 eqs)
            centroid_3d[t, 0] = [0, 0, 0]
            conf_3d[t, 0] = 0.0
            continue
        A = np.stack(A)
        _, _, vt = np.linalg.svd(A)
        Xh = vt[-1]
        X = Xh[:3] / Xh[3]
        centroid_3d[t, 0] = X
        # Confidence: mean of valid view confidences
        conf_3d[t, 0] = float(arm_conf_2d[t, arm_conf_2d[t] > 0.05].mean())

    # Body centroid (just the mean of body Gaussian xyz — static reference)
    body_xyz = xyz[part_id == 1]
    body_center = body_xyz.mean(0)
    centroid_3d[:, 1, :] = body_center  # body is static
    conf_3d[:, 1] = 1.0

    np.save(OUT / "part_centroid_3d.npy", centroid_3d)
    np.save(OUT / "part_centroid_confidence.npy", conf_3d)
    print(f"[parts-v2] arm trajectory mean conf = {conf_3d[:, 0].mean():.3f}")
    print(f"[parts-v2] arm 3D range: x[{centroid_3d[:, 0, 0].min():.2f},{centroid_3d[:, 0, 0].max():.2f}] "
          f"y[{centroid_3d[:, 0, 1].min():.2f},{centroid_3d[:, 0, 1].max():.2f}] "
          f"z[{centroid_3d[:, 0, 2].min():.2f},{centroid_3d[:, 0, 2].max():.2f}]")
    print(f"[parts-v2] saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
