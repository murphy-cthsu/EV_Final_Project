"""Generic motion mask + part assignment for any dataset.
Replaces hardcoded scene00_masked / lego_v2_masks paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from skimage.filters import threshold_otsu

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402


def project_to_view(xyz, c2w, fov_x, H, W):
    w2c = np.linalg.inv(c2w)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c = flip @ w2c
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    valid = z > 0
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return np.stack([u, v], axis=-1), z, valid


def kmeans_simple(x: np.ndarray, K: int, n_iter: int = 50, seed: int = 0):
    """Simple Lloyd's k-means (same as train_partrigid_hier)."""
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(x.shape[0], K, replace=False)]
    for _ in range(n_iter):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.stack([
            x[labels == k].mean(0) if (labels == k).sum() > 0 else centers[k]
            for k in range(K)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels, centers


def assign_chunked(x: np.ndarray, centers: np.ndarray, chunk: int = 20000):
    """argmin_k ||x - centers_k|| in chunks (x can be ~100k rows)."""
    out = np.empty(x.shape[0], dtype=np.int32)
    for i in range(0, x.shape[0], chunk):
        d = np.linalg.norm(x[i:i+chunk, None, :] - centers[None, :, :], axis=2)
        out[i:i+chunk] = d.argmin(axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="e.g., lego_v3, hellwarrior")
    p.add_argument("--canon_ply", required=True)
    p.add_argument("--threshold_method", choices=["otsu", "pct50"], default="otsu")
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--n_parts", type=int, default=1,
                   help="number of moving parts for Stage D. 1 = legacy single-centroid "
                        "trajectory. >1 = K-means on per-pixel temporal profiles; each "
                        "part gets its own DLT trajectory (for multi-limb scenes).")
    p.add_argument("--src_video_root", default=None,
                   help="path to sv4d2 video root (auto if omitted)")
    p.add_argument("--out_suffix", default="",
                   help="suffix for output dirs, e.g. '_cleancanon_p4' to avoid "
                        "clobbering existing part assignments")
    args = p.parse_args()

    data_dir = REPO / "data/custom" / args.dataset
    out_motion = REPO / "runs_aux" / f"parts_motion_{args.dataset}{args.out_suffix}"
    out_motion.mkdir(parents=True, exist_ok=True)
    out_parts = REPO / "runs_aux" / f"part_assignment_{args.dataset}{args.out_suffix}"
    out_parts.mkdir(parents=True, exist_ok=True)

    # Source video root
    if args.src_video_root:
        src_video_root = Path(args.src_video_root)
    else:
        if args.dataset == "lego_v3":
            src_video_root = Path("/mnt/HDD_1/cthsu/lego_v3/sv4d2/lego_r7_train_iter30000")
        elif args.dataset == "hellwarrior":
            src_video_root = Path("/mnt/HDD_1/cthsu/hellwarrior/sv4d2/hellwarrior_r32_train_iter40000")
        else:
            raise ValueError(f"unknown dataset {args.dataset}")

    meta_train = json.loads((data_dir / "transforms_train.json").read_text())
    meta_test = json.loads((data_dir / "transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    H = W = 576
    N_VIEWS = max(int(f["view_idx"]) for f in all_frames) + 1
    T = args.n_frames
    print(f"[motion-{args.dataset}] views={N_VIEWS}  T={T}")

    # ===== Stage B: motion mask per view via temporal variance =====
    print(f"[motion-{args.dataset}] Stage B: motion mask")
    view_to_tag = {}
    for f in all_frames:
        view_to_tag[int(f["view_idx"])] = f.get("view_tag", None)
    for v in range(N_VIEWS):
        tag = view_to_tag.get(v)
        if tag is None: continue
        mp4 = src_video_root / f"{tag}.mp4"
        if not mp4.exists():
            print(f"  view {v} ({tag}): mp4 missing, skip")
            continue
        r = imageio.get_reader(str(mp4))
        gray_frames = np.stack([r.get_data(t).mean(-1) for t in range(T)], axis=0)
        std = gray_frames.std(axis=0)
        # FG via alpha across t
        alphas = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            flat = v * T + t
            for split in ("train", "test"):
                p = data_dir / split / f"r_{flat:05d}.png"
                if p.exists():
                    rgba = np.asarray(Image.open(p))
                    alphas[t] = (rgba[..., 3] > 127).astype(np.float32)
                    break
        fg_any = alphas.max(axis=0) > 0.5
        std_in_fg = std[fg_any]
        if std_in_fg.size < 100:
            continue
        if args.threshold_method == "otsu":
            try:
                thresh = threshold_otsu(std_in_fg)
            except Exception:
                thresh = np.median(std_in_fg)
            # Safety floor/ceiling
            moving_frac = (std > thresh)[fg_any].mean()
            if moving_frac < 0.05:
                thresh = np.percentile(std_in_fg, 95)
            elif moving_frac > 0.7:
                thresh = np.percentile(std_in_fg, 30)
        else:
            thresh = np.percentile(std_in_fg, 50)
        moving = (std > thresh) & fg_any
        static = (std <= thresh) & fg_any
        moving = binary_closing(binary_opening(moving, iterations=2), iterations=2)
        static = binary_closing(binary_opening(static, iterations=2), iterations=2)
        moving = binary_dilation(moving, iterations=1)
        out = np.stack([moving.astype(np.uint8), static.astype(np.uint8)], axis=0)
        np.save(out_motion / f"view{v}_part_masks.npy", out)
        if (v + 1) % 10 == 0:
            print(f"  view {v+1}/{N_VIEWS} masked  (thresh={thresh:.2f}, moving={int(moving.sum())}, static={int(static.sum())})")

    # ===== Stage C: per-Gaussian voting =====
    print(f"[motion-{args.dataset}] Stage C: per-Gaussian voting")
    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.canon_ply), og_number_points=0); break
        except Exception:
            g = None
    xyz = g.get_xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    print(f"  canonical N={N}")

    cams_by_view = {}
    for f in all_frames:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        cams_by_view[v] = np.asarray(f["transform_matrix"], dtype=np.float64)

    motion_masks = {}; static_masks = {}
    for v in range(N_VIEWS):
        p = out_motion / f"view{v}_part_masks.npy"
        if p.exists():
            m = np.load(p)
            motion_masks[v] = m[0].astype(bool)
            static_masks[v] = m[1].astype(bool)

    votes_moving = np.zeros(N, dtype=np.int32)
    votes_static = np.zeros(N, dtype=np.int32)
    votes_total = np.zeros(N, dtype=np.int32)
    for v, m_moving in motion_masks.items():
        m_static = static_masks[v]
        uv, z, valid = project_to_view(xyz, cams_by_view[v], fov_x, H, W)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        in_m = m_moving[vi, ui] & valid
        in_s = m_static[vi, ui] & valid
        votes_moving += in_m.astype(np.int32)
        votes_static += in_s.astype(np.int32)
        votes_total += (in_m | in_s).astype(np.int32)

    n_total_views = len(motion_masks)
    # 1/3 threshold for voted moving (was 3 for 5 views, scale up for 57)
    move_thresh = max(3, n_total_views // 4)
    static_thresh = max(3, n_total_views // 4)
    part_id = np.full(N, 2, dtype=np.int32)
    part_id[(votes_moving > votes_static) & (votes_moving >= move_thresh)] = 0
    part_id[(votes_static > votes_moving) & (votes_static >= static_thresh)] = 1
    n_arm = int((part_id == 0).sum())
    n_body = int((part_id == 1).sum())
    n_unassigned = int((part_id == 2).sum())
    print(f"  arm={n_arm} ({n_arm/N*100:.1f}%)  body={n_body} ({n_body/N*100:.1f}%)  unassigned={n_unassigned} ({n_unassigned/N*100:.1f}%)")

    arm_weights = votes_moving / np.maximum(votes_total, 1)
    np.save(out_parts / "gaussian_arm_weights.npy", arm_weights.astype(np.float32))
    np.save(out_parts / "part_id.npy", part_id)

    # ===== Stage D: 3D centroid trajectory =====
    print(f"[motion-{args.dataset}] Stage D: 3D trajectory (n_parts={args.n_parts})")

    Ks, exts = [], []
    for v in range(N_VIEWS):
        if v not in cams_by_view:
            Ks.append(None); exts.append(None); continue
        c2w = cams_by_view[v]
        w2c = np.linalg.inv(c2w)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = flip @ w2c
        fx = (W / 2) / np.tan(fov_x / 2)
        K_int = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]])
        Ks.append(K_int); exts.append(w2c[:3, :])

    def dlt_triangulate(pts_2d, confs_2d, conf_min=0.05):
        """pts_2d (N_VIEWS, 2), confs_2d (N_VIEWS,) -> X (3,) or None."""
        A = []
        for v in range(N_VIEWS):
            if Ks[v] is None or confs_2d[v] < conf_min:
                continue
            P = Ks[v] @ exts[v]
            u, vp = pts_2d[v]
            A.append(u * P[2] - P[0])
            A.append(vp * P[2] - P[1])
        if len(A) < 4:
            return None
        A = np.stack(A)
        _, _, vt = np.linalg.svd(A)
        Xh = vt[-1]
        return Xh[:3] / Xh[3]

    body_xyz = xyz[part_id == 1]
    body_center = body_xyz.mean(0) if len(body_xyz) > 0 else xyz.mean(0)

    if args.n_parts <= 1:
        arm_centroid_2d = np.zeros((T, N_VIEWS, 2), dtype=np.float32)
        arm_conf_2d = np.zeros((T, N_VIEWS), dtype=np.float32)
        for v, tag in view_to_tag.items():
            if tag is None or v not in motion_masks:
                continue
            mp4 = src_video_root / f"{tag}.mp4"
            if not mp4.exists():
                continue
            r = imageio.get_reader(str(mp4))
            gray_frames = np.stack([r.get_data(t).mean(-1) for t in range(T)], axis=0)
            median_frame = np.median(gray_frames, axis=0)
            for t in range(T):
                diff = np.abs(gray_frames[t] - median_frame)
                diff_in_motion = diff * motion_masks[v].astype(np.float32)
                if diff_in_motion.sum() < 10:
                    continue
                ys, xs = np.where(diff_in_motion > 0)
                ws = diff_in_motion[ys, xs]
                cx = np.sum(xs * ws) / np.sum(ws)
                cy = np.sum(ys * ws) / np.sum(ws)
                arm_centroid_2d[t, v] = [cx, cy]
                arm_conf_2d[t, v] = float(min(1.0, ws.sum() / 50000.0))

        centroid_3d = np.zeros((T, 2, 3), dtype=np.float32)
        conf_3d = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            X = dlt_triangulate(arm_centroid_2d[t], arm_conf_2d[t])
            if X is None:
                continue
            centroid_3d[t, 0] = X
            confs = arm_conf_2d[t]
            conf_3d[t, 0] = float(confs[confs > 0.05].mean())
        centroid_3d[:, 1, :] = body_center
        conf_3d[:, 1] = 1.0
    else:
        # ===== Multi-part Stage D: K-means on per-pixel temporal profiles =====
        # The (T,)-dim |gray - median| profile of a pixel encodes WHEN it moves.
        # Pixels of the same physical part share timing across ALL views, so a
        # single global K-means gives cross-view part correspondence for free.
        P_parts = args.n_parts
        rng = np.random.default_rng(0)
        cache = {}   # v -> (ys, xs, prof float16 (n, T))
        samples = []
        for v, tag in sorted(view_to_tag.items()):
            if tag is None or v not in motion_masks:
                continue
            mp4 = src_video_root / f"{tag}.mp4"
            if not mp4.exists():
                continue
            r = imageio.get_reader(str(mp4))
            gray_frames = np.stack([r.get_data(t).mean(-1) for t in range(T)], axis=0)
            diff = np.abs(gray_frames - np.median(gray_frames, axis=0))  # (T, H, W)
            ys, xs = np.where(motion_masks[v])
            prof = diff[:, ys, xs].T.astype(np.float32)  # (n, T)
            mag = prof.sum(1)
            keep = mag > 1.0
            ys, xs, prof = ys[keep], xs[keep], prof[keep]
            if prof.shape[0] < 50:
                continue
            cache[v] = (ys, xs, prof.astype(np.float16))
            profn = prof / np.linalg.norm(prof, axis=1, keepdims=True).clip(1e-6)
            idx = rng.choice(profn.shape[0], min(2000, profn.shape[0]), replace=False)
            samples.append(profn[idx])
        X_all = np.concatenate(samples, axis=0)
        print(f"  profile k-means: {X_all.shape[0]} px samples from {len(cache)} views, K={P_parts}")
        _, prof_centers = kmeans_simple(X_all, P_parts, seed=0)

        arm_centroid_2d = np.zeros((T, N_VIEWS, P_parts, 2), dtype=np.float32)
        arm_conf_2d = np.zeros((T, N_VIEWS, P_parts), dtype=np.float32)
        label_maps = {}
        for v, (ys, xs, prof16) in cache.items():
            prof = prof16.astype(np.float32)
            profn = prof / np.linalg.norm(prof, axis=1, keepdims=True).clip(1e-6)
            lab = assign_chunked(profn, prof_centers)
            label_map = np.full((H, W), -1, dtype=np.int8)
            label_map[ys, xs] = lab
            label_maps[v] = label_map
            np.save(out_motion / f"view{v}_part_label.npy", label_map)
            for t in range(T):
                w_t = prof[:, t]
                for p_i in range(P_parts):
                    m = lab == p_i
                    wsum = float(w_t[m].sum())
                    if wsum < 10:
                        continue
                    arm_centroid_2d[t, v, p_i] = [
                        float((xs[m] * w_t[m]).sum() / wsum),
                        float((ys[m] * w_t[m]).sum() / wsum),
                    ]
                    arm_conf_2d[t, v, p_i] = min(1.0, wsum / 20000.0)

        centroid_3d = np.zeros((T, P_parts + 1, 3), dtype=np.float32)
        conf_3d = np.zeros((T, P_parts + 1), dtype=np.float32)
        for t in range(T):
            for p_i in range(P_parts):
                X = dlt_triangulate(arm_centroid_2d[t, :, p_i], arm_conf_2d[t, :, p_i])
                if X is None:
                    continue
                centroid_3d[t, p_i] = X
                cs = arm_conf_2d[t, :, p_i]
                conf_3d[t, p_i] = float(cs[cs > 0.05].mean())
        # Fill DLT-failed frames from nearest valid frame (avoids wild init jumps)
        for p_i in range(P_parts):
            valid_t = np.where(conf_3d[:, p_i] > 0)[0]
            if valid_t.size == 0:
                print(f"  WARNING: part {p_i} never triangulated")
                continue
            for t in range(T):
                if conf_3d[t, p_i] == 0:
                    centroid_3d[t, p_i] = centroid_3d[valid_t[np.argmin(np.abs(valid_t - t))], p_i]
        centroid_3d[:, -1, :] = body_center
        conf_3d[:, -1] = 1.0

        # Per-Gaussian motion-part label by multi-view voting on label maps
        votes = np.zeros((N, P_parts), dtype=np.int32)
        for v, label_map in label_maps.items():
            uv, z, valid = project_to_view(xyz, cams_by_view[v], fov_x, H, W)
            ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
            vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
            lab_g = label_map[vi, ui].astype(np.int32)
            ok = valid & (lab_g >= 0)
            np.add.at(votes, (np.where(ok)[0], lab_g[ok]), 1)
        gauss_part = np.where(votes.sum(1) > 0, votes.argmax(1), -1).astype(np.int32)
        gauss_part[part_id != 0] = -1  # only arm Gaussians carry a motion part
        np.save(out_parts / "gaussian_motion_part.npy", gauss_part)
        sizes = [int((gauss_part == p_i).sum()) for p_i in range(P_parts)]
        print(f"  per-Gaussian motion parts (arm only): {sizes}, unlabelled arm = "
              f"{int(((gauss_part < 0) & (part_id == 0)).sum())}")

    np.save(out_parts / "part_centroid_3d.npy", centroid_3d)
    np.save(out_parts / "part_centroid_confidence.npy", conf_3d)
    print(f"  arm conf mean = {conf_3d[:, 0].mean():.3f}, 3D range x[{centroid_3d[:, 0, 0].min():.2f}, {centroid_3d[:, 0, 0].max():.2f}]")
    print(f"[motion-{args.dataset}] saved to {out_parts}")


if __name__ == "__main__":
    raise SystemExit(main())
