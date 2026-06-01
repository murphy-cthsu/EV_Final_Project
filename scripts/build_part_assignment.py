"""General part-assignment builder for the 57-view SV4D/d-3dgs scenes.

Produces exactly the 3 files train_partrigid_hier.py consumes from --part_dir:
  gaussian_arm_weights.npy     (N,)     per-Gaussian "moving" weight in [0,1]
  part_centroid_3d.npy         (T,2,3)  [moving_centroid, static_centroid] per frame
  part_centroid_confidence.npy (T,2)    multi-view triangulation confidence

Mechanism (scene-agnostic, replaces the lego-hardcoded arm/body scripts):
  1. Per view: temporal std of the SV4D video -> motion map; |frame_t - time_mean|
     -> per-frame motion location.
  2. Per canonical Gaussian: project to all views, sample the motion map, average
     over views -> moving weight. >0.5 gets K-means-clustered downstream.
  3. Per frame t: triangulate the 2D motion centroid across all views -> 3D moving
     centroid trajectory (warm-starts the SE(3) translation). Static centroid =
     mean of the non-moving Gaussians (constant).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # Blender c2w -> OpenCV


def load_canon(path: Path):
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(path), og_number_points=0)
            return g.get_xyz.detach().cpu().numpy()
        except (IndexError, RuntimeError, ValueError):
            continue
    raise RuntimeError(f"can't load canonical {path}")


def cam_intrinsics(fov_x, H, W):
    fx = (W / 2) / np.tan(fov_x / 2)
    return fx, fx, W / 2, H / 2


def project(xyz, c2w, fov_x, H, W):
    w2c = FLIP @ np.linalg.inv(c2w)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], -1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx, fy, cx, cy = cam_intrinsics(fov_x, H, W)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
    return u, v, z


def ray_world(u, v, c2w, fov_x, H, W):
    fx, fy, cx, cy = cam_intrinsics(fov_x, H, W)
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R = (c2w @ FLIP)[:3, :3]
    d = R @ d_cam
    d = d / (np.linalg.norm(d) + 1e-9)
    o = c2w[:3, 3]
    return o, d


def triangulate(rays):
    """rays: list of (o, d). Returns (point, mean_ray_distance)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    p = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    resid = np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)) for o, d in rays])
    return p, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--src", required=True, help="dataset root, e.g. /mnt/HDD_1/cthsu/hellwarrior")
    ap.add_argument("--canon_ply", default=None)
    ap.add_argument("--move_thresh", type=float, default=0.5,
                    help="per-Gaussian moving weight = frac of views with motion above this percentile")
    ap.add_argument("--n_frames", type=int, default=21)
    args = ap.parse_args()

    scene_dir = REPO / "data/custom" / args.scene
    canon = Path(args.canon_ply) if args.canon_ply else \
        sorted((REPO / "outputs/custom" / f"{args.scene}_canonical" / "point_cloud").glob("iteration_*"),
               key=lambda p: int(p.name.split("_")[1]))[-1] / "point_cloud.ply"
    out = REPO / "runs_aux" / f"part_assignment_{args.scene}"
    out.mkdir(parents=True, exist_ok=True)

    xyz = load_canon(canon)
    N = xyz.shape[0]
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    allf = (json.loads((scene_dir / "transforms_test.json").read_text())["frames"]
            + meta["frames"])
    fov_x = meta["camera_angle_x"]
    # one camera per view (frame_idx 0 suffices; cameras static over t)
    cams = {}
    for f in allf:
        cams.setdefault(int(f["view_idx"]), np.asarray(f["transform_matrix"], dtype=np.float64))
    V = len(cams)
    src_iter = sorted((Path(args.src) / "sv4d2").glob("*"))[0]
    print(f"[part] scene={args.scene} N={N} V={V} canon={canon.name}")

    H = W = 576
    T = args.n_frames
    # accumulate per-Gaussian motion samples + prep per-(v,t) motion centroids
    move_votes = np.zeros(N); seen = np.zeros(N)
    motion_centroids = {}  # v -> list over t of (u,v) or None
    for vi in sorted(cams):
        c2w = cams[vi]
        tag_frames = [f for f in allf if int(f["view_idx"]) == vi]
        tag = None
        # recover the elev_az tag from src by matching view order isn't stored; use mp4 by index order
        # Instead: read the mp4 named in the dataset's source order via transforms video field if present
        # Fallback: use the sv4d mp4 list aligned to view order in transforms_sv4d2_math.json
        # We reload from the SV4D video directly using the same tag ordering as build_scene_dataset.
        # Simplest robust path: read the per-(v,t) SV4D png we already wrote.
        vid = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            flat = vi * T + t
            for split in ("train", "test"):
                p = scene_dir / split / f"r_{flat:05d}.png"
                if p.exists():
                    im = iio.imread(p).astype(np.float32) / 255.0
                    rgb = im[..., :3] * im[..., 3:4] + (1 - im[..., 3:4]) if im.shape[-1] == 4 else im[..., :3]
                    vid[t] = rgb.mean(-1)
                    break
        time_mean = vid.mean(0)
        motion_std = vid.std(0)  # (H,W)
        thr = np.percentile(motion_std[motion_std > 1e-4], 70) if (motion_std > 1e-4).any() else 1e9
        # per-Gaussian sample
        u, v, z = project(xyz, c2w, fov_x, H, W)
        ui = np.round(u).astype(int); vj = np.round(v).astype(int)
        inb = (z > 0) & (ui >= 0) & (ui < W) & (vj >= 0) & (vj < H)
        samp = np.zeros(N)
        samp[inb] = motion_std[vj[inb], ui[inb]]
        move_votes += (samp > thr).astype(np.float32) * inb
        seen += inb.astype(np.float32)
        # per-t motion centroid (for triangulation)
        cent = []
        for t in range(T):
            diff = np.abs(vid[t] - time_mean)
            diff[motion_std <= thr] = 0  # restrict to moving region
            s = diff.sum()
            if s < 1e-3:
                cent.append(None)
            else:
                ys, xs = np.mgrid[0:H, 0:W]
                cu = (diff * xs).sum() / s; cv = (diff * ys).sum() / s
                cent.append((cu, cv))
        motion_centroids[vi] = cent

    arm_weights = move_votes / np.maximum(seen, 1)
    print(f"[part] mean arm_weight={arm_weights.mean():.3f}  frac>0.5={np.mean(arm_weights>0.5):.3f}")

    # triangulate moving centroid per frame
    centroid_3d = np.zeros((T, 2, 3)); conf = np.zeros((T, 2))
    static_c = xyz[arm_weights < 0.5].mean(0) if (arm_weights < 0.5).any() else xyz.mean(0)
    reproj_errs = []
    for t in range(T):
        rays = []
        for vi in sorted(cams):
            c = motion_centroids[vi][t]
            if c is None:
                continue
            rays.append(ray_world(c[0], c[1], cams[vi], fov_x, H, W))
        if len(rays) >= 2:
            p, resid = triangulate(rays)
            centroid_3d[t, 0] = p; conf[t, 0] = 1.0 / (1.0 + resid)
            reproj_errs.append(resid)
        else:
            centroid_3d[t, 0] = static_c; conf[t, 0] = 0.0
        centroid_3d[t, 1] = static_c; conf[t, 1] = 1.0

    np.save(out / "gaussian_arm_weights.npy", arm_weights.astype(np.float32))
    np.save(out / "part_centroid_3d.npy", centroid_3d.astype(np.float32))
    np.save(out / "part_centroid_confidence.npy", conf.astype(np.float32))
    disp = np.linalg.norm(centroid_3d[:, 0] - centroid_3d[0, 0], axis=-1)
    print(f"[part] arm centroid trajectory range: {disp.min():.3f}..{disp.max():.3f} "
          f"(0=no motion warm-start)")
    print(f"[part] mean triangulation residual: {np.mean(reproj_errs) if reproj_errs else float('nan'):.4f}")
    print(f"[part] saved -> {out}")


if __name__ == "__main__":
    main()
