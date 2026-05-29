"""GT-free 3D-consistency metric: Gaussian mask-projection coverage.

Given a trained SC-GS canonical, project every canonical Gaussian into each of
the V training views (at frame 0) and look up the GT FG alpha mask at the
projected pixel. A Gaussian that lands inside the FG mask in all V views is
"3D-consistent" (the 5 views agree it represents a real surface point). A
Gaussian that lands inside the FG mask in only 1-2 views is likely a
per-view-specific artifact the model produced to fit one view's hallucination.

The metric is the histogram of (#views_in_FG) over all Gaussians. Higher mass
at V (=5) = better 3D consistency. Mass at low counts = artifact Gaussians.

This is GT-FREE: it uses only the SAM-2 alpha masks (geometric / silhouette
info, no photometric GT) as the "this is where the foreground is" reference.
A method that successfully suppresses VGM-induced per-view inconsistency
should shift the histogram mass toward higher counts.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/gaussian_mv_consistency.py \\
        --scene_dir   data/custom/scene00_masked \\
        --model_path  outputs/custom/scene00_v5_node \\
        --label v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

import torch  # noqa: E402

# SC-GS uses its own scene/gaussian/deform classes
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402


def make_intrinsics(W: int, H: int, fov_x: float) -> np.ndarray:
    """OpenCV-style K for a square FOV camera, principal point at image center."""
    fx = 0.5 * W / np.tan(0.5 * fov_x)
    fy = fx  # square pixels, same FOV
    K = np.array([[fx, 0, W / 2.0],
                  [0, fy, H / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return K


def blender_c2w_to_opencv_w2c(c2w_blender: np.ndarray) -> np.ndarray:
    """Convert Blender-convention c2w (cam looks down -Z) to OpenCV w2c
    (cam looks down +Z, +X right, +Y down)."""
    # Flip y and z axes of the camera frame
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_cv = c2w_blender @ flip
    w2c = np.linalg.inv(c2w_cv)
    return w2c


def project_points(xyz_world: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                   W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Project N world-space points to a camera. Returns
    (uv (N, 2), in_front_mask (N,))."""
    pts_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1))], axis=1)  # (N, 4)
    pts_cam = pts_h @ w2c.T  # (N, 4)
    z = pts_cam[:, 2]
    in_front = z > 1e-3
    # safe divide
    z_safe = np.where(in_front, z, 1.0)
    pts_img = pts_cam[:, :3] / z_safe[:, None]
    pts_2d = pts_img[:, :2] @ K[:2, :2].T + K[:2, 2]
    return pts_2d, in_front


def load_model(model_path: Path) -> tuple[GaussianModel, DeformModel]:
    """Load a trained SC-GS model (canonical Gaussians + node deform)."""
    last_iter_dir = sorted([p for p in (model_path / "point_cloud").iterdir()
                            if p.is_dir() and p.name.startswith("iteration_")],
                           key=lambda p: int(p.name.split("_")[-1]))[-1]
    last_iter = int(last_iter_dir.name.split("_")[-1])
    print(f"[mv] loading {model_path}  iter={last_iter}")

    # Try to infer node count from saved deform tensor (varies by run)
    deform_pt = model_path / "deform" / f"iteration_{last_iter}" / "deform.pth"
    state = torch.load(deform_pt, map_location="cpu")
    node_num = state["nodes"].shape[0]
    hyper_dim = state["nodes"].shape[1] - 3  # nodes = [x, y, z, ...features]
    with_motion_mask = True  # we used both False and True; gm tolerates either

    gaussians = GaussianModel(3, fea_dim=hyper_dim, with_motion_mask=with_motion_mask)
    deform = DeformModel(K=4, deform_type="node", is_blender=False, skinning=False,
                        hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                        pred_color=False, use_hash=False, hash_time=False,
                        d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                        with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                        is_scene_static=False)
    deform.load_weights(str(model_path), iteration=last_iter)

    ply_path = last_iter_dir / "point_cloud.ply"
    gaussians.load_ply(str(ply_path), og_number_points=0)
    return gaussians, deform


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_dir", type=Path, required=True,
                   help="scene00_masked (provides transforms_train.json + alpha masks)")
    p.add_argument("--model_path", type=Path, required=True,
                   help="model_path (without _node suffix; will append)")
    p.add_argument("--label", type=str, required=True,
                   help="short label for output dir / table (e.g. 'v5')")
    p.add_argument("--out_dir", type=Path,
                   default=REPO_ROOT / "runs_aux" / "mv_consistency")
    p.add_argument("--time_idx", type=int, default=0,
                   help="frame index to evaluate deform at (default 0)")
    args = p.parse_args()

    # Resolve _node suffix if the bare path doesn't exist
    mp = args.model_path
    if not mp.exists() and mp.with_name(mp.name + "_node").exists():
        mp = mp.with_name(mp.name + "_node")
    if not mp.exists():
        raise FileNotFoundError(args.model_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read transforms_train.json to get camera intrinsics + extrinsics for all 5 views.
    # If only a partial split exists (e.g. v6 trained on view-split), we also try the
    # corresponding scene00_masked dir so all 5 views are available.
    src_json = args.scene_dir / "transforms_train.json"
    data = json.loads(src_json.read_text())
    cax = data["camera_angle_x"]
    frames = data["frames"]

    # Per-view (view_idx -> (c2w_blender, image_name))
    by_view: dict[int, dict] = {}
    n_frames_per_view = max(int(f["frame_idx"]) for f in frames) + 1
    for f in frames:
        v = int(f["view_idx"])
        t = int(f["frame_idx"])
        if t != args.time_idx:
            continue
        by_view[v] = {
            "c2w": np.asarray(f["transform_matrix"], dtype=np.float64),
            "file_path": f["file_path"],
        }
    if len(by_view) < 5:
        print(f"[mv] WARNING: only {len(by_view)} views available "
              f"at t={args.time_idx}. Some scenes may have fewer; metric is still valid.")
    V = len(by_view)
    view_ids = sorted(by_view.keys())

    # Load alpha masks for each view at the chosen time
    masks: dict[int, np.ndarray] = {}
    for v in view_ids:
        png_path = args.scene_dir / "train" / f"{Path(by_view[v]['file_path']).name}.png"
        img = np.asarray(iio.imread(png_path))
        if img.shape[-1] != 4:
            raise RuntimeError(f"expected RGBA at {png_path}, got shape {img.shape}")
        masks[v] = img[..., 3] > 127  # bool (H, W)
    H, W = masks[view_ids[0]].shape
    K = make_intrinsics(W, H, fov_x=cax)

    print(f"[mv] V={V} views, H={H} W={W}, fov_x={cax:.4f} rad")

    # Load model and produce deformed positions at the chosen time
    gaussians, deform = load_model(mp)
    xyz_canon = gaussians.get_xyz.detach()
    feat = gaussians.feature if gaussians.fea_dim > 0 else None
    motion_mask = getattr(gaussians, "motion_mask", None)
    fid = torch.tensor([float(args.time_idx) / max(n_frames_per_view - 1, 1)]).cuda()
    time_input = deform.deform.expand_time(fid)
    with torch.no_grad():
        d = deform.step(xyz_canon, time_input, feature=feat, motion_mask=motion_mask)
    xyz_deformed = (xyz_canon + d["d_xyz"]).cpu().numpy()

    # Per-Gaussian: how many views does it land inside FG?
    coverage = np.zeros(xyz_deformed.shape[0], dtype=np.int32)
    in_image_count = np.zeros(xyz_deformed.shape[0], dtype=np.int32)
    for v in view_ids:
        w2c = blender_c2w_to_opencv_w2c(by_view[v]["c2w"])
        uv, in_front = project_points(xyz_deformed, K, w2c, W, H)
        u = uv[:, 0]
        v_coord = uv[:, 1]
        in_image = (u >= 0) & (u < W) & (v_coord >= 0) & (v_coord < H) & in_front
        # for in_image pixels, look up mask
        in_image_count += in_image.astype(np.int32)
        u_clip = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        v_clip = np.clip(np.round(v_coord).astype(np.int32), 0, H - 1)
        mask_at = masks[v][v_clip, u_clip]  # (N,)
        is_fg = in_image & mask_at
        coverage += is_fg.astype(np.int32)

    # Restrict to Gaussians that project into the image in all V views (a fair
    # denominator -- otherwise off-frame Gaussians get penalised arbitrarily)
    in_all = in_image_count == V
    coverage_in_all = coverage[in_all]
    hist = np.bincount(coverage_in_all, minlength=V + 1)
    total = int(hist.sum())
    if total == 0:
        raise RuntimeError("no Gaussians project into all views; check pose conventions")

    # Headline numbers
    frac_full = float(hist[V] / total)
    frac_majority = float(hist[max(V - 1, 1):].sum() / total)  # in >= V-1 views
    frac_orphan = float(hist[:2].sum() / total)  # in 0 or 1 view only

    print(f"[mv] {args.label}: total in-image Gaussians = {total}")
    print(f"[mv]   coverage histogram (count of Gaussians in K views):")
    for k in range(V + 1):
        bar = "#" * int(40 * hist[k] / max(hist.max(), 1))
        print(f"[mv]     k={k}: {hist[k]:>6}  ({100*hist[k]/total:5.1f}%)  {bar}")
    print(f"[mv]   3D-consistency (in all V={V}):       {100*frac_full:5.2f}%")
    print(f"[mv]   majority-consistent (in >= V-1):     {100*frac_majority:5.2f}%")
    print(f"[mv]   orphan Gaussians (in 0 or 1 view):   {100*frac_orphan:5.2f}%")

    out = {
        "label": args.label,
        "model_path": str(mp),
        "n_views": int(V),
        "time_idx": int(args.time_idx),
        "total_in_image_gaussians": total,
        "coverage_histogram": [int(x) for x in hist.tolist()],
        "frac_3d_consistent": frac_full,
        "frac_majority_consistent": frac_majority,
        "frac_orphan": frac_orphan,
    }
    out_json = args.out_dir / f"{args.label}_t{args.time_idx}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[mv] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
