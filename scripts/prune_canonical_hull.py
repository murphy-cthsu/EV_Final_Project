"""Prune floater/spike Gaussians from a reconstructed canonical via multi-view
visual hull + scale cap.

SC-GS static reconstruction from 57 multiview frames leaves spiky floaters
(Gaussians whose centers sit outside the object silhouette in many views, or
which are huge/elongated). This deterministically removes them:
  - keep only Gaussians whose projected center is INSIDE the d-3dgs silhouette
    in >= keep_frac of the views where they are in-frame
  - drop Gaussians whose max scale exceeds the scale_pct percentile
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def project(xyz, c2w, fov_x, H, W):
    w2c = FLIP @ np.linalg.inv(c2w)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], -1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return u, v, z


def slice_params(g, keep):
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        p = getattr(g, name)
        setattr(g, name, nn.Parameter(p[keep]))
    if hasattr(g, "feature") and isinstance(g.feature, nn.Parameter) and g.feature.shape[0] == keep.shape[0]:
        g.feature = nn.Parameter(g.feature[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep_frac", type=float, default=0.7,
                    help="keep Gaussian if inside silhouette in >= this frac of in-frame views")
    ap.add_argument("--scale_pct", type=float, default=99.0,
                    help="drop Gaussians above this percentile of max-scale")
    ap.add_argument("--dilate", type=int, default=3, help="silhouette dilation px (tolerance)")
    args = ap.parse_args()

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(args.src, og_number_points=0); break
        except Exception:
            g = None
    if g is None:
        raise RuntimeError(f"can't load {args.src}")
    xyz = g.get_xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    scales = g.get_scaling.detach().cpu().numpy()  # (N,3) world scale
    print(f"[prune] N={N}")

    scene_dir = REPO / "data/custom" / args.scene
    d3_dir = REPO / "outputs/custom" / f"{args.scene}_d3dgs_ref" / "renders"
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    allf = (json.loads((scene_dir / "transforms_test.json").read_text())["frames"] + meta["frames"])
    fov_x = meta["camera_angle_x"]
    cams = {}
    for f in allf:
        cams.setdefault(int(f["view_idx"]), np.asarray(f["transform_matrix"], dtype=np.float64))
    H = W = 576
    from scipy.ndimage import binary_dilation

    inside = np.zeros(N); inframe = np.zeros(N)
    for vi, c2w in cams.items():
        d3 = iio.imread(d3_dir / f"{vi*21:05d}.png").astype(np.float32) / 255.0
        sil = d3[..., :3].mean(-1) < 0.98
        if args.dilate > 0:
            sil = binary_dilation(sil, iterations=args.dilate)
        u, v, z = project(xyz, c2w, fov_x, H, W)
        ui = np.round(u).astype(int); vj = np.round(v).astype(int)
        inb = (z > 0) & (ui >= 0) & (ui < W) & (vj >= 0) & (vj < H)
        inframe += inb
        hit = np.zeros(N, bool)
        hit[inb] = sil[vj[inb], ui[inb]]
        inside += hit
    frac_inside = inside / np.maximum(inframe, 1)
    keep_hull = frac_inside >= args.keep_frac
    smax = scales.max(1)
    keep_scale = smax <= np.percentile(smax, args.scale_pct)
    keep = keep_hull & keep_scale
    print(f"[prune] hull keep={keep_hull.mean():.3f}  scale keep={keep_scale.mean():.3f}  "
          f"final keep={keep.mean():.3f} ({int(keep.sum())}/{N})")

    keep_t = torch.from_numpy(keep)
    slice_params(g, keep_t)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    g.save_ply(args.dst)
    print(f"[prune] saved -> {args.dst}")


if __name__ == "__main__":
    main()
