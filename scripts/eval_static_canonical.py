"""Attribution ladder Rung 0: frozen canonical with ZERO motion.

Renders the clean canonical at every (cam, t) with NO deformation and evaluates
vs d-3dgs clean GT and vs SV4D supervision. Answers: how much of the Phase 2
PSNR comes purely from having a clean canonical, before any motion is learned?

  Phase 2 full (K=100 + smart + scale + xyzres) ... 20.28 dB vs d-3dgs
  THIS (static, no motion)                       ... ? dB vs d-3dgs
  difference = value contributed by the learned part-rigid motion
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="lego_v2")
    ap.add_argument("--canon_ply", default=None,
                    help="default: outputs/custom/<scene>_canonical/.../point_cloud.ply (latest iter)")
    args = ap.parse_args()

    SCENE = REPO / "data/custom" / args.scene
    D3DGS = REPO / "outputs/custom" / f"{args.scene}_d3dgs_ref" / "renders"
    if args.canon_ply:
        CANON = Path(args.canon_ply)
    else:
        cdir = None
        for suffix in ("_canonical", "_canonical_node"):
            cand = REPO / "outputs/custom" / f"{args.scene}{suffix}" / "point_cloud"
            if cand.is_dir() and list(cand.glob("iteration_*")):
                cdir = cand
        if cdir is None:
            raise FileNotFoundError(f"no canonical point_cloud for scene {args.scene}")
        iters = sorted(cdir.glob("iteration_*"), key=lambda p: int(p.name.split("_")[1]))
        CANON = iters[-1] / "point_cloud.ply"
    print(f"[static] scene={args.scene}  canon={CANON}")

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(CANON), og_number_points=0)
            break
        except (IndexError, RuntimeError, ValueError):
            g = None
    if g is None:
        raise RuntimeError("can't load canonical")
    N = g.get_xyz.shape[0]
    print(f"[static] canonical N={N}")

    meta_tr = json.loads((SCENE / "transforms_train.json").read_text())
    meta_te = json.loads((SCENE / "transforms_test.json").read_text())
    frames = meta_tr["frames"] + meta_te["frames"]
    fov_x = meta_tr["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser = _A(); pipe = PipelineParams(parser).extract(parser.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    zero = torch.zeros(N, 3, device="cuda")
    drot = (torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda"))

    vs_d3, vs_sv = [], []
    per_t = {}
    for f in frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]
        cam = SCGSCamera(colmap_id=0, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(0.0).float())
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=zero, d_rotation=drot,
                         d_scaling=zero, d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

        name = f"{Path(f['file_path']).name}.png"
        sp = None
        for split in ("train", "test"):
            c = SCENE / split / name
            if c.exists(): sp = c; break
        sa = np.asarray(iio.imread(sp), dtype=np.float32) / 255.0
        a = sa[..., 3:4]; sv_rgb = sa[..., :3] * a + 1.0 * (1 - a)

        d3 = np.asarray(iio.imread(D3DGS / f"{v*21+t:05d}.png"), dtype=np.float32) / 255.0
        if d3.shape[-1] == 4:
            ad = d3[..., 3:4]; d3_rgb = d3[..., :3] * ad + 1 * (1 - ad)
        else:
            d3_rgb = d3[..., :3]

        pd = psnr(img, d3_rgb); ps = psnr(img, sv_rgb)
        vs_d3.append(pd); vs_sv.append(ps)
        per_t.setdefault(t, []).append(pd)

    d3a = np.array(vs_d3); sva = np.array(vs_sv)
    print(f"[static] vs d-3dgs (CLEAN GT) : mean={d3a.mean():.3f}  median={np.median(d3a):.3f}  std={d3a.std():.3f}")
    print(f"[static] vs SV4D (supervision): mean={sva.mean():.3f}")
    # per-t: shows the canonical is at a reference pose (best near that t)
    ts = sorted(per_t)
    best_t = max(ts, key=lambda t: np.mean(per_t[t]))
    print(f"[static] best t (canonical reference pose) = t{best_t}: {np.mean(per_t[best_t]):.2f} dB")
    print(f"[static] worst t (trajectory extreme)      = t{min(ts, key=lambda t: np.mean(per_t[t]))}: "
          f"{min(np.mean(per_t[t]) for t in ts):.2f} dB")


if __name__ == "__main__":
    main()
