"""Render clean-ref 4D-GS WITHOUT baseplate Gaussians (z > z_cutoff).

Same as render_clean_ref_fine_grid.py but filters out low-z Gaussians (baseplate)
to remove the bbox/foreground mismatch with SAM-2-masked SV4D renders.

Output: runs_aux/clean_gt_fine_nobase/renders/r_v{V}_f{F:03d}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402


def filter_gaussians_by_z(g: GaussianModel, z_min: float):
    """In-place: keep only Gaussians with z > z_min."""
    xyz = g.get_xyz.detach()
    keep = (xyz[:, 2] > z_min)
    n_before = xyz.shape[0]
    n_keep = int(keep.sum())
    print(f"[nobase] filtering z > {z_min}: keep {n_keep}/{n_before} ({100*n_keep/n_before:.1f}%)")
    # Index all per-Gaussian tensors
    import torch.nn as nn
    g._xyz = nn.Parameter(g._xyz[keep])
    g._features_dc = nn.Parameter(g._features_dc[keep])
    g._features_rest = nn.Parameter(g._features_rest[keep])
    g._scaling = nn.Parameter(g._scaling[keep])
    g._rotation = nn.Parameter(g._rotation[keep])
    g._opacity = nn.Parameter(g._opacity[keep])
    if hasattr(g, "max_radii2D") and g.max_radii2D is not None and g.max_radii2D.shape[0] == n_before:
        g.max_radii2D = g.max_radii2D[keep]
    # self.feature is a Parameter (not property) — the hyper coord buffer
    if hasattr(g, "feature") and isinstance(g.feature, nn.Parameter) and g.feature.shape[0] == n_before:
        g.feature = nn.Parameter(g.feature[keep])
    if hasattr(g, "motion_mask") and g.motion_mask is not None and g.motion_mask.shape[0] == n_before:
        g.motion_mask = g.motion_mask[keep]
    return keep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, default=REPO_ROOT / "outputs/custom/lego_clean_ref")
    p.add_argument("--our_scene",  type=Path, default=REPO_ROOT / "data/custom/scene00_masked")
    p.add_argument("--out_dir",    type=Path, default=REPO_ROOT / "runs_aux/clean_gt_fine_nobase")
    p.add_argument("--n_fid", type=int, default=100)
    p.add_argument("--z_min", type=float, default=-0.15,
                   help="Gaussian Z cutoff (keep only z > z_min). z<-0.15 ≈ baseplate band.")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    mp = args.model_path
    if not mp.exists() and mp.with_name(mp.name + "_node").exists():
        mp = mp.with_name(mp.name + "_node")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "renders").mkdir(exist_ok=True)

    pc_iters = sorted([int(pp.name.split("_")[-1])
                       for pp in (mp / "point_cloud").iterdir()
                       if pp.name.startswith("iteration_")])
    iter_use = pc_iters[-1] if args.iteration == -1 else args.iteration
    print(f"[nobase] loading {mp.name} iter={iter_use}")

    gaussians = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    ply_path = mp / "point_cloud" / f"iteration_{iter_use}" / "point_cloud.ply"
    gaussians.load_ply(str(ply_path), og_number_points=0)
    N_orig = gaussians.get_xyz.shape[0]
    keep_mask = filter_gaussians_by_z(gaussians, args.z_min)

    deform_state = torch.load(mp / "deform" / f"iteration_{iter_use}" / "deform.pth",
                              map_location=args.device)
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=iter_use)

    data = json.loads((args.our_scene / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    H = W = 576
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    cams_by_view = {}
    for f in data["frames"]:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        cams_by_view[v] = (R, Tr)
    V = len(cams_by_view)
    print(f"[nobase] views={V}  N_fid={args.n_fid}  z_min={args.z_min}")

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)

    fid_grid = np.linspace(0.0, 1.0, args.n_fid)
    from PIL import Image as PILImage
    for v in sorted(cams_by_view.keys()):
        R, Tr = cams_by_view[v]
        for fi, fid_val in enumerate(fid_grid):
            dummy = torch.zeros(3, H, W, dtype=torch.float32)
            dummy_alpha = torch.zeros(1, H, W, dtype=torch.float32)
            cam = SCGSCamera(colmap_id=fi, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                             image=dummy, gt_alpha_mask=dummy_alpha,
                             image_name=f"v{v}_f{fi}", uid=fi,
                             fid=torch.tensor(float(fid_val)).float())
            time_input = deform.deform.expand_time(cam.fid.to(args.device))
            with torch.no_grad():
                d = deform.step(gaussians.get_xyz.detach(), time_input,
                                feature=gaussians.feature,
                                motion_mask=getattr(gaussians, "motion_mask", None),
                                is_training=False)
                pkg = render(cam, gaussians, pipe, bg,
                             d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                             d_scaling=d["d_scaling"],
                             d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
                             d_rot_as_res=deform.d_rot_as_res)
            img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
            img_u8 = (img * 255).astype(np.uint8)
            PILImage.fromarray(img_u8).save(args.out_dir / "renders" / f"r_v{v}_f{fi:03d}.png")
        print(f"[nobase] view {v} done")

    print(f"[nobase] total {V * args.n_fid} frames -> {args.out_dir / 'renders'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
