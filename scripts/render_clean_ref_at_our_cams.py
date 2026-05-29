"""Render a clean-reference 4D-GS (trained on D-NeRF clean data) at our 5 SV4D
camera viewpoints. Produces "clean GT at our viewpoints" for honest evaluation.

Outputs to runs_aux/clean_gt_at_sv4d_cams/ — 105 PNG renders (5 views × 21
times) named r_NNNNN.png to match our convention.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/render_clean_ref_at_our_cams.py \\
        --model_path outputs/custom/lego_clean_ref \\
        --our_scene  data/custom/scene00_masked \\
        --out_dir    runs_aux/clean_gt_at_sv4d_cams
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True,
                   help="path to clean-reference model (will auto-append _node)")
    p.add_argument("--our_scene", type=Path, required=True,
                   help="scene00_masked dir with transforms_train.json (our 5 cameras x 21 times)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--iteration", type=int, default=-1,
                   help="checkpoint iter (-1 = latest)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    mp = args.model_path
    if not mp.exists() and mp.with_name(mp.name + "_node").exists():
        mp = mp.with_name(mp.name + "_node")
    if not mp.exists():
        raise FileNotFoundError(args.model_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "renders").mkdir(exist_ok=True)

    # Find latest iteration
    pc_iters = sorted([int(p.name.split("_")[-1])
                       for p in (mp / "point_cloud").iterdir()
                       if p.name.startswith("iteration_")])
    iter_use = pc_iters[-1] if args.iteration == -1 else args.iteration
    print(f"[clean_ref] loading {mp.name}  iter={iter_use}")

    # Load Gaussians + deform
    gaussians = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    ply_path = mp / "point_cloud" / f"iteration_{iter_use}" / "point_cloud.ply"
    gaussians.load_ply(str(ply_path), og_number_points=0)
    N = gaussians.get_xyz.shape[0]
    print(f"[clean_ref] gaussians: {N}")

    # The deform model: load whatever shape was saved
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
    print(f"[clean_ref] deform loaded: node_num={node_num}  hyper_dim={hyper_dim}")

    # Load our scene's cameras
    data = json.loads((args.our_scene / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    T = max(int(f["frame_idx"]) for f in data["frames"]) + 1
    H = W = 576
    print(f"[clean_ref] our scene: fov_x={fov_x:.4f}  T={T}  W=H={W}")

    # Pipeline setup
    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    # Render at each (cam, time) of OUR scene00 set
    summary = []
    for f in data["frames"]:
        v_idx = int(f["view_idx"])
        t_idx = int(f["frame_idx"])
        flat_idx = v_idx * T + t_idx

        # Build SC-GS Camera at our view position
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        matrix = np.linalg.inv(c2w)
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -matrix[:3, 3]
        fid_val = float(t_idx) / max(T - 1, 1)
        # Use a dummy black image (we don't compare here, just need the cam)
        dummy = torch.zeros(3, H, W, dtype=torch.float32)
        dummy_alpha = torch.zeros(1, H, W, dtype=torch.float32)
        cam = SCGSCamera(colmap_id=flat_idx, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=dummy, gt_alpha_mask=dummy_alpha,
                         image_name=f["file_path"], uid=flat_idx,
                         fid=torch.tensor(fid_val).float())

        # Run deform at this time
        xyz = gaussians.get_xyz
        time_input = deform.deform.expand_time(cam.fid.to(args.device))
        with torch.no_grad():
            d = deform.step(xyz.detach(), time_input, feature=gaussians.feature,
                            motion_mask=getattr(gaussians, "motion_mask", None),
                            is_training=False)
        d_xyz = d["d_xyz"]; d_rotation = d["d_rotation"]; d_scaling = d["d_scaling"]
        d_opacity = d.get("d_opacity"); d_color = d.get("d_color")
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background,
                         d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                         d_opacity=d_opacity, d_color=d_color,
                         d_rot_as_res=deform.d_rot_as_res)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        img_u8 = (img * 255).astype(np.uint8)

        from PIL import Image as PILImage
        PILImage.fromarray(img_u8).save(args.out_dir / "renders" / f"r_{flat_idx:05d}.png")
        summary.append({"view": v_idx, "time": t_idx, "fid": fid_val, "flat": flat_idx})

    (args.out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[clean_ref] rendered {len(summary)} frames to {args.out_dir / 'renders'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
