"""Evaluate trained vanilla SC-GS model. Reports train + test PSNR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SCGS = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--source_path", type=Path, required=True)
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    mp = args.model_path
    if not mp.exists() and mp.with_name(mp.name + "_node").exists():
        mp = mp.with_name(mp.name + "_node")
    iters = sorted([int(p.name.split("_")[-1]) for p in (mp / "point_cloud").iterdir()
                    if p.name.startswith("iteration_")])
    iter_use = iters[-1] if args.iteration == -1 else args.iteration
    print(f"[eval] {mp.name} iter={iter_use}")

    # Detect SH degree / hyper_dim from save
    deform_state = torch.load(mp / "deform" / f"iteration_{iter_use}" / "deform.pth",
                              map_location="cuda")
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    print(f"[eval] node_num={node_num} hyper_dim={hyper_dim}")

    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(str(mp / "point_cloud" / f"iteration_{iter_use}" / "point_cloud.ply"),
               og_number_points=0)
    N = g.get_xyz.shape[0]

    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=iter_use)
    print(f"[eval] gaussians N={N}")

    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    for split in (["train", "test"] if args.split == "both" else [args.split]):
        meta_path = args.source_path / f"transforms_{split}.json"
        if not meta_path.exists():
            print(f"[eval] {split} skip (no {meta_path.name})")
            continue
        meta = json.loads(meta_path.read_text())
        fov_x = meta["camera_angle_x"]
        T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
        H = W = 576
        FovY = focal2fov(fov2focal(fov_x, W), H)

        out_dir = REPO / f"runs_aux/scgs_eval/{mp.name}_{split}"
        if args.save_renders:
            out_dir.mkdir(parents=True, exist_ok=True)
        psnrs = []
        for i, f in enumerate(meta["frames"]):
            v = int(f["view_idx"]); t = int(f["frame_idx"])
            c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
            M = np.linalg.inv(c2w)
            R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
            Tr = -M[:3, 3]
            fid_val = float(t) / max(T_full - 1, 1)
            png = args.source_path / split / f"{Path(f['file_path']).name}.png"
            gt_rgba = np.asarray(iio.imread(png), dtype=np.float32) / 255.0
            a = gt_rgba[..., 3:4]
            gt_rgb = gt_rgba[..., :3] * a + 1.0 * (1 - a)

            dummy = torch.zeros(3, H, W, dtype=torch.float32)
            cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                             image=dummy, gt_alpha_mask=None,
                             image_name=f["file_path"], uid=i,
                             fid=torch.tensor(fid_val).float())
            time_input = deform.deform.expand_time(cam.fid.to("cuda"))
            with torch.no_grad():
                d = deform.step(g.get_xyz.detach(), time_input,
                                feature=g.feature,
                                motion_mask=getattr(g, "motion_mask", None),
                                is_training=False)
                pkg = render(cam, g, pipe, bg,
                             d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                             d_scaling=d["d_scaling"],
                             d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
                             d_rot_as_res=deform.d_rot_as_res)
            img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
            ps = psnr(img, gt_rgb)
            psnrs.append(ps)
            if args.save_renders:
                both = np.concatenate([gt_rgb, img], axis=1)
                Image.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).save(
                    out_dir / f"v{v}_t{t:02d}.png")
        arr = np.array(psnrs)
        print(f"[eval] {split:5s} ({len(arr)} frames): "
              f"mean PSNR = {arr.mean():.3f} ± {arr.std():.3f}  median = {np.median(arr):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
