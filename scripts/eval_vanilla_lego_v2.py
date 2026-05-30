"""Eval a vanilla SC-GS model on lego_v2 against BOTH:
  - SV4D supervision (data/custom/lego_v2/{train,test})
  - d-3dgs clean GT (outputs/custom/lego_v2_d3dgs_ref/renders)

Mirror of eval_lego_v2_hier.py but for vanilla SC-GS deform-MLP models.
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
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
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
    p.add_argument("--model_path", required=True)
    p.add_argument("--scene_dir",  default=REPO / "data/custom/lego_v2")
    p.add_argument("--d3dgs_dir",  default=REPO / "outputs/custom/lego_v2_d3dgs_ref/renders")
    p.add_argument("--iteration",  type=int, default=-1)
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    mp = Path(args.model_path)
    if not mp.exists() and mp.with_name(mp.name + "_node").exists():
        mp = mp.with_name(mp.name + "_node")
    iters = sorted([int(p.name.split("_")[-1]) for p in (mp / "point_cloud").iterdir()
                    if p.name.startswith("iteration_")])
    iter_use = iters[-1] if args.iteration == -1 else args.iteration
    print(f"[eval-vanilla-v2] {mp.name} iter={iter_use}")

    deform_state = torch.load(mp / "deform" / f"iteration_{iter_use}" / "deform.pth",
                              map_location="cuda")
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    print(f"[eval-vanilla-v2] nodes={node_num} hyper={hyper_dim}")

    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(str(mp / "point_cloud" / f"iteration_{iter_use}" / "point_cloud.ply"),
               og_number_points=0)
    N = g.get_xyz.shape[0]
    print(f"[eval-vanilla-v2] N={N}")

    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=iter_use)

    scene_dir = Path(args.scene_dir)
    d3dgs_dir = Path(args.d3dgs_dir)
    meta_train = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_test = json.loads((scene_dir / "transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    H = W = 576
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out_dir = REPO / f"runs_aux/vanilla_eval_v2/{mp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_renders:
        (out_dir / "tiles").mkdir(exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    psnr_vs_sv4d = []
    psnr_vs_d3dgs = []
    for i, f in enumerate(all_frames):
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        fid_val = float(t) / max(T_full - 1, 1)

        dummy = torch.zeros(3, H, W, dtype=torch.float32)
        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=dummy, gt_alpha_mask=None,
                         image_name="x", uid=i, fid=torch.tensor(fid_val).float())
        time_input = deform.deform.expand_time(cam.fid.to("cuda"))
        with torch.no_grad():
            d = deform.step(g.get_xyz.detach(), time_input,
                            feature=g.feature, motion_mask=getattr(g, "motion_mask", None),
                            is_training=False)
            pkg = render(cam, g, pipe, bg,
                         d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                         d_scaling=d["d_scaling"],
                         d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
                         d_rot_as_res=deform.d_rot_as_res)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

        # SV4D GT
        png_name = f"{Path(f['file_path']).name}.png"
        sv4d_path = None
        for split in ("train", "test"):
            cand = scene_dir / split / png_name
            if cand.exists():
                sv4d_path = cand; break
        sv4d_rgba = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
        a = sv4d_rgba[..., 3:4]
        sv4d_rgb = sv4d_rgba[..., :3] * a + 1.0 * (1 - a)

        # d-3dgs GT
        flat = v * T_full + t
        d3_path = d3dgs_dir / f"{flat:05d}.png"
        d3_rgba = np.asarray(iio.imread(d3_path), dtype=np.float32) / 255.0
        if d3_rgba.shape[-1] == 4:
            ad3 = d3_rgba[..., 3:4]; d3_rgb = d3_rgba[..., :3] * ad3 + 1 * (1 - ad3)
        else:
            d3_rgb = d3_rgba[..., :3]

        psnr_vs_sv4d.append(psnr(img, sv4d_rgb))
        psnr_vs_d3dgs.append(psnr(img, d3_rgb))

        if args.save_renders and v == 0:
            sep = np.ones((H, 4, 3))
            row = np.concatenate([sv4d_rgb, sep, d3_rgb, sep, img], axis=1)
            pil = Image.fromarray((row * 255).astype(np.uint8))
            top = Image.new("RGB", (pil.width, pil.height + 30), (255, 255, 255))
            d = ImageDraw.Draw(top)
            col_w = W + 4
            d.text((W // 2 - 50, 6), "SV4D GT", fill="black", font=font)
            d.text((col_w + W // 2 - 80, 6),
                   f"d-3dgs clean ({psnr_vs_d3dgs[-1]:.2f} dB)", fill="black", font=font)
            d.text((2 * col_w + W // 2 - 110, 6),
                   f"Vanilla SC-GS ({psnr_vs_sv4d[-1]:.2f} vs SV4D)",
                   fill=(180, 0, 0), font=font)
            d.text((top.width - 100, 6), f"v={v} t={t:02d}", fill="black", font=font)
            top.paste(pil, (0, 30))
            top.save(out_dir / "tiles" / f"v0_t{t:02d}.png")

    arr_sv = np.array(psnr_vs_sv4d)
    arr_d3 = np.array(psnr_vs_d3dgs)
    print()
    print(f"[eval-vanilla-v2] vs SV4D supervision : mean={arr_sv.mean():.3f} median={np.median(arr_sv):.3f}")
    print(f"[eval-vanilla-v2] vs d-3dgs CLEAN GT  : mean={arr_d3.mean():.3f} median={np.median(arr_d3):.3f}")
    print(f"[eval-vanilla-v2] gap (d3dgs - sv4d) : {arr_d3.mean() - arr_sv.mean():+.3f}")

    # Save numeric results json for the report builder
    (out_dir / "psnr_summary.json").write_text(json.dumps({
        "model": mp.name,
        "iteration": iter_use,
        "n_gaussians": int(N),
        "n_frames": int(len(all_frames)),
        "vs_sv4d_mean": float(arr_sv.mean()),
        "vs_sv4d_median": float(np.median(arr_sv)),
        "vs_sv4d_std": float(arr_sv.std()),
        "vs_d3dgs_mean": float(arr_d3.mean()),
        "vs_d3dgs_median": float(np.median(arr_d3)),
        "vs_d3dgs_std": float(arr_d3.std()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
