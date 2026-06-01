"""Fairness experiment F2: SC-GS DeformModel (16M params, the actual SC-GS
motion mechanism) trained on top of OUR frozen clean canonical.

Apples-to-apples comparison vs Phase 2:
  - Both: frozen 114k canonical Gaussians
  - Both: SV4D supervision + SAM-2 alpha + same losses
  - DIFFERENCE: SC-GS DeformModel (16M deform-MLP) vs our cluster SE(3)+LBS+xyz_res (885k)

If SC-GS DeformModel wins → our cluster SE(3) approach isn't the key
If our method wins → cluster SE(3) + smart photo IS the contribution
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams, OptimizationParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


def silhouette_loss(render_alpha, gt_alpha):
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
    inter = (a * g).sum()
    union = (a + g - a * g).sum()
    return bce + (1 - inter / (union + 1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="scgs_deform_frozen_canon")
    p.add_argument("--canon_ply", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--scene_dir", default=REPO / "data/custom/lego_v2")
    p.add_argument("--v5_render_dir", default=REPO / "outputs/custom/lego_v2_d3dgs_ref/renders")
    p.add_argument("--node_num", type=int, default=512)
    p.add_argument("--hyper_dim", type=int, default=8)
    p.add_argument("--iterations", type=int, default=8000)
    p.add_argument("--lr_deform", type=float, default=5e-4)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--lam_photo_smart", type=float, default=3.0)
    p.add_argument("--photo_smart_alpha", type=float, default=16.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = REPO / "outputs/custom" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load canonical Gaussians (FROZEN) =====
    print(f"[scgs-fc] loading canonical {args.canon_ply}")
    gaussians = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.canon_ply), og_number_points=0)
            gaussians = g; break
        except Exception:
            continue
    # FREEZE all canonical attributes
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)
    N = gaussians.get_xyz.shape[0]
    print(f"[scgs-fc] canonical N={N} (FROZEN)")

    # ===== SC-GS DeformModel (16M params, the actual SC-GS motion module) =====
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=args.hyper_dim, node_num=args.node_num,
                         pred_opacity=False, pred_color=False,
                         use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True,
                         progressive_brand_time=False, with_arap_loss=True,
                         max_d_scale=-1, enable_densify_prune=False, is_scene_static=False)
    # Set up training args for deform
    opt_parser = _A()
    opt = OptimizationParams(opt_parser)
    opt = opt.extract(opt_parser.parse_args([]))
    # Need spatial_lr_scale for deform — set to 1.0 (object-centric)
    deform.spatial_lr_scale = 1.0
    deform.train_setting(opt)
    n_params = sum(p.numel() for p in deform.deform.parameters())
    print(f"[scgs-fc] DeformModel params: {n_params}")

    # ===== Cameras + GT =====
    data = json.loads((Path(args.scene_dir)/"transforms_train.json").read_text())
    test_meta = json.loads((Path(args.scene_dir)/"transforms_test.json").read_text())
    data["frames"] = data["frames"] + test_meta["frames"]
    T_full = max(int(f["frame_idx"]) for f in data["frames"]) + 1
    fov_x = data["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    cams, gt_alphas, gt_rgbs, times = [], [], [], []
    for i, f in enumerate(data["frames"]):
        ti = int(f["frame_idx"])
        png_name = f"{Path(f['file_path']).name}.png"
        png = None
        for split in ("train", "test"):
            cand = Path(args.scene_dir) / split / png_name
            if cand.exists(): png = cand; break
        rgba = np.asarray(iio.imread(png))
        alpha = (rgba[..., 3] > 127).astype(np.float32)
        rgb = (rgba[..., :3].astype(np.float32) / 255.0)
        rgb = rgb * alpha[..., None] + 1.0 * (1 - alpha[..., None])
        gt_alphas.append(torch.from_numpy(alpha).to(args.device))
        gt_rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1).to(args.device))
        times.append(ti)
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        img_t = torch.from_numpy(rgba[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1)
        alpha_t = torch.from_numpy((rgba[..., 3:4] / 255.0).astype(np.float32)).permute(2, 0, 1)
        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=img_t, gt_alpha_mask=alpha_t,
                         image_name=Path(f['file_path']).stem, uid=i,
                         fid=torch.tensor(float(ti) / max(T_full - 1, 1)).float())
        cams.append(cam)

    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)

    # Smart photo weights (using d-3dgs clean ref)
    smart_w = []
    for i, f in enumerate(data["frames"]):
        v = int(f["view_idx"]); ti = int(f["frame_idx"])
        v5p = Path(args.v5_render_dir) / f"{v*T_full + ti:05d}.png"
        v5_rgba = np.asarray(iio.imread(v5p), dtype=np.float32) / 255.0
        v5_a = v5_rgba[..., 3:4] if v5_rgba.shape[-1] == 4 else np.ones_like(v5_rgba[..., :1])
        v5_rgb = v5_rgba[..., :3] * v5_a + 1.0 * (1 - v5_a)
        gt_rgb_np = gt_rgbs[i].permute(1, 2, 0).cpu().numpy()
        residual = np.abs(gt_rgb_np - v5_rgb).mean(axis=-1)
        w = np.exp(-args.photo_smart_alpha * residual)
        smart_w.append(torch.from_numpy(w.astype(np.float32)).to(args.device))
    print(f"[scgs-fc] smart photo pre-built ({len(smart_w)} weight maps)")

    print(f"[scgs-fc] training {args.iterations} iters")
    t0 = time.time()
    for it in range(1, args.iterations + 1):
        idx = np.random.randint(len(cams))
        cam = cams[idx]
        t = int(times[idx])

        # SC-GS forward: deform predicts (dxyz, drot, dscale) per Gaussian
        time_input = deform.deform.expand_time(cam.fid.to(args.device))
        d = deform.step(gaussians.get_xyz.detach(), time_input,
                        feature=gaussians.feature,
                        motion_mask=getattr(gaussians, "motion_mask", None),
                        is_training=True, iteration=it)
        pkg = render(cam, gaussians, pipe, bg,
                     d_xyz=d["d_xyz"], d_rotation=d["d_rotation"], d_scaling=d["d_scaling"],
                     d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
                     d_rot_as_res=deform.d_rot_as_res)
        img = pkg["render"]; alpha = pkg["alpha"]
        gt_alpha = gt_alphas[idx]
        L_silh = silhouette_loss(alpha[0], gt_alpha)

        gt_rgb = gt_rgbs[idx]
        w_pix = smart_w[idx] * gt_alpha
        err = (img - gt_rgb).abs().mean(dim=0)
        L_photo_smart = (err * w_pix).sum() / w_pix.sum().clamp(min=1)

        loss = args.lam_silh * L_silh + args.lam_photo_smart * L_photo_smart

        # SC-GS deform has its own optim
        deform.optimizer.zero_grad()
        loss.backward()
        deform.optimizer.step()
        deform.update_learning_rate(it)

        if it % 500 == 0:
            print(f"[scgs-fc] it {it:5d} loss={loss:.4f} silh={L_silh:.4f} photo={L_photo_smart:.4f} ({time.time()-t0:.0f}s)")

    # Save deform model
    save_iter = args.iterations
    deform.save_weights(str(out_dir), save_iter)
    # Save canonical for eval (just symlink it actually)
    pc_dir = out_dir / "point_cloud" / f"iteration_{save_iter}"
    pc_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.canon_ply, pc_dir / "point_cloud.ply")
    print(f"[scgs-fc] saved {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
