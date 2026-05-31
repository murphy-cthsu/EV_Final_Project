"""Eval a hier-trained model on lego_v2 against TWO GTs:
  1. SV4D supervision data (data/custom/lego_v2/train+test)
  2. d-3dgs clean reference (outputs/custom/lego_v2_d3dgs_ref/renders)
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
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

CANON_DEFAULT = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"


def aa2mat_np(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--canon_ply", default=None,
                   help="canonical ply path (auto-detect from state config if omitted)")
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    s = np.load(REPO / f"outputs/custom/partrigid_{args.label}/partrigid_state.npz", allow_pickle=True)
    trans = s["trans"]; aa = s["aa"]; centers = s["arm_centers"]; lbs = s["lbs_weights"]
    K, T_train, _ = trans.shape
    have_scale = "scale" in s.files
    scale_per_kt = s["scale"] if have_scale else None
    have_xyz_res = "xyz_residual" in s.files
    xyz_res_kt = s["xyz_residual"] if have_xyz_res else None
    have_rot_res = "rot_residual" in s.files
    rot_res_kt = s["rot_residual"] if have_rot_res else None
    arm_idx_res = s["arm_idx_for_residual"] if "arm_idx_for_residual" in s.files else None
    print(f"[eval-v2] label={args.label}  K={K}  T_train={T_train}  scale={have_scale}  xyz_res={have_xyz_res}  rot_res={have_rot_res}")

    # Resolve canon: CLI > state config > default
    if args.canon_ply:
        canon_path = Path(args.canon_ply)
    elif "config" in s.files:
        cfg = s["config"].item() if hasattr(s["config"], "item") else s["config"]
        canon_path = Path(cfg["canon_ply"]) if isinstance(cfg, dict) and cfg.get("canon_ply") else CANON_DEFAULT
    else:
        canon_path = CANON_DEFAULT
    print(f"[eval-v2] canon={canon_path}")

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(canon_path), og_number_points=0)
            print(f"[eval-v2] loaded with fea_dim={fdim}")
            break
        except (IndexError, RuntimeError, ValueError):
            g = None
    if g is None:
        raise RuntimeError(f"can't load canonical {canon_path}")
    xyz_canon = g.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    print(f"[eval-v2] canonical N={N}")

    scene_dir = REPO / "data/custom/lego_v2"
    d3dgs_dir = REPO / "outputs/custom/lego_v2_d3dgs_ref/renders"
    meta_train = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_test = json.loads((scene_dir / "transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out_dir = REPO / f"runs_aux/lego_v2_eval/{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_renders:
        (out_dir / "tiles").mkdir(exist_ok=True)

    psnr_vs_sv4d = []
    psnr_vs_d3dgs = []
    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]
        tl = min(t, T_train - 1)

        # Deformation
        R_all = aa2mat_np(aa[:, tl, :])
        T_all = trans[:, tl, :]
        rel = xyz_canon[:, None, :] - centers[None, :, :]
        rotated = np.einsum("kij,nkj->nki", R_all, rel)
        new_per = rotated + centers[None, :, :] + T_all[None, :, :]
        weighted = (lbs[..., None] * new_per).sum(axis=1)
        w_total = lbs.sum(axis=1, keepdims=True).clip(min=0, max=1)
        new_xyz = weighted + (1 - w_total) * xyz_canon
        # Apply per-Gaussian per-time XYZ residual if available
        if have_xyz_res and arm_idx_res is not None and len(arm_idx_res) > 0:
            new_xyz[arm_idx_res] = new_xyz[arm_idx_res] + xyz_res_kt[:, tl, :]
        d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
        d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_sc = torch.zeros(N, 3, device="cuda")
        if have_scale:
            scale_blend = lbs @ scale_per_kt[:, tl, :]
            d_sc = torch.from_numpy(scale_blend.astype(np.float32)).cuda()
        # Per-Gaussian rotation residual → d_rotation_bias
        d_rot_bias = None
        if have_rot_res and arm_idx_res is not None and len(arm_idx_res) > 0:
            aa_res = rot_res_kt[:, tl, :]  # (N_arm, 3) axis-angle
            theta = np.linalg.norm(aa_res, axis=-1, keepdims=True).clip(min=1e-8)
            axis = aa_res / theta
            half = theta * 0.5
            q_res = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)  # (N_arm, 4)
            q_full = np.zeros((N, 4), dtype=np.float32)
            q_full[:, 0] = 1.0  # identity
            q_full[arm_idx_res] = q_res.astype(np.float32)
            d_rot_bias = torch.from_numpy(q_full).cuda()

        cam = SCGSCamera(colmap_id=0, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(0.0).float())
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz_t, d_rotation=d_rot, d_scaling=d_sc,
                         d_rot_as_res=True, d_rotation_bias=d_rot_bias)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

        # SV4D GT
        png_name = f"{Path(f['file_path']).name}.png"
        sv4d_path = None
        for split in ("train", "test"):
            cand = scene_dir / split / png_name
            if cand.exists(): sv4d_path = cand; break
        sv4d_rgba = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
        a = sv4d_rgba[..., 3:4]
        sv4d_rgb = sv4d_rgba[..., :3] * a + 1.0 * (1 - a)

        # d-3dgs GT
        flat = v * 21 + t
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
            from PIL import ImageDraw, ImageFont
            pil = Image.fromarray((row * 255).astype(np.uint8))
            top = Image.new("RGB", (pil.width, pil.height + 30), (255, 255, 255))
            d = ImageDraw.Draw(top)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            col_w = W + 4
            d.text((W // 2 - 50, 6), "SV4D GT", fill="black", font=font)
            d.text((col_w + W // 2 - 70, 6),
                   f"d-3dgs clean ({psnr_vs_d3dgs[-1]:.2f} dB)", fill="black", font=font)
            d.text((2 * col_w + W // 2 - 100, 6),
                   f"Ours ({psnr_vs_sv4d[-1]:.2f} dB vs SV4D)", fill=(180, 0, 0), font=font)
            d.text((top.width - 100, 6), f"v={v} t={t:02d}", fill="black", font=font)
            top.paste(pil, (0, 30))
            top.save(out_dir / "tiles" / f"v0_t{t:02d}.png")

    arr_sv = np.array(psnr_vs_sv4d)
    arr_d3 = np.array(psnr_vs_d3dgs)
    print()
    print(f"[eval-v2] vs SV4D (supervision data) : mean={arr_sv.mean():.3f}  median={np.median(arr_sv):.3f}  std={arr_sv.std():.3f}")
    print(f"[eval-v2] vs d-3dgs (CLEAN GT)        : mean={arr_d3.mean():.3f}  median={np.median(arr_d3):.3f}  std={arr_d3.std():.3f}")
    print(f"[eval-v2] gap (d3dgs - sv4d): {arr_d3.mean() - arr_sv.mean():+.3f} dB")
    print(f"  → d-3dgs higher means we're predicting closer to clean GT than to noisy training data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
