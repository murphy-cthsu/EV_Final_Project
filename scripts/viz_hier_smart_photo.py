"""Build visualization for hier_K3_smart (smart photometric filter result).

Three panels per (v, t):
  - SV4D GT
  - hier_K3_smart render (the new model with v5-filtered L1)
  - hier_K3_fixed render (baseline without smart photo, for comparison)

Also: show what the v5 filter weight looks like (sanity-check the filter).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"


def aa2mat_np(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def render_hier(label, gaussians, xyz_canon, R_cam, Tcam, fov_x, FovY, H, W, t, pipe, bg, N):
    state = np.load(REPO / f"outputs/custom/partrigid_{label}/partrigid_state.npz", allow_pickle=True)
    trans = state["trans"]; aa = state["aa"]; centers = state["arm_centers"]
    lbs = state["lbs_weights"]
    T_train = trans.shape[1]
    tl = min(t, T_train - 1)
    R_all = aa2mat_np(aa[:, tl, :])
    T_all = trans[:, tl, :]
    rel = xyz_canon[:, None, :] - centers[None, :, :]
    rotated = np.einsum("kij,nkj->nki", R_all, rel)
    new_per = rotated + centers[None, :, :] + T_all[None, :, :]
    weighted = (lbs[..., None] * new_per).sum(axis=1)
    w_total = lbs.sum(axis=1, keepdims=True).clip(min=0, max=1)
    new_xyz = weighted + (1 - w_total) * xyz_canon
    d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
    d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
    d_sc = torch.zeros(N, 3, device="cuda")
    cam = SCGSCamera(colmap_id=0, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                     image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                     image_name="x", uid=0, fid=torch.tensor(0.0).float())
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg,
                     d_xyz=d_xyz_t, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    out_dir = REPO / "runs_aux/hier_smart_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiles").mkdir(exist_ok=True)
    (out_dir / "filter_viz").mkdir(exist_ok=True)

    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]

    scene_dir = REPO / "data/custom/scene00_masked"
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    fov_x = meta["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    v5_dir = REPO / "outputs/custom/scene00_v5_node/train/ours_30000/renders"

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    tiles_v0 = []
    psnrs_smart, psnrs_fix = [], []
    for f in meta["frames"]:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"])
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]

        # GT
        gt_path = scene_dir / "train" / f"{Path(f['file_path']).name}.png"
        gt_rgba = np.asarray(iio.imread(gt_path), dtype=np.float32) / 255.0
        gt_alpha = gt_rgba[..., 3:4]
        gt_rgb = gt_rgba[..., :3] * gt_alpha + 1.0 * (1 - gt_alpha)

        # v5 ref
        v5_path = v5_dir / f"{v*T_full + t:05d}.png"
        v5_rgba = np.asarray(iio.imread(v5_path), dtype=np.float32) / 255.0
        v5_a = v5_rgba[..., 3:4] if v5_rgba.shape[-1] == 4 else np.ones_like(v5_rgba[..., :1])
        v5_rgb = v5_rgba[..., :3] * v5_a + 1.0 * (1 - v5_a)
        residual = np.abs(gt_rgb - v5_rgb).mean(axis=-1)
        weight = np.exp(-8.0 * residual)

        # Renders
        pred_smart = render_hier("hier_K3_smart", gaussians, xyz_canon, R_cam, Tcam,
                                  fov_x, FovY, H, W, t, pipe, bg, N)
        pred_fix = render_hier("hier_K3_fixed", gaussians, xyz_canon, R_cam, Tcam,
                                fov_x, FovY, H, W, t, pipe, bg, N)
        psnrs_smart.append(psnr(pred_smart, gt_rgb))
        psnrs_fix.append(psnr(pred_fix, gt_rgb))

        if v == 0:
            # 4-col tile: GT | hier_K3_fixed (no smart) | hier_K3_smart | filter weight
            sep = np.ones((H, 4, 3))
            wt_vis = (weight[..., None].repeat(3, axis=-1))  # grayscale
            row = np.concatenate([gt_rgb, sep, pred_fix, sep, pred_smart, sep, wt_vis], axis=1)
            pil = Image.fromarray((row * 255).astype(np.uint8))
            top = Image.new("RGB", (pil.width, pil.height + 30), (255, 255, 255))
            d = ImageDraw.Draw(top)
            col_w = W + 4
            d.text((W // 2 - 40, 6), "SV4D GT", fill="black", font=font)
            d.text((col_w + W // 2 - 100, 6),
                   f"K=3 baseline ({psnrs_fix[-1]:.2f} dB)", fill="black", font=font)
            d.text((2 * col_w + W // 2 - 100, 6),
                   f"K=3 + smart photo ({psnrs_smart[-1]:.2f} dB)",
                   fill=(180, 0, 0), font=font)
            d.text((3 * col_w + W // 2 - 70, 6),
                   f"v5 filter weight", fill="black", font=font)
            d.text((top.width - 100, 6), f"v={v} t={t:02d}", fill="black", font=font)
            top.paste(pil, (0, 30))
            top.save(out_dir / "tiles" / f"v0_t{t:02d}.png")
            tiles_v0.append(top)

            # Also save the filter weight standalone for §5.6 inspection
            Image.fromarray((wt_vis * 255).astype(np.uint8)).save(
                out_dir / "filter_viz" / f"v0_t{t:02d}.png")

    # Per-view GIF for v=0
    if tiles_v0:
        tiles_v0[0].save(out_dir / "gallery_v0.gif", save_all=True,
                          append_images=tiles_v0[1:], duration=400, loop=0)

    print(f"hier_K3_fixed (baseline)   : mean PSNR = {np.mean(psnrs_fix):.3f}")
    print(f"hier_K3_smart (v5 filter)  : mean PSNR = {np.mean(psnrs_smart):.3f}")
    print(f"  uplift                  : {np.mean(psnrs_smart) - np.mean(psnrs_fix):+.3f} dB")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
