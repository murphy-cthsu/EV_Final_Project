"""Visualize a vanilla SC-GS (node) model trained with NO prior canonical.

Produces three artifacts under runs_aux/vanilla_noprior_viz/<scene>/:
  train_view{V}.gif        3-col [SV4D GT | clean d-3dgs GT | vanilla] over time
  novel_orbit.gif          turntable novel view (vanilla only) + time advancing
  keyframes.png            rows = keyframe t, cols = [SV4D | clean GT | vanilla]

Usage:
  python scripts/viz_vanilla_noprior.py \
      --model outputs/custom/lego_v2_vanilla_noprior_node \
      --scene_dir data/custom/lego_v2 \
      --d3dgs_dir outputs/custom/lego_v2_d3dgs_ref/renders \
      --scene lego_v2 --view 0
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

H = W = 576


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


# --- standard D-NeRF spherical pose (blender convention) ---
def pose_spherical(theta_deg, phi_deg, radius):
    def t_trans(t):
        m = np.eye(4); m[2, 3] = t; return m
    def rot_phi(p):
        c, s = math.cos(p), math.sin(p)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    def rot_theta(th):
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]])
    c2w = t_trans(radius)
    c2w = rot_phi(phi_deg / 180.0 * math.pi) @ c2w
    c2w = rot_theta(theta_deg / 180.0 * math.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def c2w_to_RT(c2w):
    M = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
    T = -M[:3, 3]
    return R, T


def load_model(model_path):
    mp = Path(model_path)
    iters = sorted(int(p.name.split("_")[-1]) for p in (mp / "point_cloud").iterdir()
                   if p.name.startswith("iteration_"))
    it = iters[-1]
    ds = torch.load(mp / "deform" / f"iteration_{it}" / "deform.pth",
                    map_location="cuda", weights_only=False)
    node_num = ds["nodes"].shape[0]; hyper_dim = ds["nodes"].shape[1] - 3
    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(str(mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"),
               og_number_points=0)
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=it)
    print(f"[viz] {mp.name} iter={it} N={g.get_xyz.shape[0]} nodes={node_num}")
    return g, deform, it


def render_at(g, deform, pipe, bg, R, T, fov_x, fov_y, fid_val):
    dummy = torch.zeros(3, H, W, dtype=torch.float32)
    cam = SCGSCamera(colmap_id=0, R=R, T=T, FoVx=fov_x, FoVy=fov_y,
                     image=dummy, gt_alpha_mask=None, image_name="x", uid=0,
                     fid=torch.tensor(fid_val).float())
    time_input = deform.deform.expand_time(cam.fid.to("cuda"))
    with torch.no_grad():
        d = deform.step(g.get_xyz.detach(), time_input, feature=g.feature,
                        motion_mask=getattr(g, "motion_mask", None), is_training=False)
        pkg = render(cam, g, pipe, bg, d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                     d_scaling=d["d_scaling"], d_opacity=d.get("d_opacity"),
                     d_color=d.get("d_color"), d_rot_as_res=deform.d_rot_as_res)
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def load_rgba_white(path):
    a = np.asarray(iio.imread(path), dtype=np.float32) / 255.0
    if a.shape[-1] == 4:
        al = a[..., 3:4]; return a[..., :3] * al + 1.0 * (1 - al)
    return a[..., :3]


def label(img, text):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    top = Image.new("RGB", (pil.width, pil.height + 24), (255, 255, 255))
    top.paste(pil, (0, 24))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    ImageDraw.Draw(top).text((4, 3), text, fill=(0, 0, 0), font=font)
    return np.asarray(top, dtype=np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--d3dgs_dir", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--orbit_elev", type=float, default=-20.0)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    scene_dir = Path(args.scene_dir); d3dgs_dir = Path(args.d3dgs_dir)
    out = REPO / "runs_aux/vanilla_noprior_viz" / args.scene
    out.mkdir(parents=True, exist_ok=True)

    meta_tr = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = meta_tr["frames"] + meta_te["frames"]
    fov_x = meta_tr["camera_angle_x"]
    fov_y = focal2fov(fov2focal(fov_x, W), H)
    T_full = max(int(f["frame_idx"]) for f in frames) + 1
    radius = float(np.linalg.norm(np.array(frames[0]["transform_matrix"])[:3, 3]))

    g, deform, it = load_model(args.model)
    _pp = _A(); pipe = PipelineParams(_pp).extract(_pp.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # index frames for chosen view: t -> (sv4d_path, flat)
    view_frames = {}
    for f in frames:
        if int(f["view_idx"]) != args.view:
            continue
        t = int(f["frame_idx"])
        name = f"{Path(f['file_path']).name}.png"
        sv = None
        for sp in ("train", "test"):
            c = scene_dir / sp / name
            if c.exists():
                sv = c; break
        view_frames[t] = (sv, args.view * T_full + t,
                           np.asarray(f["transform_matrix"], dtype=np.float64))
    ts = sorted(view_frames)

    # (1) train-view 3-col animation -------------------------------------
    gif1 = []
    psnrs = []
    for t in ts:
        sv_path, flat, c2w = view_frames[t]
        R, T = c2w_to_RT(c2w)
        img = render_at(g, deform, pipe, bg, R, T, fov_x, fov_y, t / max(T_full - 1, 1))
        sv = load_rgba_white(sv_path)
        d3 = load_rgba_white(d3dgs_dir / f"{flat:05d}.png")
        psnrs.append(psnr(img, d3))
        sep = np.ones((H + 24, 6, 3))
        row = np.concatenate([
            label(sv, "SV4D supervision"), sep,
            label(d3, "clean d-3dgs GT"), sep,
            label(img, f"vanilla SC-GS  t={t:02d}  {psnrs[-1]:.1f}dB")], axis=1)
        gif1.append((np.clip(row, 0, 1) * 255).astype(np.uint8))
    iio.imwrite(out / f"train_view{args.view}.gif", np.stack(gif1),
                duration=int(1000 / args.fps), loop=0)
    print(f"[viz] train_view{args.view}.gif  meanPSNR(vanilla vs cleanGT)={np.mean(psnrs):.2f}")

    # (2) novel-view turntable (vanilla only) ----------------------------
    N = 48
    azis = np.linspace(0, 360, N, endpoint=False)
    tmap = np.round(np.linspace(0, T_full - 1, N)).astype(int)
    gif2 = []
    for az, t in zip(azis, tmap):
        c2w = pose_spherical(az, args.orbit_elev, radius)
        R, T = c2w_to_RT(c2w)
        img = render_at(g, deform, pipe, bg, R, T, fov_x, fov_y, t / max(T_full - 1, 1))
        gif2.append((np.clip(label(img, f"vanilla SC-GS novel view  az={az:.0f} t={t:02d}"),
                             0, 1) * 255).astype(np.uint8))
    iio.imwrite(out / "novel_orbit.gif", np.stack(gif2),
                duration=int(1000 / args.fps), loop=0)
    print("[viz] novel_orbit.gif")

    # (3) keyframe grid PNG ----------------------------------------------
    key_ts = [ts[int(round(x))] for x in np.linspace(0, len(ts) - 1, 4)]
    rows = []
    for t in key_ts:
        sv_path, flat, c2w = view_frames[t]
        R, T = c2w_to_RT(c2w)
        img = render_at(g, deform, pipe, bg, R, T, fov_x, fov_y, t / max(T_full - 1, 1))
        sv = load_rgba_white(sv_path)
        d3 = load_rgba_white(d3dgs_dir / f"{flat:05d}.png")
        sep = np.ones((H + 24, 6, 3))
        row = np.concatenate([
            label(sv, f"SV4D  t={t:02d}"), sep,
            label(d3, "clean d-3dgs GT"), sep,
            label(img, "vanilla SC-GS")], axis=1)
        rows.append((np.clip(row, 0, 1) * 255).astype(np.uint8))
    hsep = (np.ones((6, rows[0].shape[1], 3)) * 255).astype(np.uint8)
    stacked = rows[0]
    for r in rows[1:]:
        stacked = np.concatenate([stacked, hsep, r], axis=0)
    Image.fromarray(stacked).save(out / "keyframes.png")
    print("[viz] keyframes.png")


if __name__ == "__main__":
    raise SystemExit(main())
