"""Visualize how mask supervision guides the Gaussians.

Mask provenance per scene:
  - lego_v2 (5-view):       dataset alpha = SAM-2 video-predictor mask
                            (scripts/sam2_mask_lego_v2.py)
  - lego_v3 / hellwarrior:  dataset alpha = non-white threshold
                            (scripts/build_scene_dataset.py — NO SAM)
Either way the alpha is what L_silh trains against, so the same panels answer
"are the Gaussians following the mask guidance?".

Outputs to runs_aux/mask_guidance_<scene>_<label>/:
  panel_v{v}_t{t}.png : [SV4D frame | gt alpha | rendered alpha | disagreement
                         | motion mask overlay | projected Gaussian parts]
  iou_vs_azimuth.png  : silhouette IoU per view (elev=0 ring highlighted)
  iou_table.txt       : per-view IoU numbers

Run (scgs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/viz_mask_guidance.py \
      --label hellwarrior_cleancanon_ctrl_rotfix --scene hellwarrior \
      --part_dir runs_aux/part_assignment_hellwarrior_cleancanon_p4 \
      --motion_dir runs_aux/parts_motion_hellwarrior_cleancanon_p4 \
      --views 0 7 14 28 42 --times 0 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

PART_COLORS = np.array([(255, 60, 60), (60, 200, 60), (80, 120, 255),
                        (255, 200, 0), (200, 60, 255)], np.uint8)


def aa2mat_np(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def project_np(xyz, c2w, fov_x, H, W):
    w2c = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(c2w)
    cam = (w2c @ np.concatenate([xyz, np.ones((len(xyz), 1))], -1).T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return np.stack([u, v], -1), z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--scene", default="hellwarrior")
    p.add_argument("--canon_ply", default=None)
    p.add_argument("--part_dir", default=None,
                   help="for gaussian part colors (gaussian_arm_weights / gaussian_motion_part)")
    p.add_argument("--motion_dir", default=None,
                   help="parts_motion dir with view{v}_part_masks.npy (Stage B)")
    p.add_argument("--views", type=int, nargs="+", default=[0, 7, 14, 28, 42])
    p.add_argument("--times", type=int, nargs="+", default=[0, 10])
    p.add_argument("--iou_t", type=int, default=10, help="frame for the IoU-vs-view sweep")
    args = p.parse_args()

    s = np.load(REPO / f"outputs/custom/partrigid_{args.label}/partrigid_state.npz", allow_pickle=True)
    trans = s["trans"]; aa = s["aa"]; centers = s["arm_centers"]; lbs = s["lbs_weights"]
    T_train = trans.shape[1]
    scale_per_kt = s["scale"] if "scale" in s.files else None
    xyz_res_kt = s["xyz_residual"] if "xyz_residual" in s.files else None
    arm_idx_res = s["arm_idx_for_residual"] if "arm_idx_for_residual" in s.files else None
    cfg = s["config"].item() if "config" in s.files else {}
    canon_path = Path(args.canon_ply or cfg["canon_ply"])

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(canon_path), og_number_points=0); break
        except (IndexError, RuntimeError, ValueError):
            g = None
    xyz_canon = g.get_xyz.detach().cpu().numpy()
    N = len(xyz_canon)

    # Gaussian part colors: prefer 4-part motion labels, fall back to arm/body
    part_dir = Path(args.part_dir or cfg["part_dir"])
    gp_file = part_dir / "gaussian_motion_part.npy"
    if gp_file.exists():
        gauss_part = np.load(gp_file)          # {-1, 0..P-1}; -1 = body/static
        part_legend = "colors = motion parts (grey = static body)"
    else:
        arm_w = np.load(part_dir / "gaussian_arm_weights.npy")
        gauss_part = np.where(arm_w > 0.5, 0, -1)
        part_legend = "red = moving (arm), grey = static body"

    scene_dir = REPO / "data/custom" / args.scene
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = {}
    view_az = {}
    view_el = {}
    for f in meta["frames"] + meta_te["frames"]:
        v, t = int(f["view_idx"]), int(f["frame_idx"])
        frames[(v, t)] = f
        view_az[v] = float(f.get("azimuth_deg", v))
        view_el[v] = float(f.get("elevation_deg", 0))
    fov_x = meta["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    n_views = max(v for v, _ in frames) + 1

    motion_dir = Path(args.motion_dir) if args.motion_dir else None

    out = REPO / f"runs_aux/mask_guidance_{args.scene}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)

    def render_alpha_rgb(v, t):
        f = frames[(v, t)]
        c2w = np.asarray(f["transform_matrix"], np.float64)
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]
        tl = min(t, T_train - 1)
        R_all = aa2mat_np(aa[:, tl, :]); T_all = trans[:, tl, :]
        rel = xyz_canon[:, None, :] - centers[None, :, :]
        new_per = np.einsum("kij,nkj->nki", R_all, rel) + centers[None] + T_all[None]
        w_total = lbs.sum(1, keepdims=True).clip(0, 1)
        new_xyz = (lbs[..., None] * new_per).sum(1) + (1 - w_total) * xyz_canon
        if xyz_res_kt is not None and arm_idx_res is not None and len(arm_idx_res):
            new_xyz[arm_idx_res] += xyz_res_kt[:, tl, :]
        d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
        d_rot = torch.zeros(N, 4, device="cuda")   # rotfix (d_rot_zero) convention
        d_sc = torch.zeros(N, 3, device="cuda")
        if scale_per_kt is not None:
            d_sc = torch.from_numpy((lbs @ scale_per_kt[:, tl, :]).astype(np.float32)).cuda()
        cam = SCGSCamera(colmap_id=0, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(0.0).float())
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz_t, d_rotation=d_rot,
                         d_scaling=d_sc, d_rot_as_res=True)
        rgb = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        alpha = torch.clamp(pkg["alpha"], 0, 1).cpu().numpy()[0]
        return rgb, alpha, new_xyz, c2w

    def load_gt(v, t):
        f = frames[(v, t)]
        png = scene_dir / (f["file_path"].lstrip("./") + ".png")
        rgba = np.asarray(Image.open(png), np.float32) / 255.0
        a = rgba[..., 3]
        rgb = rgba[..., :3] * a[..., None] + (1 - a[..., None])
        return rgb, a > 0.5

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    def labeled(img01, text):
        pil = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        d.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
        d.text((6, 4), text, fill=(255, 255, 255), font=font)
        return np.asarray(pil, np.float32) / 255.0

    # ---------- panels ----------
    for v in args.views:
        for t in args.times:
            if (v, t) not in frames:
                continue
            rgb_r, alpha_r, new_xyz, c2w = render_alpha_rgb(v, t)
            rgb_gt, m_gt = load_gt(v, t)
            m_r = alpha_r > 0.5
            inter = (m_r & m_gt).sum(); union = (m_r | m_gt).sum()
            iou = inter / max(union, 1)

            # disagreement: red = we render outside gt (leak), blue = gt not covered (miss)
            dis = np.ones((H, W, 3), np.float32)
            dis[m_r & ~m_gt] = [1, 0.2, 0.2]
            dis[m_gt & ~m_r] = [0.2, 0.4, 1]
            dis[m_r & m_gt] = [0.85, 0.85, 0.85]

            # motion-mask overlay (Stage B guidance for smart-photo gating / voting)
            mm_img = rgb_gt.copy()
            if motion_dir is not None and (motion_dir / f"view{v}_part_masks.npy").exists():
                mm = np.load(motion_dir / f"view{v}_part_masks.npy")
                mov, stat = mm[0] > 0, mm[1] > 0
                mm_img[mov] = mm_img[mov] * 0.4 + np.array([1, 0, 0]) * 0.6
                mm_img[stat] = mm_img[stat] * 0.6 + np.array([0, 1, 1]) * 0.4

            # projected DEFORMED Gaussians colored by part
            pg = rgb_gt.copy() * 0.55 + 0.45
            uv, z = project_np(new_xyz, c2w, fov_x, H, W)
            order = np.argsort(-z)  # far first so near points overwrite
            ui = np.clip(uv[order, 0].astype(int), 0, W - 1)
            vi = np.clip(uv[order, 1].astype(int), 0, H - 1)
            lab = gauss_part[order]
            col = np.where(lab[:, None] >= 0,
                           PART_COLORS[np.clip(lab, 0, len(PART_COLORS) - 1)],
                           np.array([130, 130, 130], np.uint8)).astype(np.float32) / 255.0
            pg[vi, ui] = col

            row = np.concatenate([
                labeled(rgb_gt, f"SV4D supervision  v{v} (az {view_az[v]:.0f}) t{t}"),
                labeled(np.repeat(m_gt[..., None], 3, -1).astype(np.float32),
                        "gt alpha (silhouette guidance)"),
                labeled(np.repeat(alpha_r[..., None], 3, -1), "our rendered alpha"),
                labeled(dis, f"disagree  IoU={iou:.3f}  red=leak blue=miss"),
                labeled(mm_img, "Stage B motion mask (red=moving cyan=static)"),
                labeled(pg, f"deformed Gaussians ({part_legend})"),
            ], axis=1)
            Image.fromarray((row * 255).astype(np.uint8)).save(out / f"panel_v{v:02d}_t{t:02d}.png")
            print(f"  panel v{v} t{t}: silhouette IoU = {iou:.3f}")

    # ---------- IoU vs view sweep ----------
    t = args.iou_t
    ious, azs, els = [], [], []
    for v in range(n_views):
        if (v, t) not in frames:
            continue
        _, alpha_r, _, _ = render_alpha_rgb(v, t)
        _, m_gt = load_gt(v, t)
        m_r = alpha_r > 0.5
        iou = (m_r & m_gt).sum() / max((m_r | m_gt).sum(), 1)
        ious.append(iou); azs.append(view_az[v]); els.append(view_el[v])
    ious = np.array(ious); azs = np.array(azs); els = np.array(els)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    e0 = els == 0
    ax.scatter(azs[~e0], ious[~e0], s=18, c="silver", label="other elevations")
    if e0.any():
        o = np.argsort(azs[e0])
        ax.plot(azs[e0][o], ious[e0][o], "o-", color="crimson", label="elev = 0 ring")
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel(f"silhouette IoU @ t={t}")
    ax.set_title(f"{args.scene} / {args.label} — mask-guidance fidelity vs view")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "iou_vs_azimuth.png", dpi=130)

    with open(out / "iou_table.txt", "w") as fh:
        for v, (a_, e_, i_) in enumerate(zip(azs, els, ious)):
            fh.write(f"view {v:2d}  az {a_:6.1f}  elev {e_:5.1f}  IoU {i_:.3f}\n")
        fh.write(f"\nmean IoU = {ious.mean():.3f}  min = {ious.min():.3f} "
                 f"(view {int(ious.argmin())})  max = {ious.max():.3f}\n")
    print(f"[viz] mean silhouette IoU = {ious.mean():.3f} "
          f"(min {ious.min():.3f} @ view {int(ious.argmin())})")
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
