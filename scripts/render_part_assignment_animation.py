"""Render Part-Rigid model with Gaussians colored by their part assignment.

Color scheme:
  - arm_weights == 1.0  → red    (pure arm)
  - arm_weights == 0.0  → blue   (pure body / static)
  - boundary (0.1<w<0.9) → purple (LBS soft region)
  - unassigned          → gray   (if available)

Saves:
  - One GIF per view (5 GIFs) with the animated colored render
  - One PNG showing canonical viz with color overlay (no motion)
  - One contact sheet across views at t=0

Run AFTER the part-rigid training has produced partrigid_state.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


SH_C0 = 0.28209479177387814  # SH degree 0 normalization constant


def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / theta
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    th = theta[..., None]
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def color_from_weights(arm_weights: np.ndarray, part_id: np.ndarray | None = None) -> np.ndarray:
    """Convert per-Gaussian arm_weight to RGB.
    arm_w=1 → red, arm_w=0 → blue, smooth interp purple in between.
    If part_id provided, mark unassigned (part_id==2) as gray.
    """
    N = arm_weights.shape[0]
    # lerp blue -> red via purple
    arm_w = np.clip(arm_weights, 0, 1)[:, None]
    blue = np.array([0.15, 0.30, 0.95])   # cool blue
    red  = np.array([0.95, 0.15, 0.15])   # warm red
    rgb = (1 - arm_w) * blue + arm_w * red
    if part_id is not None:
        unassigned = (part_id == 2)
        rgb[unassigned] = np.array([0.55, 0.55, 0.55])  # gray
    return rgb.astype(np.float32)


def override_color(g: GaussianModel, rgb: np.ndarray):
    """Replace Gaussians' DC SH features so rendered color == `rgb` (with SH order 0)."""
    N = rgb.shape[0]
    # rendered color (no SH dir) ≈ 0.5 + SH_C0 * f_dc
    f_dc = (rgb - 0.5) / SH_C0
    f_dc_t = torch.from_numpy(f_dc).float().cuda().reshape(N, 1, 3)
    g._features_dc = nn.Parameter(f_dc_t)
    # zero the SH rest so direction doesn't change color
    g._features_rest = nn.Parameter(torch.zeros_like(g._features_rest))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--canon_ply", default=REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply")
    p.add_argument("--partrigid_state", default=REPO_ROOT / "outputs/custom/partrigid_lbs_photo1/partrigid_state.npz")
    p.add_argument("--sv4d_meta", default=REPO_ROOT / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--out_dir", default=REPO_ROOT / "runs_aux/part_assignment_anim")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    # Load part-rigid state
    state = np.load(args.partrigid_state, allow_pickle=True)
    arm_trans = state["arm_trans"]; arm_aa = state["arm_aa"]
    arm_pivot = state["arm_pivot"]
    part_id = state["part_id"]
    if "arm_weights" in state.files:
        arm_weights = state["arm_weights"]
        print(f"[anim] LBS weights mean={arm_weights.mean():.3f} (range [{arm_weights.min():.2f}, {arm_weights.max():.2f}])")
    else:
        arm_weights = (part_id == 0).astype(np.float32)
        print(f"[anim] no LBS — using hard part ID")
    T_train = arm_trans.shape[0]
    print(f"[anim] T={T_train}  N_arm_hard={int((part_id==0).sum())}  N_body_hard={int((part_id==1).sum())}  N_unassigned={int((part_id==2).sum())}")

    # Categorize for stats
    n_pure_arm = int((arm_weights > 0.9).sum())
    n_pure_body = int((arm_weights < 0.1).sum())
    n_boundary = int(((arm_weights >= 0.1) & (arm_weights <= 0.9)).sum())
    print(f"[anim] arm_w>0.9 (red): {n_pure_arm}  |  arm_w<0.1 (blue): {n_pure_body}  |  boundary (purple): {n_boundary}")

    # Load canonical Gaussians, override color
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(args.canon_ply), og_number_points=0)
    N = gaussians.get_xyz.shape[0]
    print(f"[anim] canonical N={N}")

    rgb_colors = color_from_weights(arm_weights, part_id)
    override_color(gaussians, rgb_colors)

    # Load cameras
    meta = json.loads(Path(args.sv4d_meta).read_text())
    fov_x = meta["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)

    cams_by_view = {}
    for f in meta["frames"]:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        cams_by_view[v] = (R, Tr)
    V = len(cams_by_view)

    # Render pipeline
    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()

    from PIL import Image as PILImage, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Render canonical (no motion) for each view, contact sheet
    canon_renders = []
    for v in sorted(cams_by_view.keys()):
        Rv, Tv = cams_by_view[v]
        dummy = torch.zeros(3, H, W, dtype=torch.float32)
        cam = SCGSCamera(colmap_id=v, R=Rv, T=Tv, FoVx=fov_x, FoVy=FovY,
                         image=dummy, gt_alpha_mask=None,
                         image_name=f"v{v}", uid=v, fid=torch.tensor(0.0).float())
        d_xyz = torch.zeros(N, 3, device="cuda")
        d_rotation = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_scaling = torch.zeros(N, 3, device="cuda")
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg,
                         d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                         d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        canon_renders.append(img)
    # Contact sheet
    pad = 8
    cs = np.ones((H + 30, V * W + (V - 1) * pad, 3), dtype=np.float32)
    for v, img in enumerate(canon_renders):
        cs[30:, v * (W + pad):v * (W + pad) + W] = img
    cs_pil = PILImage.fromarray((cs * 255).astype(np.uint8))
    d = ImageDraw.Draw(cs_pil)
    for v in range(V):
        d.text((v * (W + pad) + 10, 6), f"view {v}", fill="black", font=font)
    d.text((cs_pil.width - 350, 6),
           f"red=arm({n_pure_arm})  blue=body({n_pure_body})  purple=LBS-boundary({n_boundary})",
           fill="black", font=font)
    cs_pil.save(out_dir / "canonical_part_assignment_contact_sheet.png")
    print(f"[anim] saved contact sheet")

    # Per-view animation over t=0..T_full-1
    for v in sorted(cams_by_view.keys()):
        Rv, Tv = cams_by_view[v]
        view_frames = []
        for t in range(T_full):
            t_lo = min(t, T_train - 1)
            # Apply LBS deformation
            R_t = axis_angle_to_matrix_np(arm_aa[t_lo:t_lo+1])[0]
            rel = xyz_canon - arm_pivot
            rotated = rel @ R_t.T
            arm_xyz = rotated + arm_pivot + arm_trans[t_lo]
            new_xyz = arm_weights[:, None] * arm_xyz + (1 - arm_weights[:, None]) * xyz_canon
            d_xyz_np = new_xyz - xyz_canon

            d_xyz_t = torch.from_numpy(d_xyz_np).float().cuda()
            d_rotation = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
            d_scaling = torch.zeros(N, 3, device="cuda")
            dummy = torch.zeros(3, H, W, dtype=torch.float32)
            cam = SCGSCamera(colmap_id=t, R=Rv, T=Tv, FoVx=fov_x, FoVy=FovY,
                             image=dummy, gt_alpha_mask=None,
                             image_name=f"v{v}_t{t}", uid=t, fid=torch.tensor(0.0).float())
            with torch.no_grad():
                pkg = render(cam, gaussians, pipe, bg,
                             d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                             d_rot_as_res=True)
            img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
            img_u8 = (img * 255).astype(np.uint8)
            pil = PILImage.fromarray(img_u8)
            d = ImageDraw.Draw(pil)
            d.text((10, 10), f"view {v}  t={t:02d}/{T_full-1}", fill="black", font=font)
            view_frames.append(pil)

        gif_path = out_dir / f"part_anim_v{v}.gif"
        view_frames[0].save(gif_path, save_all=True, append_images=view_frames[1:],
                             duration=300, loop=0)
        view_frames[0].save(out_dir / "frames" / f"v{v}_t00.png")
        view_frames[T_full // 2].save(out_dir / "frames" / f"v{v}_t{T_full//2:02d}.png")
        view_frames[-1].save(out_dir / "frames" / f"v{v}_t{T_full-1:02d}.png")
        print(f"[anim] view {v}: {len(view_frames)} frames -> {gif_path.name}")

    print(f"[anim] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
