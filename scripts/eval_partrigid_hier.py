"""Eval hierarchical part-rigid model. Computes PSNR vs SV4D + saves renders.

Supports:
  - K-cluster sub-decomposition (state["trans"], state["aa"] shape (K, T, 3))
  - LBS weights (state["lbs_weights"] shape (N, K))
  - Optional rotation propagation (--use_rot_prop)
  - Optional color tint (auto-detected from state)
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
SCGS_ROOT = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"


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


def axis_angle_to_quaternion_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / theta
    half = theta * 0.5
    w = np.cos(half)
    s = np.sin(half)
    return np.concatenate([w, axis * s], axis=-1)


def psnr(img, gt):
    mse = ((img - gt) ** 2).mean()
    return -10 * math.log10(max(float(mse), 1e-12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="hier_v1")
    p.add_argument("--scene", default="scene00_masked",
                   help="evaluate against all 105 frames in scene00_masked (full eval, not split)")
    p.add_argument("--use_rot_prop", action="store_true",
                   help="apply rotation propagation in eval (should match training flag)")
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    scene_dir = REPO / "data" / "custom" / args.scene
    state = np.load(REPO / f"outputs/custom/partrigid_{args.label}/partrigid_state.npz",
                    allow_pickle=True)
    trans = state["trans"]; aa = state["aa"]           # (K, T, 3) each
    arm_centers = state["arm_centers"]                  # (K, 3)
    lbs_weights = state["lbs_weights"]                  # (N, K)
    K = trans.shape[0]; T_train = trans.shape[1]
    have_color_tint = "color_tint" in state.files
    color_tint = state["color_tint"] if have_color_tint else None
    have_scale = "scale" in state.files
    scale_per_kt = state["scale"] if have_scale else None  # (K, T, 3)
    if "config" in state.files:
        cfg = state["config"].item() if hasattr(state["config"], "item") else state["config"]
        if isinstance(cfg, dict):
            args.use_rot_prop = args.use_rot_prop or bool(cfg.get("use_rot_prop", False))
    print(f"[eval-hier] label={args.label}  K={K}  T_train={T_train}  "
          f"color_tint={have_color_tint}  rot_prop={args.use_rot_prop}")

    # Load canonical
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    if have_color_tint:
        f_dc_canonical = gaussians._features_dc.detach().clone()

    # Pre-compute per-cluster rotations + quaternions across t
    R_kt = np.stack([axis_angle_to_matrix_np(aa[:, t, :]) for t in range(T_train)], axis=0)  # (T, K, 3, 3)
    q_kt = np.stack([axis_angle_to_quaternion_np(aa[:, t, :]) for t in range(T_train)], axis=0)  # (T, K, 4)
    lbs_w_t = torch.from_numpy(lbs_weights).float().cuda()

    # Meta
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    fov_x = meta["camera_angle_x"]
    H = W = 576
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out_dir = REPO / f"runs_aux/hier_eval/{args.label}_{args.scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "renders").mkdir(exist_ok=True)

    psnr_list = []
    for i, f in enumerate(meta["frames"]):
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]
        t_lo = min(t, T_train - 1)

        # LBS deformation
        R_all = R_kt[t_lo]            # (K, 3, 3)
        T_all = trans[:, t_lo, :]     # (K, 3)
        rel = xyz_canon[:, None, :] - arm_centers[None, :, :]   # (N, K, 3)
        rotated = np.einsum("kij,nkj->nki", R_all, rel)         # (N, K, 3)
        new_per_cluster = rotated + arm_centers[None, :, :] + T_all[None, :, :]
        weighted = (lbs_weights[..., None] * new_per_cluster).sum(axis=1)  # (N, 3)
        w_total = lbs_weights.sum(axis=1, keepdims=True).clip(min=0, max=1)
        new_xyz = weighted + (1 - w_total) * xyz_canon
        d_xyz_np = new_xyz - xyz_canon

        d_xyz_t = torch.from_numpy(d_xyz_np).float().cuda()
        d_rotation = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_scaling = torch.zeros(N, 3, device="cuda")
        if have_scale:
            scale_blend = lbs_w_t @ torch.from_numpy(scale_per_kt[:, t_lo, :]).float().cuda()  # (N, 3)
            d_scaling = d_scaling + scale_blend

        d_rotation_bias = None
        if args.use_rot_prop:
            q_clusters = torch.from_numpy(q_kt[t_lo]).float().cuda()   # (K, 4)
            q_blend = lbs_w_t @ q_clusters                              # (N, 4)
            body_w_t = (1 - lbs_w_t.sum(dim=1, keepdim=True)).clamp(min=0)
            identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
            q_blend = q_blend + body_w_t * identity_q
            d_rotation_bias = q_blend / q_blend.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        if have_color_tint:
            tint = torch.from_numpy(color_tint[t_lo]).float().cuda()
            gaussians._features_dc.data = f_dc_canonical + tint[None, None, :]

        gt_path = scene_dir / "train" / f"{Path(f['file_path']).name}.png"
        gt_rgba = np.asarray(iio.imread(gt_path)).astype(np.float32) / 255.0
        alpha = gt_rgba[..., 3:4]
        gt_rgb = gt_rgba[..., :3] * alpha + np.array([1, 1, 1]) * (1 - alpha)
        gt_tensor = torch.from_numpy(gt_rgb.astype(np.float32)).permute(2, 0, 1).cuda()

        cam = SCGSCamera(colmap_id=i, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=gt_tensor, gt_alpha_mask=None,
                         image_name=f["file_path"], uid=i, fid=torch.tensor(0.0).float())

        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg,
                         d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                         d_rot_as_res=True, d_rotation_bias=d_rotation_bias)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        ps = psnr(img, gt_rgb)
        psnr_list.append(ps)
        if args.save_renders:
            both = np.concatenate([gt_rgb, img], axis=1)
            Image.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).save(
                out_dir / "renders" / f"v{v}_t{t:02d}.png")

    arr = np.array(psnr_list)
    print()
    print(f"[eval-hier] === {args.label} on {args.scene} (all {len(arr)} frames) ===")
    print(f"[eval-hier] mean PSNR = {arr.mean():.3f} +/- {arr.std():.3f}")
    print(f"[eval-hier] median PSNR = {np.median(arr):.3f}")
    print(f"[eval-hier] min/max = {arr.min():.3f} / {arr.max():.3f}")

    (out_dir / "psnr_per_frame.json").write_text(json.dumps({
        "label": args.label, "scene": args.scene,
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "std": float(arr.std()), "per_frame": arr.tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
