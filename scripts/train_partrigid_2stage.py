"""Curriculum 2-stage training with semantic 3-part decomposition.

Three Gaussian groups (from runs_aux/part_assignment/sub_assignment.npy):
  0 = body       (frozen identity)
  1 = arm-shaft  (own SE(3) trajectory T_shaft, R_shaft)
  2 = bucket     (own SE(3) trajectory T_bucket, R_bucket)

Two-stage curriculum:
  Stage 1 (curriculum learning, easy task):
    - Freeze arm-shaft SE(3) at identity
    - Train ONLY bucket SE(3) with full loss (silhouette + smart photo + smooth)
    - Idea: learn the most prominent motion first
  Stage 2 (joint refinement):
    - Unfreeze arm-shaft SE(3), continue with both learnable
    - Lower LR for bucket (it's already close), normal LR for arm-shaft
    - Add ARAP regularizer between bucket and arm-shaft (kinematic coupling)

Compare against:
  - K=100 hier_smart_3x baseline (18.89 dB)
  - Original part-rigid (18.03 dB)

DOF: 2 parts × 21 t × 6 = 252 (much less than K=100's 12,600)
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
SCGS = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PART_DIR = REPO / "runs_aux/part_assignment"
SCENE = REPO / "data/custom/scene00_masked"
V5_RENDER_DIR = REPO / "outputs/custom/scene00_v5_node/train/ours_30000/renders"


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    K = torch.zeros((aa.shape[0], 3, 3), device=aa.device, dtype=aa.dtype)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype)[None].expand_as(K)
    th = theta[..., None]
    return I + th.sin() * K + (1 - th.cos()) * (K @ K)


class TwoPartRigid(nn.Module):
    def __init__(self, T, shaft_pivot, bucket_pivot):
        super().__init__()
        self.T = T
        self.shaft_trans = nn.Parameter(torch.zeros(T, 3))
        self.shaft_aa    = nn.Parameter(torch.zeros(T, 3))
        self.bucket_trans = nn.Parameter(torch.zeros(T, 3))
        self.bucket_aa    = nn.Parameter(torch.zeros(T, 3))
        self.register_buffer("shaft_pivot", torch.as_tensor(shaft_pivot, dtype=torch.float32))
        self.register_buffer("bucket_pivot", torch.as_tensor(bucket_pivot, dtype=torch.float32))

    def deform_part(self, t, xyz, pivot, trans, aa):
        R = axis_angle_to_matrix(aa[t:t+1])[0]
        rel = xyz - pivot
        rotated = rel @ R.T
        return rotated + pivot + trans[t]


def silhouette_loss(render_alpha, gt_alpha):
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
    inter = (a * g).sum()
    union = (a + g - a * g).sum()
    return bce + (1 - inter / (union + 1e-6))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="2stage_v1")
    p.add_argument("--stage1_iter", type=int, default=2000)
    p.add_argument("--stage2_iter", type=int, default=4000)
    p.add_argument("--lr_trans", type=float, default=2e-3)
    p.add_argument("--lr_rot", type=float, default=5e-3)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--lam_smooth", type=float, default=1.0)
    p.add_argument("--lam_arap", type=float, default=0.5,
                   help="Stage 2 only: penalty for |shaft_aa - bucket_aa|² (kinematic chain coupling)")
    p.add_argument("--lam_photo_smart", type=float, default=3.0)
    p.add_argument("--photo_smart_alpha", type=float, default=8.0)
    p.add_argument("--stage2_lr_scale_bucket", type=float, default=0.25,
                   help="LR for bucket params in Stage 2 (scaled down since init from Stage 1)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = REPO / "outputs/custom" / f"partrigid_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2stage] loading canonical")
    g = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    g.load_ply(str(CANON), og_number_points=0)
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(g, attr):
            getattr(g, attr).requires_grad_(False)
    xyz_canon = g.get_xyz.detach().to(args.device)
    N = xyz_canon.shape[0]

    sub_assign = np.load(PART_DIR / "sub_assignment.npy")  # (N,) 0=body, 1=shaft, 2=bucket
    centroid_3d = np.load(PART_DIR / "part_centroid_3d.npy")
    T = centroid_3d.shape[0]
    is_body = torch.from_numpy(sub_assign == 0).to(args.device)
    is_shaft = torch.from_numpy(sub_assign == 1).to(args.device)
    is_bucket = torch.from_numpy(sub_assign == 2).to(args.device)
    print(f"[2stage] N={N}, T={T}, body={int(is_body.sum())} shaft={int(is_shaft.sum())} bucket={int(is_bucket.sum())}")

    shaft_pivot_np = xyz_canon[is_shaft].mean(0).cpu().numpy()
    bucket_pivot_np = xyz_canon[is_bucket].mean(0).cpu().numpy()
    print(f"[2stage] shaft pivot  = {shaft_pivot_np}")
    print(f"[2stage] bucket pivot = {bucket_pivot_np}")

    model = TwoPartRigid(T, shaft_pivot_np, bucket_pivot_np).to(args.device)

    # Load data
    data = json.loads((SCENE / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    cams, gt_alphas, gt_rgbs, times = [], [], [], []
    for i, f in enumerate(data["frames"]):
        ti = int(f["frame_idx"])
        png = SCENE / "train" / f"{Path(f['file_path']).name}.png"
        rgba = np.asarray(iio.imread(png))
        alpha = (rgba[..., 3] > 127).astype(np.float32)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        rgb = rgb * alpha[..., None] + 1.0 * (1 - alpha[..., None])
        gt_alphas.append(torch.from_numpy(alpha).to(args.device))
        gt_rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1).to(args.device))
        times.append(ti)
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=torch.from_numpy(rgba[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1),
                         gt_alpha_mask=torch.from_numpy((rgba[..., 3:4] / 255.0).astype(np.float32)).permute(2, 0, 1),
                         image_name=Path(f['file_path']).stem, uid=i,
                         fid=torch.tensor(float(ti) / max(T - 1, 1)).float())
        cams.append(cam)

    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)

    # Smart photo weights
    smart_w = []
    for i, f in enumerate(data["frames"]):
        v = int(f["view_idx"]); ti = int(f["frame_idx"])
        v5p = V5_RENDER_DIR / f"{v*T + ti:05d}.png"
        v5_rgba = np.asarray(iio.imread(v5p), dtype=np.float32) / 255.0
        v5_a = v5_rgba[..., 3:4] if v5_rgba.shape[-1] == 4 else np.ones_like(v5_rgba[..., :1])
        v5_rgb = v5_rgba[..., :3] * v5_a + 1.0 * (1 - v5_a)
        gt_rgb_np = gt_rgbs[i].permute(1, 2, 0).cpu().numpy()
        residual = np.abs(gt_rgb_np - v5_rgb).mean(axis=-1)
        w = np.exp(-args.photo_smart_alpha * residual)
        smart_w.append(torch.from_numpy(w.astype(np.float32)).to(args.device))
    print(f"[2stage] smart photo: {len(smart_w)} weight maps")

    def train_one_stage(stage_name, n_iters, lr_groups, log_prefix):
        optim = torch.optim.Adam(lr_groups)
        t0 = time.time()
        for it in range(1, n_iters + 1):
            idx = np.random.randint(len(cams))
            cam = cams[idx]; t = int(times[idx])

            # Apply per-part SE(3) to corresponding Gaussians
            new_xyz = xyz_canon.clone()
            # Shaft
            shaft_new = model.deform_part(t, xyz_canon[is_shaft],
                                           model.shaft_pivot, model.shaft_trans, model.shaft_aa)
            new_xyz[is_shaft] = shaft_new
            # Bucket
            bucket_new = model.deform_part(t, xyz_canon[is_bucket],
                                            model.bucket_pivot, model.bucket_trans, model.bucket_aa)
            new_xyz[is_bucket] = bucket_new
            d_xyz = new_xyz - xyz_canon
            d_rot = torch.zeros(N, 4, device=args.device) - torch.tensor([1, 0, 0, 0], device=args.device)
            d_sc = torch.zeros(N, 3, device=args.device)

            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
            img = pkg["render"]; alpha = pkg["alpha"]
            L_silh = silhouette_loss(alpha[0], gt_alphas[idx])
            L_smooth = ((model.shaft_trans[1:] - model.shaft_trans[:-1]) ** 2).mean() + \
                       ((model.shaft_aa[1:] - model.shaft_aa[:-1]) ** 2).mean() + \
                       ((model.bucket_trans[1:] - model.bucket_trans[:-1]) ** 2).mean() + \
                       ((model.bucket_aa[1:] - model.bucket_aa[:-1]) ** 2).mean()
            # ARAP coupling (Stage 2 only)
            L_arap = ((model.shaft_aa - model.bucket_aa) ** 2).mean() + \
                     ((model.shaft_trans - model.bucket_trans) ** 2).mean()
            # Smart photo
            w_pix = smart_w[idx] * gt_alphas[idx]
            err = (img - gt_rgbs[idx]).abs().mean(dim=0)
            L_photo = (err * w_pix).sum() / w_pix.sum().clamp(min=1)

            loss = (args.lam_silh * L_silh + args.lam_smooth * L_smooth +
                    args.lam_arap * L_arap + args.lam_photo_smart * L_photo)
            optim.zero_grad(); loss.backward(); optim.step()
            if it % 500 == 0:
                print(f"[{log_prefix}] it {it:>5d}  loss={loss:.4f}  silh={L_silh:.4f}  "
                      f"photo={L_photo:.4f}  smooth={L_smooth:.4f}  arap={L_arap:.4f}  "
                      f"({time.time()-t0:.0f}s)")

    # ===== Stage 1: bucket only =====
    print(f"\n[2stage] === Stage 1 (bucket only, {args.stage1_iter} iters) ===")
    stage1_groups = [
        {"params": [model.bucket_trans], "lr": args.lr_trans},
        {"params": [model.bucket_aa],    "lr": args.lr_rot},
    ]
    train_one_stage("stage1", args.stage1_iter, stage1_groups, "S1")

    # ===== Stage 2: joint refinement =====
    print(f"\n[2stage] === Stage 2 (joint bucket+shaft, {args.stage2_iter} iters) ===")
    stage2_groups = [
        {"params": [model.bucket_trans], "lr": args.lr_trans * args.stage2_lr_scale_bucket},
        {"params": [model.bucket_aa],    "lr": args.lr_rot   * args.stage2_lr_scale_bucket},
        {"params": [model.shaft_trans],  "lr": args.lr_trans},
        {"params": [model.shaft_aa],     "lr": args.lr_rot},
    ]
    train_one_stage("stage2", args.stage2_iter, stage2_groups, "S2")

    # Save
    state = {
        "shaft_trans": model.shaft_trans.detach().cpu().numpy(),
        "shaft_aa": model.shaft_aa.detach().cpu().numpy(),
        "shaft_pivot": shaft_pivot_np,
        "bucket_trans": model.bucket_trans.detach().cpu().numpy(),
        "bucket_aa": model.bucket_aa.detach().cpu().numpy(),
        "bucket_pivot": bucket_pivot_np,
        "sub_assignment": sub_assign,
        "config": vars(args),
    }
    np.savez(out_dir / "partrigid_state.npz", **state)
    print(f"[2stage] saved to {out_dir}/partrigid_state.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
