"""Phase 2.5: Mini per-Gaussian deform-MLP (replace cluster SE(3) + LBS).

Borrows SC-GS philosophy:
  - Each Gaussian gets its own (xyz, t) → (dxyz, dscale) prediction
  - Small MLP, frozen canonical, smart photo + silh loss

Key differences vs SC-GS:
  - MLP much smaller (~50k params, vs SC-GS deform-MLP ~1M)
  - Bounded output (tanh × max_disp) prevents Gaussian explosion
  - Smart photo filter (v5/d-3dgs residual weighting) suppresses VGM noise
  - Frozen canonical structure preserved
  - Only arm-region Gaussians get MLP deform (body stays canonical)

DOF: ~50k MLP params (vs cluster SE(3) 12.6k + xyz_res 866k = 879k for K=100)
     Massive DOF REDUCTION + continuous deform field (no LBS blur).
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
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


class TinyDeformMLP(nn.Module):
    def __init__(self, T_freqs=8, hidden=64, max_disp=0.3):
        super().__init__()
        self.T_freqs = T_freqs
        self.max_disp = max_disp
        in_dim = 3 + 2 * T_freqs * 1  # xyz + sin/cos(t × 2^k)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 6),  # dxyz (3) + dscale (3)
        )
        # Initialize last layer to zero → identity init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def time_enc(self, t_normalized):
        freqs = (2 ** torch.arange(self.T_freqs, device=t_normalized.device)).float()
        # t_normalized: scalar
        ang = 2 * math.pi * t_normalized * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (2*T_freqs,)

    def forward(self, xyz, t_normalized):
        N = xyz.shape[0]
        t_enc = self.time_enc(t_normalized).unsqueeze(0).expand(N, -1)
        feat = torch.cat([xyz, t_enc], dim=-1)
        out = self.mlp(feat)
        dxyz = torch.tanh(out[:, 0:3]) * self.max_disp
        dscale = torch.tanh(out[:, 3:6]) * 0.5
        return dxyz, dscale


def silhouette_loss(render_alpha, gt_alpha):
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
    inter = (a * g).sum()
    union = (a + g - a * g).sum()
    return bce + (1 - inter / (union + 1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="deformmlp_v1")
    p.add_argument("--canon_ply", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--part_dir", default=REPO / "runs_aux/part_assignment_lego_v2")
    p.add_argument("--scene_dir", default=REPO / "data/custom/lego_v2")
    p.add_argument("--v5_render_dir", default=REPO / "outputs/custom/lego_v2_d3dgs_ref/renders")
    p.add_argument("--iterations", type=int, default=8000)
    p.add_argument("--lr_mlp", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--T_freqs", type=int, default=8)
    p.add_argument("--max_disp", type=float, default=0.3)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--lam_photo_smart", type=float, default=3.0)
    p.add_argument("--photo_smart_alpha", type=float, default=16.0)
    p.add_argument("--lam_smooth_t", type=float, default=1.0,
                   help="Smoothness on consecutive t outputs (regularizer)")
    p.add_argument("--use_test_too", action="store_true", default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = REPO / "outputs/custom" / f"deformmlp_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical
    gaussians = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.canon_ply), og_number_points=0)
            gaussians = g; break
        except Exception:
            continue
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)
    xyz_canon = gaussians.get_xyz.detach().to(args.device)
    N = xyz_canon.shape[0]
    print(f"[deformmlp] canonical N={N}")

    # Arm mask
    arm_weights = np.load(Path(args.part_dir) / "gaussian_arm_weights.npy")
    arm_mask = arm_weights > 0.1   # soft inclusion
    arm_idx = np.where(arm_mask)[0]
    arm_idx_t = torch.from_numpy(arm_idx).long().to(args.device)
    N_arm = len(arm_idx)
    print(f"[deformmlp] arm Gaussians (w>0.1): {N_arm}")

    # Center + normalize arm xyz for MLP input
    arm_xyz_canon = xyz_canon[arm_idx_t]
    arm_xyz_mean = arm_xyz_canon.mean(0, keepdim=True)
    arm_xyz_scale = arm_xyz_canon.std() * 2
    arm_xyz_norm = (arm_xyz_canon - arm_xyz_mean) / arm_xyz_scale

    # Init MLP
    model = TinyDeformMLP(T_freqs=args.T_freqs, hidden=args.hidden, max_disp=args.max_disp).to(args.device)
    n_mlp_params = sum(p.numel() for p in model.parameters())
    print(f"[deformmlp] MLP params: {n_mlp_params}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr_mlp)

    # Load data
    data = json.loads((Path(args.scene_dir)/"transforms_train.json").read_text())
    if args.use_test_too:
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
            if cand.exists():
                png = cand; break
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

    # Smart photo weights
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
    print(f"[deformmlp] smart photo weights pre-computed for {len(smart_w)} frames")

    t0 = time.time()
    for it in range(1, args.iterations + 1):
        idx = np.random.randint(len(cams))
        cam = cams[idx]
        t = int(times[idx])
        t_norm = torch.tensor(float(t) / max(T_full - 1, 1), device=args.device)

        # Predict per-Gaussian deform for arm region
        dxyz_arm, dscale_arm = model(arm_xyz_norm, t_norm)
        # Apply
        new_xyz = xyz_canon.clone()
        new_xyz[arm_idx_t] = xyz_canon[arm_idx_t] + dxyz_arm
        d_xyz = new_xyz - xyz_canon
        d_rot = torch.zeros(N, 4, device=args.device) - torch.tensor([1, 0, 0, 0], device=args.device)
        d_scaling = torch.zeros(N, 3, device=args.device)
        d_scaling[arm_idx_t] = dscale_arm

        pkg = render(cam, gaussians, pipe, bg,
                     d_xyz=d_xyz, d_rotation=d_rot, d_scaling=d_scaling, d_rot_as_res=True)
        img = pkg["render"]; alpha = pkg["alpha"]
        gt_alpha = gt_alphas[idx]
        L_silh = silhouette_loss(alpha[0], gt_alpha)

        gt_rgb = gt_rgbs[idx]
        w_pix = smart_w[idx] * gt_alpha
        err = (img - gt_rgb).abs().mean(dim=0)
        L_photo_smart = (err * w_pix).sum() / w_pix.sum().clamp(min=1)

        # Temporal smoothness: render at neighbor t, compute |dxyz(t) - dxyz(t±1)|
        L_smooth_t = torch.tensor(0.0, device=args.device)
        if args.lam_smooth_t > 0:
            t_next = torch.tensor(float(min(t+1, T_full-1)) / max(T_full - 1, 1), device=args.device)
            dxyz_next, _ = model(arm_xyz_norm, t_next)
            L_smooth_t = ((dxyz_arm - dxyz_next) ** 2).mean()

        loss = (args.lam_silh * L_silh +
                args.lam_photo_smart * L_photo_smart +
                args.lam_smooth_t * L_smooth_t)
        optim.zero_grad(); loss.backward(); optim.step()
        if it % 500 == 0:
            print(f"[deformmlp] it {it:5d}  loss={loss:.4f}  silh={L_silh:.4f}  "
                  f"photo={L_photo_smart:.4f}  smooth_t={L_smooth_t:.4f}  ({time.time()-t0:.0f}s)")

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "arm_idx": arm_idx,
        "arm_xyz_mean": arm_xyz_mean.cpu().numpy(),
        "arm_xyz_scale": float(arm_xyz_scale.item()),
        "T_freqs": args.T_freqs,
        "hidden": args.hidden,
        "max_disp": args.max_disp,
        "T_full": T_full,
    }, out_dir / "model.pt")
    print(f"[deformmlp] saved {out_dir}/model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
