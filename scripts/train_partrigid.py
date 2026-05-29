"""Stage E (P3+P4+P5): Part-rigid 4D Gaussian training.

Frozen canonical (from P1) + per-Gaussian part ID (from Stage C) + 3D arm
trajectory target (from Stage D). We learn only per-(time) SE(3) for the arm
part; body stays static. Loss is multi-signal, **no raw RGB photometric**.

Architecture:
    Canonical Gaussians (xyz, color, scale, rotation, opacity)   -- FROZEN
    Part IDs (N_g,) in {0=arm, 1=body, -1=ignore}                 -- FROZEN
    arm_T: nn.Parameter (T, 3) -- per-time translation
    arm_aa: nn.Parameter (T, 3) -- per-time axis-angle rotation
      Initialized from Stage D centroid trajectory (translation only).

Forward at time t:
    For each Gaussian g:
        if part_id[g] == 0 (arm):
            d_xyz[g] = R(arm_aa[t]) @ (xyz[g] - arm_pivot) + arm_pivot + arm_T[t] - xyz[g]
        else:
            d_xyz[g] = 0
    Render via SC-GS renderer; compute losses.

Loss:
    L = lam_silh * L_silhouette                       (BCE + IoU)
      + lam_traj * L_part_traj                        (||C_render - C_target||^2)
      + lam_smooth * L_temporal_smooth                (||T(t+1) - T(t)||^2)
      + lam_arap * L_arap                             (TODO P6; skipped in P5)

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/train_partrigid.py \\
        --iterations 3000 --label partrigid_v1
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

CANON_PLY = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PART_DIR = REPO_ROOT / "runs_aux" / "part_assignment"
PARTS_MOTION_DIR = REPO_ROOT / "runs_aux" / "parts_motion"
SCENE = REPO_ROOT / "data" / "custom" / "scene00_masked"


def make_K(W, H, fov_x):
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    return np.array([[fx, 0, W/2.0], [0, fx, H/2.0], [0, 0, 1.0]], dtype=np.float64)


def c2w_b_to_w2c_cv(c2w_b):
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    return np.linalg.inv(c2w_b @ flip)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(T, 3) axis-angle to (T, 3, 3) rotation matrices, via Rodrigues."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (T, 1)
    axis = aa / theta
    K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[0], 3, 3)
    th = theta.unsqueeze(-1)  # (T,1,1)
    R = I + th.sin() * K + (1 - th.cos()) * (K @ K)
    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(T,3,3) -> (T,4) quaternion w,x,y,z."""
    m00 = R[:, 0, 0]; m11 = R[:, 1, 1]; m22 = R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    # Always positive branch (assumes well-conditioned)
    s = torch.sqrt(torch.clamp(tr + 1, min=1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class PartRigidModel(nn.Module):
    def __init__(self, T: int, arm_pivot: torch.Tensor, arm_trans_init: torch.Tensor):
        super().__init__()
        self.T = T
        # Per-time translation + axis-angle for the arm part
        self.arm_trans = nn.Parameter(arm_trans_init.clone().float())  # (T, 3)
        self.arm_aa = nn.Parameter(torch.zeros(T, 3, dtype=torch.float32))  # identity init
        self.register_buffer("arm_pivot", arm_pivot.float())  # (3,)

    def rotation_matrices(self) -> torch.Tensor:
        return axis_angle_to_matrix(self.arm_aa)  # (T, 3, 3)

    def deform_arm(self, t: int, xyz_arm: torch.Tensor) -> torch.Tensor:
        """Apply T(t) to arm canonical Gaussians. Returns d_xyz (N_arm, 3)."""
        R = axis_angle_to_matrix(self.arm_aa[t:t+1])[0]    # (3, 3)
        T = self.arm_trans[t]                              # (3,)
        rel = xyz_arm - self.arm_pivot                     # (N, 3)
        rotated = rel @ R.T                                # (N, 3)
        new_xyz = rotated + self.arm_pivot + T             # (N, 3)
        return new_xyz - xyz_arm                           # d_xyz


def silhouette_loss(render_alpha: torch.Tensor, gt_alpha: torch.Tensor) -> torch.Tensor:
    """BCE + soft IoU."""
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
    inter = (a * g).sum()
    union = (a + g - a * g).sum()
    iou = inter / (union + 1e-6)
    return bce + (1 - iou)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--label", default="partrigid_v1")
    p.add_argument("--lr_trans", type=float, default=2e-3)
    p.add_argument("--lr_rot", type=float, default=5e-3)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--lam_traj", type=float, default=0.1)
    p.add_argument("--lam_smooth", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = REPO_ROOT / "outputs/custom" / f"partrigid_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load canonical Gaussians (frozen) =====
    print(f"[stage_E] loading canonical {CANON_PLY.name}")
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON_PLY), og_number_points=0)
    # Freeze all canonical params
    for attr in ["_xyz", "_features_dc", "_features_rest",
                 "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)
    xyz_canon = gaussians.get_xyz.detach().to(args.device)
    N = xyz_canon.shape[0]
    print(f"[stage_E] N={N}")

    # ===== Load part assignments + 3D centroid trajectory =====
    part_id = np.load(PART_DIR / "gaussian_part_ids.npy")  # (N,)
    centroid_3d = np.load(PART_DIR / "part_centroid_3d.npy")  # (T, 2, 3)
    conf = np.load(PART_DIR / "part_centroid_confidence.npy")  # (T, 2)
    T = centroid_3d.shape[0]
    part_id_t = torch.from_numpy(part_id).long().to(args.device)
    arm_mask = (part_id_t == 0)
    body_mask = (part_id_t == 1)
    n_arm = int(arm_mask.sum().item())
    n_body = int(body_mask.sum().item())
    print(f"[stage_E] parts: arm={n_arm}, body={n_body}, total={N}")
    print(f"[stage_E] arm centroid sweep z={centroid_3d[:,0,2].max()-centroid_3d[:,0,2].min():.3f}")

    arm_pivot = torch.tensor(centroid_3d[0, 0], dtype=torch.float32, device=args.device)
    arm_trans_init = torch.tensor(centroid_3d[:, 0] - centroid_3d[0, 0],
                                  dtype=torch.float32, device=args.device)
    centroid_3d_t = torch.tensor(centroid_3d, dtype=torch.float32, device=args.device)
    conf_t = torch.tensor(conf, dtype=torch.float32, device=args.device)

    model = PartRigidModel(T, arm_pivot, arm_trans_init).to(args.device)
    optim = torch.optim.Adam([
        {"params": [model.arm_trans], "lr": args.lr_trans},
        {"params": [model.arm_aa],    "lr": args.lr_rot},
    ])
    print(f"[stage_E] learnable params: arm_trans (T,3)=({T},3) arm_aa (T,3)=({T},3) = {T*6} unknowns")

    # ===== Load cameras + per-(view, time) GT silhouette =====
    data = json.loads((SCENE / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    H = W = 576
    K_np = make_K(W, H, fov_x)
    K_t = torch.from_numpy(K_np).float().to(args.device)
    cams_meta = []
    gt_alpha_per_cam = []
    for f in data["frames"]:
        v = int(f["view_idx"]); ti = int(f["frame_idx"])
        png = SCENE / "train" / f"{Path(f['file_path']).name}.png"
        rgba = np.asarray(iio.imread(png))
        alpha = (rgba[..., 3] > 127).astype(np.float32)
        cams_meta.append({
            "view_idx": v, "time": ti,
            "c2w": np.asarray(f["transform_matrix"], dtype=np.float64),
            "fid": float(ti) / max(T - 1, 1),
        })
        gt_alpha_per_cam.append(torch.from_numpy(alpha).to(args.device))

    # ===== SC-GS renderer needs a Camera-like object =====
    # Build a thin shim: use SC-GS's Camera class
    from scene.cameras import Camera as SCGSCamera
    from utils.graphics_utils import focal2fov, fov2focal  # noqa
    cams = []
    for i, m in enumerate(cams_meta):
        c2w = m["c2w"]
        matrix = np.linalg.inv(c2w)
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -matrix[:3, 3]
        # SC-GS Camera signature is large; let's just create one with all needed args
        FovY = focal2fov(fov2focal(fov_x, W), H)
        FovX = fov_x
        rgba_path = SCENE / "train" / f"{Path(data['frames'][i]['file_path']).name}.png"
        rgba = np.asarray(iio.imread(rgba_path))
        img_t = torch.from_numpy(rgba[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1)
        alpha_t = torch.from_numpy((rgba[..., 3:4] / 255.0).astype(np.float32)).permute(2, 0, 1)
        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=FovX, FoVy=FovX,
                         image=img_t, gt_alpha_mask=alpha_t,
                         image_name=Path(data['frames'][i]['file_path']).stem,
                         uid=i, fid=torch.tensor(m["fid"]).float())
        cams.append(cam)

    # ===== Pipeline params (renderer settings) =====
    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe_args = parser_pipe.parse_args([])
    pipe = pp.extract(pipe_args)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)

    # ===== Training loop =====
    print(f"[stage_E] training for {args.iterations} iters")
    log_every = 100
    t0 = time.time()
    losses_history = []
    for it in range(1, args.iterations + 1):
        idx = np.random.randint(len(cams))
        cam = cams[idx]
        t = int(cams_meta[idx]["time"])

        # Compute d_xyz for arm Gaussians; zeros for body/unassigned
        d_xyz = torch.zeros_like(xyz_canon)
        if n_arm > 0:
            arm_d = model.deform_arm(t, xyz_canon[arm_mask])
            d_xyz[arm_mask] = arm_d
        d_rotation = torch.zeros(N, 4, device=args.device)  # identity, no rotation per-Gaussian
        d_rotation[:, 0] = 1.0  # w-component for identity quat
        d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device=args.device)  # residual format expected
        d_scaling = torch.zeros(N, 3, device=args.device)

        render_pkg = render(cam, gaussians, pipe, background,
                            d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                            d_rot_as_res=True)
        img = render_pkg["render"]      # (3, H, W)
        alpha = render_pkg["alpha"]     # (1, H, W)

        gt_alpha = gt_alpha_per_cam[idx]  # (H, W)

        # L_silhouette
        L_silh = silhouette_loss(alpha[0], gt_alpha)

        # L_part_traj: deformed arm centroid should match target
        with torch.no_grad():
            target = centroid_3d_t[t, 0]  # (3,)
            cc = conf_t[t, 0]
        if n_arm > 0:
            deformed_arm_xyz = xyz_canon[arm_mask] + d_xyz[arm_mask]
            pred_centroid = deformed_arm_xyz.mean(dim=0)
            L_traj = cc * ((pred_centroid - target) ** 2).sum()
        else:
            L_traj = torch.tensor(0.0, device=args.device)

        # L_temporal_smooth on arm_trans + arm_aa
        L_smooth = ((model.arm_trans[1:] - model.arm_trans[:-1]) ** 2).mean() + \
                   ((model.arm_aa[1:] - model.arm_aa[:-1]) ** 2).mean()

        loss = args.lam_silh * L_silh + args.lam_traj * L_traj + args.lam_smooth * L_smooth

        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % log_every == 0:
            elapsed = time.time() - t0
            losses_history.append({
                "iter": it, "loss": float(loss), "L_silh": float(L_silh),
                "L_traj": float(L_traj), "L_smooth": float(L_smooth),
                "arm_aa_max": float(model.arm_aa.detach().abs().max()),
                "arm_trans_max": float(model.arm_trans.detach().abs().max()),
            })
            print(f"[stage_E] iter {it:>5d}  loss={loss:.4f}  "
                  f"silh={L_silh:.4f}  traj={L_traj:.4f}  smooth={L_smooth:.4f}  "
                  f"|aa|={float(model.arm_aa.detach().abs().max()):.3f}  "
                  f"|trans|={float(model.arm_trans.detach().abs().max()):.3f}  "
                  f"({elapsed:.0f}s)")

    # Save final state
    state = {
        "arm_trans": model.arm_trans.detach().cpu().numpy(),
        "arm_aa": model.arm_aa.detach().cpu().numpy(),
        "arm_pivot": model.arm_pivot.detach().cpu().numpy(),
        "part_id": part_id,
        "config": vars(args),
        "losses_history": losses_history,
    }
    np.savez(out_dir / "partrigid_state.npz", **state)
    print(f"[stage_E] saved {out_dir}/partrigid_state.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
