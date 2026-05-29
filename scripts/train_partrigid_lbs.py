"""Stage E v2 — Part-rigid 4D Gaussian training with LBS soft weights.

Same as train_partrigid.py but uses per-Gaussian soft arm weight (from
build_part_lbs_weights.py) instead of hard part ID. Optionally adds masked +
blurred photometric loss (low-freq + erosion mask) which constrains rotation
without exposing the model to VGM high-freq texture hallucination.

Per-Gaussian deformation (LBS, 2 parts: arm + body):
    d_xyz(g) = w_arm(g) * (R(t) @ (xyz(g) - arm_pivot) + arm_pivot + T(t) - xyz(g))

Body is implicit (static, weight = 1 - w_arm). Boundary Gaussians smoothly
blend between the arm SE(3) transform and identity → no tearing.

Optional loss term:
    L_blurred_photo = L1(blur(render) * eroded_mask, blur(gt) * eroded_mask)

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/train_partrigid_lbs.py \\
        --iterations 5000 --label lbs_v1 --lam_photo_blur 0.0   # LBS only
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/train_partrigid_lbs.py \\
        --iterations 5000 --label lbs_photo --lam_photo_blur 1.0  # LBS + blurred photo
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
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

CANON_PLY = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PART_DIR = REPO_ROOT / "runs_aux" / "part_assignment"
PARTS_MOTION_DIR = REPO_ROOT / "runs_aux" / "parts_motion"
SCENE = REPO_ROOT / "data" / "custom" / "scene00_masked"


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[0], 3, 3)
    th = theta.unsqueeze(-1)
    return I + th.sin() * K + (1 - th.cos()) * (K @ K)


class PartRigidLBSModel(nn.Module):
    def __init__(self, T: int, arm_pivot: torch.Tensor, arm_trans_init: torch.Tensor):
        super().__init__()
        self.T = T
        self.arm_trans = nn.Parameter(arm_trans_init.clone().float())
        self.arm_aa = nn.Parameter(torch.zeros(T, 3, dtype=torch.float32))
        self.register_buffer("arm_pivot", arm_pivot.float())

    def deform(self, t: int, xyz_canon: torch.Tensor, arm_weights: torch.Tensor) -> torch.Tensor:
        """Compute d_xyz for all Gaussians using LBS with 2 parts."""
        R = axis_angle_to_matrix(self.arm_aa[t:t+1])[0]
        T = self.arm_trans[t]
        rel = xyz_canon - self.arm_pivot
        rotated = rel @ R.T
        arm_xyz = rotated + self.arm_pivot + T  # full arm-rigid position
        # body keeps canonical position
        # LBS: weight * arm_xyz + (1 - weight) * canonical
        new_xyz = arm_weights[:, None] * arm_xyz + (1 - arm_weights[:, None]) * xyz_canon
        return new_xyz - xyz_canon


def silhouette_loss(render_alpha: torch.Tensor, gt_alpha: torch.Tensor) -> torch.Tensor:
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
    inter = (a * g).sum()
    union = (a + g - a * g).sum()
    return bce + (1 - inter / (union + 1e-6))


def gaussian_blur(img: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Per-channel 2D Gaussian blur via separable conv."""
    if img.dim() == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)
    sz = max(int(2 * round(2.5 * sigma)) + 1, 3)
    x = torch.arange(sz, device=img.device, dtype=img.dtype) - sz // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    C = img.shape[1]
    k_h = k.view(1, 1, 1, sz).expand(C, 1, 1, sz)
    k_v = k.view(1, 1, sz, 1).expand(C, 1, sz, 1)
    img = F.conv2d(img, k_h, padding=(0, sz // 2), groups=C)
    img = F.conv2d(img, k_v, padding=(sz // 2, 0), groups=C)
    return img.squeeze(0) if img.shape[0] == 1 else img


def erode_mask(mask: torch.Tensor, ksize: int = 9) -> torch.Tensor:
    """Binary erosion: 1 only where ALL pixels in window of radius ksize//2 are 1."""
    if mask.dim() == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        m = mask.unsqueeze(0)
    else:
        m = mask
    pad = ksize // 2
    eroded = -F.max_pool2d(-m, kernel_size=ksize, stride=1, padding=pad)
    return eroded.squeeze(0).squeeze(0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--label", default="lbs_v1")
    p.add_argument("--lr_trans", type=float, default=2e-3)
    p.add_argument("--lr_rot", type=float, default=5e-3)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--lam_traj", type=float, default=0.1)
    p.add_argument("--lam_smooth", type=float, default=1.0)
    p.add_argument("--lam_photo_blur", type=float, default=0.0,
                   help="Weight for masked + blurred photometric (rotation constraint). 0 = disabled.")
    p.add_argument("--blur_sigma", type=float, default=8.0)
    p.add_argument("--erode_ksize", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = REPO_ROOT / "outputs/custom" / f"partrigid_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Canonical Gaussians (frozen) =====
    print(f"[LBS_train] loading canonical")
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON_PLY), og_number_points=0)
    for attr in ["_xyz", "_features_dc", "_features_rest",
                 "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)
    xyz_canon = gaussians.get_xyz.detach().to(args.device)
    N = xyz_canon.shape[0]

    # ===== LBS weights + 3D trajectory =====
    arm_weights = np.load(PART_DIR / "gaussian_arm_weights.npy")  # (N,) float
    arm_weights_t = torch.from_numpy(arm_weights).float().to(args.device)
    centroid_3d = np.load(PART_DIR / "part_centroid_3d.npy")  # (T, 2, 3)
    conf = np.load(PART_DIR / "part_centroid_confidence.npy")
    T = centroid_3d.shape[0]
    print(f"[LBS_train] N={N}, T={T}, mean arm weight={arm_weights.mean():.3f}, "
          f"boundary Gaussians [0.1, 0.9]: {((arm_weights > 0.1) & (arm_weights < 0.9)).sum()}")

    arm_pivot = torch.tensor(centroid_3d[0, 0], dtype=torch.float32, device=args.device)
    arm_trans_init = torch.tensor(centroid_3d[:, 0] - centroid_3d[0, 0],
                                  dtype=torch.float32, device=args.device)
    centroid_3d_t = torch.tensor(centroid_3d, dtype=torch.float32, device=args.device)
    conf_t = torch.tensor(conf, dtype=torch.float32, device=args.device)

    model = PartRigidLBSModel(T, arm_pivot, arm_trans_init).to(args.device)
    optim = torch.optim.Adam([
        {"params": [model.arm_trans], "lr": args.lr_trans},
        {"params": [model.arm_aa],    "lr": args.lr_rot},
    ])

    # ===== Cameras + GT alpha + (optional) GT RGB =====
    data = json.loads((SCENE / "transforms_train.json").read_text())
    fov_x = data["camera_angle_x"]
    H = W = 576
    cams = []
    gt_alphas = []
    gt_rgbs = []
    times = []
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)
    for i, f in enumerate(data["frames"]):
        v = int(f["view_idx"]); ti = int(f["frame_idx"])
        png = SCENE / "train" / f"{Path(f['file_path']).name}.png"
        rgba = np.asarray(iio.imread(png))
        alpha = (rgba[..., 3] > 127).astype(np.float32)
        rgb = (rgba[..., :3].astype(np.float32) / 255.0)
        # composite GT onto white background to match training scene
        rgb = rgb * alpha[..., None] + 1.0 * (1 - alpha[..., None])
        gt_alphas.append(torch.from_numpy(alpha).to(args.device))
        gt_rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1).to(args.device))
        times.append(ti)
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        matrix = np.linalg.inv(c2w)
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -matrix[:3, 3]
        img_t = torch.from_numpy(rgba[..., :3].astype(np.float32) / 255.0).permute(2, 0, 1)
        alpha_t = torch.from_numpy((rgba[..., 3:4] / 255.0).astype(np.float32)).permute(2, 0, 1)
        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=img_t, gt_alpha_mask=alpha_t,
                         image_name=Path(f['file_path']).stem, uid=i,
                         fid=torch.tensor(float(ti) / max(T - 1, 1)).float())
        cams.append(cam)

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)

    print(f"[LBS_train] {len(cams)} cameras")
    print(f"[LBS_train] loss config: silh={args.lam_silh}, traj={args.lam_traj}, "
          f"smooth={args.lam_smooth}, photo_blur={args.lam_photo_blur}")
    print(f"[LBS_train] training for {args.iterations} iters")

    t0 = time.time()
    losses_history = []
    for it in range(1, args.iterations + 1):
        idx = np.random.randint(len(cams))
        cam = cams[idx]
        t = int(times[idx])

        # LBS deformation
        d_xyz = model.deform(t, xyz_canon, arm_weights_t)
        d_rotation = torch.zeros(N, 4, device=args.device)
        d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device=args.device)
        d_scaling = torch.zeros(N, 3, device=args.device)

        pkg = render(cam, gaussians, pipe, background,
                     d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                     d_rot_as_res=True)
        img = pkg["render"]
        alpha = pkg["alpha"]

        gt_alpha = gt_alphas[idx]
        L_silh = silhouette_loss(alpha[0], gt_alpha)

        target = centroid_3d_t[t, 0]
        cc = conf_t[t, 0]
        # Weighted arm centroid (LBS-aware): each Gaussian contributes its arm_weight
        deformed_xyz = xyz_canon + d_xyz
        w_sum = arm_weights_t.sum().clamp(min=1e-6)
        pred_centroid = (arm_weights_t[:, None] * deformed_xyz).sum(0) / w_sum
        L_traj = cc * ((pred_centroid - target) ** 2).sum()

        L_smooth = ((model.arm_trans[1:] - model.arm_trans[:-1]) ** 2).mean() + \
                   ((model.arm_aa[1:] - model.arm_aa[:-1]) ** 2).mean()

        L_photo = torch.tensor(0.0, device=args.device)
        if args.lam_photo_blur > 0:
            gt_rgb = gt_rgbs[idx]
            if args.blur_sigma > 0.5:
                rd = gaussian_blur(img, sigma=args.blur_sigma)
                gd = gaussian_blur(gt_rgb, sigma=args.blur_sigma)
            else:
                rd = img; gd = gt_rgb
            if args.erode_ksize > 1:
                eroded = erode_mask(gt_alpha, ksize=args.erode_ksize)
            else:
                eroded = gt_alpha
            mask3 = eroded.unsqueeze(0).expand_as(rd)
            L_photo = ((rd - gd).abs() * mask3).sum() / mask3.sum().clamp(min=1)

        loss = (args.lam_silh * L_silh + args.lam_traj * L_traj +
                args.lam_smooth * L_smooth + args.lam_photo_blur * L_photo)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % 200 == 0:
            losses_history.append({
                "iter": it, "loss": float(loss), "silh": float(L_silh),
                "traj": float(L_traj), "smooth": float(L_smooth), "photo": float(L_photo),
                "aa_max": float(model.arm_aa.detach().abs().max()),
                "trans_max": float(model.arm_trans.detach().abs().max()),
            })
            print(f"[LBS_train] it {it:>5d}  loss={loss:.4f}  silh={L_silh:.4f}  "
                  f"traj={L_traj:.4f}  photo={L_photo:.4f}  "
                  f"|aa|={float(model.arm_aa.detach().abs().max()):.3f}  "
                  f"|trans|={float(model.arm_trans.detach().abs().max()):.3f}  "
                  f"({time.time()-t0:.0f}s)")

    state = {
        "arm_trans": model.arm_trans.detach().cpu().numpy(),
        "arm_aa": model.arm_aa.detach().cpu().numpy(),
        "arm_pivot": model.arm_pivot.detach().cpu().numpy(),
        "arm_weights": arm_weights,
        "part_id": np.where(arm_weights > 0.5, 0, 1).astype(np.int32),
        "config": vars(args),
        "losses_history": losses_history,
    }
    np.savez(out_dir / "partrigid_state.npz", **state)
    print(f"[LBS_train] saved {out_dir}/partrigid_state.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
