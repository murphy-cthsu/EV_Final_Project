"""Stage E v3: Hierarchical part-rigid + LBS (Gemini's path-3 recommendation).

K-means cluster the arm Gaussians into K_arm sub-parts. Each sub-part has its
own (translation, rotation) SE(3) trajectory. LBS over K_arm clusters with
Gaussian-kernel weights based on distance to cluster center. Body Gaussians
stay static. Optional time-varying color tint (4D-SH lite).

DOF budget:
    K_arm sub-parts × T × 6 SE(3) = 10 × 21 × 6 = 1260 motion DOF
    + (optional) T × 3 color tint = 63 appearance DOF
    Total: ~1.3k DOF (vs vanilla 16M, vs our previous 126)

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/train_partrigid_hier.py \\
        --iterations 8000 --label hier_v1 --k_arm 10
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

CANON_PLY_DEFAULT = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PART_DIR_DEFAULT = REPO_ROOT / "runs_aux" / "part_assignment"
SCENE_DEFAULT = REPO_ROOT / "data" / "custom" / "scene00_masked"


def axis_angle_to_quaternion(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 4) unit quaternion (w, x, y, z)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    half = theta * 0.5
    s = half.sin()
    w = half.cos()
    return torch.cat([w, axis * s], dim=-1)


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


def kmeans_simple(x: np.ndarray, K: int, n_iter: int = 50, seed: int = 0):
    """Simple Lloyd's k-means."""
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(x.shape[0], K, replace=False)]
    for _ in range(n_iter):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.stack([
            x[labels == k].mean(0) if (labels == k).sum() > 0 else centers[k]
            for k in range(K)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels, centers


class HierarchicalPartRigidModel(nn.Module):
    def __init__(self, T: int, K_arm: int, arm_centers: torch.Tensor,
                 sub_trans_init: torch.Tensor, color_tint: bool = False,
                 per_time_scale: bool = False,
                 xyz_residual_n: int = 0):
        super().__init__()
        self.T = T
        self.K = K_arm
        # Per-(k, t) translation + axis-angle. Init: translation = full-arm
        # centroid sweep (every cluster shares same init for now). Rotation = 0.
        self.trans = nn.Parameter(sub_trans_init.clone().float())  # (K, T, 3)
        self.aa = nn.Parameter(torch.zeros(K_arm, T, 3, dtype=torch.float32))
        self.register_buffer("centers", arm_centers.float())  # (K, 3)
        # Optional time-varying color tint (additive, single RGB per time)
        self.color_tint_enabled = color_tint
        if color_tint:
            self.color_tint = nn.Parameter(torch.zeros(T, 3, dtype=torch.float32))
        # Optional per-(cluster, time) 3D scale residual — added to Gaussian
        # _scaling via LBS-weighted blend. Addresses streaking artifact when
        # canonical Gaussian shape doesn't track cluster rotation.
        self.per_time_scale_enabled = per_time_scale
        if per_time_scale:
            self.scale = nn.Parameter(torch.zeros(K_arm, T, 3, dtype=torch.float32))
        # Per-Gaussian per-time XYZ residual (only for arm Gaussians)
        self.xyz_residual_enabled = xyz_residual_n > 0
        if self.xyz_residual_enabled:
            self.xyz_residual = nn.Parameter(torch.zeros(xyz_residual_n, T, 3, dtype=torch.float32))
        # Per-Gaussian per-time ROTATION residual (axis-angle, only for arm Gaussians)
        # Initialized at 0 → identity rotation
        self.rot_residual_n = xyz_residual_n  # reuse for compat
        self.rot_residual_enabled = False
        # Will be lazily set up by main() if --use_rot_residual

    def deform_arm(self, t: int, arm_xyz: torch.Tensor, arm_weights: torch.Tensor) -> torch.Tensor:
        """LBS over K clusters with canonical fallback for sub-unity weights.
        arm_xyz:     (N_arm, 3) canonical positions of arm Gaussians
        arm_weights: (N_arm, K) soft weights (not necessarily summing to 1)
        Returns: (N_arm, 3) deformed positions.
        """
        R_all = axis_angle_to_matrix(self.aa[:, t, :])   # (K, 3, 3)
        T_all = self.trans[:, t, :]                       # (K, 3)
        rel = arm_xyz.unsqueeze(1) - self.centers.unsqueeze(0)  # (N_arm, K, 3)
        rotated = torch.einsum("kij,nkj->nki", R_all, rel)      # (N_arm, K, 3)
        new_per_cluster = rotated + self.centers.unsqueeze(0) + T_all.unsqueeze(0)  # (N_arm, K, 3)
        weighted = (arm_weights.unsqueeze(-1) * new_per_cluster).sum(dim=1)  # (N_arm, 3)
        # Canonical fallback for weight residual (lerp between deformed and canonical,
        # NOT between deformed and origin). Without this, boundary Gaussians collapse to origin.
        w_total = arm_weights.sum(dim=1, keepdim=True).clamp(min=0, max=1)  # (N_arm, 1)
        return weighted + (1 - w_total) * arm_xyz


def silhouette_loss(render_alpha, gt_alpha, outside_weight: float = 1.0):
    """Silhouette loss with optional outside-mask down-weighting.

    outside_weight=1.0 (default): symmetric penalty everywhere (legacy behavior).
    outside_weight<1: penalty for rendering OUTSIDE gt_alpha mask is reduced.
      Use case: when canonical contains structure (e.g., baseplate) that the
      SAM-2 supervision mask excludes — we don't want silhouette gradient to
      fight the canonical's existing geometry. GENERIC for any
      "supervision-mask is subset of canonical-scene" mismatch.
    """
    a = render_alpha.clamp(1e-6, 1 - 1e-6)
    g = gt_alpha.clamp(1e-6, 1 - 1e-6)
    if outside_weight < 1.0:
        fg = (gt_alpha > 0.5).float()
        w = fg + (1 - fg) * outside_weight
        bce = -((g * a.log() + (1 - g) * (1 - a).log()) * w).mean()
        inter = (a * g).sum()
        union = (w * (a + g - a * g)).sum()
        return bce + (1 - inter / (union + 1e-6))
    else:
        bce = -(g * a.log() + (1 - g) * (1 - a).log()).mean()
        inter = (a * g).sum()
        union = (a + g - a * g).sum()
        return bce + (1 - inter / (union + 1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=8000)
    p.add_argument("--label", default="hier_v1")
    p.add_argument("--k_arm", type=int, default=10, help="number of arm sub-parts")
    p.add_argument("--lbs_K", type=int, default=3, help="LBS K-nearest for soft weights")
    p.add_argument("--lbs_sigma", type=float, default=0.15,
                   help="LBS kernel temperature (in scene units)")
    p.add_argument("--lr_trans", type=float, default=2e-3)
    p.add_argument("--lr_rot", type=float, default=5e-3)
    p.add_argument("--lr_color", type=float, default=1e-3)
    p.add_argument("--lam_silh", type=float, default=1.0)
    p.add_argument("--silh_outside_weight", type=float, default=1.0,
                   help="Weight for silhouette penalty OUTSIDE gt_alpha mask. "
                        "Set to <1 (e.g., 0.1) when canonical has structure outside "
                        "the SAM mask (e.g., baseplate not in mask) to avoid silhouette "
                        "fighting canonical. GENERIC, not lego-specific.")
    p.add_argument("--lam_traj", type=float, default=0.1)
    p.add_argument("--arap_cross_part", type=float, default=1.0,
                   help="ARAP weight multiplier for cluster pairs in DIFFERENT motion "
                        "parts (joints). 1.0 = legacy uniform; e.g. 0.1 lets joints "
                        "articulate while limbs stay rigid. Needs multi-part part_dir.")
    p.add_argument("--zero_traj_init", action="store_true",
                   help="L1b diagnostic: init per-cluster translations to zero instead "
                        "of the Stage D centroid trajectory (fully removes Stage D).")
    p.add_argument("--lam_smooth", type=float, default=1.0)
    p.add_argument("--lam_arap", type=float, default=0.5,
                   help="ARAP-like inter-cluster smoothness (penalises adjacent sub-parts disagreeing)")
    p.add_argument("--lam_photo_blur", type=float, default=0.0)
    p.add_argument("--blur_sigma", type=float, default=6.0)
    p.add_argument("--erode_ksize", type=int, default=21)
    p.add_argument("--use_color_tint", action="store_true",
                   help="Per-time global color tint (cheap 4D-SH approximation)")
    p.add_argument("--use_rot_prop", action="store_true",
                   help="Apply per-cluster rotation to Gaussian quaternion via LBS (Tier 1 free fix)")
    p.add_argument("--lam_photo_smart", type=float, default=0.0,
                   help="Smart photometric loss weight: L1(pred, gt) * filter_weight, "
                        "where filter = exp(-alpha * |gt - v5_render|). "
                        "Suppresses pixels where v5 fits-all canonical disagrees with GT "
                        "(likely VGM artifacts).")
    p.add_argument("--photo_smart_alpha", type=float, default=8.0,
                   help="Filter sharpness for smart photometric")
    p.add_argument("--view_reliability_beta", type=float, default=0.0,
                   help="Cone-weighted supervision (poster takeaway #1): per-view "
                        "photometric trust = exp(-beta*(az_dist/180 + 0.5*elev/30)). "
                        "0 = uniform (current). >0 down-weights off-axis noisy views so "
                        "the part-rigid prior extrapolates them from reliable views. "
                        "Attacks the SV4D-noise bottleneck (oracle gap), not model capacity.")
    p.add_argument("--view_reliability_on_silh", action="store_true",
                   help="Also apply the reliability weight to the silhouette loss "
                        "(off-axis alpha drifts too on hellwarrior). Default: photo only.")
    p.add_argument("--v5_render_dir", type=str,
                   default="outputs/custom/scene00_v5_node/train/ours_30000/renders",
                   help="Per-(view, time) renders from the fits-all v5 canonical (§3) "
                        "OR the d-3dgs clean reference for lego_v2")
    p.add_argument("--d_rot_zero", action="store_true",
                   help="Correct identity rotation residual (d_rot=0). Legacy default subtracts (1,0,0,0) from raw quats, scrambling orientations.")
    p.add_argument("--motion_gated_smart_photo", action="store_true",
                   help="When set, smart-photo weight applies only on static pixels "
                        "(motion mask == 0). Moving pixels get weight = 1. Useful when "
                        "reference is a static canonical render (Option B leak-free).")
    p.add_argument("--canon_ply",  default=None, help="Override canonical ply path")
    p.add_argument("--part_dir",   default=None, help="Override part assignment dir")
    p.add_argument("--scene_dir",  default=None, help="Override scene dir (with transforms_train.json)")
    p.add_argument("--use_test_too", action="store_true",
                   help="Also include test split frames in training (for lego_v2 where we eval against d-3dgs)")
    p.add_argument("--use_per_time_scale", action="store_true",
                   help="Per-(cluster, time) 3D scale residual (+K*T*3 DOF). "
                        "Lets Gaussians in each cluster stretch differently at each time. "
                        "Addresses streaking when canonical shape doesn't track rotation.")
    p.add_argument("--lr_scale", type=float, default=1e-3)
    p.add_argument("--lam_scale_smooth", type=float, default=1.0,
                   help="Temporal smoothness for per-time scale residual")
    p.add_argument("--use_xyz_residual", action="store_true",
                   help="Per-Gaussian per-time XYZ residual (+N_arm*T*3 DOF). "
                        "Adds local micro-deformation on top of cluster SE(3). Directly "
                        "addresses bucket streaking when SE(3)+scale is insufficient.")
    p.add_argument("--lr_xyz_res", type=float, default=5e-4)
    p.add_argument("--lam_xyz_res_smooth", type=float, default=5.0,
                   help="Temporal smoothness for per-Gaussian XYZ residual (high to prevent overfit)")
    p.add_argument("--lam_xyz_res_l2", type=float, default=1.0,
                   help="L2 regularizer on XYZ residual magnitude (keep small)")
    p.add_argument("--lam_lpips", type=float, default=0.0,
                   help="LPIPS perceptual loss weight (sharpens edges, fixes blur from smart photo)")
    p.add_argument("--canon_finetune", action="store_true",
                   help="Unfreeze canonical scale/rotation/features for mild adaptation (Path 2)")
    p.add_argument("--lr_canon", type=float, default=1e-4,
                   help="Learning rate for canonical fine-tune (very small)")
    p.add_argument("--use_rot_residual", action="store_true",
                   help="Per-Gaussian per-time ROTATION residual axis-angle (+N_arm*T*3 DOF). "
                        "Rotates each arm Gaussian's orientation to track arm rotation. "
                        "Should fix anisotropic streaking when bucket rotates.")
    p.add_argument("--lr_rot_res", type=float, default=5e-4)
    p.add_argument("--lam_rot_res_smooth", type=float, default=5.0,
                   help="Temporal smoothness for per-Gaussian rotation residual")
    p.add_argument("--lam_rot_res_l2", type=float, default=2.0,
                   help="L2 regularizer on rotation residual magnitude (keep rotations small)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = REPO_ROOT / "outputs/custom" / f"partrigid_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Resolve paths (CLI overrides) + load canonical Gaussians (frozen) =====
    CANON_PLY = Path(args.canon_ply) if args.canon_ply else CANON_PLY_DEFAULT
    PART_DIR  = Path(args.part_dir)  if args.part_dir  else PART_DIR_DEFAULT
    SCENE     = Path(args.scene_dir) if args.scene_dir else SCENE_DEFAULT
    print(f"[hier] CANON={CANON_PLY.name}  PART={PART_DIR.name}  SCENE={SCENE.name}")
    print(f"[hier] loading canonical")
    # Try fea_dim=8 then 2 then 0 (different canonicals have different feature dims)
    gaussians = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(CANON_PLY), og_number_points=0)
            gaussians = g
            print(f"[hier] loaded canonical with fea_dim={fdim}")
            break
        except (IndexError, RuntimeError, ValueError):
            continue
    if gaussians is None:
        raise RuntimeError(f"failed to load canonical {CANON_PLY}")
    for attr in ["_xyz", "_features_dc", "_features_rest",
                 "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)
    # Path 2: unfreeze scale/rotation/features for canonical fine-tune
    if args.canon_finetune:
        gaussians._scaling.requires_grad_(True)
        gaussians._rotation.requires_grad_(True)
        gaussians._features_dc.requires_grad_(True)
        print(f"[hier] Path 2: canonical fine-tune ENABLED (scale/rot/features unfrozen, lr={args.lr_canon})")
    xyz_canon = gaussians.get_xyz.detach().to(args.device)
    N = xyz_canon.shape[0]

    # Time-varying color tint optionally unfreezes _features_dc
    if args.use_color_tint:
        gaussians._features_dc.requires_grad_(True)
        f_dc_canonical = gaussians._features_dc.detach().clone()

    # ===== LBS weights (global arm vs body) + 3D trajectory =====
    arm_weights_global = np.load(PART_DIR / "gaussian_arm_weights.npy")
    centroid_3d = np.load(PART_DIR / "part_centroid_3d.npy")  # (T, P+1, 3); last entry = body
    conf = np.load(PART_DIR / "part_centroid_confidence.npy")
    T = centroid_3d.shape[0]
    n_motion_parts = centroid_3d.shape[1] - 1
    gauss_part_np = None
    if n_motion_parts > 1:
        gauss_part_np = np.load(PART_DIR / "gaussian_motion_part.npy")  # (N,) in {-1, 0..P-1}
        print(f"[hier] multi-part Stage D: {n_motion_parts} motion parts "
              f"(labelled Gaussians: {(gauss_part_np >= 0).sum()})")
    print(f"[hier] N={N}, T={T}, mean global arm_weight={arm_weights_global.mean():.3f}")

    # ===== K-means cluster the arm Gaussians =====
    arm_mask_np = arm_weights_global > 0.5  # hard threshold for clustering
    arm_xyz_np = xyz_canon[arm_mask_np].cpu().numpy()
    print(f"[hier] arm Gaussians (w>0.5): {arm_mask_np.sum()}")
    labels, centers = kmeans_simple(arm_xyz_np, K=args.k_arm, n_iter=50, seed=args.seed)
    print(f"[hier] k-means done. cluster sizes: {[int((labels == k).sum()) for k in range(args.k_arm)]}")
    arm_centers_t = torch.from_numpy(centers).float().to(args.device)

    # ===== Build per-Gaussian LBS weights over K clusters =====
    # For each Gaussian, distance to each cluster center -> softmax / Gaussian kernel
    # Body Gaussians (arm_weight_global < 0.5) get zero weight on all clusters (so deformation = 0)
    arm_w_global_t = torch.from_numpy(arm_weights_global).float().to(args.device)
    # Compute per-Gaussian distances to all cluster centers
    diff = xyz_canon.unsqueeze(1) - arm_centers_t.unsqueeze(0)  # (N, K, 3)
    dists = diff.norm(dim=-1)                                   # (N, K)
    # Gaussian-kernel + only keep top K_lbs nearest clusters
    K_lbs = args.lbs_K
    sigma = args.lbs_sigma
    raw_w = torch.exp(-(dists ** 2) / (2 * sigma ** 2))         # (N, K)
    # Mask all but top K_lbs nearest
    topk = raw_w.topk(K_lbs, dim=1)
    mask_keep = torch.zeros_like(raw_w)
    mask_keep.scatter_(1, topk.indices, 1.0)
    raw_w = raw_w * mask_keep
    # Normalise per Gaussian
    lbs_weights = raw_w / raw_w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, K)
    # Modulate by global arm weight (body Gaussians have ~0 sum, so deformation -> 0)
    lbs_weights = lbs_weights * arm_w_global_t.unsqueeze(1)              # (N, K)

    print(f"[hier] LBS weights stats: mean sum per Gaussian = {lbs_weights.sum(1).mean():.3f} "
          f"(body ~0, arm ~1)")

    # ===== Init translation from centroid trajectory (per part if multi-part) =====
    centroid_3d_t = torch.tensor(centroid_3d, dtype=torch.float32, device=args.device)
    part_of_cluster = None
    if n_motion_parts > 1:
        # Majority-vote a motion part per k-means cluster (used for per-part
        # trajectory init and for part-aware ARAP).
        arm_part_labels = gauss_part_np[arm_mask_np]  # aligned with kmeans `labels`
        part_of_cluster = np.zeros(args.k_arm, dtype=np.int64)
        for k in range(args.k_arm):
            member = arm_part_labels[labels == k]
            member = member[member >= 0]
            if member.size > 0:
                part_of_cluster[k] = np.bincount(member, minlength=n_motion_parts).argmax()
            else:
                d0 = np.linalg.norm(centroid_3d[0, :n_motion_parts] - centers[k], axis=1)
                part_of_cluster[k] = int(d0.argmin())
        counts = np.bincount(part_of_cluster, minlength=n_motion_parts)
        print(f"[hier] clusters per motion part: {counts.tolist()}")
    if args.zero_traj_init:
        sub_trans_init = torch.zeros(args.k_arm, T, 3, device=args.device)
        print("[hier] zero_traj_init: Stage D trajectory NOT used for init")
    elif n_motion_parts > 1:
        sub_trans_init = torch.stack([
            centroid_3d_t[:, int(p)] - centroid_3d_t[0, int(p)] for p in part_of_cluster
        ])  # (K, T, 3)
    else:
        arm_trans_global = centroid_3d_t[:, 0] - centroid_3d_t[0, 0]  # (T, 3) for the WHOLE arm
        sub_trans_init = arm_trans_global.unsqueeze(0).expand(args.k_arm, T, 3).contiguous()  # (K, T, 3)

    # Pre-compute fixed per-part trajectory-loss weights (multi-part Stage D)
    traj_part_w = None
    if n_motion_parts > 1:
        gauss_part_t = torch.from_numpy(gauss_part_np).long().to(args.device)
        w_base = lbs_weights.sum(1)
        traj_part_w = []
        for p_i in range(n_motion_parts):
            w_p = w_base * (gauss_part_t == p_i).float()
            traj_part_w.append((w_p, float(w_p.sum().item())))

    # Index arm Gaussians (lbs sum > 1e-4) for per-Gaussian XYZ residual
    nontrivial_mask_np = (lbs_weights.sum(1) > 1e-4).cpu().numpy()
    arm_idx_for_residual = np.where(nontrivial_mask_np)[0]
    N_arm_residual = len(arm_idx_for_residual)
    arm_idx_tensor = torch.from_numpy(arm_idx_for_residual).long().to(args.device)
    print(f"[hier] arm Gaussians eligible for XYZ residual: {N_arm_residual}")

    n_residual_pool = N_arm_residual if (args.use_xyz_residual or args.use_rot_residual) else 0
    model = HierarchicalPartRigidModel(T=T, K_arm=args.k_arm,
                                        arm_centers=arm_centers_t,
                                        sub_trans_init=sub_trans_init,
                                        color_tint=args.use_color_tint,
                                        per_time_scale=args.use_per_time_scale,
                                        xyz_residual_n=n_residual_pool).to(args.device)
    # Set up rotation residual parameter (axis-angle, N_arm × T × 3)
    if args.use_rot_residual:
        model.rot_residual = nn.Parameter(
            torch.zeros(N_arm_residual, T, 3, dtype=torch.float32).to(args.device))
        model.rot_residual_enabled = True
    param_groups = [
        {"params": [model.trans], "lr": args.lr_trans},
        {"params": [model.aa],    "lr": args.lr_rot},
    ]
    if args.use_color_tint:
        param_groups.append({"params": [model.color_tint], "lr": args.lr_color})
    if args.use_per_time_scale:
        param_groups.append({"params": [model.scale], "lr": args.lr_scale})
    if args.use_xyz_residual:
        param_groups.append({"params": [model.xyz_residual], "lr": args.lr_xyz_res})
    if args.use_rot_residual:
        param_groups.append({"params": [model.rot_residual], "lr": args.lr_rot_res})
    # Path 2: canonical fine-tune params with small lr
    if args.canon_finetune:
        param_groups.append({"params": [gaussians._scaling], "lr": args.lr_canon})
        param_groups.append({"params": [gaussians._rotation], "lr": args.lr_canon})
        param_groups.append({"params": [gaussians._features_dc], "lr": args.lr_canon})
    optim = torch.optim.Adam(param_groups)

    n_motion_dof = args.k_arm * T * 6
    n_appear_dof = T * 3 if args.use_color_tint else 0
    print(f"[hier] DOF: motion={n_motion_dof}  appearance={n_appear_dof}  total={n_motion_dof+n_appear_dof}")

    # ===== Cameras =====
    data = json.loads((SCENE / "transforms_train.json").read_text())
    # Optionally merge test frames into training set (when eval is against external GT, not split test)
    if args.use_test_too:
        test_meta_path = SCENE / "transforms_test.json"
        if test_meta_path.exists():
            test_meta = json.loads(test_meta_path.read_text())
            data["frames"] = data["frames"] + test_meta["frames"]
            print(f"[hier] merged {len(test_meta['frames'])} test frames into train (total {len(data['frames'])})")
    fov_x = data["camera_angle_x"]
    H = W = 576
    cams = []
    gt_alphas = []
    gt_rgbs = []
    times = []
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)
    for i, f in enumerate(data["frames"]):
        ti = int(f["frame_idx"])
        # Try train/ then test/ (file_path may use either after split)
        png_name = f"{Path(f['file_path']).name}.png"
        for split in ("train", "test"):
            cand = SCENE / split / png_name
            if cand.exists():
                png = cand
                break
        else:
            raise FileNotFoundError(f"frame not found: {png_name}")
        rgba = np.asarray(iio.imread(png))
        alpha = (rgba[..., 3] > 127).astype(np.float32)
        rgb = (rgba[..., :3].astype(np.float32) / 255.0)
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

    # ===== Pre-compute smart-photometric filter weights from v5 fits-all canonical =====
    # When --motion_gated_smart_photo is set: weight only applies to static pixels
    # (m_v == 0). On moving pixels weight = 1 (no filter), so motion learns from
    # raw photo signal without being suppressed by a static reference.
    smart_photo_weights = None
    if args.lam_photo_smart > 0:
        v5_dir = REPO_ROOT / args.v5_render_dir
        if not v5_dir.exists():
            raise FileNotFoundError(f"v5 render dir missing: {v5_dir}")
        # Optional per-view static motion mask (precomputed by motion_parts_generic.py)
        per_view_motion = {}
        if args.motion_gated_smart_photo:
            # Recompute per-view static motion mask = pixels that move at any t.
            # Use SAME logic as motion_parts_generic: temporal std + Otsu inside FG.
            from skimage.filters import threshold_otsu
            view_set = sorted({int(f["view_idx"]) for f in data["frames"]})
            for v in view_set:
                frames_v = [None] * T
                for f in data["frames"]:
                    if int(f["view_idx"]) != v: continue
                    ti = int(f["frame_idx"])
                    flat_idx = v * T + ti
                    candidate = SCENE / "train" / f"r_{flat_idx:05d}.png"
                    if not candidate.exists():
                        candidate = SCENE / "test" / f"r_{flat_idx:05d}.png"
                    rgba = np.asarray(iio.imread(candidate), dtype=np.float32) / 255.0
                    a = rgba[..., 3:4] if rgba.shape[-1] == 4 else np.ones_like(rgba[..., :1])
                    rgb = rgba[..., :3] * a + 1.0 * (1 - a)
                    frames_v[ti] = (rgb, a[..., 0])
                # Stack
                stk = np.stack([fr[0] for fr in frames_v if fr is not None], axis=0)
                alp = np.stack([fr[1] for fr in frames_v if fr is not None], axis=0)
                std = stk.std(axis=0).mean(axis=-1)  # (H, W)
                fg_any = alp.max(axis=0) > 0.5
                vals = std[fg_any]
                thresh = float(threshold_otsu(vals)) if vals.size > 0 else 0.05
                moving = (std > thresh) & fg_any
                frac = moving.sum() / max(fg_any.sum(), 1)
                if frac > 0.7 or frac < 0.05:  # safety clamp
                    thresh = float(np.percentile(vals, 70)) if vals.size > 0 else 0.05
                    moving = (std > thresh) & fg_any
                per_view_motion[v] = moving.astype(np.float32)
                print(f"  [motion-gated] view {v}: moving_frac={moving.sum()/max(fg_any.sum(),1):.3f}")
        smart_photo_weights = []
        for i, f in enumerate(data["frames"]):
            v = int(f["view_idx"]); ti = int(f["frame_idx"])
            flat_idx = v * T + ti
            v5_path = v5_dir / f"{flat_idx:05d}.png"
            if not v5_path.exists():
                raise FileNotFoundError(f"missing v5 render: {v5_path}")
            v5_rgba = np.asarray(iio.imread(v5_path), dtype=np.float32) / 255.0
            v5_alpha = v5_rgba[..., 3:4] if v5_rgba.shape[-1] == 4 else np.ones_like(v5_rgba[..., :1])
            v5_rgb = v5_rgba[..., :3] * v5_alpha + 1.0 * (1 - v5_alpha)
            gt_rgb_np = gt_rgbs[i].permute(1, 2, 0).cpu().numpy()
            residual = np.abs(gt_rgb_np - v5_rgb).mean(axis=-1)  # (H, W)
            weight = np.exp(-args.photo_smart_alpha * residual)
            if args.motion_gated_smart_photo and v in per_view_motion:
                weight = np.where(per_view_motion[v] > 0.5, 1.0, weight).astype(np.float32)
            smart_photo_weights.append(torch.from_numpy(weight.astype(np.float32)).to(args.device))
        # Diagnostics
        avg_weight = float(torch.stack(smart_photo_weights).mean())
        print(f"[hier] smart photo: pre-built {len(smart_photo_weights)} weight maps  "
              f"(mean weight={avg_weight:.3f}, low = filtered-as-artifact; "
              f"motion-gated={args.motion_gated_smart_photo})")

    # ===== Per-frame view-reliability weight (cone-weighted supervision) =====
    # Implements poster takeaway #1: trust views by their position in the
    # reliability cone. Off-axis = noisy (SV4D's failure axis) -> low weight ->
    # the part-rigid prior fills them in from reliable near-input views.
    view_rel_w = torch.ones(len(cams), device=args.device)
    if args.view_reliability_beta > 0:
        for i, f in enumerate(data["frames"]):
            az = float(f.get("azimuth_deg", 0.0)) % 360.0
            az_dist = min(az, 360.0 - az) / 180.0           # [0,1]
            el = abs(float(f.get("elevation_deg", 0.0))) / 30.0  # [0,1]
            view_rel_w[i] = float(np.exp(-args.view_reliability_beta * (az_dist + 0.5 * el)))
        uniq = {}
        for i, f in enumerate(data["frames"]):
            uniq[float(f.get("azimuth_deg", 0.0))] = float(view_rel_w[i])
        print(f"[hier] view-reliability (beta={args.view_reliability_beta}): "
              f"az->weight " + ", ".join(f"{a:.0f}:{w:.2f}" for a, w in sorted(uniq.items())))

    print(f"[hier] training {args.iterations} iters with {len(cams)} cameras")
    t0 = time.time()
    arm_mask_t = arm_w_global_t > 0.5  # use canonical arm-mask for ARAP

    # Precompute K-nearest cluster adjacency for ARAP (neighbouring clusters)
    cluster_dist = torch.cdist(arm_centers_t, arm_centers_t)  # (K, K)
    n_neigh = min(3, args.k_arm)
    cluster_neigh = cluster_dist.topk(n_neigh, dim=1, largest=False).indices  # (K, n_neigh)
    # Part-aware ARAP: full strength within a motion part, down-weighted across
    # part boundaries (joints) so limbs stay rigid but joints can articulate.
    arap_pair_w = torch.ones(args.k_arm, cluster_neigh.shape[1], device=args.device)
    if args.arap_cross_part != 1.0 and part_of_cluster is not None:
        poc = torch.from_numpy(part_of_cluster).to(args.device)
        for k_neigh in range(1, cluster_neigh.shape[1]):
            same = poc == poc[cluster_neigh[:, k_neigh]]
            arap_pair_w[:, k_neigh] = torch.where(
                same, torch.ones_like(arap_pair_w[:, 0]),
                torch.full_like(arap_pair_w[:, 0], args.arap_cross_part))
        n_cross = int((arap_pair_w[:, 1:] < 1).sum().item())
        print(f"[hier] part-aware ARAP: {n_cross}/{args.k_arm * (cluster_neigh.shape[1]-1)} "
              f"cross-part pairs down-weighted to {args.arap_cross_part}")

    for it in range(1, args.iterations + 1):
        idx = np.random.randint(len(cams))
        cam = cams[idx]
        t = int(times[idx])

        # Compute new positions for arm Gaussians via LBS
        # For efficiency only compute on arm_mask Gaussians (rest are static)
        new_xyz = xyz_canon.clone()
        # Apply LBS over all clusters for the moving (arm) Gaussians
        # The body Gaussians have lbs_weights summing to ~0 so the SE(3) contribution is ~0
        # For numerical efficiency, compute deformation only where lbs_weights sum > epsilon
        nontrivial = lbs_weights.sum(1) > 1e-4
        if nontrivial.sum() > 0:
            new_pos = model.deform_arm(t, xyz_canon[nontrivial], lbs_weights[nontrivial])
            new_xyz = new_xyz.clone()
            new_xyz[nontrivial] = new_pos
        d_xyz = new_xyz - xyz_canon
        # Per-Gaussian per-time XYZ residual (only on arm-eligible Gaussians)
        if args.use_xyz_residual:
            d_xyz_clone = d_xyz.clone()
            d_xyz_clone[arm_idx_tensor] = d_xyz_clone[arm_idx_tensor] + model.xyz_residual[:, t, :]
            d_xyz = d_xyz_clone
        d_rotation = torch.zeros(N, 4, device=args.device)
        if not args.d_rot_zero:
            d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device=args.device)
        d_scaling = torch.zeros(N, 3, device=args.device)
        # Per-(cluster, time) scale residual via LBS-weighted blend
        if args.use_per_time_scale:
            scale_blend = lbs_weights @ model.scale[:, t, :]      # (N, 3)
            d_scaling = d_scaling + scale_blend

        # Tier 1 — rotation propagation: per-Gaussian quaternion from LBS-weighted
        # cluster rotations, applied via d_rotation_bias (multiplicative composition).
        d_rotation_bias = None
        if args.use_rot_prop:
            q_clusters = axis_angle_to_quaternion(model.aa[:, t, :])    # (K, 4)
            q_blend = lbs_weights @ q_clusters                           # (N, 4)
            body_w = (1 - lbs_weights.sum(dim=1, keepdim=True)).clamp(min=0)
            identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args.device)
            q_blend = q_blend + body_w * identity_q
            d_rotation_bias = q_blend / q_blend.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Per-Gaussian rotation residual (axis-angle → quaternion, only arm Gaussians)
        if args.use_rot_residual:
            q_res = axis_angle_to_quaternion(model.rot_residual[:, t, :])  # (N_arm, 4)
            identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args.device)
            q_full = identity_q.unsqueeze(0).expand(N, 4).clone()
            q_full[arm_idx_tensor] = q_res
            if d_rotation_bias is None:
                d_rotation_bias = q_full
            else:
                # Compose: q_full * d_rotation_bias (quaternion multiplication)
                from scene.gaussian_model import quaternion_multiply
                d_rotation_bias = quaternion_multiply(q_full, d_rotation_bias)
                d_rotation_bias = d_rotation_bias / d_rotation_bias.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Optional color tint
        if args.use_color_tint:
            # set per-time tint on the canonical _features_dc
            tint = model.color_tint[t]  # (3,)
            gaussians._features_dc.data = f_dc_canonical + tint[None, None, :]

        pkg = render(cam, gaussians, pipe, background,
                     d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling,
                     d_rot_as_res=True, d_rotation_bias=d_rotation_bias)
        img = pkg["render"]
        alpha = pkg["alpha"]
        gt_alpha = gt_alphas[idx]
        L_silh = silhouette_loss(alpha[0], gt_alpha, outside_weight=args.silh_outside_weight)
        if args.view_reliability_on_silh:
            L_silh = L_silh * view_rel_w[idx]

        # Trajectory loss: cluster-union centroid tracks Stage D target
        # (per motion part when multi-part Stage D is available)
        if n_motion_parts > 1:
            L_traj = torch.tensor(0.0, device=args.device)
            for p_i, (w_p, denom) in enumerate(traj_part_w):
                if denom < 1e-6:
                    continue
                pred = (w_p.unsqueeze(1) * new_xyz).sum(0) / denom
                L_traj = L_traj + float(conf[t, p_i]) * ((pred - centroid_3d_t[t, p_i]) ** 2).sum()
        else:
            target = centroid_3d_t[t, 0]
            cc = conf[t, 0]
            pred = (lbs_weights.sum(1).unsqueeze(1) * new_xyz).sum(0) / lbs_weights.sum(1).sum().clamp(min=1e-6)
            L_traj = float(cc) * ((pred - target) ** 2).sum()

        # Temporal smoothness (per cluster)
        L_smooth = ((model.trans[:, 1:, :] - model.trans[:, :-1, :]) ** 2).mean() + \
                   ((model.aa[:, 1:, :] - model.aa[:, :-1, :]) ** 2).mean()
        if args.use_color_tint:
            L_smooth = L_smooth + ((model.color_tint[1:] - model.color_tint[:-1]) ** 2).mean()
        # Temporal smoothness for per-time scale (avoid jitter)
        L_scale_smooth = torch.tensor(0.0, device=args.device)
        if args.use_per_time_scale:
            L_scale_smooth = ((model.scale[:, 1:, :] - model.scale[:, :-1, :]) ** 2).mean()
        # Per-Gaussian XYZ residual: smoothness + L2 magnitude regularizers
        L_xyz_res_smooth = torch.tensor(0.0, device=args.device)
        L_xyz_res_l2 = torch.tensor(0.0, device=args.device)
        if args.use_xyz_residual:
            L_xyz_res_smooth = ((model.xyz_residual[:, 1:, :] - model.xyz_residual[:, :-1, :]) ** 2).mean()
            L_xyz_res_l2 = (model.xyz_residual ** 2).mean()
        L_rot_res_smooth = torch.tensor(0.0, device=args.device)
        L_rot_res_l2 = torch.tensor(0.0, device=args.device)
        if args.use_rot_residual:
            L_rot_res_smooth = ((model.rot_residual[:, 1:, :] - model.rot_residual[:, :-1, :]) ** 2).mean()
            L_rot_res_l2 = (model.rot_residual ** 2).mean()

        # ARAP-like: adjacent clusters should not differ too much
        # (pair weights down-weight cross-part joints when --arap_cross_part < 1)
        L_arap = 0.0
        for k_neigh in range(1, cluster_neigh.shape[1]):
            nbrs = cluster_neigh[:, k_neigh]  # (K,) indices
            w_pair = arap_pair_w[:, k_neigh].view(-1, 1, 1)
            L_arap = L_arap + (w_pair * (model.trans - model.trans[nbrs]) ** 2).mean()
            L_arap = L_arap + (w_pair * (model.aa - model.aa[nbrs]) ** 2).mean()
        L_arap = L_arap / max(cluster_neigh.shape[1] - 1, 1)

        # Optional blurred photometric
        L_photo = torch.tensor(0.0, device=args.device)
        if args.lam_photo_blur > 0:
            gt_rgb = gt_rgbs[idx]
            if args.blur_sigma > 0.5:
                # simple separable Gaussian blur
                sz = max(int(2 * round(2.5 * args.blur_sigma)) + 1, 3)
                x = torch.arange(sz, device=args.device, dtype=torch.float32) - sz // 2
                kk = torch.exp(-0.5 * (x / args.blur_sigma) ** 2); kk = kk / kk.sum()
                kh = kk.view(1, 1, 1, sz).expand(3, 1, 1, sz)
                kv = kk.view(1, 1, sz, 1).expand(3, 1, sz, 1)
                rd = F.conv2d(img.unsqueeze(0), kh, padding=(0, sz // 2), groups=3)
                rd = F.conv2d(rd, kv, padding=(sz // 2, 0), groups=3).squeeze(0)
                gd = F.conv2d(gt_rgb.unsqueeze(0), kh, padding=(0, sz // 2), groups=3)
                gd = F.conv2d(gd, kv, padding=(sz // 2, 0), groups=3).squeeze(0)
            else:
                rd = img; gd = gt_rgb
            if args.erode_ksize > 1:
                eroded = -F.max_pool2d(-gt_alpha.unsqueeze(0).unsqueeze(0),
                                        kernel_size=args.erode_ksize, stride=1,
                                        padding=args.erode_ksize // 2).squeeze(0).squeeze(0)
            else:
                eroded = gt_alpha
            mask3 = eroded.unsqueeze(0).expand_as(rd)
            L_photo = ((rd - gd).abs() * mask3).sum() / mask3.sum().clamp(min=1)

        # Smart photometric (artifact-filtered L1) — Gemini path 2 with VGM filter
        L_photo_smart = torch.tensor(0.0, device=args.device)
        if args.lam_photo_smart > 0 and smart_photo_weights is not None:
            gt_rgb = gt_rgbs[idx]
            w_pix = smart_photo_weights[idx]                          # (H, W) confidence
            # Restrict to FG via gt_alpha to avoid penalizing white background
            fg_w = w_pix * gt_alpha                                   # (H, W)
            err = (img - gt_rgb).abs().mean(dim=0)                    # (H, W)
            L_photo_smart = (err * fg_w).sum() / fg_w.sum().clamp(min=1)
            L_photo_smart = L_photo_smart * view_rel_w[idx]

        # LPIPS perceptual loss (alex backbone, robust to small offsets + sharpens edges)
        L_lpips = torch.tensor(0.0, device=args.device)
        if args.lam_lpips > 0:
            if not hasattr(args, "_lpips_model"):
                import lpips
                args._lpips_model = lpips.LPIPS(net="alex").to(args.device)
                print(f"[hier] LPIPS model loaded")
            gt_rgb = gt_rgbs[idx]
            # mask to FG region to focus loss
            fg = (gt_alpha > 0.5).float().unsqueeze(0).expand_as(img)
            pred_fg = img * fg + (1 - fg)         # composite over white bg
            gt_fg   = gt_rgb * fg + (1 - fg)
            with torch.cuda.amp.autocast(enabled=False):
                L_lpips = args._lpips_model(
                    pred_fg.unsqueeze(0) * 2 - 1,
                    gt_fg.unsqueeze(0) * 2 - 1).mean()

        loss = (args.lam_silh * L_silh + args.lam_traj * L_traj +
                args.lam_smooth * L_smooth + args.lam_arap * L_arap +
                args.lam_photo_blur * L_photo +
                args.lam_photo_smart * L_photo_smart +
                args.lam_scale_smooth * L_scale_smooth +
                args.lam_xyz_res_smooth * L_xyz_res_smooth +
                args.lam_xyz_res_l2 * L_xyz_res_l2 +
                args.lam_rot_res_smooth * L_rot_res_smooth +
                args.lam_rot_res_l2 * L_rot_res_l2 +
                args.lam_lpips * L_lpips)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % 500 == 0:
            print(f"[hier] it {it:>5d}  loss={loss:.4f}  silh={L_silh:.4f}  "
                  f"traj={L_traj:.4f}  smooth={L_smooth:.4f}  arap={float(L_arap):.4f}  "
                  f"photo={L_photo:.4f}  photo_smart={float(L_photo_smart):.4f}  "
                  f"({time.time()-t0:.0f}s)")

    state = {
        "trans": model.trans.detach().cpu().numpy(),
        "aa": model.aa.detach().cpu().numpy(),
        "arm_idx_for_residual": arm_idx_for_residual if args.use_xyz_residual else np.array([]),
        "arm_centers": model.centers.detach().cpu().numpy(),
        "lbs_weights": lbs_weights.detach().cpu().numpy(),
        "arm_weights_global": arm_weights_global,
        "config": vars(args),
    }
    if args.use_color_tint:
        state["color_tint"] = model.color_tint.detach().cpu().numpy()
    if args.use_per_time_scale:
        state["scale"] = model.scale.detach().cpu().numpy()
    if args.use_xyz_residual:
        state["xyz_residual"] = model.xyz_residual.detach().cpu().numpy()
    if args.use_rot_residual:
        state["rot_residual"] = model.rot_residual.detach().cpu().numpy()
    np.savez(out_dir / "partrigid_state.npz", **state)
    print(f"[hier] saved {out_dir}/partrigid_state.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
