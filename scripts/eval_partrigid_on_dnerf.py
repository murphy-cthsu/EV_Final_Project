"""P5 evaluation: render part-rigid model at D-NeRF lego test cam+time pairs
and compute PSNR/SSIM/LPIPS against clean D-NeRF GT.

Inputs:
    - Canonical Gaussians (outputs/custom/canonical_static_node/.../point_cloud.ply)
    - Part-rigid state (outputs/custom/partrigid_*/partrigid_state.npz)
    - D-NeRF lego test set (data/dnerf/lego/transforms_test.json + test/*.png)

For each test (cam, t) pair:
    1. Build SC-GS Camera at that pose with that t.
    2. Compute d_xyz for arm Gaussians via part-rigid model.
    3. Render.
    4. Compare to D-NeRF GT (composite onto black background to match D-NeRF).

Reports mean PSNR/SSIM/LPIPS over the 20 test frames.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/eval_partrigid_on_dnerf.py \\
        --partrigid_label v1
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO_ROOT / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

CANON = REPO_ROOT / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
DNERF_LEGO = REPO_ROOT / "data" / "dnerf" / "lego"


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


def psnr(img, gt):
    mse = ((img - gt) ** 2).mean()
    return -10 * np.log10(max(mse, 1e-12))


def ssim_naive(img, gt):
    """Simple per-channel mean SSIM proxy via gaussian window."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img, gt, channel_axis=2, data_range=1.0))
    except Exception:
        return float("nan")


def lpips_score(img, gt, lpips_fn):
    if lpips_fn is None:
        return float("nan")
    a = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    b = torch.from_numpy(gt).permute(2, 0, 1)[None].float().cuda() * 2 - 1
    with torch.no_grad():
        return float(lpips_fn(a, b).item())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partrigid_label", default="v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_renders", action="store_true")
    p.add_argument("--background", choices=["black", "white"], default="black",
                   help="D-NeRF GT has black bg; black is correct for fair PSNR.")
    args = p.parse_args()

    state = np.load(REPO_ROOT / f"outputs/custom/partrigid_{args.partrigid_label}/partrigid_state.npz",
                    allow_pickle=True)
    arm_trans = state["arm_trans"]       # (T, 3)
    arm_aa = state["arm_aa"]              # (T, 3)
    arm_pivot = state["arm_pivot"]        # (3,)
    part_id = state["part_id"]            # (N,)
    T_train = arm_trans.shape[0]
    print(f"[eval] part-rigid model: T_train={T_train}, arm pivot={arm_pivot}")

    # ===== Load canonical Gaussians =====
    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    print(f"[eval] canonical N={N}")
    arm_mask_np = (part_id == 0)
    print(f"[eval] arm mask: {arm_mask_np.sum()} Gaussians")

    # ===== D-NeRF lego test set =====
    test_meta = json.loads((DNERF_LEGO / "transforms_test.json").read_text())
    fov_x = test_meta["camera_angle_x"]
    print(f"[eval] D-NeRF test: {len(test_meta['frames'])} frames, fov_x={fov_x:.4f}")

    # Optional LPIPS model
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(args.device)
        lpips_fn.eval()
    except Exception as e:
        print(f"[eval] LPIPS unavailable ({e}); skipping LPIPS")
        lpips_fn = None

    # ===== Pipeline setup =====
    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe_args = parser_pipe.parse_args([])
    pipe = pp.extract(pipe_args)
    bg_color = [0, 0, 0] if args.background == "black" else [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    H = W = 800  # D-NeRF default render size
    # Determine D-NeRF image dims from first test image
    first_png = DNERF_LEGO / "test" / "r_000.png"
    first_img = iio.imread(first_png)
    H, W = first_img.shape[:2]
    print(f"[eval] image size: {W}x{H}")

    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    out_dir = REPO_ROOT / f"runs_aux/partrigid_eval/{args.partrigid_label}"
    if args.save_renders:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(test_meta["frames"]):
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        matrix = np.linalg.inv(c2w)
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -matrix[:3, 3]
        t = float(f.get("time", 0.0))
        # Map continuous t in [0,1] to discrete training time bin
        # arm_trans/aa were trained at T discrete times
        t_idx_f = t * (T_train - 1)
        t_lo = int(np.floor(t_idx_f))
        t_hi = min(t_lo + 1, T_train - 1)
        w = float(t_idx_f - t_lo)
        # Linear interpolate translation; for rotation, interpolate axis-angle directly
        trans_t = (1 - w) * arm_trans[t_lo] + w * arm_trans[t_hi]
        aa_t = (1 - w) * arm_aa[t_lo] + w * arm_aa[t_hi]

        # Compute d_xyz for arm Gaussians
        R_t = axis_angle_to_matrix_np(aa_t[None])[0]
        d_xyz_np = np.zeros_like(xyz_canon)
        rel = xyz_canon[arm_mask_np] - arm_pivot
        rotated = rel @ R_t.T
        new_xyz = rotated + arm_pivot + trans_t
        d_xyz_np[arm_mask_np] = new_xyz - xyz_canon[arm_mask_np]

        d_xyz_t = torch.from_numpy(d_xyz_np).float().to(args.device)
        d_rotation = torch.zeros(N, 4, device=args.device)
        d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device=args.device)
        d_scaling = torch.zeros(N, 3, device=args.device)

        # Build Camera
        gt_path = DNERF_LEGO / "test" / f"{Path(f['file_path']).name}.png"
        gt_img = np.asarray(iio.imread(gt_path)).astype(np.float32) / 255.0
        if gt_img.shape[-1] == 4:
            # D-NeRF: alpha-blend onto chosen bg color
            alpha = gt_img[..., 3:4]
            gt_rgb = gt_img[..., :3] * alpha + np.array(bg_color) * (1 - alpha)
        else:
            gt_rgb = gt_img
        gt_tensor = torch.from_numpy(gt_rgb.astype(np.float32)).permute(2, 0, 1).cuda()

        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=gt_tensor, gt_alpha_mask=None,
                         image_name=f["file_path"], uid=i, fid=torch.tensor(t).float())

        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background,
                         d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                         d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

        ps = psnr(img, gt_rgb)
        ss = ssim_naive(img, gt_rgb)
        lp = lpips_score(img, gt_rgb, lpips_fn)
        psnr_list.append(ps); ssim_list.append(ss); lpips_list.append(lp)
        print(f"[eval] t={t:.3f} frame {i:>2d}: PSNR={ps:.2f}  SSIM={ss:.3f}  LPIPS={lp:.3f}")

        if args.save_renders:
            from PIL import Image as PILImage
            both = np.concatenate([gt_rgb, img], axis=1)
            PILImage.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{i:03d}.png")

    psnr_arr = np.array(psnr_list)
    ssim_arr = np.array(ssim_list)
    lpips_arr = np.array([v for v in lpips_list if not math.isnan(v)])
    print()
    print(f"[eval] === RESULTS on D-NeRF lego test set ({len(psnr_list)} frames) ===")
    print(f"[eval] mean PSNR : {psnr_arr.mean():.3f} +/- {psnr_arr.std():.3f}")
    print(f"[eval] mean SSIM : {ssim_arr.mean():.3f} +/- {ssim_arr.std():.3f}")
    print(f"[eval] mean LPIPS: {lpips_arr.mean():.3f} +/- {lpips_arr.std():.3f}")
    print(f"[eval] median PSNR: {np.median(psnr_arr):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
