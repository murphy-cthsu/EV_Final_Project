"""P5 evaluation (SV4D-internal): render part-rigid model at the same held-out
test pairs we used for vanilla SC-GS (view-split or temporal-split). Camera
convention guaranteed to align since both come from the SV4D converter.

Comparable baselines:
    vanilla SC-GS on scene00_split (view-split)    : best 13.61 dB
    vanilla SC-GS on scene00_split_t (temporal)    : best 25.75 dB

We render the part-rigid model at the held-out frames and report mean PSNR.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partrigid_label", default="v1")
    p.add_argument("--scene", default="scene00_split_t",
                   choices=["scene00_split", "scene00_split_t"])
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    scene_dir = REPO_ROOT / "data" / "custom" / args.scene
    state = np.load(REPO_ROOT / f"outputs/custom/partrigid_{args.partrigid_label}/partrigid_state.npz",
                    allow_pickle=True)
    arm_trans = state["arm_trans"]; arm_aa = state["arm_aa"]
    arm_pivot = state["arm_pivot"]; part_id = state["part_id"]
    T_train = arm_trans.shape[0]
    print(f"[eval] part-rigid model: T_train={T_train}")

    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    arm_mask_np = (part_id == 0)
    print(f"[eval] canonical N={N}, arm Gaussians={arm_mask_np.sum()}")

    # Test set
    test_meta = json.loads((scene_dir / "transforms_test.json").read_text())
    fov_x = test_meta["camera_angle_x"]
    print(f"[eval] scene {args.scene}: {len(test_meta['frames'])} test frames")

    H = W = 576
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out_dir = REPO_ROOT / f"runs_aux/partrigid_eval_sv4d/{args.partrigid_label}_{args.scene}"
    if args.save_renders:
        out_dir.mkdir(parents=True, exist_ok=True)

    psnr_list = []
    for i, f in enumerate(test_meta["frames"]):
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        matrix = np.linalg.inv(c2w)
        R = -np.transpose(matrix[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -matrix[:3, 3]
        t_idx = int(f["frame_idx"])
        # Direct discrete lookup, no interpolation
        t_lo = min(t_idx, T_train - 1)

        R_t = axis_angle_to_matrix_np(arm_aa[t_lo:t_lo+1])[0]
        d_xyz_np = np.zeros_like(xyz_canon)
        rel = xyz_canon[arm_mask_np] - arm_pivot
        rotated = rel @ R_t.T
        new_xyz = rotated + arm_pivot + arm_trans[t_lo]
        d_xyz_np[arm_mask_np] = new_xyz - xyz_canon[arm_mask_np]

        d_xyz_t = torch.from_numpy(d_xyz_np).float().cuda()
        d_rotation = torch.zeros(N, 4, device="cuda")
        d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device="cuda")
        d_scaling = torch.zeros(N, 3, device="cuda")

        gt_path = scene_dir / "train" / f"{Path(f['file_path']).name}.png"
        gt_rgba = np.asarray(iio.imread(gt_path)).astype(np.float32) / 255.0
        # Composite onto white background to match training setup
        alpha = gt_rgba[..., 3:4]
        gt_rgb = gt_rgba[..., :3] * alpha + np.array([1, 1, 1]) * (1 - alpha)
        gt_tensor = torch.from_numpy(gt_rgb.astype(np.float32)).permute(2, 0, 1).cuda()

        cam = SCGSCamera(colmap_id=i, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=gt_tensor, gt_alpha_mask=None,
                         image_name=f["file_path"], uid=i, fid=torch.tensor(0.0).float())

        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg,
                         d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                         d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        ps = psnr(img, gt_rgb)
        psnr_list.append(ps)
        if args.save_renders:
            from PIL import Image as PILImage
            both = np.concatenate([gt_rgb, img], axis=1)
            PILImage.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{i:03d}.png")
        print(f"[eval] frame {i:>2d} view={int(f['view_idx'])} t={t_idx}: PSNR={ps:.2f}")

    arr = np.array(psnr_list)
    print()
    print(f"[eval] === scene={args.scene}  partrigid={args.partrigid_label} ===")
    print(f"[eval] mean PSNR = {arr.mean():.3f} +/- {arr.std():.3f}  ({len(arr)} frames)")
    print(f"[eval] median PSNR = {np.median(arr):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
