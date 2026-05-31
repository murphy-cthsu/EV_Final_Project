"""Eval deform-MLP model against SV4D + d-3dgs."""

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
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

# import deform-MLP class
sys.path.insert(0, str(REPO / "scripts"))
from train_partrigid_deformmlp import TinyDeformMLP


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="deformmlp_v1")
    p.add_argument("--canon_ply", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--save_renders", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(REPO / f"outputs/custom/deformmlp_{args.label}/model.pt", map_location="cuda")
    arm_idx = ckpt["arm_idx"]
    arm_xyz_mean = torch.from_numpy(ckpt["arm_xyz_mean"]).float().cuda()
    arm_xyz_scale = float(ckpt["arm_xyz_scale"])
    T_full = int(ckpt["T_full"])

    model = TinyDeformMLP(T_freqs=ckpt["T_freqs"], hidden=ckpt["hidden"], max_disp=ckpt["max_disp"]).cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.canon_ply), og_number_points=0); break
        except Exception:
            g = None
    xyz_canon = g.get_xyz.detach().to("cuda")
    N = xyz_canon.shape[0]
    arm_idx_t = torch.from_numpy(arm_idx).long().cuda()
    arm_xyz_canon = xyz_canon[arm_idx_t]
    arm_xyz_norm = (arm_xyz_canon - arm_xyz_mean) / arm_xyz_scale

    scene_dir = REPO / "data/custom/lego_v2"
    d3dgs_dir = REPO / "outputs/custom/lego_v2_d3dgs_ref/renders"
    meta_train = json.loads((scene_dir/"transforms_train.json").read_text())
    meta_test = json.loads((scene_dir/"transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out_dir = REPO / f"runs_aux/deformmlp_eval/{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_renders:
        (out_dir / "tiles").mkdir(exist_ok=True)

    psnr_vs_sv4d = []; psnr_vs_d3 = []
    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        t_norm = torch.tensor(float(t) / max(T_full - 1, 1), device="cuda")

        with torch.no_grad():
            dxyz_arm, dscale_arm = model(arm_xyz_norm, t_norm)
        new_xyz = xyz_canon.clone()
        new_xyz[arm_idx_t] = xyz_canon[arm_idx_t] + dxyz_arm
        d_xyz = new_xyz - xyz_canon
        d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_sc = torch.zeros(N, 3, device="cuda")
        d_sc[arm_idx_t] = dscale_arm

        cam = SCGSCamera(colmap_id=0, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(0.0).float())
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

        # SV4D GT
        png_name = f"{Path(f['file_path']).name}.png"
        sv4d_path = None
        for split in ("train", "test"):
            cand = scene_dir / split / png_name
            if cand.exists(): sv4d_path = cand; break
        rgba = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
        a = rgba[..., 3:4]
        sv4d = rgba[..., :3] * a + 1 * (1 - a)
        # d-3dgs GT
        d3 = np.asarray(iio.imread(d3dgs_dir / f"{v*T_full+t:05d}.png"), dtype=np.float32) / 255.0
        if d3.shape[-1] == 4:
            ad3 = d3[..., 3:4]; d3 = d3[..., :3] * ad3 + 1 * (1 - ad3)

        psnr_vs_sv4d.append(psnr(img, sv4d))
        psnr_vs_d3.append(psnr(img, d3))
        if args.save_renders and v == 0:
            sep = np.ones((H, 4, 3))
            row = np.concatenate([sv4d, sep, d3, sep, img], axis=1)
            Image.fromarray((row * 255).astype(np.uint8)).save(out_dir / "tiles" / f"v0_t{t:02d}.png")

    arr_sv = np.array(psnr_vs_sv4d); arr_d3 = np.array(psnr_vs_d3)
    print(f"\n[deformmlp-eval] vs SV4D : mean={arr_sv.mean():.3f}  median={np.median(arr_sv):.3f}")
    print(f"[deformmlp-eval] vs d-3dgs: mean={arr_d3.mean():.3f}  median={np.median(arr_d3):.3f}")
    print(f"[deformmlp-eval] gap     : {arr_d3.mean() - arr_sv.mean():+.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
