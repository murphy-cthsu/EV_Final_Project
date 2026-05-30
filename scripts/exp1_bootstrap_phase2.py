"""Exp 1: Iterative bootstrap denoising.

Round 0: Phase 2 (K=100+smart+scale) trained on SV4D supervision — current state.
Round 1: Render Phase 2 at all (cam, t) → save as bootstrap_v1/r_NNNNN.png
         These renders are CLOSER to clean GT than SV4D (model resists noise).
         Replace SV4D supervision with these "denoised" renders.
         Re-train Phase 2 with the new (cleaner) supervision.
         Smart photo filter still uses d-3dgs as reference (unchanged).

Hypothesis: lower noise in training → motion fits better → PSNR vs clean GT 提升.

Output:
  data/custom/lego_v2_bootstrap/{train,test}/r_NNNNN.png  (RGBA: bootstrapped)
  outputs/custom/partrigid_lego_v2_K100_bootstrap/        (re-trained model)
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

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

R0_LABEL = "lego_v2_K100_smart_scale"        # baseline (round 0)
R1_LABEL = "lego_v2_K100_bootstrap_r1"        # bootstrap round 1
SCENE = REPO / "data/custom/lego_v2"
BOOTSTRAP_DIR = REPO / "data/custom/lego_v2_bootstrap"
CANON = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"


def aa2mat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def main():
    # Step 1: Render Phase 2 (round 0) model at all 105 (cam, t)
    print("[exp1] step 1: rendering Phase 2 round 0 at all (cam, t)…")
    s = np.load(REPO / f"outputs/custom/partrigid_{R0_LABEL}/partrigid_state.npz", allow_pickle=True)
    trans = s["trans"]; aa = s["aa"]; centers = s["arm_centers"]; lbs = s["lbs_weights"]
    have_scale = "scale" in s.files
    scale_kt = s["scale"] if have_scale else None
    K_, T_train = trans.shape[0], trans.shape[1]

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(CANON), og_number_points=0); break
        except Exception:
            g = None
    xyz_canon = g.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    print(f"[exp1] canonical N={N}")

    meta_train = json.loads((SCENE / "transforms_train.json").read_text())
    meta_test = json.loads((SCENE / "transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Set up bootstrap dir (mirror lego_v2 structure, NEW images)
    for d in (BOOTSTRAP_DIR / "train", BOOTSTRAP_DIR / "test"):
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)

    # Render Phase 2 model + bake into bootstrap_dir using SAME alpha as original (SAM-2 mask)
    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R_cam = -np.transpose(M[:3, :3]); R_cam[:, 0] = -R_cam[:, 0]
        Tcam = -M[:3, 3]
        tl = min(t, T_train - 1)
        R_all = aa2mat(aa[:, tl, :])
        T_all = trans[:, tl, :]
        rel = xyz_canon[:, None, :] - centers[None, :, :]
        rotated = np.einsum("kij,nkj->nki", R_all, rel)
        new_per = rotated + centers[None, :, :] + T_all[None, :, :]
        weighted = (lbs[..., None] * new_per).sum(axis=1)
        w_total = lbs.sum(axis=1, keepdims=True).clip(min=0, max=1)
        new_xyz = weighted + (1 - w_total) * xyz_canon

        d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
        d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_sc = torch.zeros(N, 3, device="cuda")
        if have_scale:
            sb = lbs @ scale_kt[:, tl, :]
            d_sc = torch.from_numpy(sb.astype(np.float32)).cuda()
        cam = SCGSCamera(colmap_id=0, R=R_cam, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(0.0).float())
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz_t, d_rotation=d_rot,
                         d_scaling=d_sc, d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        img_u8 = (img * 255).astype(np.uint8)

        # Get original SAM-2 alpha (reuse)
        orig_png_name = f"{Path(f['file_path']).name}.png"
        orig_path = None
        for split in ("train", "test"):
            cand = SCENE / split / orig_png_name
            if cand.exists():
                orig_path = cand
                tgt_split = split
                break
        orig_rgba = np.asarray(Image.open(orig_path))
        alpha = orig_rgba[..., 3:4]  # keep SAM-2 mask
        rgba_new = np.concatenate([img_u8, alpha], axis=-1)
        Image.fromarray(rgba_new, mode="RGBA").save(BOOTSTRAP_DIR / tgt_split / orig_png_name)

    # Copy transforms json
    shutil.copy(SCENE / "transforms_train.json", BOOTSTRAP_DIR / "transforms_train.json")
    shutil.copy(SCENE / "transforms_test.json",  BOOTSTRAP_DIR / "transforms_test.json")
    print(f"[exp1] bootstrap dataset built at {BOOTSTRAP_DIR}")

    # Step 2: Re-train Phase 2 on bootstrapped supervision
    print("[exp1] step 2: re-train on bootstrapped supervision…")
    cmd = [
        "/home/cthsu/miniconda3/envs/scgs/bin/python",
        "scripts/train_partrigid_hier.py",
        "--label", R1_LABEL,
        "--canon_ply", str(CANON),
        "--part_dir", "runs_aux/part_assignment_lego_v2",
        "--scene_dir", str(BOOTSTRAP_DIR),
        "--v5_render_dir", "outputs/custom/lego_v2_d3dgs_ref/renders",
        "--use_test_too",
        "--k_arm", "100", "--lbs_K", "6", "--lam_arap", "1.0",
        "--lam_photo_smart", "3.0", "--use_per_time_scale",
        "--iterations", "8000",
    ]
    subprocess.run(cmd, check=True)

    # Step 3: Eval new model against ORIGINAL SV4D + d-3dgs (use lego_v2, not bootstrap)
    print("[exp1] step 3: eval bootstrap model against ORIGINAL SV4D + d-3dgs…")
    cmd = [
        "/home/cthsu/miniconda3/envs/scgs/bin/python",
        "scripts/eval_lego_v2_hier.py",
        "--label", R1_LABEL,
        "--canon_ply", str(CANON),
        "--save_renders",
    ]
    subprocess.run(cmd, check=True)
    print("[exp1] DONE")


if __name__ == "__main__":
    raise SystemExit(main())
