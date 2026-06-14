"""Render the provided jumpingjacks_shake_lefthand SC-GS model (mlp deform) into a
self-supervised D-NeRF scene our part-rigid pipeline can train on.

NO clean reference (by design): supervision = the model's own orbit renders.
Poses are synthesized (pose_spherical), so render and supervision are aligned by
construction. Clean alpha comes straight from the rasterizer.

Writes:
  data/custom/jumpingjacks_shake_lefthand/{train,test}/r_{flat:05d}.png   (RGBA)
  data/custom/jumpingjacks_shake_lefthand/transforms_{train,test}.json
  runs_aux/jjshake_videos/{view_tag}.mp4         (per-view RGB, for Stage B/D)
  outputs/custom/jjshake_selfref/renders/{flat:05d}.png   (RGB white, v5 plain-photo ref)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
from viz_vanilla_noprior import pose_spherical, c2w_to_RT  # noqa: E402

H = W = 576
T = 21
FOV_X = 0.6911112070083618
RADIUS = 4.03
MODEL = REPO / "outputs/custom/jumpingjacks_shake_lefthand"
SCENE = REPO / "data/custom/jumpingjacks_shake_lefthand"
VIDDIR = REPO / "runs_aux/jjshake_videos"
SELFREF = REPO / "outputs/custom/jjshake_selfref/renders"
AZIS = [0, 20, 40, 60, 80, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]


def main():
    for d in (SCENE / "train", SCENE / "test", VIDDIR, SELFREF):
        d.mkdir(parents=True, exist_ok=True)

    g = GaussianModel(3, fea_dim=0, with_motion_mask=False)
    g.load_ply(str(MODEL / "point_cloud/iteration_40000/point_cloud.ply"), og_number_points=0)
    deform = DeformModel(deform_type="mlp", is_blender=True, d_rot_as_res=True)
    deform.load_weights(str(MODEL), iteration=40000)
    N = g.get_xyz.shape[0]
    _pp = _A(); pipe = PipelineParams(_pp).extract(_pp.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    fov_y = focal2fov(fov2focal(FOV_X, W), H)
    print(f"[build] model N={N}  views={len(AZIS)}  T={T}")

    train_frames, test_frames = [], []
    for vi, az in enumerate(AZIS):
        tag = f"elev_0_az_{az}"
        c2w = pose_spherical(az, 0.0, RADIUS)
        R, Tt = c2w_to_RT(c2w)
        vid_rgb = []
        for t in range(T):
            fid = t / (T - 1)
            cam = SCGSCamera(colmap_id=0, R=R, T=Tt, FoVx=FOV_X, FoVy=fov_y,
                             image=torch.zeros(3, H, W), gt_alpha_mask=None,
                             image_name="x", uid=0, fid=torch.tensor(fid).float())
            time_input = torch.tensor([[fid]], device="cuda").expand(N, -1).float()
            with torch.no_grad():
                d = deform.step(g.get_xyz.detach(), time_input)
                pkg = render(cam, g, pipe, bg, d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                             d_scaling=d["d_scaling"], d_rot_as_res=deform.d_rot_as_res)
            rgb = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
            alpha = torch.clamp(pkg["alpha"], 0, 1).cpu().numpy()
            if alpha.ndim == 3:
                alpha = alpha[0]
            flat = vi * T + t
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            rgba_u8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            is_test = (t % 4 == 0)
            split = "test" if is_test else "train"
            Image.fromarray(rgba_u8, "RGBA").save(SCENE / split / f"r_{flat:05d}.png")
            Image.fromarray(rgb_u8, "RGB").save(SELFREF / f"{flat:05d}.png")
            vid_rgb.append(rgb_u8)
            frame_meta = {
                "file_path": f"./{split}/r_{flat:05d}",
                "view_idx": vi, "frame_idx": t, "view_tag": tag,
                "time": fid, "transform_matrix": c2w.tolist(),
            }
            (test_frames if is_test else train_frames).append(frame_meta)
        iio.imwrite(VIDDIR / f"{tag}.mp4", np.stack(vid_rgb), fps=10)
        print(f"  view {vi:2d} {tag} done")

    for split, frames in (("train", train_frames), ("test", test_frames)):
        (SCENE / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": FOV_X, "frames": frames}, indent=2))
    print(f"[build] train frames={len(train_frames)}  test frames={len(test_frames)}")
    print(f"[build] scene -> {SCENE}")


if __name__ == "__main__":
    raise SystemExit(main())
