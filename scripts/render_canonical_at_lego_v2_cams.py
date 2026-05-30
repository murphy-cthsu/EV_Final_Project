"""Render the provided canonical ply at our 5 lego_v2 cameras to verify
spatial alignment + visual quality before proceeding with Phase 2 training."""

from __future__ import annotations

import json
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


def main():
    ply = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
    g = GaussianModel(3, fea_dim=0, with_motion_mask=False)
    g.load_ply(str(ply), og_number_points=0)
    N = g.get_xyz.shape[0]
    print(f"[verify] loaded canonical N={N}")

    meta = json.loads((REPO / "data/custom/lego_v2_frame0_d3dgs/transforms_train.json").read_text())
    fov_x = meta["camera_angle_x"]
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out = REPO / "runs_aux/canonical_verify"
    out.mkdir(parents=True, exist_ok=True)

    for f in meta["frames"]:
        v = int(f["view_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tcam = -M[:3, 3]
        cam = SCGSCamera(colmap_id=v, R=R, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name=f"v{v}", uid=v, fid=torch.tensor(0.0).float())
        d_xyz = torch.zeros(N, 3, device="cuda")
        d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
        d_sc = torch.zeros(N, 3, device="cuda")
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f"v{v}_canonical.png")
        # Also load d-3dgs ref + sv4d for comparison
        import imageio
        d3_r = imageio.get_reader(f"/mnt/HDD_1/cthsu/lego/d-3dgs_video/000001_v00{v}/000001_v00{v}.mp4")
        sv_r = imageio.get_reader(f"/mnt/HDD_1/cthsu/lego/sv4d2/000001_v00{v}.mp4")
        d3 = d3_r.get_data(0)
        sv = sv_r.get_data(0)
        sep = np.ones((H, 4, 3), dtype=np.uint8) * 255
        trio = np.concatenate([sv[..., :3], sep, d3[..., :3], sep, (img * 255).astype(np.uint8)], axis=1)
        Image.fromarray(trio).save(out / f"v{v}_compare.png")
        print(f"[verify] view {v} rendered")
    print(f"[verify] outputs in {out}")


if __name__ == "__main__":
    raise SystemExit(main())
