"""Render frozen canonical at lego_v2's 5 cameras x 21 times (static, identical
per view, just replicated). Output: outputs/custom/lego_v2_canonical_static_render/
Used as a leak-free smart-photo reference (Option B)."""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as SCGSCamera
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal

H = W = 576
SCENE = REPO / "data/custom/lego_v2"
OUT = REPO / "outputs/custom/lego_v2_canonical_static_render"
PLY = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = GaussianModel(3, fea_dim=0, with_motion_mask=False)
    g.load_ply(str(PLY), og_number_points=0)
    meta_t = json.loads((SCENE/"transforms_train.json").read_text())
    meta_te = json.loads((SCENE/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    print(f"[canon-static] {T_full} frames, {len(all_frames)} (view,t) pairs")

    # group frames by view, use first frame's camera (camera is identical across t)
    seen_view = {}
    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        if v not in seen_view:
            c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
            M = np.linalg.inv(c2w)
            R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
            Tcam = -M[:3, 3]
            cam = SCGSCamera(colmap_id=0, R=R, T=Tcam, FoVx=fov_x, FoVy=FovY,
                             image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                             image_name="x", uid=0, fid=torch.tensor(0.0).float())
            d_xyz = torch.zeros_like(g.get_xyz)
            d_rot = torch.zeros(g.get_xyz.shape[0], 4, device="cuda")
            d_sc = torch.zeros_like(g.get_xyz)
            with torch.no_grad():
                pkg = render(cam, g, pipe, bg, d_xyz=d_xyz, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
            img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
            seen_view[v] = (img * 255).astype(np.uint8)
            print(f"  rendered view {v}")
        flat = v * T_full + t
        Image.fromarray(seen_view[v]).save(OUT / f"{flat:05d}.png")
    print(f"[canon-static] wrote {len(all_frames)} pngs to {OUT}")


if __name__ == "__main__":
    main()
