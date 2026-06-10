"""Render clean canonical at all (v, t) pairs of a 57-view dataset for smart-photo reference."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import argparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as SCGSCamera
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal

H = W = 576

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--canon_ply", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(args.canon_ply, og_number_points=0)
    scene = Path(args.scene_dir)
    meta_t = json.loads((scene/"transforms_train.json").read_text())
    meta_te = json.loads((scene/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    seen = {}
    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        if v not in seen:
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
            seen[v] = (img*255).astype(np.uint8)
        flat = v*T_full + t
        Image.fromarray(seen[v]).save(out / f"{flat:05d}.png")
    print(f"wrote {len(all_frames)} pngs ({len(seen)} unique views)")

if __name__ == "__main__":
    main()
