"""Render vanilla SC-GS (lego_v2_vanilla_sam_node, fit on SV4D) at the 5 cameras
x 21 frames -> outputs/custom/lego_v2_vanilla_render/  (used as leak-free
smart-photo reference, replacing the d-3dgs renders that leaked GT info)."""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import imageio.v3 as iio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

H = W = 576
SCENE = REPO / "data/custom/lego_v2"
OUT = REPO / "outputs/custom/lego_v2_vanilla_render"
MODEL = REPO / "outputs/custom/lego_v2_vanilla_sam_node"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    it = max(int(p.name.split("_")[-1]) for p in (MODEL/"point_cloud").iterdir() if p.name.startswith("iter"))
    print(f"[render-vanilla] using iteration {it}")
    deform_state = torch.load(MODEL/"deform"/f"iteration_{it}"/"deform.pth", map_location="cuda", weights_only=False)
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(str(MODEL/"point_cloud"/f"iteration_{it}"/"point_cloud.ply"), og_number_points=0)
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                          hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                          pred_color=False, use_hash=False, hash_time=False,
                          d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                          with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                          is_scene_static=False)
    deform.load_weights(str(MODEL), iteration=it)

    meta_t = json.loads((SCENE/"transforms_train.json").read_text())
    meta_te = json.loads((SCENE/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    print(f"[render-vanilla] {T_full} frames, {len(all_frames)} (view, t) pairs")

    for f in all_frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tcam = -M[:3, 3]
        fid = float(t) / max(T_full - 1, 1)
        cam = SCGSCamera(colmap_id=0, R=R, T=Tcam, FoVx=fov_x, FoVy=FovY,
                         image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                         image_name="x", uid=0, fid=torch.tensor(fid).float())
        time_input = deform.deform.expand_time(cam.fid.to("cuda"))
        with torch.no_grad():
            d = deform.step(g.get_xyz.detach(), time_input,
                            feature=g.feature, motion_mask=getattr(g, "motion_mask", None),
                            is_training=False)
            pkg = render(cam, g, pipe, bg,
                         d_xyz=d["d_xyz"], d_rotation=d["d_rotation"], d_scaling=d["d_scaling"],
                         d_opacity=d.get("d_opacity"), d_color=d.get("d_color"),
                         d_rot_as_res=deform.d_rot_as_res)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        img8 = (img * 255).astype(np.uint8)
        flat = v * T_full + t
        iio.imwrite(OUT / f"{flat:05d}.png", img8)
    print(f"[render-vanilla] wrote {len(all_frames)} renders to {OUT}")


if __name__ == "__main__":
    main()
