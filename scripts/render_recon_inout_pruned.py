"""Re-render the VGM-supervised recon at input vs off-axis with floater
Gaussians (low-opacity + oversized streaks) pruned, for a clean block ①."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
import torch  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
import eval_region_psnr as E  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402

MODEL = REPO / "outputs/custom/lego_v2_vanilla_sam_node"
VIEWS = {"sharp": "elev_0_az_0", "offaxis": "elev_0_az_270"}
T = 10


def prune(g, op_thr=0.10, scale_pct=97.0, dark_thr=0.18):
    op = g.get_opacity.detach().squeeze(-1)
    sc = g.get_scaling.detach().max(1).values
    smax = np.percentile(sc.cpu().numpy(), scale_pct)
    C0 = 0.28209479177387814
    fdc = g.get_features.detach()[:, 0, :]            # SH DC term (N,3)
    bright = (0.5 + C0 * fdc).clamp(0, 1).mean(1)     # per-Gaussian base brightness
    keep = (op > op_thr) & (sc < smax) & (bright > dark_thr)
    idx = torch.where(keep)[0]
    for name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        t = getattr(g, name, None)
        if t is not None and t.shape[0] == keep.shape[0]:
            setattr(g, name, t[idx].contiguous())
    for name in ["motion_mask", "feature"]:
        t = getattr(g, name, None)
        if isinstance(t, torch.Tensor) and t.shape[0] == keep.shape[0]:
            setattr(g, name, t[idx].contiguous())
    return int(keep.sum()), int(keep.numel())


def main():
    out = REPO / "runs_aux/recon_inout"; out.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPO / "data/custom/lego_v3/transforms_train.json").read_text())
    mt = json.loads((REPO / "data/custom/lego_v3/transforms_test.json").read_text())
    allf = m["frames"] + mt["frames"]
    fov_x = m["camera_angle_x"]; FovY = focal2fov(fov2focal(fov_x, 576), 576)
    by_key = {(f["view_tag"], int(f["frame_idx"])): f for f in allf}
    _pq = _A(); pp = PipelineParams(_pq); pipe = pp.extract(_pq.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    mp = MODEL
    iters = sorted(int(p.name.split("_")[-1]) for p in (mp / "point_cloud").iterdir()
                   if p.name.startswith("iteration_"))
    it = iters[-1]
    ds = torch.load(mp / "deform" / f"iteration_{it}" / "deform.pth", map_location="cuda")
    node_num = ds["nodes"].shape[0]; hyper_dim = ds["nodes"].shape[1] - 3
    g = E.load_gauss(mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply")
    kept, tot = prune(g)
    print(f"  pruned floaters: kept {kept}/{tot}")
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=it)

    for k, tag in VIEWS.items():
        f = by_key[(tag, T)]
        cam = E.make_cam(f, fov_x, FovY, float(T) / 20.0)
        time_input = deform.deform.expand_time(cam.fid.to("cuda"))
        with torch.no_grad():
            d = deform.step(g.get_xyz.detach(), time_input, feature=g.feature,
                            motion_mask=getattr(g, "motion_mask", None), is_training=False)
            pkg = render(cam, g, pipe, bg, d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                         d_scaling=d["d_scaling"], d_opacity=d.get("d_opacity"),
                         d_color=d.get("d_color"), d_rot_as_res=deform.d_rot_as_res)
        img = torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f"pruned_{tag}.png")
        print("  saved pruned", tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
