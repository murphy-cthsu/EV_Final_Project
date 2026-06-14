"""Region-decomposed PSNR vs clean d-3dgs GT on lego_v2.

Answers: "is the PSNR win just from inheriting the baseplate from the frozen
canonical (SV4D supervision has no baseplate)?"

Regions per (view, time), same masks applied to ALL models:
  digger     = SV4D SAM-2 alpha > 0.5           (the supervised object)
  baseplate  = d-3dgs foreground AND NOT digger (in GT but never supervised)
  background = neither                           (white)
d-3dgs foreground = non-white pixels (RGB renders, white bg).

Models: vanilla SC-GS / F1 warm-start / F2 frozen-canon deform-MLP / ours.
Per-region score = mean over frames of per-frame PSNR on region pixels
(frames with <50 region px skipped), matching the per-frame protocol of
eval_lego_v2_hier.py / eval_vanilla_lego_v2.py.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

CANON = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
H = W = 576
MIN_PX = 50


def aa2mat_np(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def psnr_masked(a, b, m):
    if m.sum() < MIN_PX:
        return None
    mse = ((a[m] - b[m]) ** 2).mean()
    return -10 * math.log10(max(mse, 1e-12))


def load_gauss(ply_path):
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(ply_path), og_number_points=0)
            return g
        except (IndexError, RuntimeError, ValueError):
            pass
    raise RuntimeError(f"can't load {ply_path}")


def make_cam(f, fov_x, FovY, fid_val):
    c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
    M = np.linalg.inv(c2w)
    R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
    Tr = -M[:3, 3]
    return SCGSCamera(colmap_id=0, R=R, T=Tr, FoVx=fov_x, FoVy=FovY,
                      image=torch.zeros(3, H, W), gt_alpha_mask=None,
                      image_name="x", uid=0, fid=torch.tensor(fid_val).float())


# ---------- model wrappers: each returns render_frame(f, T_full) -> HxWx3 ----------

def make_deformmlp_renderer(model_path, pipe, bg):
    mp = Path(model_path)
    iters = sorted(int(p.name.split("_")[-1]) for p in (mp / "point_cloud").iterdir()
                   if p.name.startswith("iteration_"))
    it = iters[-1]
    ds = torch.load(mp / "deform" / f"iteration_{it}" / "deform.pth", map_location="cuda")
    node_num = ds["nodes"].shape[0]; hyper_dim = ds["nodes"].shape[1] - 3
    g = load_gauss(mp / "point_cloud" / f"iteration_{it}" / "point_cloud.ply")
    deform = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                         hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                         pred_color=False, use_hash=False, hash_time=False,
                         d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                         with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                         is_scene_static=False)
    deform.load_weights(str(mp), iteration=it)
    print(f"  [{mp.name}] iter={it} N={g.get_xyz.shape[0]} nodes={node_num}")

    def render_frame(f, fov_x, FovY, T_full):
        t = int(f["frame_idx"])
        cam = make_cam(f, fov_x, FovY, float(t) / max(T_full - 1, 1))
        time_input = deform.deform.expand_time(cam.fid.to("cuda"))
        with torch.no_grad():
            d = deform.step(g.get_xyz.detach(), time_input, feature=g.feature,
                            motion_mask=getattr(g, "motion_mask", None), is_training=False)
            pkg = render(cam, g, pipe, bg, d_xyz=d["d_xyz"], d_rotation=d["d_rotation"],
                         d_scaling=d["d_scaling"], d_opacity=d.get("d_opacity"),
                         d_color=d.get("d_color"), d_rot_as_res=deform.d_rot_as_res)
        return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

    return render_frame


def make_partrigid_renderer(label, pipe, bg, d_rot_zero=True):
    s = np.load(REPO / f"outputs/custom/partrigid_{label}/partrigid_state.npz", allow_pickle=True)
    trans = s["trans"]; aa = s["aa"]; centers = s["arm_centers"]; lbs = s["lbs_weights"]
    K, T_train, _ = trans.shape
    scale_kt = s["scale"] if "scale" in s.files else None
    xyz_res = s["xyz_residual"] if "xyz_residual" in s.files else None
    arm_idx = s["arm_idx_for_residual"] if "arm_idx_for_residual" in s.files else None
    canon = CANON
    if "config" in s.files:
        cfg = s["config"].item() if hasattr(s["config"], "item") else s["config"]
        if isinstance(cfg, dict) and cfg.get("canon_ply"):
            canon = Path(cfg["canon_ply"])
    g = load_gauss(canon)
    xyz_canon = g.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    print(f"  [partrigid_{label}] K={K} N={N} canon={canon.name}")

    def render_frame(f, fov_x, FovY, T_full):
        t = int(f["frame_idx"]); tl = min(t, T_train - 1)
        R_all = aa2mat_np(aa[:, tl, :]); T_all = trans[:, tl, :]
        rel = xyz_canon[:, None, :] - centers[None, :, :]
        rotated = np.einsum("kij,nkj->nki", R_all, rel)
        new_per = rotated + centers[None, :, :] + T_all[None, :, :]
        weighted = (lbs[..., None] * new_per).sum(axis=1)
        w_total = lbs.sum(axis=1, keepdims=True).clip(min=0, max=1)
        new_xyz = weighted + (1 - w_total) * xyz_canon
        if xyz_res is not None and arm_idx is not None and len(arm_idx) > 0:
            new_xyz[arm_idx] = new_xyz[arm_idx] + xyz_res[:, tl, :]
        d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
        d_rot = torch.zeros(N, 4, device="cuda")
        if not d_rot_zero:
            d_rot = d_rot - torch.tensor([1, 0, 0, 0], device="cuda")
        d_sc = torch.zeros(N, 3, device="cuda")
        if scale_kt is not None:
            d_sc = torch.from_numpy((lbs @ scale_kt[:, tl, :]).astype(np.float32)).cuda()
        cam = make_cam(f, fov_x, FovY, 0.0)
        with torch.no_grad():
            pkg = render(cam, g, pipe, bg, d_xyz=d_xyz_t, d_rotation=d_rot,
                         d_scaling=d_sc, d_rot_as_res=True)
        return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)

    return render_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="lego_v2")
    p.add_argument("--ours_label", default="lego_v2_A1_leakfree_B")
    p.add_argument("--out", default="runs_aux/region_psnr_lego_v2.json")
    args = p.parse_args()

    scene_dir = REPO / "data/custom" / args.scene
    d3dgs_dir = REPO / "outputs/custom" / f"{args.scene}_d3dgs_ref" / "renders"
    meta_train = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_test = json.loads((scene_dir / "transforms_test.json").read_text())
    all_frames = meta_train["frames"] + meta_test["frames"]
    fov_x = meta_train["camera_angle_x"]
    FovY = focal2fov(fov2focal(fov_x, W), H)
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    print("[region-eval] loading models...")
    models = {
        "vanilla":   make_deformmlp_renderer(REPO / "outputs/custom/lego_v2_vanilla_sam_node", pipe, bg),
        "F1_warm":   make_deformmlp_renderer(REPO / "outputs/custom/lego_v2_F1_vanilla_warmstart_node", pipe, bg),
        "F2_frozen": make_deformmlp_renderer(REPO / "outputs/custom/F2_scgs_frozen", pipe, bg),
        "ours":      make_partrigid_renderer(args.ours_label, pipe, bg, d_rot_zero=True),
    }

    regions = ("full", "digger", "baseplate", "background")
    acc = {m: {r: [] for r in regions} for m in models}
    px_share = {r: [] for r in regions[1:]}

    for i, f in enumerate(all_frames):
        v = int(f["view_idx"]); t = int(f["frame_idx"])

        # masks
        png_name = f"{Path(f['file_path']).name}.png"
        sv4d_path = next(scene_dir / sp / png_name for sp in ("train", "test")
                         if (scene_dir / sp / png_name).exists())
        sv = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
        digger = sv[..., 3] > 0.5
        d3 = np.asarray(iio.imread(d3dgs_dir / f"{v * T_full + t:05d}.png"), dtype=np.float32) / 255.0
        d3_rgb = d3[..., :3]
        d3_fg = np.any(d3_rgb < 250.0 / 255.0, axis=-1)
        baseplate = d3_fg & ~digger
        background = ~d3_fg & ~digger
        masks = {"full": np.ones((H, W), bool), "digger": digger,
                 "baseplate": baseplate, "background": background}
        for r in regions[1:]:
            px_share[r].append(masks[r].mean())

        for name, rf in models.items():
            img = rf(f, fov_x, FovY, T_full)
            for r in regions:
                val = psnr_masked(img, d3_rgb, masks[r])
                if val is not None:
                    acc[name][r].append(val)
        if i % 21 == 0:
            print(f"  frame {i}/{len(all_frames)} (v={v} t={t})")

    print(f"\n[region-eval] {args.scene}: per-region PSNR vs clean d-3dgs GT (mean over frames)")
    print(f"  region pixel share: " + "  ".join(f"{r}={np.mean(px_share[r])*100:.1f}%" for r in regions[1:]))
    hdr = f"{'model':>12} | " + " | ".join(f"{r:>10}" for r in regions)
    print(hdr); print("-" * len(hdr))
    summary = {}
    for name in models:
        row = {r: (float(np.mean(acc[name][r])) if acc[name][r] else None) for r in regions}
        summary[name] = row
        print(f"{name:>12} | " + " | ".join(f"{row[r]:>10.2f}" for r in regions))

    out = REPO / args.out
    out.write_text(json.dumps({"scene": args.scene, "ours_label": args.ours_label,
                               "pixel_share": {r: float(np.mean(px_share[r])) for r in regions[1:]},
                               "psnr": summary}, indent=2))
    print(f"\n[region-eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
