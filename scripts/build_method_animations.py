"""Build per-view GIF animations comparing:
  Vanilla SC-GS (16M deform-MLP) vs Phase 2 ours (K=100+smart α=16 + xyz_res)

Each frame: [SV4D GT | d-3dgs CLEAN GT | Model render]
Output:
  runs_aux/method_animations/
    vanilla_v{0-4}.gif         per-view 21-frame GIF (vanilla SC-GS)
    ours_v{0-4}.gif            per-view 21-frame GIF (our best)
    sidebyside_v0.gif          (5-row) GT | d-3dgs | vanilla | ours
    contact_sheet_t{0,10,20}.png  5-views × {vanilla, ours} contact sheet
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

OUT = REPO / "runs_aux/method_animations"
SCENE = REPO / "data/custom/lego_v2"
D3DGS = REPO / "outputs/custom/lego_v2_d3dgs_ref/renders"
H = W = 576


def aa2mat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def setup_pipe(fov_x):
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    FovY = focal2fov(fov2focal(fov_x, W), H)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    return pipe, FovY, bg


def render_vanilla(g, deform, R, T, fov_x, FovY, fid, pipe, bg, N):
    cam = SCGSCamera(colmap_id=0, R=R, T=T, FoVx=fov_x, FoVy=FovY,
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
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def render_hier(s, g, xyz_canon, R, T, fov_x, FovY, t, pipe, bg, N):
    trans = s["trans"]; aa = s["aa"]; centers = s["arm_centers"]; lbs = s["lbs_weights"]
    T_train = trans.shape[1]
    tl = min(t, T_train - 1)
    R_all = aa2mat(aa[:, tl, :])
    T_all = trans[:, tl, :]
    rel = xyz_canon[:, None, :] - centers[None, :, :]
    rotated = np.einsum("kij,nkj->nki", R_all, rel)
    new_per = rotated + centers[None, :, :] + T_all[None, :, :]
    weighted = (lbs[..., None] * new_per).sum(axis=1)
    w_total = lbs.sum(axis=1, keepdims=True).clip(min=0, max=1)
    new_xyz = weighted + (1 - w_total) * xyz_canon
    if "xyz_residual" in s.files and "arm_idx_for_residual" in s.files:
        ai = s["arm_idx_for_residual"]
        if len(ai) > 0:
            new_xyz[ai] = new_xyz[ai] + s["xyz_residual"][:, tl, :]
    d_xyz_t = torch.from_numpy((new_xyz - xyz_canon).astype(np.float32)).cuda()
    d_rot = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
    d_sc = torch.zeros(N, 3, device="cuda")
    if "scale" in s.files:
        sb = lbs @ s["scale"][:, tl, :]
        d_sc = torch.from_numpy(sb.astype(np.float32)).cuda()
    cam = SCGSCamera(colmap_id=0, R=R, T=T, FoVx=fov_x, FoVy=FovY,
                     image=torch.zeros(3, H, W).cuda(), gt_alpha_mask=None,
                     image_name="x", uid=0, fid=torch.tensor(0.0).float())
    with torch.no_grad():
        pkg = render(cam, g, pipe, bg,
                     d_xyz=d_xyz_t, d_rotation=d_rot, d_scaling=d_sc, d_rot_as_res=True)
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def load_gt(scene, d3dgs, v, t):
    flat = v * 21 + t
    # SV4D
    sv4d_path = None
    for split in ("train", "test"):
        for cand in (scene / split / f"r_{flat:05d}.png",):
            if cand.exists(): sv4d_path = cand; break
        if sv4d_path: break
    rgba = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
    a = rgba[..., 3:4]
    sv4d = rgba[..., :3] * a + 1.0 * (1 - a)
    # d-3dgs
    d3 = np.asarray(iio.imread(d3dgs / f"{flat:05d}.png"), dtype=np.float32) / 255.0
    if d3.shape[-1] == 4:
        ad3 = d3[..., 3:4]; d3 = d3[..., :3] * ad3 + 1 * (1 - ad3)
    return sv4d, d3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames_vanilla").mkdir(exist_ok=True)
    (OUT / "frames_ours").mkdir(exist_ok=True)

    # Load vanilla
    vmp = REPO / "outputs/custom/lego_v2_vanilla_sam_node"
    iter_v = max(int(p.name.split("_")[-1]) for p in (vmp/"point_cloud").iterdir() if p.name.startswith("iter"))
    deform_state = torch.load(vmp/"deform"/f"iteration_{iter_v}"/"deform.pth", map_location="cuda")
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    g_v = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g_v.load_ply(str(vmp/"point_cloud"/f"iteration_{iter_v}"/"point_cloud.ply"), og_number_points=0)
    Nv = g_v.get_xyz.shape[0]
    deform_v = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                            hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                            pred_color=False, use_hash=False, hash_time=False,
                            d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                            with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                            is_scene_static=False)
    deform_v.load_weights(str(vmp), iteration=iter_v)
    print(f"[anim] vanilla loaded: N={Nv} nodes={node_num}")

    # Load ours (best A1)
    canon = REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
    g_o = None
    for fdim in (8, 2, 0):
        try:
            g_o = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g_o.load_ply(str(canon), og_number_points=0); break
        except Exception:
            g_o = None
    xyz_canon = g_o.get_xyz.detach().cpu().numpy()
    No = xyz_canon.shape[0]
    state = np.load(REPO/"outputs/custom/partrigid_lego_v2_alpha16/partrigid_state.npz", allow_pickle=True)
    print(f"[anim] ours loaded: canon N={No}")

    # Setup
    meta_t = json.loads((SCENE/"transforms_train.json").read_text())
    meta_te = json.loads((SCENE/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    pipe, FovY, bg = setup_pipe(fov_x)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Group frames by view
    by_view = {v: [None]*T_full for v in range(5)}
    for f in all_frames:
        by_view[int(f["view_idx"])][int(f["frame_idx"])] = f

    # Per-view animation
    per_view_psnr = {"vanilla": {v: [] for v in range(5)}, "ours": {v: [] for v in range(5)}}

    for v in range(5):
        frames_van = []
        frames_ours = []
        for t in range(T_full):
            f = by_view[v][t]
            c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
            M = np.linalg.inv(c2w)
            R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
            Tcam = -M[:3, 3]
            fid = float(t) / max(T_full - 1, 1)

            sv4d, d3 = load_gt(SCENE, D3DGS, v, t)
            pred_v = render_vanilla(g_v, deform_v, R, Tcam, fov_x, FovY, fid, pipe, bg, Nv)
            pred_o = render_hier(state, g_o, xyz_canon, R, Tcam, fov_x, FovY, t, pipe, bg, No)

            per_view_psnr["vanilla"][v].append(psnr(pred_v, d3))
            per_view_psnr["ours"][v].append(psnr(pred_o, d3))

            sep = np.ones((H, 4, 3))
            row_v = np.concatenate([sv4d, sep, d3, sep, pred_v], axis=1)
            row_o = np.concatenate([sv4d, sep, d3, sep, pred_o], axis=1)
            for row, kind, frames in [(row_v, "vanilla", frames_van), (row_o, "ours", frames_ours)]:
                pil = Image.fromarray((row * 255).astype(np.uint8))
                top = Image.new("RGB", (pil.width, pil.height + 28), (255, 255, 255))
                d = ImageDraw.Draw(top)
                col_w = W + 4
                d.text((W//2 - 50, 5), "SV4D GT", fill="black", font=font)
                d.text((col_w + W//2 - 70, 5), "d-3dgs clean", fill="black", font=font)
                d.text((2*col_w + W//2 - 100, 5),
                       f"{kind} ({per_view_psnr[kind][v][-1]:.2f} dB)",
                       fill=(180,0,0) if kind=="ours" else (50,50,50), font=font)
                d.text((top.width - 70, 5), f"v={v} t={t:02d}", fill="black", font=font)
                top.paste(pil, (0, 28))
                frames.append(top)
            # Save still for keyframes
            if t in (0, 10, 20):
                Image.fromarray((row_v*255).astype(np.uint8)).save(OUT/"frames_vanilla"/f"v{v}_t{t:02d}.png")
                Image.fromarray((row_o*255).astype(np.uint8)).save(OUT/"frames_ours"/f"v{v}_t{t:02d}.png")

        frames_van[0].save(OUT/f"vanilla_v{v}.gif", save_all=True,
                            append_images=frames_van[1:], duration=350, loop=0)
        frames_ours[0].save(OUT/f"ours_v{v}.gif", save_all=True,
                             append_images=frames_ours[1:], duration=350, loop=0)
        vp = np.mean(per_view_psnr["vanilla"][v])
        op = np.mean(per_view_psnr["ours"][v])
        print(f"[anim] view {v} mean PSNR vs d-3dgs: vanilla={vp:.2f}  ours={op:.2f}  Δ={op-vp:+.2f}")

    # Overall
    all_v = np.array(sum(per_view_psnr["vanilla"].values(), []))
    all_o = np.array(sum(per_view_psnr["ours"].values(), []))
    print(f"\n[anim] OVERALL vs d-3dgs CLEAN GT:")
    print(f"  vanilla SC-GS : mean={all_v.mean():.3f}  median={np.median(all_v):.3f}")
    print(f"  ours (A1)     : mean={all_o.mean():.3f}  median={np.median(all_o):.3f}")
    print(f"  Δ             : {all_o.mean() - all_v.mean():+.3f} dB")
    print(f"\n[anim] outputs in {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
