"""Motion-specific evaluation metrics — prove the model LEARNED motion, not
just fit per-frame appearance.

Metrics:
  1. Frame-to-frame Δ match (SSIM of |pred[t+1]-pred[t]| vs |GT[t+1]-GT[t]|)
     High = motion direction + magnitude correct.
  2. Motion-region IoU: where the model's render shows temporal variance
     vs where GT shows it. High = "what's moving" agrees with GT.
  3. Static-region stability: temporal jitter in static regions of GT.
     Low = body/cabin doesn't wobble.
  4. Motion magnitude correlation: per-pixel temporal std, Pearson correlation
     between pred and GT — does the model match the "intensity of motion"?

Computed per view × across-time, aggregated.
Compares vanilla SC-GS vs Phase 2 ours.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


def aa2mat(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    ax = aa / th
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]
    K[:, 1, 0] = ax[:, 2];  K[:, 1, 2] = -ax[:, 0]
    K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    return I + np.sin(th[..., None]) * K + (1 - np.cos(th[..., None])) * (K @ K)


def render_vanilla(g, deform, R, T, fov_x, FovY, fid, pipe, bg, H, W):
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


def render_hier(s, g, xyz_canon, R, T, fov_x, FovY, t, pipe, bg, H, W, N):
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


def main():
    SCENE = REPO / "data/custom/lego_v2"
    D3DGS = REPO / "outputs/custom/lego_v2_d3dgs_ref/renders"
    OUT = REPO / "runs_aux/motion_metric_eval"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load models
    vmp = REPO / "outputs/custom/lego_v2_vanilla_sam_node"
    iter_v = max(int(p.name.split("_")[-1]) for p in (vmp/"point_cloud").iterdir() if p.name.startswith("iter"))
    deform_state = torch.load(vmp/"deform"/f"iteration_{iter_v}"/"deform.pth", map_location="cuda")
    node_num = deform_state["nodes"].shape[0]
    hyper_dim = deform_state["nodes"].shape[1] - 3
    g_v = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g_v.load_ply(str(vmp/"point_cloud"/f"iteration_{iter_v}"/"point_cloud.ply"), og_number_points=0)
    deform_v = DeformModel(K=4, deform_type="node", is_blender=True, skinning=False,
                            hyper_dim=hyper_dim, node_num=node_num, pred_opacity=False,
                            pred_color=False, use_hash=False, hash_time=False,
                            d_rot_as_res=True, local_frame=True, progressive_brand_time=False,
                            with_arap_loss=True, max_d_scale=-1, enable_densify_prune=False,
                            is_scene_static=False)
    deform_v.load_weights(str(vmp), iteration=iter_v)
    print(f"[motion] vanilla loaded")

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
    s_o = np.load(REPO/"outputs/custom/partrigid_lego_v2_alpha16/partrigid_state.npz", allow_pickle=True)
    print(f"[motion] ours loaded")

    meta_t = json.loads((SCENE/"transforms_train.json").read_text())
    meta_te = json.loads((SCENE/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    by_view = {v: [None]*T_full for v in range(5)}
    for f in all_frames:
        by_view[int(f["view_idx"])][int(f["frame_idx"])] = f

    # Render all (v, t) for both methods + load GT
    print("[motion] rendering all frames…")
    renders = {"vanilla": {}, "ours": {}, "gt": {}}
    for v in range(5):
        renders["vanilla"][v] = []
        renders["ours"][v] = []
        renders["gt"][v] = []
        for t in range(T_full):
            f = by_view[v][t]
            c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
            M = np.linalg.inv(c2w)
            R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
            Tcam = -M[:3, 3]
            fid = float(t) / max(T_full - 1, 1)
            renders["vanilla"][v].append(
                render_vanilla(g_v, deform_v, R, Tcam, fov_x, FovY, fid, pipe, bg, H, W))
            renders["ours"][v].append(
                render_hier(s_o, g_o, xyz_canon, R, Tcam, fov_x, FovY, t, pipe, bg, H, W, No))
            d3 = np.asarray(iio.imread(D3DGS / f"{v*T_full+t:05d}.png"), dtype=np.float32) / 255
            if d3.shape[-1] == 4:
                a = d3[..., 3:4]; d3 = d3[..., :3]*a + 1*(1-a)
            renders["gt"][v].append(d3)
        print(f"[motion] view {v} rendered")

    # === Compute motion metrics per view ===
    def temporal_std(seq):  # seq: (T, H, W, 3)
        return np.stack(seq, axis=0).std(axis=0).mean(axis=-1)  # (H, W) grayscale std

    def frame_deltas(seq):
        deltas = []
        for t in range(len(seq) - 1):
            d = np.abs(seq[t+1] - seq[t]).mean(axis=-1)  # (H, W)
            deltas.append(d)
        return np.stack(deltas, axis=0)  # (T-1, H, W)

    results = {}
    for method in ["vanilla", "ours"]:
        print(f"\n[motion] === Computing {method} motion metrics ===")
        m1 = []  # frame-to-frame Δ SSIM
        m2 = []  # motion-region IoU
        m3 = []  # static-region jitter
        m4 = []  # motion magnitude correlation
        for v in range(5):
            seq_p = renders[method][v]
            seq_g = renders["gt"][v]
            std_p = temporal_std(seq_p)
            std_g = temporal_std(seq_g)
            # M1: frame-delta SSIM
            d_p = frame_deltas(seq_p)
            d_g = frame_deltas(seq_g)
            ssim_deltas = []
            for t in range(len(d_p)):
                s = ssim(d_p[t], d_g[t], data_range=1.0)
                ssim_deltas.append(s)
            m1.append(np.mean(ssim_deltas))

            # M2: motion-region IoU (top quartile of std)
            thresh_g = np.percentile(std_g[std_g > 0], 75) if (std_g > 0).any() else 0
            thresh_p = np.percentile(std_p[std_p > 0], 75) if (std_p > 0).any() else 0
            mm_g = std_g > thresh_g
            mm_p = std_p > thresh_p
            inter = (mm_g & mm_p).sum()
            union = (mm_g | mm_p).sum()
            m2.append(inter / max(union, 1))

            # M3: static-region jitter (mean of pred std where GT std is low)
            static_g = std_g < np.percentile(std_g, 25)
            jitter = std_p[static_g].mean() if static_g.any() else 0
            m3.append(jitter)

            # M4: per-pixel motion magnitude correlation (Pearson)
            mask = (std_g + std_p) > 1e-4
            if mask.sum() > 100:
                from scipy.stats import pearsonr
                r, _ = pearsonr(std_p[mask], std_g[mask])
                m4.append(r)
            else:
                m4.append(0)
        results[method] = {
            "m1_frame_delta_ssim_mean": float(np.mean(m1)),
            "m2_motion_iou_mean": float(np.mean(m2)),
            "m3_static_jitter_mean": float(np.mean(m3)),
            "m4_motion_mag_corr_mean": float(np.mean(m4)),
            "per_view_m1": [float(x) for x in m1],
            "per_view_m2": [float(x) for x in m2],
            "per_view_m3": [float(x) for x in m3],
            "per_view_m4": [float(x) for x in m4],
        }
        for k in ["m1_frame_delta_ssim_mean", "m2_motion_iou_mean", "m3_static_jitter_mean", "m4_motion_mag_corr_mean"]:
            print(f"  {k:40s} = {results[method][k]:.4f}")

    print(f"\n[motion] === COMPARISON (vs d-3dgs CLEAN GT motion) ===")
    print(f"  {'Metric':<35}  {'Vanilla':<12}  {'Ours (A1)':<12}  {'Δ':<10}  Winner")
    for k, name, higher_better in [
        ("m1_frame_delta_ssim_mean", "Frame-Δ SSIM (motion match)", True),
        ("m2_motion_iou_mean",       "Motion-region IoU",          True),
        ("m3_static_jitter_mean",    "Static-region jitter",       False),
        ("m4_motion_mag_corr_mean",  "Motion magnitude correlation", True),
    ]:
        vv = results["vanilla"][k]; oo = results["ours"][k]
        winner = "ours" if (oo > vv if higher_better else oo < vv) else "vanilla"
        marker = "✅" if winner == "ours" else "❌"
        print(f"  {name:<35}  {vv:<12.4f}  {oo:<12.4f}  {oo-vv:<+10.4f}  {winner} {marker}")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[motion] saved {OUT}/summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
