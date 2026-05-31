"""Multi-metric evaluation comparing methods on lego_v2 against d-3dgs CLEAN GT.

Metrics computed per (view, time):
  PSNR             — pixel L2 (current default, has known bias)
  SSIM             — structural similarity (sensitive to local structure)
  LPIPS            — perceptual similarity (AlexNet features, robust to small offsets)
  DINOv2-feat L2   — semantic feature distance (patch-level)
  FG-IoU           — foreground silhouette overlap (binary mask)
  Edge L1          — Sobel-edge L1 (sharpness proxy; high blur → low edge magnitude)

Eval two methods side-by-side:
  vanilla_path: path to vanilla SC-GS model
  ours_path:    path to our hier+smart+scale+xyz_res model
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
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


def psnr_fn(a, b):
    """a, b shape (H, W, 3) [0,1]."""
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def ssim_fn(a, b):
    """Lightweight SSIM via skimage."""
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, channel_axis=-1, data_range=1.0)


def lpips_fn(a, b, lpips_model):
    """a, b [0,1] (H, W, 3) → LPIPS distance (low = similar)."""
    at = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    bt = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    with torch.no_grad():
        d = lpips_model(at, bt)
    return float(d.item())


def sobel_edge_l1(a, b):
    """L1 of Sobel-edge magnitudes — penalizes blur (low edge gradient)."""
    at = torch.from_numpy(a.mean(-1)).float().unsqueeze(0).unsqueeze(0).cuda()
    bt = torch.from_numpy(b.mean(-1)).float().unsqueeze(0).unsqueeze(0).cuda()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda").reshape(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device="cuda").reshape(1, 1, 3, 3)
    ea_x = F.conv2d(at, kx, padding=1); ea_y = F.conv2d(at, ky, padding=1)
    eb_x = F.conv2d(bt, kx, padding=1); eb_y = F.conv2d(bt, ky, padding=1)
    edge_a = (ea_x.abs() + ea_y.abs()).squeeze().cpu().numpy()
    edge_b = (eb_x.abs() + eb_y.abs()).squeeze().cpu().numpy()
    # L1 between edge maps (high = more edge-mismatch, captures blur)
    return float(np.abs(edge_a - edge_b).mean()), float(edge_a.mean()), float(edge_b.mean())


def fg_iou(a, b, thresh=0.95):
    """IoU on foreground: 1 - alpha-equivalent (where pixel != white-bg)."""
    a_fg = (np.abs(a - 1).sum(-1) > 0.05).astype(np.float32)
    b_fg = (np.abs(b - 1).sum(-1) > 0.05).astype(np.float32)
    inter = (a_fg * b_fg).sum()
    union = ((a_fg + b_fg).clip(max=1)).sum()
    return float(inter / max(union, 1e-6))


def dinov2_feat_dist(a, b, dino_model):
    """Patch-level DINOv2 feature L2 distance (semantic similarity)."""
    # Resize to 224
    at = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().cuda()
    bt = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float().cuda()
    at = F.interpolate(at, size=224, mode="bilinear", align_corners=False)
    bt = F.interpolate(bt, size=224, mode="bilinear", align_corners=False)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").reshape(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").reshape(1, 3, 1, 1)
    at = (at - mean) / std
    bt = (bt - mean) / std
    with torch.no_grad():
        fa = dino_model.forward_features(at)
        fb = dino_model.forward_features(bt)
        # Use patch tokens (B, N, D) — skip CLS
        if isinstance(fa, dict):
            fa = fa["x_norm_patchtokens"] if "x_norm_patchtokens" in fa else list(fa.values())[0]
            fb = fb["x_norm_patchtokens"] if "x_norm_patchtokens" in fb else list(fb.values())[0]
        # Cosine similarity per patch
        fa = F.normalize(fa, dim=-1); fb = F.normalize(fb, dim=-1)
        sim = (fa * fb).sum(-1).mean()
    return float(1 - sim.item())  # distance, low = similar


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


def load_d3dgs(d3dgs_dir, v, t, T_full=21):
    flat = v * T_full + t
    rgba = np.asarray(iio.imread(d3dgs_dir / f"{flat:05d}.png"), dtype=np.float32) / 255.0
    if rgba.shape[-1] == 4:
        a = rgba[..., 3:4]; return rgba[..., :3] * a + 1 * (1 - a)
    return rgba[..., :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_label", default="lego_v2_vanilla_sam_node")
    p.add_argument("--ours_label", default="partrigid_lego_v2_alpha16")
    p.add_argument("--canon_ply", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--scene_dir", default=REPO / "data/custom/lego_v2")
    p.add_argument("--d3dgs_dir", default=REPO / "outputs/custom/lego_v2_d3dgs_ref/renders")
    p.add_argument("--out_dir", default=REPO / "runs_aux/multi_metric_eval")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load metrics models
    print("[multi] loading LPIPS (alex)...")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").cuda()
    print("[multi] loading DINOv2 (vits14)...")
    try:
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
        use_dino = True
    except Exception as e:
        print(f"[multi] DINOv2 unavailable: {e}; skipping")
        dino_model = None
        use_dino = False

    # Load vanilla
    vmp = REPO / f"outputs/custom/{args.vanilla_label}"
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
    print(f"[multi] vanilla loaded: N={g_v.get_xyz.shape[0]}")

    # Load ours
    g_o = None
    for fdim in (8, 2, 0):
        try:
            g_o = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g_o.load_ply(str(args.canon_ply), og_number_points=0); break
        except Exception:
            g_o = None
    xyz_canon = g_o.get_xyz.detach().cpu().numpy()
    No = xyz_canon.shape[0]
    s_o = np.load(REPO / f"outputs/custom/{args.ours_label}/partrigid_state.npz", allow_pickle=True)
    print(f"[multi] ours loaded: canon N={No}")

    # Setup
    meta_t = json.loads((args.scene_dir/"transforms_train.json").read_text())
    meta_te = json.loads((args.scene_dir/"transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    rows = []
    for i, f in enumerate(all_frames):
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tcam = -M[:3, 3]
        fid = float(t) / max(T_full - 1, 1)

        gt = load_d3dgs(args.d3dgs_dir, v, t, T_full)
        pred_v = render_vanilla(g_v, deform_v, R, Tcam, fov_x, FovY, fid, pipe, bg, H, W)
        pred_o = render_hier(s_o, g_o, xyz_canon, R, Tcam, fov_x, FovY, t, pipe, bg, H, W, No)

        row = {"view": v, "time": t}
        for name, pred in [("vanilla", pred_v), ("ours", pred_o)]:
            row[f"{name}_psnr"]  = psnr_fn(pred, gt)
            row[f"{name}_ssim"]  = ssim_fn(pred, gt)
            row[f"{name}_lpips"] = lpips_fn(pred, gt, lpips_model)
            edge_diff, edge_pred, edge_gt = sobel_edge_l1(pred, gt)
            row[f"{name}_edge_diff"] = edge_diff
            row[f"{name}_edge_pred"] = edge_pred
            row[f"{name}_iou"]   = fg_iou(pred, gt)
            if use_dino:
                row[f"{name}_dino"] = dinov2_feat_dist(pred, gt, dino_model)
        rows.append(row)
        if i % 20 == 0:
            print(f"[multi] {i+1}/{len(all_frames)} (v={v} t={t})")

    # Aggregate
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("view", "time")}
    summary = {}
    metrics_to_print = ["psnr", "ssim", "lpips", "edge_diff", "edge_pred", "iou"] + (["dino"] if use_dino else [])
    print(f"\n[multi] === FINAL: 105 frames vs d-3dgs CLEAN GT ===")
    print(f"  {'Metric':<12}  {'Vanilla SC-GS':<18}  {'Ours (A1)':<18}  {'Δ':<10}  Winner")
    for m in metrics_to_print:
        vk = f"vanilla_{m}"; ok = f"ours_{m}"
        vv = arr[vk].mean(); oo = arr[ok].mean()
        higher_better = m in ("psnr", "ssim", "iou", "edge_pred")
        if higher_better:
            winner = "ours" if oo > vv else "vanilla"
            delta = oo - vv
        else:
            winner = "ours" if oo < vv else "vanilla"
            delta = oo - vv  # negative = ours better
        summary[m] = {"vanilla": float(vv), "ours": float(oo), "delta": float(delta), "winner": winner}
        marker = "✅" if winner == "ours" else "❌"
        print(f"  {m:<12}  {vv:<18.4f}  {oo:<18.4f}  {delta:<+10.4f}  {winner} {marker}")
    print()
    if "edge_pred" in arr:
        gt_edge = sobel_edge_l1(np.zeros_like(gt), gt)[2]
        print(f"  (reference d-3dgs edge magnitude: {gt_edge:.4f} — both methods should match this)")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    rows_serializable = [{k: (float(v) if hasattr(v, "item") else v) for k, v in r.items()} for r in rows]
    (out / "per_frame.json").write_text(json.dumps(rows_serializable, indent=2))
    print(f"\n[multi] saved {out}/summary.json + per_frame.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
