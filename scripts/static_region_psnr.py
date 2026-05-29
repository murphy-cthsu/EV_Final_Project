"""C) Static-region PSNR.

Mask out moving (bucket+arm) region using per-pixel temporal variance over
the 21 SV4D frames at each view. Evaluate PSNR of our part-rigid model and
(optionally) clean ref at the *static body* region only — this measures pure
structural reconstruction quality, decoupled from the animation mismatch.

Outputs:
  - static_masks/view{V}.png        (binary masks per view)
  - static_psnr_table.json
  - static_vs_full_bar.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"


def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / theta
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    th = theta[..., None]
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def load_sv4d_rgb(p: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a)


def compute_static_mask(sv4d_dir: Path, view: int, T_full: int,
                        var_thresh_pct: float = 30.0):
    """Static mask: pixels with low temporal variance + foreground (alpha>0 in any frame)."""
    imgs = []
    alphas = []
    for t in range(T_full):
        rgba = np.asarray(Image.open(sv4d_dir / f"r_{view*T_full+t:05d}.png").convert("RGBA"),
                          dtype=np.float32) / 255.0
        imgs.append(rgba[..., :3] * rgba[..., 3:4] + 1.0 * (1 - rgba[..., 3:4]))
        alphas.append(rgba[..., 3])
    imgs = np.stack(imgs, axis=0)        # [T, H, W, 3]
    alphas = np.stack(alphas, axis=0)    # [T, H, W]
    # Temporal variance — sum across RGB channels
    var = imgs.var(axis=0).mean(axis=-1)  # [H, W]
    fg_any = (alphas > 0.1).any(axis=0)   # [H, W]
    fg_pixels = var[fg_any]
    # Static = bottom `var_thresh_pct`% of variance among FG pixels
    if fg_pixels.size > 0:
        thresh = np.percentile(fg_pixels, var_thresh_pct)
    else:
        thresh = 0.0
    static = (var < thresh) & fg_any
    return static.astype(np.float32), var, fg_any


def psnr(pred, gt, mask):
    """PSNR of (pred, gt) over `mask>0.5` region. Mask shape [H, W]."""
    m = mask > 0.5
    if not m.any(): return float("nan")
    diff2 = ((pred - gt) ** 2).mean(axis=-1)
    mse = diff2[m].mean()
    return -10 * math.log10(max(float(mse), 1e-12))


def render_partrigid(gaussians, xyz_canon, arm_trans, arm_aa, arm_pivot, arm_weights,
                     v_R, v_T, fov_x, FovY, H, W, t_idx, T_train, pipe, bg, N):
    t_lo = min(t_idx, T_train - 1)
    R_t = axis_angle_to_matrix_np(arm_aa[t_lo:t_lo+1])[0]
    rel = xyz_canon - arm_pivot
    rotated = rel @ R_t.T
    arm_xyz = rotated + arm_pivot + arm_trans[t_lo]
    new_xyz = arm_weights[:, None] * arm_xyz + (1 - arm_weights[:, None]) * xyz_canon
    d_xyz_np = new_xyz - xyz_canon

    d_xyz_t = torch.from_numpy(d_xyz_np).float().cuda()
    d_rotation = torch.zeros(N, 4, device="cuda")
    d_rotation = d_rotation - torch.tensor([1, 0, 0, 0], device="cuda")
    d_scaling = torch.zeros(N, 3, device="cuda")
    dummy = torch.zeros(3, H, W, dtype=torch.float32)
    cam = SCGSCamera(colmap_id=0, R=v_R, T=v_T, FoVx=fov_x, FoVy=FovY,
                     image=dummy, gt_alpha_mask=None,
                     image_name="x", uid=0, fid=torch.tensor(0.0).float())
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg,
                     d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                     d_rot_as_res=True)
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--partrigid_label", default="lbs_photo1")
    p.add_argument("--sv4d_train_meta", default=REPO / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--sv4d_train_dir",  default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--out_dir",         default=REPO / "runs_aux/static_region_C")
    p.add_argument("--var_thresh_pct",  type=float, default=30.0,
                   help="bottom-percentile of FG temporal variance counted as static")
    p.add_argument("--clean_fine_dir",  default=REPO / "runs_aux/clean_gt_fine/renders",
                   help="fine-grid clean-ref renders for upper-bound comparison")
    p.add_argument("--matching_map",    default=REPO / "runs_aux/alignment_A/matching_map.json",
                   help="if exists, use matched fid for clean ref upper bound; else use fid==t/(T-1)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "static_masks").mkdir(exist_ok=True)
    (out_dir / "per_frame_renders").mkdir(exist_ok=True)

    # Load SV4D meta
    meta = json.loads(Path(args.sv4d_train_meta).read_text())
    fov_x = meta["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
    H = W = 576
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)

    # 1) compute static masks per view
    print("[C] computing per-view static masks (temporal variance)…")
    static_masks = {}
    for v in range(5):
        m, var, fg = compute_static_mask(Path(args.sv4d_train_dir), v, T_full,
                                          var_thresh_pct=args.var_thresh_pct)
        static_masks[v] = m
        Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / "static_masks" / f"view{v}.png")
        # Also save variance heatmap for diagnosis
        vmax = max(var.max(), 1e-8)
        vis = (var / vmax * 255).astype(np.uint8)
        Image.fromarray(vis).save(out_dir / "static_masks" / f"view{v}_variance.png")
        print(f"[C] view {v}: static area = {int(m.sum())}/{int(fg.sum())} fg px  ({m.sum()/max(fg.sum(),1)*100:.1f}%)")

    # 2) load part-rigid state + canonical
    state = np.load(REPO / f"outputs/custom/partrigid_{args.partrigid_label}/partrigid_state.npz",
                    allow_pickle=True)
    arm_trans = state["arm_trans"]; arm_aa = state["arm_aa"]
    arm_pivot = state["arm_pivot"]
    if "arm_weights" in state.files:
        arm_weights = state["arm_weights"]
        print(f"[C] part-rigid LBS weights mean={arm_weights.mean():.3f}")
    else:
        arm_weights = (state["part_id"] == 0).astype(np.float32)
    T_train = arm_trans.shape[0]

    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Build per-view R, T
    cams_by_view = {}
    for f in meta["frames"]:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        cams_by_view[v] = (R, Tr)

    # 3) matching map (for clean ref upper bound)
    if Path(args.matching_map).exists():
        match = json.loads(Path(args.matching_map).read_text())
        print(f"[C] using matching map from {args.matching_map}")
    else:
        match = None
        print("[C] no matching map — skipping clean ref upper bound")

    # 4) eval per frame
    results = {"per_frame": [], "per_view_static": {}, "per_view_full": {},
               "per_view_clean_static": {}}
    psnr_static_partrigid = {v: [] for v in range(5)}
    psnr_full_partrigid = {v: [] for v in range(5)}
    psnr_static_clean = {v: [] for v in range(5)}
    for f in meta["frames"]:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        R, Tr = cams_by_view[v]
        gt_path = Path(args.sv4d_train_dir) / f"{Path(f['file_path']).name}.png"
        gt = load_sv4d_rgb(gt_path)
        if gt.shape[:2] != (H, W):
            gt = np.asarray(Image.fromarray((gt*255).astype(np.uint8)).resize((W, H)), dtype=np.float32)/255

        pred = render_partrigid(gaussians, xyz_canon, arm_trans, arm_aa, arm_pivot,
                                arm_weights, R, Tr, fov_x, FovY, H, W, t, T_train,
                                pipe, bg, N)
        sm = static_masks[v]
        fg_any = (sm > 0) | (np.abs(gt - 1).sum(-1) > 0.05)
        ps_static = psnr(pred, gt, sm)
        ps_full = psnr(pred, gt, fg_any.astype(np.float32))
        psnr_static_partrigid[v].append(ps_static)
        psnr_full_partrigid[v].append(ps_full)

        # clean ref upper bound (if matching available)
        ps_clean_static = float("nan")
        if match is not None:
            key = f"v{v}_t{t}"
            if key in match:
                fi = match[key]["best_fid_idx"]
                clean_p = Path(args.clean_fine_dir) / f"r_v{v}_f{fi:03d}.png"
                clean = np.asarray(Image.open(clean_p).convert("RGB"), dtype=np.float32) / 255.0
                if clean.shape[:2] != (H, W):
                    clean = np.asarray(Image.fromarray((clean*255).astype(np.uint8)).resize((W, H)),
                                        dtype=np.float32)/255
                ps_clean_static = psnr(clean, gt, sm)
                psnr_static_clean[v].append(ps_clean_static)

        results["per_frame"].append({
            "v": v, "t": t,
            "psnr_static_partrigid": ps_static,
            "psnr_full_partrigid": ps_full,
            "psnr_static_clean_aligned": ps_clean_static,
        })

    for v in range(5):
        results["per_view_static"][str(v)]      = float(np.mean(psnr_static_partrigid[v]))
        results["per_view_full"][str(v)]        = float(np.mean(psnr_full_partrigid[v]))
        if psnr_static_clean[v]:
            results["per_view_clean_static"][str(v)] = float(np.mean(psnr_static_clean[v]))

    all_static = [r["psnr_static_partrigid"] for r in results["per_frame"]]
    all_full   = [r["psnr_full_partrigid"]   for r in results["per_frame"]]
    all_clean  = [r["psnr_static_clean_aligned"] for r in results["per_frame"]
                  if not math.isnan(r["psnr_static_clean_aligned"])]
    results["overall"] = {
        "psnr_static_partrigid_mean": float(np.mean(all_static)),
        "psnr_full_partrigid_mean":   float(np.mean(all_full)),
        "psnr_static_clean_aligned_mean": float(np.mean(all_clean)) if all_clean else None,
    }
    (out_dir / "static_psnr_table.json").write_text(json.dumps(results, indent=2))

    # Bar chart: static vs full per view
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(5); w = 0.28
    ax.bar(xs - w, [results["per_view_full"][str(v)] for v in range(5)], w,
           label="Part-rigid LBS (full FG)", color="#1f77b4")
    ax.bar(xs, [results["per_view_static"][str(v)] for v in range(5)], w,
           label="Part-rigid LBS (static region only)", color="#2ca02c")
    if any(psnr_static_clean[v] for v in range(5)):
        ax.bar(xs + w, [results["per_view_clean_static"].get(str(v), 0) for v in range(5)], w,
               label="Clean ref @ aligned fid (static, upper bound)", color="#d62728")
    ax.set_xticks(xs); ax.set_xticklabels([f"v{v}" for v in range(5)])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Static vs full-FG PSNR  (var_thresh={args.var_thresh_pct}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "static_vs_full_bar.png", dpi=130)
    plt.close()

    print()
    print(f"[C] === Static-region PSNR summary ===")
    print(f"[C] Part-rigid full-FG  PSNR = {results['overall']['psnr_full_partrigid_mean']:.3f}")
    print(f"[C] Part-rigid STATIC   PSNR = {results['overall']['psnr_static_partrigid_mean']:.3f}")
    if results['overall']['psnr_static_clean_aligned_mean']:
        print(f"[C] Clean ref STATIC    PSNR = {results['overall']['psnr_static_clean_aligned_mean']:.3f}  (upper bound)")
    print(f"[C] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
