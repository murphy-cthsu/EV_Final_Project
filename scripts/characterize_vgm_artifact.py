"""Characterize VGM artifacts using a trained SC-GS canonical as a 3D probe.

The idea: train a single canonical-4D-Gaussian model that has to fit ALL V views
across T timesteps (the SC-GS train run with no held-out view -- our v5). The
per-pixel residual |gt - render| inside the foreground mask is then the portion
of the VGM output that "no single 3D model can simultaneously explain". This is
a direct lower bound on multi-view + temporal inconsistency of the VGM.

Inputs (already on disk from prior runs):
    data/custom/scene00_masked/train/r_NNNNN.png    RGBA (alpha = SAM-2 mask)
    outputs/custom/scene00_v5_node/train/ours_30000/renders/NNNNN.png   RGBA
    outputs/custom/scene00_v5_node/train/ours_30000/gt/NNNNN.png        RGB

Outputs:
    runs_aux/vgm_artifact/
        per_view_per_time_residual.npy   float32 (V, T) mean |Δ| per cell
        per_view_per_time_psnr.npy       float32 (V, T) PSNR per cell
        heatmap_residual.png             V x T grid
        per_view_curve.png               1 line per view, residual vs t
        per_time_curve.png               1 line summarizing across views vs t
        spatial_avg_residual.png         where on the image artifacts cluster
        worst_frames_panel.png           top-9 highest-residual frames
        SUMMARY.md                       short narrative + the numbers

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/characterize_vgm_artifact.py \\
        --scene_dir   data/custom/scene00_masked \\
        --render_dir  outputs/custom/scene00_v5_node/train/ours_30000 \\
        --out_dir     runs_aux/vgm_artifact \\
        --n_views 5 --n_frames 21
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rgba(path: Path) -> np.ndarray:
    a = np.asarray(iio.imread(path))
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 3:
        a = np.concatenate([a, 255 * np.ones((*a.shape[:2], 1), dtype=a.dtype)], axis=-1)
    return a.astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_dir", type=Path, required=True,
                   help="data/custom/scene00_masked (has source RGBA + mask)")
    p.add_argument("--render_dir", type=Path, required=True,
                   help="outputs/custom/.../train/ours_30000")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_views", type=int, required=True)
    p.add_argument("--n_frames", type=int, required=True)
    args = p.parse_args()

    V, T = args.n_views, args.n_frames
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[characterize] {V} views x {T} frames = {V*T} cells")
    print(f"[characterize] scene = {args.scene_dir}")
    print(f"[characterize] render = {args.render_dir}")

    # Per-cell metrics. residual = mean |gt-render| inside FG mask, intensity scale [0, 255].
    residual = np.zeros((V, T), dtype=np.float32)
    psnr_mat = np.zeros((V, T), dtype=np.float32)
    fg_frac_mat = np.zeros((V, T), dtype=np.float32)
    # Spatial accumulator (averaged over all cells, mask-weighted)
    spatial_sum = np.zeros((576, 576), dtype=np.float64)
    spatial_count = np.zeros((576, 576), dtype=np.float64)
    # For "worst frames" panel
    cell_records: list[dict] = []

    for v in range(V):
        for t in range(T):
            flat_idx = v * T + t
            # Source (defines the FG alpha mask)
            src = load_rgba(args.scene_dir / "train" / f"r_{flat_idx:05d}.png")
            alpha = src[..., 3].astype(np.float32) / 255.0
            fg = alpha > 0.5
            # Rendered + GT from the render dir
            r_path = args.render_dir / "renders" / f"{flat_idx:05d}.png"
            g_path = args.render_dir / "gt" / f"{flat_idx:05d}.png"
            r = load_rgba(r_path)[..., :3].astype(np.float32)
            g = load_rgba(g_path)[..., :3].astype(np.float32)
            if r.shape != g.shape:
                raise RuntimeError(f"shape mismatch at flat_idx={flat_idx}: "
                                   f"render={r.shape} gt={g.shape}")
            diff = np.abs(r - g).mean(axis=-1)  # (H, W) intensity [0, 255]
            if fg.sum() > 0:
                masked_resid = float(diff[fg].mean())
                # PSNR on FG region only
                mse = ((r - g) ** 2)[fg].mean() / (255.0 ** 2)
                psnr = -10 * np.log10(max(mse, 1e-12))
            else:
                masked_resid = float("nan")
                psnr = float("nan")
            residual[v, t] = masked_resid
            psnr_mat[v, t] = psnr
            fg_frac_mat[v, t] = float(fg.mean())

            # Spatial accumulator (mask-weighted)
            spatial_sum += diff * fg
            spatial_count += fg.astype(np.float64)

            cell_records.append({
                "view": v, "time": t, "flat_idx": flat_idx,
                "residual": masked_resid, "psnr": float(psnr),
                "fg_frac": float(fg.mean()),
            })
        print(f"[characterize] view {v}: "
              f"residual mean={residual[v].mean():.3f}  PSNR mean={psnr_mat[v].mean():.2f}")

    # Save arrays
    np.save(out / "per_view_per_time_residual.npy", residual)
    np.save(out / "per_view_per_time_psnr.npy", psnr_mat)
    np.save(out / "per_view_per_time_fg_frac.npy", fg_frac_mat)

    # Heatmap of per-cell residual
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(residual, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("frame (time)")
    ax.set_ylabel("view")
    ax.set_title("VGM artifact map: per-(view, time) FG residual  |gt - render|  [0..255]")
    ax.set_yticks(range(V))
    ax.set_xticks(range(0, T, 2))
    plt.colorbar(im, ax=ax, label="mean |Δ| (intensity units)")
    fig.tight_layout()
    fig.savefig(out / "heatmap_residual.png", dpi=140)
    plt.close(fig)

    # Per-view residual curve vs time
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in range(V):
        ax.plot(range(T), residual[v], marker="o", markersize=3, label=f"view {v}")
    ax.set_xlabel("frame (time)")
    ax.set_ylabel("mean |Δ| over FG (intensity units)")
    ax.set_title("Per-view residual vs. time  (v5 canonical fits all 5 views)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_view_curve.png", dpi=140)
    plt.close(fig)

    # Per-time mean +/- std across views
    per_time_mean = residual.mean(axis=0)
    per_time_std = residual.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(T), per_time_mean, "o-", color="black", label="mean across views")
    ax.fill_between(range(T), per_time_mean - per_time_std, per_time_mean + per_time_std,
                    color="black", alpha=0.15, label="±1 std")
    ax.set_xlabel("frame (time)")
    ax.set_ylabel("mean |Δ| over FG (intensity units)")
    ax.set_title("Cross-view spread of residual vs. time")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_time_curve.png", dpi=140)
    plt.close(fig)

    # Spatial avg residual (where on the image artifacts cluster)
    spatial_avg = np.where(spatial_count > 0, spatial_sum / spatial_count, 0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(spatial_avg, cmap="magma", interpolation="nearest")
    ax.set_title("Spatial mean residual over FG pixels (avg across all cells)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |Δ|")
    fig.tight_layout()
    fig.savefig(out / "spatial_avg_residual.png", dpi=140)
    plt.close(fig)

    # Worst-frames panel: top 9 highest-residual cells
    worst = sorted(cell_records, key=lambda r: -r["residual"])[:9]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, rec in zip(axes.flat, worst):
        idx = rec["flat_idx"]
        r_img = load_rgba(args.render_dir / "renders" / f"{idx:05d}.png")[..., :3]
        g_img = load_rgba(args.render_dir / "gt" / f"{idx:05d}.png")[..., :3]
        # GT | render side by side
        sbs = np.concatenate([g_img, r_img], axis=1)
        ax.imshow(sbs)
        ax.set_title(f"view {rec['view']}  t={rec['time']}\n"
                     f"resid={rec['residual']:.1f}  PSNR={rec['psnr']:.1f}",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("Worst-9 cells: GT | render (v5 trained on all 5 views)",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(out / "worst_frames_panel.png", dpi=140)
    plt.close(fig)

    # SUMMARY.md
    per_view_mean = residual.mean(axis=1)
    per_view_psnr_mean = psnr_mat.mean(axis=1)
    overall = float(residual.mean())
    overall_psnr = float(psnr_mat.mean())
    md = ["# VGM artifact characterization — scene00 (SV4D 2.0 output, 5 views × 21 frames)",
          "",
          "Source: v5 SC-GS checkpoint trained on **all 5 views** (no held-out). The residual",
          "between the v5 render and the GT inside the FG mask is what no single canonical 3D",
          "Gaussian model could simultaneously explain across all 5 views; it is a direct",
          "lower-bound on the VGM's multi-view + temporal inconsistency.",
          "",
          "## Per-view summary",
          "",
          "| view | mean |Δ| (FG, intensity 0-255) | mean PSNR (dB) |",
          "|---:|---:|---:|",
          ]
    for v in range(V):
        md.append(f"| {v} | {per_view_mean[v]:.3f} | {per_view_psnr_mean[v]:.2f} |")
    md.append(f"| **all** | **{overall:.3f}** | **{overall_psnr:.2f}** |")
    md.extend([
        "",
        "## Cross-view spread per timestep",
        "",
        f"- Per-time mean residual range: [{per_time_mean.min():.3f}, {per_time_mean.max():.3f}]",
        f"- Per-time std (across views) range: [{per_time_std.min():.3f}, {per_time_std.max():.3f}]",
        f"- Time of peak residual: t={int(per_time_mean.argmax())}  (mean={per_time_mean.max():.3f})",
        f"- Time of min residual: t={int(per_time_mean.argmin())}  (mean={per_time_mean.min():.3f})",
        "",
        "## Worst single cells",
        "",
        "| rank | view | time | residual | PSNR |",
        "|---:|---:|---:|---:|---:|",
    ])
    for i, rec in enumerate(worst):
        md.append(f"| {i+1} | {rec['view']} | {rec['time']} | "
                  f"{rec['residual']:.2f} | {rec['psnr']:.2f} |")
    md.extend([
        "",
        "## Reading the result",
        "",
        "- If **per-view means are nearly identical**: no single view is systematically",
        "  worse → the VGM's per-view quality is uniform; inconsistency is",
        "  distributed.",
        "- If **per-view spread is large at specific timesteps**: the VGM's multi-view",
        "  disagreement is *time-localized* (e.g. peaks of motion). CVCG / MV gating",
        "  would help most at those timesteps.",
        "- If **spatial map concentrates on edges/silhouette**: the VGM hallucinates",
        "  near object boundaries (typical SV4D failure). Mask-based losses already",
        "  help here.",
        "- If **residual scale << 10 intensity units**: the VGM is actually quite",
        "  3D-consistent and CVCG-class methods have very little to recover; the",
        "  hold-out gap is then dominated by **camera-baseline sparsity (A3)** which",
        "  no training-time mitigation can fix.",
        "",
        "## Files",
        "",
        "- `per_view_per_time_{residual,psnr,fg_frac}.npy` — raw matrices",
        "- `heatmap_residual.png` — V × T heatmap",
        "- `per_view_curve.png` — per-view residual vs time",
        "- `per_time_curve.png` — mean ± std across views vs time",
        "- `spatial_avg_residual.png` — where on the image artifacts cluster",
        "- `worst_frames_panel.png` — GT | render for the 9 highest-residual cells",
        "",
    ])
    (out / "SUMMARY.md").write_text("\n".join(md))
    print(f"[characterize] done. {out}")
    print(f"[characterize] overall mean |Δ| (FG)  = {overall:.3f}  (intensity units)")
    print(f"[characterize] overall mean PSNR (FG) = {overall_psnr:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
