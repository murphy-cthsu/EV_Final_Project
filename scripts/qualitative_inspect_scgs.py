"""Qualitative SC-GS failure inspection.

For a finished SC-GS run, emits three still-image artifacts that are easier to
compare than the side-by-side mp4s:

  1. worst_frames_contact_sheet.png
       Grid of the N worst test-split frames (by PSNR), each row
       [GT | pred | per-pixel error heatmap].

  2. joint_zoom_<frame>.png
       For the single worst frame: find the K highest-error patches with NMS,
       crop+zoom each, lay them out as [GT | pred | error] tiles per patch.

  3. temporal_strobe.png
       From the time-interpolation render dir: a strobe overlay (every Nth
       frame alpha-blended) AND a per-pixel temporal-std heatmap, both
       designed to reveal limb-end ghosting / trail artifacts.

Inputs are the raw per-frame PNGs SC-GS render.py already produced:
    <run_dir>/test/ours_<iter>/{gt,renders}/*.png
    <run_dir>/test/interpolate_<iter>/renders/*.png
Worst-frame ordering comes from <run_dir>/inspection/per_frame_psnr.json.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/qualitative_inspect_scgs.py \\
        --run_dir outputs/jumpingjacks_scgs_default_node \\
        --iteration 30000
    # or do all four at once:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/qualitative_inspect_scgs.py --all_scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SCENE_DIRS = [
    REPO_ROOT / "outputs" / f"{s}_scgs_default_node"
    for s in ("jumpingjacks", "bouncingballs", "hellwarrior", "standup")
]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def per_pixel_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """L2 error across channels, shape (H, W)."""
    return np.sqrt(((gt - pred) ** 2).sum(axis=-1))


def normalize_for_heatmap(err: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    """Scale to [0,1] using 99th percentile so a few outliers don't crush contrast."""
    if vmax is None:
        vmax = float(np.percentile(err, 99))
        if vmax <= 1e-8:
            vmax = float(err.max()) if err.max() > 0 else 1.0
    return np.clip(err / vmax, 0.0, 1.0), vmax


def find_topk_patches(err: np.ndarray, patch: int, k: int) -> list[tuple[int, int, float]]:
    """K highest-mean-error patch centers with NMS so they don't overlap.

    Returns list of (cy, cx, mean_err) in image coordinates.
    """
    h, w = err.shape
    half = patch // 2
    # integral image for O(1) box-sum
    integ = np.pad(err.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(half, h - half)
    xs = np.arange(half, w - half)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0, y1 = yy - half, yy + half
    x0, x1 = xx - half, xx + half
    box_sum = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    mean = box_sum / (patch * patch)

    picks: list[tuple[int, int, float]] = []
    flat = mean.flatten()
    order = np.argsort(-flat)
    taken = np.zeros_like(mean, dtype=bool)
    for idx in order:
        ry, rx = divmod(int(idx), mean.shape[1])
        if taken[ry, rx]:
            continue
        cy, cx = ys[ry], xs[rx]
        picks.append((int(cy), int(cx), float(mean[ry, rx])))
        # suppress neighbors within patch radius
        ly = max(0, ry - patch)
        hy = min(mean.shape[0], ry + patch + 1)
        lx = max(0, rx - patch)
        hx = min(mean.shape[1], rx + patch + 1)
        taken[ly:hy, lx:hx] = True
        if len(picks) >= k:
            break
    return picks


def make_contact_sheet(run_dir: Path, iteration: int, top_n: int, out_path: Path) -> dict:
    pf_path = run_dir / "inspection" / "per_frame_psnr.json"
    pf = json.loads(pf_path.read_text())  # already sorted worst→best
    worst = pf[:top_n]

    test_dir = run_dir / "test" / f"ours_{iteration}"
    gt_dir = test_dir / "gt"
    pred_dir = test_dir / "renders"

    fig, axes = plt.subplots(top_n, 3, figsize=(9, 3 * top_n), squeeze=False)
    scene_vmax = 0.0  # share heatmap scale across rows for honest comparison
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = []
    for entry in worst:
        frame = entry["frame"]
        gt = load_rgb(gt_dir / f"{frame}.png")
        pred = load_rgb(pred_dir / f"{frame}.png")
        err = per_pixel_error(gt, pred)
        scene_vmax = max(scene_vmax, float(np.percentile(err, 99)))
        cached.append((gt, pred, err, entry))

    for row, (gt, pred, err, entry) in enumerate(cached):
        err_norm, _ = normalize_for_heatmap(err, vmax=scene_vmax)
        axes[row, 0].imshow(gt)
        axes[row, 0].set_title(f"GT  frame {entry['frame']}  PSNR={entry['psnr']:.2f}", fontsize=9)
        axes[row, 1].imshow(pred)
        axes[row, 1].set_title("SC-GS pred", fontsize=9)
        im = axes[row, 2].imshow(err_norm, cmap="inferno", vmin=0, vmax=1)
        axes[row, 2].set_title(f"|GT - pred| (vmax={scene_vmax:.3f})", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{run_dir.name} — worst {top_n} test frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"top_n": top_n, "scene_vmax": scene_vmax, "frames": [c[3] for c in cached]}


def make_joint_zoom(run_dir: Path, iteration: int, patch: int, num_crops: int,
                    out_path: Path) -> dict:
    pf_path = run_dir / "inspection" / "per_frame_psnr.json"
    worst_frame = json.loads(pf_path.read_text())[0]
    frame = worst_frame["frame"]

    test_dir = run_dir / "test" / f"ours_{iteration}"
    gt = load_rgb(test_dir / "gt" / f"{frame}.png")
    pred = load_rgb(test_dir / "renders" / f"{frame}.png")
    err = per_pixel_error(gt, pred)
    err_norm, vmax = normalize_for_heatmap(err)

    picks = find_topk_patches(err, patch=patch, k=num_crops)

    # Layout: top row = whole-frame GT/pred/err with crop boxes overlaid;
    # subsequent rows = one crop triplet per pick.
    nrows = 1 + num_crops
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows), squeeze=False)
    axes[0, 0].imshow(gt)
    axes[0, 0].set_title(f"GT  frame {frame}  PSNR={worst_frame['psnr']:.2f}", fontsize=9)
    axes[0, 1].imshow(pred)
    axes[0, 1].set_title("SC-GS pred", fontsize=9)
    axes[0, 2].imshow(err_norm, cmap="inferno", vmin=0, vmax=1)
    axes[0, 2].set_title(f"|GT - pred| (vmax={vmax:.3f})", fontsize=9)

    colors = ["lime", "cyan", "magenta", "yellow", "orange"]
    half = patch // 2
    for i, (cy, cx, mean_e) in enumerate(picks):
        color = colors[i % len(colors)]
        for col in range(3):
            rect = plt.Rectangle((cx - half, cy - half), patch, patch,
                                 fill=False, edgecolor=color, linewidth=1.5)
            axes[0, col].add_patch(rect)

        gt_crop = gt[cy - half: cy + half, cx - half: cx + half]
        pred_crop = pred[cy - half: cy + half, cx - half: cx + half]
        err_crop = err_norm[cy - half: cy + half, cx - half: cx + half]
        axes[i + 1, 0].imshow(gt_crop, interpolation="nearest")
        axes[i + 1, 0].set_title(f"crop {i+1} GT (mean err={mean_e:.3f})",
                                 fontsize=9, color=color)
        axes[i + 1, 1].imshow(pred_crop, interpolation="nearest")
        axes[i + 1, 1].set_title("pred", fontsize=9, color=color)
        axes[i + 1, 2].imshow(err_crop, cmap="inferno", vmin=0, vmax=1,
                              interpolation="nearest")
        axes[i + 1, 2].set_title("error", fontsize=9, color=color)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{run_dir.name} — high-error region zoom ({patch}×{patch} patches)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"worst_frame": frame, "picks": picks, "patch": patch}


def make_temporal_strobe(run_dir: Path, iteration: int, step: int,
                         out_path: Path) -> dict:
    interp_dir = run_dir / "test" / f"interpolate_{iteration}" / "renders"
    frames = sorted(interp_dir.glob("*.png"))
    if not frames:
        return {"error": f"no frames in {interp_dir}"}
    sampled = frames[::step]
    stack = np.stack([load_rgb(p) for p in sampled], axis=0)  # (T, H, W, 3)

    # strobe overlay: min over time on a white canvas highlights dark/colored
    # foreground accumulating; equivalent to alpha-blending non-white pixels.
    # Using min preserves silhouettes at each pose; pure alpha-blend washes out.
    strobe = stack.min(axis=0)

    # temporal std: per-pixel std across the time stack, summed over channels.
    # High std = the model can't keep this pixel stable across temporal warp.
    tstd = stack.std(axis=0).sum(axis=-1)
    tstd_norm, vmax = normalize_for_heatmap(tstd)

    # also compute std weighted only over the body region (non-near-white pixels
    # in the strobe) so background noise doesn't dominate the colormap.
    fg_mask = (strobe.mean(axis=-1) < 0.95)
    fg_tstd = tstd.copy()
    fg_tstd[~fg_mask] = 0.0
    fg_norm, fg_vmax = normalize_for_heatmap(fg_tstd)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    axes[0, 0].imshow(strobe)
    axes[0, 0].set_title(f"strobe overlay (min over {len(sampled)} frames, step={step})",
                         fontsize=9)
    axes[0, 1].imshow(tstd_norm, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title(f"temporal std heatmap (vmax={vmax:.3f})", fontsize=9)
    axes[0, 2].imshow(fg_norm, cmap="inferno", vmin=0, vmax=1)
    axes[0, 2].set_title(f"temporal std, foreground only (vmax={fg_vmax:.3f})",
                         fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{run_dir.name} — time-interpolation failure surface", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "num_frames": len(frames),
        "sampled": len(sampled),
        "step": step,
        "tstd_vmax": vmax,
        "fg_tstd_vmax": fg_vmax,
    }


def run_one(run_dir: Path, iteration: int, top_n: int, patch: int, num_crops: int,
            strobe_step: int) -> dict:
    out_dir = run_dir / "inspection" / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[qual] {run_dir.name}: contact sheet (worst {top_n})...")
    cs = make_contact_sheet(run_dir, iteration, top_n,
                            out_dir / "worst_frames_contact_sheet.png")

    print(f"[qual] {run_dir.name}: joint zoom (patch={patch}, k={num_crops})...")
    worst_frame = json.loads((run_dir / "inspection" / "per_frame_psnr.json").read_text())[0]["frame"]
    jz = make_joint_zoom(run_dir, iteration, patch, num_crops,
                         out_dir / f"joint_zoom_{worst_frame}.png")

    print(f"[qual] {run_dir.name}: temporal strobe (step={strobe_step})...")
    ts = make_temporal_strobe(run_dir, iteration, strobe_step,
                              out_dir / "temporal_strobe.png")

    summary = {"contact_sheet": cs, "joint_zoom": jz, "temporal_strobe": ts}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path,
                   help="Single scene run dir (omit if --all_scenes)")
    p.add_argument("--all_scenes", action="store_true",
                   help="Process all 4 D-NeRF scene outputs in outputs/")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--top_n", type=int, default=4,
                   help="Number of worst frames in the contact sheet")
    p.add_argument("--patch", type=int, default=96,
                   help="Joint-zoom patch size (pixels)")
    p.add_argument("--num_crops", type=int, default=3,
                   help="Number of high-error patches to zoom into")
    p.add_argument("--strobe_step", type=int, default=10,
                   help="Sample every Nth frame from interpolate_<iter>/renders")
    args = p.parse_args()

    if args.all_scenes:
        targets = [d for d in ALL_SCENE_DIRS if d.exists()]
    elif args.run_dir is not None:
        targets = [args.run_dir.resolve()]
    else:
        p.error("Pass --run_dir or --all_scenes")
        return 2

    for run_dir in targets:
        if not (run_dir / "inspection" / "per_frame_psnr.json").exists():
            print(f"[qual] SKIP {run_dir.name}: per_frame_psnr.json missing "
                  "(run inspect_scgs_failure.py first)")
            continue
        run_one(run_dir, args.iteration, args.top_n, args.patch,
                args.num_crops, args.strobe_step)

    print("[qual] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
