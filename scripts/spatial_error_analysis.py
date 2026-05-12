"""Quantitative spatial-error analysis on SC-GS test renders.

Two complementary statistics over all test frames in a run:

1. core-vs-periphery error ratio
       Foreground mask = pixels where GT brightness > eps.
       Core mask      = foreground morphologically eroded by `core_erode` pixels.
       Periphery mask = foreground & ~core.
       Bar: mean |GT-pred| in {core, periphery}, plus the ratio.
       Hypothesis: articulation drift concentrates in periphery →
       periphery error should be substantially larger than core error.

2. radial error profile (normalized distance from connected-component centroid)
       Per frame, label connected components of the foreground mask, take each
       component's centroid + bbox diagonal as a length scale. For each fg
       pixel, normalized distance = ||pixel - centroid|| / (bbox_diag / 2).
       Bin error by normalized distance; plot mean ± 1 σ.
       Generalizes to multi-object scenes (e.g. bouncingballs: 4 components).

Both run on the existing test/ours_<iter>/{gt,renders}/*.png — no new model
runs or extra segmentation needed.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/spatial_error_analysis.py \\
        --run_dir outputs/jumpingjacks_scgs_default_node --iteration 30000
    # or all 4 D-NeRF scenes:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/spatial_error_analysis.py --all_scenes
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
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SCENE_DIRS = [
    REPO_ROOT / "outputs" / f"{s}_scgs_default_node"
    for s in ("jumpingjacks", "bouncingballs", "hellwarrior", "standup")
]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def per_pixel_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(((gt - pred) ** 2).sum(axis=-1))


def core_periphery(fg_mask: np.ndarray, erode: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (core_mask, periphery_mask) from a binary foreground mask."""
    if erode <= 0:
        return fg_mask.copy(), np.zeros_like(fg_mask)
    core = ndi.binary_erosion(fg_mask, iterations=erode)
    periph = fg_mask & ~core
    return core, periph


def radial_profile_per_frame(err: np.ndarray, fg_mask: np.ndarray, n_bins: int,
                              max_norm_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """For each fg pixel, normalized distance from its connected-component centroid.

    Returns (bin_edges, bin_means_with_NaN).
    """
    labels, n_cc = ndi.label(fg_mask)
    if n_cc == 0:
        edges = np.linspace(0, max_norm_dist, n_bins + 1)
        return edges, np.full(n_bins, np.nan)

    # accumulate (normalized_dist, error) for every fg pixel
    all_norm = []
    all_err = []
    ys, xs = np.indices(fg_mask.shape)
    for k in range(1, n_cc + 1):
        mask_k = labels == k
        if not mask_k.any():
            continue
        cy = ys[mask_k].mean()
        cx = xs[mask_k].mean()
        # bbox diagonal as the length scale
        y_min, y_max = ys[mask_k].min(), ys[mask_k].max()
        x_min, x_max = xs[mask_k].min(), xs[mask_k].max()
        diag = float(np.hypot(y_max - y_min, x_max - x_min))
        scale = max(diag / 2.0, 1.0)

        dy = ys[mask_k] - cy
        dx = xs[mask_k] - cx
        d = np.hypot(dy, dx) / scale
        all_norm.append(d)
        all_err.append(err[mask_k])

    all_norm_arr = np.concatenate(all_norm)
    all_err_arr = np.concatenate(all_err)
    edges = np.linspace(0, max_norm_dist, n_bins + 1)
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (all_norm_arr >= edges[i]) & (all_norm_arr < edges[i + 1])
        if sel.any():
            means[i] = float(all_err_arr[sel].mean())
    return edges, means


def analyze_run(run_dir: Path, iteration: int, fg_thresh: float,
                core_erode: int, n_bins: int, max_norm_dist: float) -> dict:
    test_dir = run_dir / "test" / f"ours_{iteration}"
    gt_dir = test_dir / "gt"
    pred_dir = test_dir / "renders"
    frames = sorted(p.stem for p in gt_dir.glob("*.png"))

    # aggregate over frames
    core_errs, periph_errs = [], []
    radial_per_frame = []   # list of (n_bins,) arrays
    edges = None
    per_frame_records = []
    for fid in frames:
        gt = load_rgb(gt_dir / f"{fid}.png")
        pred = load_rgb(pred_dir / f"{fid}.png")
        err = per_pixel_error(gt, pred)

        # Foreground = pixels brighter than threshold (D-NeRF background is black after compositing)
        fg = gt.max(axis=-1) > fg_thresh
        core_m, periph_m = core_periphery(fg, core_erode)

        c_e = float(err[core_m].mean()) if core_m.any() else float("nan")
        p_e = float(err[periph_m].mean()) if periph_m.any() else float("nan")
        core_errs.append(c_e)
        periph_errs.append(p_e)

        edges_f, means_f = radial_profile_per_frame(err, fg, n_bins, max_norm_dist)
        if edges is None:
            edges = edges_f
        radial_per_frame.append(means_f)

        per_frame_records.append({
            "frame": fid,
            "core_err": c_e,
            "periphery_err": p_e,
            "ratio": (p_e / c_e) if c_e and c_e > 0 else float("nan"),
            "core_area_frac": float(core_m.mean()),
            "periphery_area_frac": float(periph_m.mean()),
        })

    radial_stack = np.stack(radial_per_frame, axis=0)
    radial_mean = np.nanmean(radial_stack, axis=0)
    radial_std = np.nanstd(radial_stack, axis=0)

    return {
        "scene": run_dir.name,
        "n_frames": len(frames),
        "core_err_mean": float(np.nanmean(core_errs)),
        "core_err_std": float(np.nanstd(core_errs)),
        "periphery_err_mean": float(np.nanmean(periph_errs)),
        "periphery_err_std": float(np.nanstd(periph_errs)),
        "periphery_over_core": float(np.nanmean(periph_errs) / np.nanmean(core_errs)),
        "radial_edges": edges.tolist() if edges is not None else None,
        "radial_mean": radial_mean.tolist(),
        "radial_std": radial_std.tolist(),
        "per_frame": per_frame_records,
        "config": {
            "fg_thresh": fg_thresh,
            "core_erode": core_erode,
            "n_bins": n_bins,
            "max_norm_dist": max_norm_dist,
        },
    }


def plot_core_periphery_bar(results: list[dict], out_path: Path) -> None:
    scenes = [r["scene"].replace("_scgs_default_node", "") for r in results]
    core_m = [r["core_err_mean"] for r in results]
    core_s = [r["core_err_std"] for r in results]
    perip_m = [r["periphery_err_mean"] for r in results]
    perip_s = [r["periphery_err_std"] for r in results]
    ratios = [r["periphery_over_core"] for r in results]

    x = np.arange(len(scenes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, core_m, w, yerr=core_s, label="core (eroded body)",
                color="#4477AA", capsize=3)
    b2 = ax.bar(x + w / 2, perip_m, w, yerr=perip_s, label="periphery (limb tips / edges)",
                color="#EE6677", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("mean per-pixel error |GT − pred|")
    ax.set_title("SC-GS-default: reconstruction error concentrates at the periphery")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    # annotate ratios
    for xi, r in zip(x, ratios):
        ax.annotate(f"×{r:.1f}", xy=(xi, max(core_m[xi], perip_m[xi])),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#222")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_radial_profiles(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
    for i, r in enumerate(results):
        edges = np.asarray(r["radial_edges"])
        mid = 0.5 * (edges[:-1] + edges[1:])
        mean = np.asarray(r["radial_mean"])
        std = np.asarray(r["radial_std"])
        scene = r["scene"].replace("_scgs_default_node", "")
        color = colors[i % len(colors)]
        ax.plot(mid, mean, color=color, lw=2, label=scene, marker="o", ms=4)
        ax.fill_between(mid, np.clip(mean - std, 0, None), mean + std,
                        color=color, alpha=0.15)
    ax.set_xlabel("normalized distance from object centroid  (0 = center, 1 ≈ bbox edge)")
    ax.set_ylabel("mean per-pixel error |GT − pred|")
    ax.set_title("Error grows with radial distance — articulation drift is peripheral")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_per_scene_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # per-scene single-row bar
    fig, ax = plt.subplots(figsize=(5, 4))
    cats = ["core", "periphery"]
    vals = [result["core_err_mean"], result["periphery_err_mean"]]
    errs = [result["core_err_std"], result["periphery_err_std"]]
    bars = ax.bar(cats, vals, yerr=errs, color=["#4477AA", "#EE6677"], capsize=4)
    ratio = result["periphery_over_core"]
    ax.set_title(f"{result['scene']}\nperiphery/core = ×{ratio:.2f}")
    ax.set_ylabel("mean per-pixel error")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "core_vs_periphery.png", dpi=130)
    plt.close(fig)

    # per-scene radial profile
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.asarray(result["radial_edges"])
    mid = 0.5 * (edges[:-1] + edges[1:])
    mean = np.asarray(result["radial_mean"])
    std = np.asarray(result["radial_std"])
    ax.plot(mid, mean, "#222266", lw=2, marker="o", ms=4)
    ax.fill_between(mid, np.clip(mean - std, 0, None), mean + std,
                    color="#222266", alpha=0.2)
    ax.set_xlabel("normalized radial distance")
    ax.set_ylabel("mean per-pixel error")
    ax.set_title(f"{result['scene']} — radial error profile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_profile.png", dpi=130)
    plt.close(fig)

    (out_dir / "spatial_error.json").write_text(json.dumps(result, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path)
    p.add_argument("--all_scenes", action="store_true")
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--fg_thresh", type=float, default=0.05,
                   help="Foreground luminance threshold (D-NeRF bg is black)")
    p.add_argument("--core_erode", type=int, default=30,
                   help="Erosion iterations for the 'core body' mask")
    p.add_argument("--n_bins", type=int, default=12,
                   help="Number of bins for radial profile")
    p.add_argument("--max_norm_dist", type=float, default=1.2,
                   help="Max normalized radial distance to bin (>1 to include edges)")
    p.add_argument("--cross_scene_out", type=Path,
                   default=REPO_ROOT / "outputs" / "_cross_scene_failure",
                   help="Where to write the cross-scene aggregate figure")
    args = p.parse_args()

    if args.all_scenes:
        targets = [d for d in ALL_SCENE_DIRS if d.exists()]
    elif args.run_dir is not None:
        targets = [args.run_dir.resolve()]
    else:
        p.error("Pass --run_dir or --all_scenes")
        return 2

    results = []
    for run_dir in targets:
        if not (run_dir / "test" / f"ours_{args.iteration}" / "gt").exists():
            print(f"[spatial] SKIP {run_dir.name}: no test gt for iter {args.iteration}")
            continue
        print(f"[spatial] analyzing {run_dir.name}...")
        r = analyze_run(run_dir, args.iteration, args.fg_thresh,
                        args.core_erode, args.n_bins, args.max_norm_dist)
        results.append(r)
        out_dir = run_dir / "inspection" / "qualitative"
        write_per_scene_outputs(r, out_dir)
        print(f"[spatial]   core={r['core_err_mean']:.4f}±{r['core_err_std']:.4f}  "
              f"periph={r['periphery_err_mean']:.4f}±{r['periphery_err_std']:.4f}  "
              f"ratio={r['periphery_over_core']:.2f}")

    if results:
        args.cross_scene_out.mkdir(parents=True, exist_ok=True)
        plot_core_periphery_bar(results, args.cross_scene_out / "core_vs_periphery_cross_scene.png")
        plot_radial_profiles(results, args.cross_scene_out / "radial_profile_cross_scene.png")
        agg = {
            "scenes": results,
            "config": results[0]["config"],
            "summary": [
                {
                    "scene": r["scene"].replace("_scgs_default_node", ""),
                    "core_err": r["core_err_mean"],
                    "periphery_err": r["periphery_err_mean"],
                    "periphery_over_core": r["periphery_over_core"],
                }
                for r in results
            ],
        }
        (args.cross_scene_out / "spatial_error_summary.json").write_text(
            json.dumps(agg, indent=2)
        )
        print(f"[spatial] cross-scene plot -> {args.cross_scene_out}")

    print("[spatial] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
