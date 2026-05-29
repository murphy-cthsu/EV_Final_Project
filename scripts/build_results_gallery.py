"""Build a unified visualization gallery for the final report.

Produces (in runs_aux/results_gallery/):
    1. part_rigid_motion.gif      -- part-rigid renders animated through time
    2. comparison_3col.gif        -- GT | vanilla SC-GS | part-rigid (per-frame)
    3. canonical_quality.png      -- frozen canonical at training view (PSNR 39.4)
    4. arm_trajectory.png         -- 3D plot of learned arm centroid path vs target

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_results_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "runs_aux" / "results_gallery"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    a = np.asarray(iio.imread(path))
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.astype(np.uint8)


def build_part_rigid_motion_gif():
    """All 25 part-rigid renders (GT | render side-by-side) as GIF."""
    src = REPO_ROOT / "runs_aux" / "partrigid_eval_sv4d" / "v1_scene00_split_t"
    files = sorted(src.glob("*.png"))
    frames = [load_rgb(f) for f in files]
    iio.imwrite(OUT / "part_rigid_motion.gif",
                np.stack(frames, axis=0), duration=150, loop=0)
    print(f"[gallery] wrote part_rigid_motion.gif  ({len(frames)} frames)")


def build_3col_comparison_gif():
    """3-col comparison: GT | vanilla SC-GS (t_vanilla) | part-rigid.

    All three on temporal-split test set (25 frames, 5 views x 5 held-out times).
    We use t_vanilla's test renders for the middle column.
    """
    pr_dir = REPO_ROOT / "runs_aux" / "partrigid_eval_sv4d" / "v1_scene00_split_t"
    van_renders = REPO_ROOT / "outputs/custom/scene00_t_vanilla_node/test/ours_30000/renders"
    van_gt = REPO_ROOT / "outputs/custom/scene00_t_vanilla_node/test/ours_30000/gt"

    # Find a common set of indices both have
    pr_files = sorted(pr_dir.glob("*.png"))
    if not pr_files or not van_renders.is_dir():
        print(f"[gallery] skipping 3col GIF (missing inputs)")
        return

    frames = []
    n = min(25, len(pr_files))
    for i in range(n):
        # Part-rigid render: PNG is "gt | render" side-by-side already (split horizontally)
        pr_combo = load_rgb(pr_dir / f"{i:03d}.png")
        W = pr_combo.shape[1] // 2
        gt = pr_combo[:, :W]      # left half
        pr = pr_combo[:, W:]      # right half
        # Vanilla render (just RGB)
        van_path = van_renders / f"{i:05d}.png"
        van_gt_path = van_gt / f"{i:05d}.png"
        if not van_path.is_file():
            van = np.zeros_like(gt)
        else:
            van = load_rgb(van_path)
            if van.shape != gt.shape:
                from PIL import Image as PILImage
                van = np.asarray(PILImage.fromarray(van).resize((W, gt.shape[0])))
        # 3 columns
        row = np.concatenate([gt, van, pr], axis=1)
        frames.append(row)
    iio.imwrite(OUT / "comparison_3col.gif",
                np.stack(frames, axis=0), duration=200, loop=0)
    print(f"[gallery] wrote comparison_3col.gif  ({len(frames)} frames, GT|vanilla|part-rigid)")


def canonical_quality_panel():
    """Side-by-side: a canonical render vs. its GT."""
    rend = REPO_ROOT / "outputs/custom/canonical_static_node/train/ours_5000/renders/00000.png"
    gt   = REPO_ROOT / "outputs/custom/canonical_static_node/train/ours_5000/gt/00000.png"
    if not (rend.is_file() and gt.is_file()):
        print(f"[gallery] skipping canonical quality panel (missing renders)")
        return
    r = load_rgb(rend); g = load_rgb(gt)
    H = min(r.shape[0], g.shape[0]); W = min(r.shape[1], g.shape[1])
    pair = np.concatenate([g[:H, :W], r[:H, :W]], axis=1)
    from PIL import Image as PILImage
    PILImage.fromarray(pair).save(OUT / "canonical_quality.png")
    print(f"[gallery] wrote canonical_quality.png (GT | canonical render, PSNR 39.4)")


def arm_trajectory_plot():
    """3D plot: learned arm SE(3) trajectory vs the triangulated target."""
    state_path = REPO_ROOT / "outputs/custom/partrigid_v1/partrigid_state.npz"
    target_path = REPO_ROOT / "runs_aux/part_assignment/part_centroid_3d.npy"
    if not (state_path.is_file() and target_path.is_file()):
        print(f"[gallery] skipping trajectory plot (missing inputs)")
        return
    state = np.load(state_path, allow_pickle=True)
    arm_trans = state["arm_trans"]     # (T, 3)
    arm_pivot = state["arm_pivot"]
    target = np.load(target_path)[:, 0, :]  # (T, 3) arm centroid target

    # Learned trajectory = arm_pivot + arm_trans (translation applied to centroid)
    learned = arm_pivot[None] + arm_trans  # (T, 3)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], "o-",
            color="tab:blue", label="Stage D target (triangulated)", markersize=6)
    ax.plot(learned[:, 0], learned[:, 1], learned[:, 2], "s-",
            color="tab:orange", label="Stage E learned (SE(3) traj)", markersize=6)
    # Connect each pair (target_t to learned_t) to show error
    for i in range(target.shape[0]):
        ax.plot([target[i, 0], learned[i, 0]],
                [target[i, 1], learned[i, 1]],
                [target[i, 2], learned[i, 2]], "-", color="gray", alpha=0.3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Arm centroid trajectory: learned vs target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "arm_trajectory.png", dpi=140)
    plt.close(fig)
    err = np.linalg.norm(learned - target, axis=1)
    print(f"[gallery] wrote arm_trajectory.png (mean tracking error = {err.mean():.4f}, "
          f"max = {err.max():.4f})")


def summary_dashboard():
    """A single PNG that tiles the key result figures together."""
    # We tile 5 figures in a 2x3 grid:
    #   [vgm artifact heatmap]      [per-view residual curve]
    #   [spatial residual map]      [canonical quality]
    #   [arm trajectory plot]       [(empty)]
    inputs = [
        REPO_ROOT / "runs_aux/vgm_artifact/heatmap_residual.png",
        REPO_ROOT / "runs_aux/vgm_artifact/per_view_curve.png",
        REPO_ROOT / "runs_aux/vgm_artifact/spatial_avg_residual.png",
        OUT / "canonical_quality.png",
        OUT / "arm_trajectory.png",
    ]
    available = [p for p in inputs if p.is_file()]
    if len(available) < 2:
        print(f"[gallery] not enough panels for summary dashboard")
        return
    from PIL import Image as PILImage

    def load_resize(path: Path, w: int) -> np.ndarray:
        img = PILImage.open(path).convert("RGB")
        ratio = w / img.width
        h = int(img.height * ratio)
        img = img.resize((w, h))
        return np.asarray(img)

    target_w = 600
    imgs = [load_resize(p, target_w) for p in available]
    # Pad to uniform height (max)
    max_h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] < max_h:
            pad = 255 * np.ones((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        padded.append(im)
    # Tile in 2-col grid
    rows = []
    for i in range(0, len(padded), 2):
        if i + 1 < len(padded):
            rows.append(np.concatenate([padded[i], padded[i+1]], axis=1))
        else:
            blank = 255 * np.ones_like(padded[i])
            rows.append(np.concatenate([padded[i], blank], axis=1))
    dash = np.concatenate(rows, axis=0)
    PILImage.fromarray(dash).save(OUT / "summary_dashboard.png")
    print(f"[gallery] wrote summary_dashboard.png ({dash.shape})")


def main() -> int:
    build_part_rigid_motion_gif()
    build_3col_comparison_gif()
    canonical_quality_panel()
    arm_trajectory_plot()
    summary_dashboard()
    print(f"[gallery] done. open {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
