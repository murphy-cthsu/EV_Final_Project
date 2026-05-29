"""Build comparison GIFs for the view-split + temporal-split ablation runs.

For view-split: produces a 3-column GIF (GT | vanilla render | best-method render)
animated through 21 timesteps of the held-out view 2.

For temporal-split: produces a 3-column GIF (GT | vanilla render | best-method
render) animated through the 25 test cells (5 views x 5 held-out timesteps),
ordered by (view, time).

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_ablation_gifs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_rgb(path: Path) -> np.ndarray:
    a = np.asarray(iio.imread(path))
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.astype(np.uint8)


def label_image(img: np.ndarray, text: str, color=(255, 255, 0)) -> np.ndarray:
    """Crude top-bar label (no PIL drawing)."""
    H, W = img.shape[:2]
    bar = np.zeros((24, W, 3), dtype=np.uint8)
    out = np.concatenate([bar, img], axis=0)
    # No actual text drawing -- the title goes in the filename / caller.
    return out


def build_view_split_gif(out_path: Path):
    n_frames = 21
    gt_dir = REPO_ROOT / "outputs/custom/scene00_v6_split_node/test/ours_30000/gt"
    v6_dir = REPO_ROOT / "outputs/custom/scene00_v6_split_node/test/ours_30000/renders"
    v11_dir = REPO_ROOT / "outputs/custom/scene00_v11_c3cvcg_node/test/ours_30000/renders"

    frames: list[np.ndarray] = []
    for t in range(n_frames):
        gt = load_rgb(gt_dir / f"{t:05d}.png")
        v6 = load_rgb(v6_dir / f"{t:05d}.png")
        v11 = load_rgb(v11_dir / f"{t:05d}.png")
        row = np.concatenate([gt, v6, v11], axis=1)
        frames.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames, axis=0), duration=100, loop=0)
    print(f"[gif] wrote {out_path}  ({n_frames} frames; layout: GT | vanilla | +C3+CVCG)")


def build_temporal_split_gif(out_path: Path):
    """Temporal split test set is 25 frames (5 views x 5 held-out timesteps).
    They're rendered in the order they appear in transforms_test.json — flat-indexed.
    Build a GIF preserving that order."""
    n_frames = 25
    gt_dir = REPO_ROOT / "outputs/custom/scene00_t_vanilla_node/test/ours_30000/gt"
    van_dir = REPO_ROOT / "outputs/custom/scene00_t_vanilla_node/test/ours_30000/renders"
    slow_dir = REPO_ROOT / "outputs/custom/scene00_t_slow_node/test/ours_16000/renders"

    frames: list[np.ndarray] = []
    for t in range(n_frames):
        gt = load_rgb(gt_dir / f"{t:05d}.png")
        van = load_rgb(van_dir / f"{t:05d}.png")
        slow = load_rgb(slow_dir / f"{t:05d}.png")
        row = np.concatenate([gt, van, slow], axis=1)
        frames.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames, axis=0), duration=150, loop=0)
    print(f"[gif] wrote {out_path}  ({n_frames} frames; layout: GT | vanilla | +C3+CVCG slow @ iter 16k)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path,
                   default=REPO_ROOT / "runs_aux" / "ablation_gifs")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_view_split_gif(args.out_dir / "view_split_compare.gif")
    build_temporal_split_gif(args.out_dir / "temporal_split_compare.gif")
    print(f"[gif] done. open {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
