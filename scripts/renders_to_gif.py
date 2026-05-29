"""Turn SC-GS render output into per-view GIFs (render-only and render-vs-GT).

Assumes the standard SC-GS render.py output layout for a scene built by
scripts/multiview_videos_to_dnerf.py (5-digit flat indexing, V views x T frames):

    <renders_dir>/train/ours_<iter>/
        renders/00000.png ... 00104.png
        gt/00000.png ... 00104.png

The mapping back to (view, frame) is: flat_idx = view * T + frame.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/renders_to_gif.py \\
        --renders_dir outputs/custom/scene00_node/train/ours_30000 \\
        --n_views 5 --n_frames 21 --fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_frame(path: Path) -> np.ndarray:
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def write_gif(frames: list[np.ndarray], out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(fps, 1)))
    iio.imwrite(out_path, np.stack(frames, axis=0), duration=duration_ms, loop=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renders_dir", type=Path, required=True,
                   help="e.g. outputs/custom/scene00_node/train/ours_30000")
    p.add_argument("--n_views", type=int, required=True)
    p.add_argument("--n_frames", type=int, required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Defaults to <renders_dir>/gifs")
    args = p.parse_args()

    renders_dir = args.renders_dir.resolve()
    renders_pngs = renders_dir / "renders"
    gt_pngs = renders_dir / "gt"
    if not renders_pngs.is_dir():
        raise FileNotFoundError(renders_pngs)
    have_gt = gt_pngs.is_dir()

    out_dir = (args.out_dir or (renders_dir / "gifs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    V, T = args.n_views, args.n_frames
    expected = V * T
    actual = sum(1 for _ in renders_pngs.glob("*.png"))
    if actual < expected:
        raise ValueError(f"only {actual}/{expected} render PNGs in {renders_pngs}")

    print(f"[gif] {V} views x {T} frames = {expected} renders, fps={args.fps}")
    print(f"[gif] GT side-by-side: {have_gt}")

    # Per-view render-only + render|gt comparison GIFs
    for v in range(V):
        render_frames: list[np.ndarray] = []
        cmp_frames: list[np.ndarray] = []
        for t in range(T):
            idx = v * T + t
            r = load_frame(renders_pngs / f"{idx:05d}.png")
            render_frames.append(r)
            if have_gt:
                g = load_frame(gt_pngs / f"{idx:05d}.png")
                if g.shape != r.shape:
                    g = np.array(
                        iio.imread(gt_pngs / f"{idx:05d}.png"))[..., :3].astype(np.uint8)
                cmp_frames.append(np.concatenate([g, r], axis=1))

        write_gif(render_frames, out_dir / f"view_{v}_render.gif", args.fps)
        print(f"[gif]   wrote view_{v}_render.gif  ({len(render_frames)} frames)")
        if have_gt:
            write_gif(cmp_frames, out_dir / f"view_{v}_gt_vs_render.gif", args.fps)
            print(f"[gif]   wrote view_{v}_gt_vs_render.gif  (GT | render)")

    # Grid GIF: 1 x V layout, animated through T frames -- one timestep per gif frame
    grid_frames: list[np.ndarray] = []
    for t in range(T):
        row = []
        for v in range(V):
            idx = v * T + t
            row.append(load_frame(renders_pngs / f"{idx:05d}.png"))
        grid_frames.append(np.concatenate(row, axis=1))
    write_gif(grid_frames, out_dir / "all_views_grid.gif", args.fps)
    print(f"[gif]   wrote all_views_grid.gif  ({V} views side-by-side, "
          f"{T} timesteps)")

    print(f"[gif] done. open {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
