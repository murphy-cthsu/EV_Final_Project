"""Stage B.1 (motion-driven): 2-part decomposition from per-pixel temporal variance.

For each view, take all T=21 frames. A pixel that changes significantly across
time = "moving" (belongs to arm+bucket). A pixel that stays roughly the same
within the FG mask = "static" (belongs to body+treads).

No SAM-2, no anchor hand-picking — purely data-driven, leverages the fact that
our two parts ARE separated by motion (by definition).

Output: runs_aux/parts_motion/view{v}_frame0_part_masks.npy
        shape (2, H, W) uint8, channel 0 = arm (moving), 1 = body (static).

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/motion_parts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation

REPO_ROOT = Path(__file__).resolve().parent.parent


def overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = alpha * c + (1 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    src = REPO_ROOT / "data" / "custom" / "scene00_masked"
    out_dir = REPO_ROOT / "runs_aux" / "parts_motion"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((src / "transforms_train.json").read_text())
    # Group by view, sort by time
    by_view: dict[int, list[dict]] = {}
    for f in data["frames"]:
        by_view.setdefault(int(f["view_idx"]), []).append(f)

    for v in sorted(by_view.keys()):
        frames = sorted(by_view[v], key=lambda x: int(x["frame_idx"]))
        T = len(frames)
        # Load all T frames as RGBA (mask in alpha)
        stack = []
        fg = None
        for f in frames:
            png = src / "train" / f"{Path(f['file_path']).name}.png"
            rgba = np.asarray(iio.imread(png))
            if rgba.shape[-1] != 4:
                raise RuntimeError(f"need RGBA at {png}, got shape {rgba.shape}")
            stack.append(rgba[..., :3].astype(np.float32))
            cur_fg = rgba[..., 3] > 127
            fg = cur_fg if fg is None else (fg | cur_fg)  # union of FG silhouettes over time
        stack = np.stack(stack, axis=0)  # (T, H, W, 3)

        # Per-pixel temporal stddev (intensity)
        gray = stack.mean(axis=-1)  # (T, H, W)
        std = gray.std(axis=0)      # (H, W)
        # Inside FG only -- BG is white anyway
        std_fg = np.where(fg, std, 0.0)

        # Threshold: pixels with std > some threshold = moving
        # Use a percentile within FG to be scene-adaptive
        fg_pixels = std_fg[fg]
        if fg_pixels.size == 0:
            print(f"[motion] view {v}: empty FG, skipping")
            continue
        thresh = max(2.0, np.percentile(fg_pixels, 70))  # top 30% most-varying FG pixels
        moving_raw = (std_fg > thresh) & fg

        # Clean up: small opening to drop salt-and-pepper, closing to fill holes
        moving = binary_opening(moving_raw, iterations=2)
        moving = binary_closing(moving, iterations=3)
        # Dilate slightly to capture mask boundary pixels
        moving = binary_dilation(moving, iterations=2)
        moving = moving & fg

        static = fg & (~moving)

        # Sanity: if either part is degenerate, fall back to threshold sweep
        if moving.sum() < 100 or static.sum() < 100:
            print(f"[motion] view {v}: degenerate split "
                  f"moving={moving.sum()}, static={static.sum()}; "
                  f"falling back to thresh={np.percentile(fg_pixels, 50):.2f}")
            thresh2 = np.percentile(fg_pixels, 50)
            moving = (std_fg > thresh2) & fg
            moving = binary_opening(moving, iterations=1)
            moving = binary_closing(moving, iterations=2)
            static = fg & (~moving)

        out = np.stack([moving.astype(np.uint8), static.astype(np.uint8)], axis=0)
        np.save(out_dir / f"view{v}_frame0_part_masks.npy", out)

        # Visualize on frame 0
        f0 = stack[0].astype(np.uint8)
        vis = f0.copy()
        vis = overlay(vis, moving, (255, 100, 200), 0.55)   # arm = pink
        vis = overlay(vis, static, (0, 200, 255), 0.55)     # body = cyan
        Image.fromarray(vis).save(out_dir / f"view{v}_frame0_overlay.png")

        # Save std heatmap for debug
        std_norm = np.clip(std_fg / max(std_fg.max(), 1e-3) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(std_norm).save(out_dir / f"view{v}_temporal_std.png")

        print(f"[motion] view {v}: T={T}, thresh={thresh:.2f}, "
              f"moving={int(moving.sum()):>6}, static={int(static.sum()):>6}, "
              f"fg={int(fg.sum()):>6}, ratio={moving.sum()/max(fg.sum(), 1):.3f}")

    print(f"[motion] done. Inspect {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
