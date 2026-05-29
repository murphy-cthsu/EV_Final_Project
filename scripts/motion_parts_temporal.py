"""Stage B.1 extended (P2.3): per-(view, time) part masks via temporal diff.

For each view v at each timestep t:
    moving(v, t) = FG(v, t) AND  pixel_diff(frame_v_t, frame_v_0) > tau
    static(v, t) = FG(v, t) AND NOT moving(v, t)

This produces a moving-region mask per (view, time) cell. Per-cell 2D centroid
of the moving mask is what Stage D triangulates into 3D part trajectory.

Output: runs_aux/parts_motion/view{v}_part_masks_T.npy of shape (T, 2, H, W)
        and view{v}_centroids.npy of shape (T, 2, 2)  -- per-time (arm, body) 2D centroids.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/motion_parts_temporal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    src = REPO_ROOT / "data" / "custom" / "scene00_masked"
    out_dir = REPO_ROOT / "runs_aux" / "parts_motion"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((src / "transforms_train.json").read_text())
    by_view: dict[int, list[dict]] = {}
    for f in data["frames"]:
        by_view.setdefault(int(f["view_idx"]), []).append(f)

    for v in sorted(by_view.keys()):
        frames = sorted(by_view[v], key=lambda x: int(x["frame_idx"]))
        T = len(frames)
        rgbs = []
        fgs = []
        for f in frames:
            png = src / "train" / f"{Path(f['file_path']).name}.png"
            rgba = np.asarray(iio.imread(png))
            rgbs.append(rgba[..., :3].astype(np.float32))
            fgs.append(rgba[..., 3] > 127)
        rgbs = np.stack(rgbs, axis=0)   # (T, H, W, 3)
        fgs = np.stack(fgs, axis=0)     # (T, H, W)
        H, W = rgbs.shape[1:3]

        # Diff each frame to frame 0 (intensity)
        gray = rgbs.mean(axis=-1)  # (T, H, W)
        diff_to_0 = np.abs(gray - gray[0])  # (T, H, W); diff_to_0[0] is all zeros
        # Threshold from per-view distribution. We use the same percentile as
        # Stage B.1 spatial: top 30% within-FG difference per frame is "moved".
        masks_T = np.zeros((T, 2, H, W), dtype=np.uint8)
        centroids = np.zeros((T, 2, 2), dtype=np.float32)  # (T, [arm,body], [y,x])
        for t in range(T):
            d = diff_to_0[t]
            fg = fgs[t]
            if not fg.any():
                continue
            d_fg = d[fg]
            tau = max(15.0, np.percentile(d_fg, 70))
            moving = (d > tau) & fg
            # Cleanup
            moving = binary_opening(moving, iterations=2)
            moving = binary_closing(moving, iterations=3)
            moving = binary_dilation(moving, iterations=2) & fg
            static = fg & (~moving)
            masks_T[t, 0] = moving.astype(np.uint8)
            masks_T[t, 1] = static.astype(np.uint8)
            # Centroids (y, x)
            for p, m in enumerate([moving, static]):
                if m.sum() > 0:
                    ys, xs = np.where(m)
                    centroids[t, p, 0] = ys.mean()
                    centroids[t, p, 1] = xs.mean()
                else:
                    centroids[t, p] = np.nan

        # For t=0, moving mask via diff is empty (it's diff to itself). Use the
        # spatial-stddev mask from Stage B.1 as t=0 fallback.
        spatial_mask = np.load(out_dir / f"view{v}_frame0_part_masks.npy")  # (2, H, W)
        masks_T[0] = spatial_mask
        for p, m in enumerate([spatial_mask[0], spatial_mask[1]]):
            if m.sum() > 0:
                ys, xs = np.where(m.astype(bool))
                centroids[0, p, 0] = ys.mean()
                centroids[0, p, 1] = xs.mean()

        np.save(out_dir / f"view{v}_part_masks_T.npy", masks_T)
        np.save(out_dir / f"view{v}_centroids.npy", centroids)

        # Visualize: contact sheet of 7 timesteps showing moving overlay
        timesteps = [0, 3, 7, 10, 14, 17, 20]
        from PIL import Image
        from PIL import ImageDraw
        contact = []
        for t in timesteps:
            rgb = rgbs[t].astype(np.uint8)
            vis = rgb.copy().astype(np.float32)
            mm = masks_T[t, 0].astype(bool)
            sm = masks_T[t, 1].astype(bool)
            vis[mm] = 0.55 * np.array([255, 100, 200]) + 0.45 * vis[mm]
            vis[sm] = 0.55 * np.array([0, 200, 255]) + 0.45 * vis[sm]
            vis = np.clip(vis, 0, 255).astype(np.uint8)
            # Centroid markers
            cy, cx = centroids[t, 0]
            if not np.isnan(cy):
                cy_i, cx_i = int(cy), int(cx)
                R = 5
                vis[max(0,cy_i-R):cy_i+R+1, max(0,cx_i-R):cx_i+R+1] = [255, 255, 0]
            contact.append(vis)
        cs = np.concatenate(contact, axis=1)
        Image.fromarray(cs).save(out_dir / f"view{v}_temporal_contact.png")

        print(f"[motion_T] view {v}: masks shape {masks_T.shape}, "
              f"centroid arm sweep: y=({centroids[:,0,0].min():.0f}..{centroids[:,0,0].max():.0f}), "
              f"x=({centroids[:,0,1].min():.0f}..{centroids[:,0,1].max():.0f})")

    print(f"[motion_T] done. Inspect {out_dir}/view*_temporal_contact.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
