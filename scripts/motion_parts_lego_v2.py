"""Motion mask for lego_v2 — per-pixel temporal variance over 21 SV4D frames.
Output: runs_aux/parts_motion_lego_v2/view{v}_part_masks.npy + overlays."""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing, binary_dilation

REPO = Path(__file__).resolve().parent.parent


def main():
    src_videos = Path("/mnt/HDD_1/cthsu/lego/sv4d2")
    alpha_src = REPO / "data/custom/lego_v2"  # SAM-2 masked alphas
    out_dir = REPO / "runs_aux/parts_motion_lego_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    N_VIEWS, T = 5, 21
    H = W = 576

    for v in range(N_VIEWS):
        # Load all 21 RGB frames
        r = imageio.get_reader(str(src_videos / f"000001_v00{v}.mp4"))
        frames = np.stack([r.get_data(t) for t in range(T)], axis=0)  # (T, H, W, 3)
        # Load FG alpha from train OR test (whichever split has it)
        alphas = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            flat = v * T + t
            for split in ("train", "test"):
                p = alpha_src / split / f"r_{flat:05d}.png"
                if p.exists():
                    rgba = np.asarray(Image.open(p))
                    alphas[t] = (rgba[..., 3] > 127).astype(np.float32)
                    break
        fg_any = alphas.max(axis=0) > 0.5    # (H, W) — pixel ever FG?

        # Temporal std on grayscale, only inside fg
        gray = frames.mean(axis=-1)  # (T, H, W)
        std = gray.std(axis=0)       # (H, W)
        std = std * fg_any           # mask to FG region
        std_in_fg = std[fg_any]
        if std_in_fg.size > 0:
            # bottom 50% std = static body, top = moving arm/bucket
            thresh = np.percentile(std_in_fg, 50)
        else:
            thresh = 0
        moving = (std > thresh) & fg_any
        static = (std <= thresh) & fg_any
        # Cleanup
        moving = binary_closing(binary_opening(moving, iterations=2), iterations=2)
        static = binary_closing(binary_opening(static, iterations=2), iterations=2)
        # Also expand moving to ensure coverage
        moving = binary_dilation(moving, iterations=1)

        out = np.stack([moving.astype(np.uint8), static.astype(np.uint8)], axis=0)
        np.save(out_dir / f"view{v}_part_masks.npy", out)
        # Visualizations
        vis = frames[0].copy()
        # Color moving (red) + static (blue) overlays
        m_mask = moving.astype(bool)
        s_mask = static.astype(bool)
        vis_alpha = 0.45
        vis[m_mask] = ((1-vis_alpha) * vis[m_mask].astype(np.float32) +
                       vis_alpha * np.array([255, 50, 50])).astype(np.uint8)
        vis[s_mask] = ((1-vis_alpha) * vis[s_mask].astype(np.float32) +
                       vis_alpha * np.array([50, 50, 255])).astype(np.uint8)
        Image.fromarray(vis).save(out_dir / f"view{v}_frame0_overlay.png")
        std_norm = (std / max(std.max(), 1e-8) * 255).astype(np.uint8)
        Image.fromarray(std_norm).save(out_dir / f"view{v}_temporal_std.png")
        print(f"[motion] view {v}: moving area={int(moving.sum())} static area={int(static.sum())}  "
              f"fg total={int(fg_any.sum())}  thresh={thresh:.2f}")

    print(f"[motion] done. Inspect {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
