"""Extract d-3dgs_video clean ref renders to a flat directory matching v5-render
convention (r_{flat_idx:05d}.png) so train_partrigid_hier.py can use them as
smart-photo reference (clean GT proxy) and viz scripts can find them.

Output: outputs/custom/lego_v2_d3dgs_ref/renders/r_NNNNN.png  (5 × 21 = 105 PNGs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_base", default="/mnt/HDD_1/cthsu/lego/d-3dgs_video")
    p.add_argument("--out_dir", default=REPO / "outputs/custom/lego_v2_d3dgs_ref/renders")
    p.add_argument("--n_views", type=int, default=5)
    p.add_argument("--n_frames", type=int, default=21)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for v in range(args.n_views):
        mp4 = Path(args.src_base) / f"000001_v00{v}" / f"000001_v00{v}.mp4"
        r = imageio.get_reader(str(mp4))
        for t in range(args.n_frames):
            fr = r.get_data(t)
            flat = v * args.n_frames + t
            Image.fromarray(fr).save(out / f"r_{flat:05d}.png")
        print(f"[d3dgs-ref] view {v}: 21 frames -> {out.name}")
    print(f"[d3dgs-ref] total {args.n_views * args.n_frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
