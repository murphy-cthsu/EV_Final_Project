"""Extract d-3dgs_video clean ref to a flat directory for new datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lego_v3", "hellwarrior"], required=True)
    p.add_argument("--n_frames", type=int, default=21)
    args = p.parse_args()

    src_base = Path(f"/mnt/HDD_1/cthsu/{args.dataset}")
    sub = "lego_r7_train" if args.dataset == "lego_v3" else "hellwarrior_r32_train"
    src_meta = json.loads((src_base / f"camera_estimation_math_{sub}" / "transforms_sv4d2_math.json").read_text())
    d3dgs_root = src_base / "d-3dgs_video"

    out = REPO / "outputs/custom" / f"{args.dataset}_d3dgs_ref" / "renders"
    out.mkdir(parents=True, exist_ok=True)

    for v_idx, f in enumerate(src_meta["frames"]):
        tag = f["offset_tag"]
        # d-3dgs_video has subfolders per view, with mp4 inside
        sub_dir = d3dgs_root / tag
        mp4_candidates = list(sub_dir.glob("*.mp4")) if sub_dir.exists() else []
        if not mp4_candidates:
            print(f"[{args.dataset}] missing d-3dgs for {tag}, skip")
            continue
        mp4 = mp4_candidates[0]
        r = imageio.get_reader(str(mp4))
        for t in range(args.n_frames):
            fr = r.get_data(t)
            flat = v_idx * args.n_frames + t
            Image.fromarray(fr).save(out / f"{flat:05d}.png")
        if (v_idx + 1) % 10 == 0:
            print(f"[{args.dataset}] {v_idx+1}/{len(src_meta['frames'])} views done")
    print(f"[{args.dataset}] saved to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
