"""Build a frame-0 subset of lego_v2 for training the frozen clean canonical 3DGS.

Two output variants (control via --source):
  --source d3dgs    : 5 views × t=0 from /mnt/HDD_1/cthsu/lego/d-3dgs_video/ (CLEAN refs)
                       → "ideal" canonical training source
  --source sv4d2    : 5 views × t=0 from /mnt/HDD_1/cthsu/lego/sv4d2/ (v0 clean, v1-v4 noisy)
                       → "available-only" baseline

Output: data/custom/lego_v2_frame0_{source}/{train/r_NNNNN.png + transforms_train.json}

Designed for SC-GS static training (no temporal dimension) — all 5 cams at t=0.
Alpha mask: from existing SAM-2-masked lego_v2 PNGs (so canonical matches motion-train mask).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["d3dgs", "sv4d2"], default="d3dgs",
                   help="d3dgs = clean ref videos (recommended), sv4d2 = noisy SV4D")
    p.add_argument("--src_base", default="/mnt/HDD_1/cthsu/lego")
    p.add_argument("--alpha_src", default=REPO / "data/custom/lego_v2",
                   help="Source for SAM-2 alpha masks (re-uses lego_v2's masks)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_views", type=int, default=5)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = REPO / f"data/custom/lego_v2_frame0_{args.source}"
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    src_meta = json.loads(Path(f"{args.src_base}/transforms_sv4d2_math.json").read_text())
    fov_x = src_meta["camera_angle_x"]

    # Build per-view (mp4, c2w) mapping
    view_data = {}
    for f in src_meta["frames"]:
        v = int(f["view_index"])
        if args.source == "d3dgs":
            mp4 = Path(args.src_base) / f"d-3dgs_video/000001_v00{v}/000001_v00{v}.mp4"
        else:
            mp4 = Path(args.src_base) / f"sv4d2/000001_v00{v}.mp4"
        view_data[v] = {"mp4": mp4, "c2w": np.asarray(f["transform_matrix"], dtype=np.float64)}

    frames_out = []
    alpha_src_dir = Path(args.alpha_src)
    print(f"[frame0] source={args.source} → {out_dir}")
    print(f"[frame0] alpha source: {alpha_src_dir}")

    for v in range(args.n_views):
        # Read frame 0 of the source mp4
        r = imageio.get_reader(str(view_data[v]["mp4"]))
        rgb = r.get_data(0)  # (H, W, 3) uint8

        # Read alpha from existing SAM-2 masked PNG (same view t=0)
        # Since temporal split puts t=0 in test/, check both
        flat_idx = v * 21 + 0   # t=0
        alpha_path = None
        for split in ("test", "train"):
            cand = alpha_src_dir / split / f"r_{flat_idx:05d}.png"
            if cand.exists():
                alpha_path = cand
                break
        if alpha_path is None:
            raise FileNotFoundError(f"alpha src not found for v={v} t=0")
        alpha = np.asarray(Image.open(alpha_path))[..., 3]

        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        out_name = f"r_{v:05d}.png"
        Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(train_dir / out_name)

        frames_out.append({
            "file_path": f"./train/{out_name.replace('.png', '')}",
            "view_idx": v,
            "frame_idx": 0,
            "transform_matrix": view_data[v]["c2w"].tolist(),
            "time": 0.0,
            "rotation": 0,
        })
        fg_frac = (alpha > 127).mean()
        print(f"[frame0] view {v}: written, fg_frac={fg_frac*100:.1f}%")

    out_meta = {
        "camera_angle_x": fov_x,
        "n_views": args.n_views,
        "n_frames": 1,
        "source": args.source,
        "frames": frames_out,
    }
    (out_dir / "transforms_train.json").write_text(json.dumps(out_meta, indent=2))
    # Empty test json (SC-GS needs it for --eval; we don't really eval here)
    (out_dir / "transforms_test.json").write_text(json.dumps({**out_meta, "frames": []}, indent=2))
    print(f"[frame0] saved {args.n_views} PNGs + transforms_train.json -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
