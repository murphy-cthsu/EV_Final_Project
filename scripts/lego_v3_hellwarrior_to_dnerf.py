"""Convert lego_v3 or hellwarrior to D-NeRF format.

Both datasets have:
  - 57 views (7 elevations × 9 azimuths roughly, 57 actual combinations)
  - 21 frames per view (21-frame mp4)
  - sv4d2/ — SV4D 2.0 noisy generated mp4s
  - d-3dgs_video/ — Deformable-3DGS clean reference renders
  - transforms_sv4d2_math.json — camera poses

Output: data/custom/{dataset}/{train,test}/r_{flat_idx:05d}.png + transforms

Flat indexing: flat_idx = view_idx × 21 + frame_idx (view ordered by transforms list)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def extract_frames(mp4_path: Path, n: int) -> np.ndarray:
    r = imageio.get_reader(str(mp4_path))
    return np.stack([r.get_data(t) for t in range(n)], axis=0)


def compute_alpha_from_white_bg(rgb: np.ndarray, threshold: int = 25) -> np.ndarray:
    diff = np.abs(rgb.astype(np.int32) - 255).max(axis=-1)
    return (diff > threshold).astype(np.uint8) * 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lego_v3", "hellwarrior"], required=True)
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--split_mode", choices=["none", "temporal"], default="temporal")
    p.add_argument("--test_every", type=int, default=4)
    args = p.parse_args()

    src_base = Path(f"/mnt/HDD_1/cthsu/{args.dataset}")
    if args.dataset == "lego_v3":
        sub = "lego_r7_train"
        sv4d_iter = "lego_r7_train_iter30000"
    else:
        sub = "hellwarrior_r32_train"
        sv4d_iter = "hellwarrior_r32_train_iter40000"

    src_meta = json.loads((src_base / f"camera_estimation_math_{sub}" / "transforms_sv4d2_math.json").read_text())
    sv4d_dir = src_base / "sv4d2" / sv4d_iter

    out = REPO / "data/custom" / args.dataset
    if out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True)
    (out / "test").mkdir(parents=True)

    fov_x = src_meta["camera_angle_x"]
    print(f"[{args.dataset}] fov_x={fov_x:.4f} n_views={len(src_meta['frames'])} n_frames/view={args.n_frames}")
    print(f"[{args.dataset}] split={args.split_mode}")

    train_frames = []
    test_frames = []

    for v_idx, f in enumerate(src_meta["frames"]):
        tag = f["offset_tag"]
        mp4 = sv4d_dir / f"{tag}.mp4"
        if not mp4.exists():
            print(f"[{args.dataset}] SKIP — mp4 missing: {mp4}")
            continue
        frames = extract_frames(mp4, args.n_frames)
        for t in range(args.n_frames):
            rgb = frames[t]
            alpha = compute_alpha_from_white_bg(rgb)
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            flat = v_idx * args.n_frames + t

            is_test = (args.split_mode == "temporal" and t % args.test_every == 0)
            split = "test" if is_test else "train"
            png_name = f"r_{flat:05d}.png"
            Image.fromarray(rgba, mode="RGBA").save(out / split / png_name)

            frame_meta = {
                "file_path": f"./{split}/{png_name.replace('.png','')}",
                "view_idx": v_idx,
                "view_tag": tag,
                "elevation_deg": f.get("elevation_offset_deg", 0),
                "azimuth_deg": f.get("azimuth_offset_deg", 0),
                "frame_idx": t,
                "transform_matrix": f["transform_matrix"],
                "time": float(t) / max(args.n_frames - 1, 1),
                "rotation": 0,
            }
            if is_test:
                test_frames.append(frame_meta)
            else:
                train_frames.append(frame_meta)
        if (v_idx + 1) % 10 == 0:
            print(f"[{args.dataset}] {v_idx+1}/{len(src_meta['frames'])} views done")

    common = {
        "camera_angle_x": fov_x,
        "n_views": len(src_meta["frames"]),
        "n_frames": args.n_frames,
        "split_mode": args.split_mode,
        "test_every": args.test_every if args.split_mode == "temporal" else None,
    }
    (out / "transforms_train.json").write_text(json.dumps({**common, "frames": train_frames}, indent=2))
    (out / "transforms_test.json").write_text(json.dumps({**common, "frames": test_frames}, indent=2))
    print(f"[{args.dataset}] saved {len(train_frames)} train + {len(test_frames)} test -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
