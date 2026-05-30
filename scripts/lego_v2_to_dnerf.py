"""Convert lego.zip (SV4D 2.0 5-view × 21-frame) → D-NeRF format scene.

Input:
  /mnt/HDD_1/cthsu/lego/
    sv4d2/000001_v{0-4}.mp4      training source (RGB on white bg)
    d-3dgs_video/000001_v{0-4}/*.mp4   reference (clean D-3DGS render, for eval)
    transforms_sv4d2_math.json   per-view azimuth + camera pose

Output:
  data/custom/lego_v2/
    train/r_{flat_idx:05d}.png   RGBA (alpha from white-bg-diff)
    transforms_train.json        D-NeRF format
    test/...                     (held out — temporal split for benchmark)
    transforms_test.json

Flags:
  --exclude_v000   drop view 0 (the SV4D input), train on 4 noisy VGM views only.
  --split_mode {none,temporal,view}  test split. Default 'none' = no split (all 105 = train).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def extract_frames(mp4_path: Path, n_frames: int = 21) -> np.ndarray:
    """Return (T, H, W, 3) uint8."""
    r = imageio.get_reader(str(mp4_path))
    frames = []
    for t in range(n_frames):
        frames.append(r.get_data(t))
    return np.stack(frames, axis=0)


def compute_alpha_from_white_bg(rgb: np.ndarray, threshold: int = 25) -> np.ndarray:
    """Alpha = 255 where pixel differs from white by more than threshold.
    Returns (H, W) uint8.
    """
    diff = np.abs(rgb.astype(np.int32) - 255).max(axis=-1)
    return (diff > threshold).astype(np.uint8) * 255


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default=Path("/mnt/HDD_1/cthsu/lego"))
    p.add_argument("--out_dir", default=REPO / "data/custom/lego_v2")
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--n_views",  type=int, default=5)
    p.add_argument("--exclude_v000", action="store_true",
                   help="Drop view 0 (SV4D input) — 4 VGM-only views")
    p.add_argument("--split_mode", choices=["none", "temporal", "view"], default="none")
    p.add_argument("--test_view", type=int, default=2,
                   help="(split_mode=view) which view goes to test")
    p.add_argument("--test_every", type=int, default=4,
                   help="(split_mode=temporal) every Nth frame to test")
    args = p.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    if args.exclude_v000:
        out = out.parent / f"{out.name}_4v"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)
    (out / "test").mkdir(exist_ok=True)

    # Load source transforms
    src_meta = json.loads((src / "transforms_sv4d2_math.json").read_text())
    fov_x = src_meta["camera_angle_x"]
    print(f"[lego_v2] fov_x = {fov_x:.6f}, {args.n_frames} frames × {args.n_views} views")
    if args.exclude_v000:
        print(f"[lego_v2] EXCLUDING v000 — 4-view VGM-only setup")

    # Build view → (mp4 path, c2w) mapping
    view_data = {}
    for f in src_meta["frames"]:
        v = int(f["view_index"])
        mp4_path = src / "sv4d2" / f"{f['video']}.mp4"
        view_data[v] = {
            "mp4": mp4_path,
            "c2w": np.asarray(f["transform_matrix"], dtype=np.float64),
            "az": float(f.get("azimuth_offset_deg", 0)),
            "metrics": f.get("metrics", {}),
        }

    # Filter v000 if requested
    use_views = [v for v in range(args.n_views) if not (args.exclude_v000 and v == 0)]
    print(f"[lego_v2] using views: {use_views}")

    # Build frame list
    frames_train = []
    frames_test = []
    for v in use_views:
        vd = view_data[v]
        print(f"[lego_v2] view {v} az={vd['az']:.0f}° PSNR={vd['metrics'].get('psnr', '?')}")
        frames_t = extract_frames(vd["mp4"], n_frames=args.n_frames)
        for t in range(args.n_frames):
            rgb = frames_t[t]
            alpha = compute_alpha_from_white_bg(rgb)
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            flat_idx = v * args.n_frames + t
            png_name = f"r_{flat_idx:05d}.png"
            # Decide train vs test
            is_test = False
            if args.split_mode == "view":
                is_test = (v == args.test_view)
            elif args.split_mode == "temporal":
                is_test = (t % args.test_every == 0)
            target_dir = "test" if is_test else "train"
            Image.fromarray(rgba, mode="RGBA").save(out / target_dir / png_name)

            frame_meta = {
                "file_path": f"./{target_dir}/{png_name.replace('.png', '')}",
                "view_idx": v,
                "frame_idx": t,
                "azimuth_offset_deg": vd["az"],
                "transform_matrix": vd["c2w"].tolist(),
                "time": float(t) / max(args.n_frames - 1, 1),
            }
            if is_test:
                frames_test.append(frame_meta)
            else:
                frames_train.append(frame_meta)

    train_meta = {
        "camera_angle_x": fov_x,
        "n_views": len(use_views),
        "n_frames": args.n_frames,
        "frames": frames_train,
        "split_mode": args.split_mode,
        "exclude_v000": args.exclude_v000,
    }
    (out / "transforms_train.json").write_text(json.dumps(train_meta, indent=2))

    if frames_test:
        test_meta = {**train_meta, "frames": frames_test}
        (out / "transforms_test.json").write_text(json.dumps(test_meta, indent=2))

    # Also write v2 stats
    n_total = sum(args.n_frames for _ in use_views)
    print(f"[lego_v2] saved: {len(frames_train)} train + {len(frames_test)} test frames")
    print(f"[lego_v2] output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
