"""Convert a multi-view video capture into a D-NeRF-format scene SC-GS can train on.

Input layout (the user's capture):

    <src_dir>/
        000000_v000.mp4
        000000_v001.mp4
        ...
        000000_vNNN.mp4
        camera_pos.json     # {"views": [{"view_index": v, "video": "...mp4",
                            #            "estimated_c2w": <4x4 Blender-convention>}]}

Output layout (matches third_party/SC-GS/scene/dataset_readers.py:readNerfSyntheticInfo):

    <out_dir>/
        train/
            r_00000.png ... r_00104.png    # V views x T frames, flat 5-digit index
        transforms_train.json              # camera_angle_x + frames[v*T+t]
        custom_metadata.json               # provenance

Notes:
    * SC-GS's filename sort key (dataset_readers.py:292) parses the int after the
      last '_' in the basename. We therefore flatten to a single integer index
      (v*T + t) and put view/frame metadata in JSON fields, not in the filename.
    * The c2w matrices in camera_pos.json already use Blender convention (camera
      looks down -Z, +X right, +Y up) and point at the origin -- they can be
      written into transform_matrix as-is.
    * No points3d.ply is emitted: SC-GS's reader falls back to a 100k random
      point cloud in [-1.3, 1.3]^3 (dataset_readers.py:398), which works for
      object-centered orbit captures at radius ~4 with D-NeRF default FoV.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/multiview_videos_to_dnerf.py \\
        --src_dir /mnt/HDD_1/cthsu/multiview_videos \\
        --out_dir data/custom/scene00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


D_NERF_CAMERA_ANGLE_X = 0.6911112070083618  # 50mm on 36mm sensor, Blender default


def load_cam_pos(src_dir: Path) -> list[dict]:
    cam_json = src_dir / "camera_pos.json"
    if not cam_json.is_file():
        raise FileNotFoundError(cam_json)
    data = json.loads(cam_json.read_text())
    views = data["views"]
    views = sorted(views, key=lambda v: int(v["view_index"]))
    return views


def decode_video(video_path: Path) -> np.ndarray:
    """Return (T, H, W, 3) uint8."""
    arr = np.asarray(iio.imread(video_path))
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"unexpected video shape {arr.shape} from {video_path}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def convert(
    src_dir: Path,
    out_dir: Path,
    camera_angle_x: float = D_NERF_CAMERA_ANGLE_X,
    overwrite: bool = False,
) -> dict:
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()

    views = load_cam_pos(src_dir)
    V = len(views)
    if V == 0:
        raise ValueError(f"no views in {src_dir/'camera_pos.json'}")

    if out_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "train"
    train_dir.mkdir(exist_ok=True)

    # Decode all videos up front and verify T is consistent across views.
    per_view_frames: list[np.ndarray] = []
    for v in views:
        mp4 = src_dir / v["video"]
        if not mp4.is_file():
            raise FileNotFoundError(mp4)
        frames = decode_video(mp4)
        per_view_frames.append(frames)
        print(f"[convert]   view {v['view_index']}: {mp4.name} -> {frames.shape}")

    T_min = min(f.shape[0] for f in per_view_frames)
    if any(f.shape[0] != T_min for f in per_view_frames):
        print(f"[convert]   note: trimming all views to T={T_min} (shortest video)")
    H, W = per_view_frames[0].shape[1:3]
    if any((f.shape[1], f.shape[2]) != (H, W) for f in per_view_frames):
        raise ValueError("inconsistent (H, W) across views")

    frames_meta: list[dict] = []
    n_written = 0
    for v_idx, (view_obj, view_frames) in enumerate(zip(views, per_view_frames)):
        c2w = np.asarray(view_obj["estimated_c2w"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"bad c2w shape {c2w.shape} for view {v_idx}")
        for t in range(T_min):
            flat_idx = v_idx * T_min + t
            stem = f"r_{flat_idx:05d}"
            Image.fromarray(view_frames[t]).save(train_dir / f"{stem}.png")
            n_written += 1
            frames_meta.append({
                "file_path": f"./train/{stem}",
                "rotation": 0.0,
                "time": float(t) / max(T_min - 1, 1),
                "transform_matrix": c2w.tolist(),
                "view_idx": int(view_obj["view_index"]),
                "frame_idx": int(t),
                "src_video": view_obj["video"],
            })

    transforms_train = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames_meta,
    }
    (out_dir / "transforms_train.json").write_text(
        json.dumps(transforms_train, indent=2)
    )

    metadata = {
        "source": "multiview_videos_to_dnerf",
        "src_dir": str(src_dir),
        "n_views": int(V),
        "n_frames_per_view": int(T_min),
        "image_height": int(H),
        "image_width": int(W),
        "camera_angle_x": float(camera_angle_x),
        "n_train_images_written": int(n_written),
    }
    (out_dir / "custom_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src_dir", type=Path, required=True,
                   help="Directory with N videos + camera_pos.json")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output D-NeRF-format scene dir")
    p.add_argument("--camera_angle_x", type=float, default=D_NERF_CAMERA_ANGLE_X,
                   help=f"Horizontal FoV in radians (default {D_NERF_CAMERA_ANGLE_X:.4f})")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print(f"[convert] src_dir = {args.src_dir}")
    print(f"[convert] out_dir = {args.out_dir}")
    meta = convert(args.src_dir, args.out_dir,
                   camera_angle_x=args.camera_angle_x,
                   overwrite=args.overwrite)
    print(f"[convert] wrote {meta['n_train_images_written']} train images "
          f"({meta['n_views']} views x {meta['n_frames_per_view']} frames, "
          f"{meta['image_width']}x{meta['image_height']})")
    print(f"[convert] done. point SC-GS --source_path at {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
