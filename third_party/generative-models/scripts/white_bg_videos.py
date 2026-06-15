#!/usr/bin/env python3
"""Replace video backgrounds with pure white using rembg matting."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from rembg import remove
from tqdm import tqdm


def composite_white(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255.0)
    out = rgb * alpha + white * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def process_video(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {dst}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames, desc=src.name, leave=False)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgba = remove(frame_rgb)
        white_bgr = cv2.cvtColor(composite_white(rgba), cv2.COLOR_RGB2BGR)
        writer.write(white_bgr)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()


def copy_tree_except_mp4(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(
        src_root,
        dst_root,
        ignore=shutil.ignore_patterns("*.mp4"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/root/generative-models/outputs/jumpingjacks_splitting_r10_train"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/root/generative-models/outputs/jumpingjacks_splitting_r10_train_white"),
    )
    args = parser.parse_args()

    src_root = args.input_dir.resolve()
    dst_root = args.output_dir.resolve()
    videos = sorted(src_root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files under {src_root}")

    print(f"Copying non-video files to {dst_root} ...")
    copy_tree_except_mp4(src_root, dst_root)

    print(f"Processing {len(videos)} videos ...")
    for src in tqdm(videos, desc="Videos"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        process_video(src, dst)

    print(f"Done. White-background videos saved to {dst_root}")


if __name__ == "__main__":
    main()
