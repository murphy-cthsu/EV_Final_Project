"""SAM 2 hierarchical mask generation on D-NeRF canonical training images.

Produces a 2D part-id label map for each scene's r_000.png (or first train img).
Per-Gaussian projection is deferred until the canonical 3DGS exists (HANDOFF §4.7
comment: 'Project to Gaussian-level: requires the canonical 3DGS from AnySplat
first. For now, save the 2D label map.').

Outputs (each scene):
  runs_aux/<scene>_label_map_2d.pt    int64 (H, W) tensor, -1 = background
  runs_aux/<scene>_label_overlay.png  visualization for sanity-check

Usage:
    conda activate motionprior
    python scripts/sam2_seg_dnerf.py \\
        --scenes jumpingjacks hellwarrior bouncingballs standup \\
        --checkpoint checkpoints/sam2_hiera_large.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motionprior.segmentation.parts import SAM2Segmenter  # noqa: E402


def overlay(image_rgba: np.ndarray, label_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Make a colorized overlay of label_map on top of image for visual inspection."""
    rng = np.random.default_rng(0)
    # discrete distinct colors per id; -1 stays transparent
    ids = np.unique(label_map)
    palette = {-1: np.array([0, 0, 0], dtype=np.uint8)}
    for k in ids:
        if k == -1:
            continue
        palette[int(k)] = rng.integers(40, 255, size=3, dtype=np.uint8)
    color = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for k, c in palette.items():
        color[label_map == k] = c

    rgb = image_rgba[..., :3] if image_rgba.shape[-1] == 4 else image_rgba
    rgb = rgb.astype(np.float32)
    color_f = color.astype(np.float32)
    mask = (label_map != -1)[..., None]
    out = np.where(mask, alpha * color_f + (1 - alpha) * rgb, rgb)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", required=True,
                   help="Scene names under data/dnerf/")
    p.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt", type=Path)
    p.add_argument("--data_root", default="data/dnerf", type=Path)
    p.add_argument("--out_dir", default="runs_aux", type=Path)
    p.add_argument("--image_name", default="r_000.png",
                   help="Which train-split frame to segment (default r_000.png)")
    p.add_argument("--model_cfg", default="sam2_hiera_l.yaml",
                   help="Hydra config name shipped with the sam2 package")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        print(f"[sam2] FATAL: checkpoint not found at {ckpt}")
        return 2
    print(f"[sam2] checkpoint: {ckpt}  ({ckpt.stat().st_size/1e6:.1f} MB)")
    print(f"[sam2] cfg:        {args.model_cfg}")
    print(f"[sam2] scenes:     {args.scenes}")

    # Construct once; SAM2AutomaticMaskGenerator is reusable across images
    print(f"[sam2] loading model...")
    seg = SAM2Segmenter(checkpoint_path=str(ckpt), model_cfg=args.model_cfg)

    summary = []
    for scene in args.scenes:
        img_path = args.data_root / scene / "train" / args.image_name
        if not img_path.is_file():
            print(f"[sam2] {scene}: MISSING {img_path}")
            continue
        print(f"[sam2] {scene}: segmenting {img_path}")
        img = iio.imread(img_path)
        # SAM 2 expects RGB uint8 HxWx3
        if img.shape[-1] == 4:
            # composite onto white (D-NeRF train images are RGBA with transparent bg)
            rgba = img.astype(np.float32) / 255.0
            white_bg = np.ones_like(rgba[..., :3])
            img_rgb = (rgba[..., :3] * rgba[..., 3:4] + white_bg * (1 - rgba[..., 3:4])) * 255
            img_rgb = img_rgb.astype(np.uint8)
        else:
            img_rgb = img.astype(np.uint8)

        label_map = seg.segment(img_rgb)
        n_parts = int(label_map.max()) + 1 if label_map.max() >= 0 else 0

        out_label = args.out_dir / f"{scene}_label_map_2d.pt"
        out_vis = args.out_dir / f"{scene}_label_overlay.png"
        torch.save(torch.from_numpy(label_map), out_label)
        iio.imwrite(out_vis, overlay(img, label_map))

        # area fractions for the 5 biggest parts
        parts_info = []
        unique, counts = np.unique(label_map[label_map != -1], return_counts=True)
        order = np.argsort(-counts)
        for k in order[:5]:
            parts_info.append((int(unique[k]), float(counts[k] / label_map.size)))
        print(f"[sam2]   {n_parts} parts; top-5 area fractions: " +
              ", ".join(f"#{i}={a:.3f}" for i, a in parts_info))
        print(f"[sam2]   -> {out_label}")
        print(f"[sam2]   -> {out_vis}")
        summary.append({
            "scene": scene,
            "n_parts": n_parts,
            "top5": parts_info,
        })

    print("\n[sam2] summary:")
    for s in summary:
        print(f"  {s['scene']:15s}  {s['n_parts']:3d} parts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
