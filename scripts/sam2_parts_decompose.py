"""Stage B.1: SAM-2 hierarchical part decomposition on frame 0.

Runs SAM-2 AutomaticMaskGenerator on each of the V views' frame 0 image to
produce per-view part masks. Then we manually pick part count K based on
visual inspection, and link parts across views (Stage C will use this).

Usage:
    /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_parts_decompose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM2_PATH = REPO_ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_PATH))

import torch  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_CFG = "configs/sam2/sam2_hiera_l.yaml"


def make_overlay(rgb: np.ndarray, masks: list[dict], alpha: float = 0.55) -> np.ndarray:
    """Color masks distinctly and overlay on rgb. Largest masks drawn first
    (so smaller ones can cover them where they overlap)."""
    masks = sorted(masks, key=lambda m: -m["area"])
    rng = np.random.default_rng(42)
    overlay = rgb.astype(np.float32).copy()
    palette = (rng.integers(60, 240, size=(len(masks), 3))).astype(np.float32)
    for c, m in zip(palette, masks):
        seg = m["segmentation"]
        overlay[seg] = alpha * c + (1 - alpha) * overlay[seg]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def filter_masks(masks: list[dict], min_area: int, fg_alpha: np.ndarray | None,
                 fg_overlap_thresh: float = 0.5) -> list[dict]:
    """Drop tiny masks; if fg_alpha is provided (the SAM-2 video predictor's FG
    silhouette of the object), drop masks that overlap FG <= threshold."""
    out = []
    for m in masks:
        if m["area"] < min_area:
            continue
        if fg_alpha is not None:
            inter = (m["segmentation"] & fg_alpha).sum()
            if inter < fg_overlap_thresh * m["area"]:
                # mostly background, drop
                continue
        out.append(m)
    return out


def main() -> int:
    src = REPO_ROOT / "data" / "custom" / "scene00_masked"
    out_dir = REPO_ROOT / "runs_aux" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parts] loading SAM-2 {DEFAULT_CKPT}")
    sam = build_sam2(DEFAULT_CFG, str(DEFAULT_CKPT), device="cuda")
    amg = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=16,           # coarser grid → fewer over-segmentations
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
        box_nms_thresh=0.5,
        multimask_output=True,
    )

    # Read transforms_train.json to get frame 0 image paths
    data = json.loads((src / "transforms_train.json").read_text())
    frame0 = [f for f in data["frames"] if int(f["frame_idx"]) == 0]
    print(f"[parts] {len(frame0)} frame-0 images")

    summary = {}
    for f in sorted(frame0, key=lambda x: x["view_idx"]):
        v = int(f["view_idx"])
        png_path = src / "train" / f"{Path(f['file_path']).name}.png"
        img = np.asarray(iio.imread(png_path))
        rgba = img if img.shape[-1] == 4 else np.concatenate([img, 255 * np.ones((*img.shape[:2], 1), dtype=img.dtype)], axis=-1)
        rgb = rgba[..., :3]
        fg_alpha = (rgba[..., 3] > 127) if rgba.shape[-1] == 4 else None

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks_raw = amg.generate(rgb)
        # Filter: drop bg + tiny + low-FG-overlap
        masks_filt = filter_masks(masks_raw, min_area=500, fg_alpha=fg_alpha,
                                  fg_overlap_thresh=0.5)
        # Sort by area, keep top-K
        masks_filt = sorted(masks_filt, key=lambda m: -m["area"])[:10]

        print(f"[parts] view {v}: raw={len(masks_raw)}, filtered={len(masks_filt)}, "
              f"areas={[m['area'] for m in masks_filt[:6]]}")
        ov = make_overlay(rgb, masks_filt, alpha=0.55)
        Image.fromarray(ov).save(out_dir / f"view{v}_frame0_parts_overlay.png")

        # Save raw masks as numpy stack for downstream use
        mask_stack = np.stack([m["segmentation"] for m in masks_filt], axis=0).astype(np.uint8)
        np.save(out_dir / f"view{v}_frame0_part_masks.npy", mask_stack)
        summary[v] = {
            "n_parts": len(masks_filt),
            "areas": [int(m["area"]) for m in masks_filt],
            "bboxes": [list(map(int, m["bbox"])) for m in masks_filt],
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[parts] done. Inspect overlays in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
