"""Stage B.1 (final): 2-part decomposition via body-SAM + complement.

Strategy:
  - body (static): SAM-2 prompted on cabin+tread region per view
  - arm  (moves) : the existing FG (full digger) mask MINUS body mask

This sidesteps SAM-2's poor sub-part decomposition on uniformly-colored lego
parts. We only need to nail the body mask; arm follows by set-complement.

Output: runs_aux/parts_2/view{v}_frame0_part_masks.npy of shape (2, H, W) uint8
where channel 0 = arm, channel 1 = body.

Usage:
    /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_2parts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM2_PATH = REPO_ROOT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_PATH))

import torch  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_CFG = "configs/sam2/sam2_hiera_l.yaml"


# Body anchor points per view. Body = cabin + treads (the static portion).
# We use multiple positive points (cabin + treads) and negative points on the
# bucket+arm to push SAM-2 to exclude them.
BODY_ANCHORS = {
    # view: (positive_points, negative_points)
    0: ([(290, 220), (280, 360)], [(180, 80),  (250, 100)]),
    1: ([(200, 270), (150, 360)], [(380, 90),  (320, 150)]),
    2: ([(270, 300), (280, 380)], [(200, 130), (270, 200)]),
    3: ([(280, 250), (280, 360)], [(130, 100), (200, 130)]),
    4: ([(290, 270), (280, 360)], [(180, 130), (230, 180)]),
}


def overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = alpha * c + (1 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    src = REPO_ROOT / "data" / "custom" / "scene00_masked"
    out_dir = REPO_ROOT / "runs_aux" / "parts_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parts2] loading SAM-2")
    sam = build_sam2(DEFAULT_CFG, str(DEFAULT_CKPT), device="cuda")
    predictor = SAM2ImagePredictor(sam)

    data = json.loads((src / "transforms_train.json").read_text())
    frame0 = sorted([f for f in data["frames"] if int(f["frame_idx"]) == 0],
                    key=lambda x: int(x["view_idx"]))

    for f in frame0:
        v = int(f["view_idx"])
        png = src / "train" / f"{Path(f['file_path']).name}.png"
        rgba = np.asarray(iio.imread(png))
        rgb = rgba[..., :3]
        fg_mask = (rgba[..., 3] > 127)  # full digger silhouette

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)
            pos, neg = BODY_ANCHORS[v]
            pts = np.asarray(pos + neg, dtype=np.float32)
            labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=pts, point_labels=labels,
                    multimask_output=True,
                )
        # Pick best, AND with FG (body must be inside FG silhouette)
        best = int(np.argmax(scores))
        body_raw = masks[best].astype(bool)
        body = body_raw & fg_mask
        # Arm = FG MINUS body
        arm = fg_mask & (~body)

        # Stack (2, H, W) — channel 0=arm, 1=body
        out = np.stack([arm.astype(np.uint8), body.astype(np.uint8)], axis=0)
        np.save(out_dir / f"view{v}_frame0_part_masks.npy", out)

        # Visualization
        vis = rgb.copy()
        vis = overlay(vis, arm, (255, 100, 200), alpha=0.55)   # arm = pink
        vis = overlay(vis, body, (0, 200, 255), alpha=0.55)    # body = cyan
        # mark anchors
        R = 5
        for x, y in pos:
            vis[max(0, y-R):y+R+1, max(0, x-R):x+R+1] = [0, 255, 0]
        for x, y in neg:
            vis[max(0, y-R):y+R+1, max(0, x-R):x+R+1] = [255, 0, 0]
        Image.fromarray(vis).save(out_dir / f"view{v}_frame0_overlay.png")

        print(f"[parts2] view {v}: arm={int(arm.sum()):>6}  body={int(body.sum()):>6}  "
              f"fg_total={int(fg_mask.sum()):>6}  body_score={scores[best]:.3f}")

    print(f"[parts2] done. Inspect {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
