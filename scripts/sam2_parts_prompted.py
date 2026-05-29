"""Stage B.1 (revised): SAM-2 image predictor with semantic anchor points
per part. Replaces the AMG approach which couldn't sub-decompose the digger.

Parts (defaults, override via CLI):
    bucket: the scoop at the end of the arm
    arm:    the linkage connecting bucket to body
    body:   the static cabin+treads assembly

Anchor points are per-view (576x576). Default values were picked from visual
inspection of view 0 frame 0 of scene00_masked and the corresponding views.

Usage:
    /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_parts_prompted.py
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


# Per-(view, part) anchor pixel coords (x, y) on the 576x576 frame-0 image.
# Each part has one or more positive points + optional negative points to
# exclude the other parts. SAM-2's image predictor uses these to produce a
# clean per-part mask.
#
# IMPORTANT: these are starting guesses; the script visualizes the result so
# we can iterate. Adjust if the masks look wrong.
ANCHORS_DEFAULT: dict[str, dict[int, dict]] = {
    "bucket": {  # the yellow scoop at the end of the arm
        0: {"pos": [(180, 80)],  "neg": [(290, 220), (250, 100)]},
        1: {"pos": [(380, 90)],  "neg": [(200, 270), (320, 150)]},
        2: {"pos": [(200, 130)], "neg": [(270, 300), (270, 200)]},
        3: {"pos": [(130, 100)], "neg": [(280, 250), (200, 130)]},
        4: {"pos": [(180, 130)], "neg": [(290, 270), (230, 180)]},
    },
    "arm": {  # the linkage between bucket and body
        0: {"pos": [(250, 100)], "neg": [(180, 80),  (290, 220)]},
        1: {"pos": [(320, 150)], "neg": [(380, 90),  (200, 270)]},
        2: {"pos": [(270, 200)], "neg": [(200, 130), (270, 300)]},
        3: {"pos": [(200, 130)], "neg": [(130, 100), (280, 250)]},
        4: {"pos": [(230, 180)], "neg": [(180, 130), (290, 270)]},
    },
    "body": {  # cabin + treads (static during motion)
        0: {"pos": [(290, 220), (280, 360)], "neg": [(180, 80),  (250, 100)]},
        1: {"pos": [(200, 270), (150, 360)], "neg": [(380, 90),  (320, 150)]},
        2: {"pos": [(270, 300), (280, 380)], "neg": [(200, 130), (270, 200)]},
        3: {"pos": [(280, 250), (280, 360)], "neg": [(130, 100), (200, 130)]},
        4: {"pos": [(290, 270), (280, 360)], "neg": [(180, 130), (230, 180)]},
    },
}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.55) -> np.ndarray:
    color = np.array(color, dtype=np.float32)
    out = rgb.astype(np.float32).copy()
    out[mask] = alpha * color + (1 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(rgb: np.ndarray, pts_pos, pts_neg) -> np.ndarray:
    out = rgb.copy()
    R = 5
    for x, y in pts_pos:
        out[max(0, y-R):y+R+1, max(0, x-R):x+R+1] = [0, 255, 0]  # green = pos
    for x, y in pts_neg:
        out[max(0, y-R):y+R+1, max(0, x-R):x+R+1] = [255, 0, 0]  # red = neg
    return out


def main() -> int:
    src = REPO_ROOT / "data" / "custom" / "scene00_masked"
    out_dir = REPO_ROOT / "runs_aux" / "parts_prompted"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parts] loading SAM-2 {DEFAULT_CKPT}")
    sam = build_sam2(DEFAULT_CFG, str(DEFAULT_CKPT), device="cuda")
    predictor = SAM2ImagePredictor(sam)

    data = json.loads((src / "transforms_train.json").read_text())
    frame0 = sorted([f for f in data["frames"] if int(f["frame_idx"]) == 0],
                    key=lambda x: int(x["view_idx"]))

    PART_COLORS = {
        "bucket": (255, 200, 0),    # yellow-orange
        "arm":    (0, 200, 255),    # cyan
        "body":   (255, 100, 200),  # pink
    }
    parts = list(ANCHORS_DEFAULT.keys())
    n_parts = len(parts)
    H = W = 576

    # Output mask tensor: (V, P, H, W) uint8
    summary = {"parts": parts, "per_view": {}}
    for f in frame0:
        v = int(f["view_idx"])
        png = src / "train" / f"{Path(f['file_path']).name}.png"
        rgba = np.asarray(iio.imread(png))
        rgb = rgba[..., :3]

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)

        view_masks = np.zeros((n_parts, H, W), dtype=np.uint8)
        debug_overlay = rgb.copy()
        debug_with_points = rgb.copy()

        for p_idx, part in enumerate(parts):
            cfg = ANCHORS_DEFAULT[part][v]
            pos = np.asarray(cfg["pos"], dtype=np.float32)
            neg = np.asarray(cfg["neg"], dtype=np.float32) if cfg["neg"] else np.zeros((0, 2), dtype=np.float32)
            pts = np.concatenate([pos, neg], axis=0)
            labels = np.concatenate([
                np.ones(len(pos), dtype=np.int32),
                np.zeros(len(neg), dtype=np.int32),
            ])
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, scores, _ = predictor.predict(
                        point_coords=pts, point_labels=labels,
                        multimask_output=True,
                    )
            # Pick the best-scoring mask
            best = int(np.argmax(scores))
            m = masks[best].astype(np.uint8)
            view_masks[p_idx] = m
            debug_overlay = overlay_mask(debug_overlay, m.astype(bool), PART_COLORS[part], alpha=0.55)
            debug_with_points = draw_points(debug_with_points, cfg["pos"], cfg["neg"])
            print(f"[parts] view {v} {part:6s}: area={int(m.sum())}, score={scores[best]:.3f}")

        np.save(out_dir / f"view{v}_frame0_part_masks.npy", view_masks)
        Image.fromarray(debug_overlay).save(out_dir / f"view{v}_frame0_overlay.png")
        Image.fromarray(debug_with_points).save(out_dir / f"view{v}_frame0_points.png")
        summary["per_view"][v] = {
            "areas": [int(view_masks[p].sum()) for p in range(n_parts)],
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[parts] done. Inspect overlays in {out_dir}")
    print(f"[parts] If anchors are wrong, edit ANCHORS_DEFAULT in {__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
