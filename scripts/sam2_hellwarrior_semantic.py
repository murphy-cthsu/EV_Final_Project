"""SAM-2 image predictor with semantic anchor points for hellwarrior body parts.

Segments head, torso, limbs, hands, feet, and armor pieces on a single render.
Uses positive/negative point prompts per part; pixels are assigned exclusively
(most specific parts win).

Usage:
    conda run -n sc-gs python scripts/sam2_hellwarrior_semantic.py \\
        --image outputs/custom/hellwarrior_d3dgs_ref/renders/00000.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints" / "sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"

# Anchor points (x, y) on 576x576 front-facing hellwarrior render.
# Character bbox ~[157,107]–[401,508]. Viewer-left = character's right.
PART_ANCHORS: dict[str, dict] = {
    "head": {
        "pos": [(279, 128), (265, 145), (295, 145)],
        "neg": [(279, 210), (200, 180), (360, 180)],
    },
    "chest_armor": {
        "pos": [(279, 195), (255, 200), (305, 200)],
        "neg": [(279, 128), (279, 300), (180, 250)],
    },
    "loincloth_armor": {
        "pos": [(279, 318), (260, 325), (300, 325)],
        "neg": [(279, 195), (220, 400), (340, 400)],
    },
    "shoulder_armor_r": {
        "box": [155, 130, 248, 225],
        "box_only": True,
    },
    "shoulder_armor_l": {
        "box": [312, 130, 405, 225],
        "box_only": True,
    },
    "wrist_armor_r": {
        "box": [165, 248, 205, 278],
        "box_only": True,
    },
    "wrist_armor_l": {
        "box": [368, 248, 408, 278],
        "box_only": True,
    },
    "torso": {
        "pos": [(279, 245), (279, 275), (250, 260), (310, 260)],
        "neg": [(279, 128), (175, 285), (385, 285), (218, 380), (348, 360), (279, 318)],
    },
    "arm_r": {
        "pos": [(208, 235), (195, 255), (200, 210)],
        "neg": [(175, 285), (279, 245), (198, 168)],
        "subtract": ["shoulder_armor_r"],
    },
    "arm_l": {
        "pos": [(360, 215), (375, 250), (368, 275)],
        "neg": [(388, 288), (362, 168), (348, 360)],
        "subtract": ["shoulder_armor_l"],
    },
    "hand_r": {
        "pos": [(172, 288), (180, 300)],
        "neg": [(208, 235), (178, 262), (279, 245)],
    },
    "hand_l": {
        "pos": [(388, 288), (380, 300)],
        "neg": [(352, 235), (382, 262), (279, 245)],
    },
    "leg_r": {
        "pos": [(218, 380), (225, 430), (205, 420), (230, 350)],
        "neg": [(205, 480), (279, 318), (340, 390), (172, 288)],
        "subtract": ["foot_r", "loincloth_armor"],
    },
    "leg_l": {
        "box": [305, 330, 400, 505],
        "box_only": True,
        "subtract": ["foot_l", "loincloth_armor"],
    },
    "foot_r": {
        "pos": [(202, 478), (215, 465), (195, 490)],
        "neg": [(218, 395), (279, 318), (250, 430)],
    },
    "foot_l": {
        "box": [320, 455, 400, 510],
        "box_only": True,
    },
}

# All parts to segment (order only affects PART_IDS / colors).
ALL_PARTS = [
    "head", "torso",
    "chest_armor", "loincloth_armor",
    "shoulder_armor_r", "shoulder_armor_l",
    "wrist_armor_r", "wrist_armor_l",
    "arm_r", "arm_l", "hand_r", "hand_l",
    "leg_r", "leg_l", "foot_r", "foot_l",
]

# Lower = wins on overlap (specific beats coarse).
PART_PRIORITY = {
    "foot_r": 0, "foot_l": 0,
    "hand_r": 1, "hand_l": 1,
    "wrist_armor_r": 2, "wrist_armor_l": 2,
    "shoulder_armor_r": 3, "shoulder_armor_l": 3,
    "head": 4,
    "chest_armor": 5, "loincloth_armor": 5,
    "arm_r": 6, "arm_l": 6,
    "leg_r": 7, "leg_l": 7,
    "torso": 9,
}

PART_COLORS: dict[str, tuple[int, int, int]] = {
    "head": (255, 220, 80),
    "torso": (180, 120, 80),
    "chest_armor": (160, 160, 180),
    "loincloth_armor": (140, 130, 110),
    "shoulder_armor_r": (200, 180, 100),
    "shoulder_armor_l": (200, 100, 180),
    "wrist_armor_r": (120, 200, 255),
    "wrist_armor_l": (255, 120, 200),
    "arm_r": (100, 200, 120),
    "arm_l": (120, 100, 200),
    "hand_r": (80, 255, 150),
    "hand_l": (150, 80, 255),
    "leg_r": (255, 150, 100),
    "leg_l": (100, 150, 255),
    "foot_r": (255, 100, 100),
    "foot_l": (100, 100, 255),
}


def fg_mask_from_rgb(rgb: np.ndarray, thresh: int = 250) -> np.ndarray:
    return rgb.max(axis=-1) < thresh


def predict_part(predictor, cfg: dict) -> np.ndarray:
    box = np.asarray(cfg["box"], dtype=np.float32) if "box" in cfg else None
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if cfg.get("box_only") and box is not None:
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=True,
                )
            else:
                pos = np.asarray(cfg["pos"], dtype=np.float32)
                neg = np.asarray(cfg.get("neg", []), dtype=np.float32)
                if len(neg):
                    pts = np.concatenate([pos, neg], axis=0)
                    labels = np.concatenate(
                        [np.ones(len(pos), dtype=np.int32),
                         np.zeros(len(neg), dtype=np.int32)]
                    )
                else:
                    pts, labels = pos, np.ones(len(pos), dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                )
    return masks[int(np.argmax(scores))].astype(bool)


def resolve_labels(raw_masks: dict[str, np.ndarray], fg: np.ndarray) -> np.ndarray:
    """Per-pixel winner by PART_PRIORITY among masks covering that pixel."""
    h, w = fg.shape
    label_map = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.where(fg)
    for y, x in zip(ys, xs):
        cands = [n for n, m in raw_masks.items() if m[y, x]]
        if not cands:
            continue
        winner = min(cands, key=lambda n: PART_PRIORITY[n])
        label_map[y, x] = PART_IDS[winner]
    return label_map


def overlay(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for name, pid in PART_IDS.items():
        if pid < 0:
            continue
        m = label_map == pid
        if not m.any():
            continue
        c = np.array(PART_COLORS[name], dtype=np.float32)
        out[m] = alpha * c + (1 - alpha) * out[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(rgb: np.ndarray, anchors: dict) -> np.ndarray:
    out = rgb.copy()
    r = 4
    for cfg in anchors.values():
        for x, y in cfg.get("pos", []):
            out[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1] = [0, 255, 0]
        for x, y in cfg.get("neg", []):
            out[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1] = [255, 0, 0]
    return out


PART_IDS = {name: i for i, name in enumerate(ALL_PARTS)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--min_area", type=int, default=80,
                   help="Drop part masks smaller than this (px)")
    args = p.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img_path = args.image.resolve()
    out_dir = args.out_dir or (REPO / "runs_aux" / f"{img_path.stem}_sam2_semantic")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = iio.imread(img_path)
    rgb = img[..., :3] if img.shape[-1] >= 3 else img
    fg = fg_mask_from_rgb(rgb)

    print(f"[sam2] loading model from {CKPT}")
    sam = build_sam2(CFG, str(CKPT), device="cuda")
    predictor = SAM2ImagePredictor(sam)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)

    raw_masks: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for name in ALL_PARTS:
        cfg = PART_ANCHORS[name]
        raw = predict_part(predictor, cfg) & fg
        if raw.sum() < args.min_area:
            print(f"[sam2] {name:18s} SKIPPED raw area={int(raw.sum())}")
            continue
        raw_masks[name] = raw
        summary[name] = {"raw_area": int(raw.sum())}
        print(f"[sam2] {name:18s} raw={int(raw.sum()):6d}")

    for name, cfg in PART_ANCHORS.items():
        if name not in raw_masks or "subtract" not in cfg:
            continue
        for other in cfg["subtract"]:
            if other in raw_masks:
                raw_masks[name] &= ~raw_masks[other]
        print(f"[sam2] {name:18s} after subtract raw={int(raw_masks[name].sum()):6d}")

    label_map = resolve_labels(raw_masks, fg)
    part_masks = {
        name: (label_map == PART_IDS[name])
        for name in raw_masks
    }
    for name in raw_masks:
        summary[name]["final_area"] = int(part_masks[name].sum())
        print(f"[sam2] {name:18s} final={summary[name]['final_area']:6d}")

    unlabeled_fg = fg & (label_map == -1)
    if unlabeled_fg.sum() > 0:
        label_map[unlabeled_fg] = PART_IDS["torso"]
        part_masks["torso"] = part_masks.get("torso", np.zeros_like(fg)) | unlabeled_fg
        print(f"[sam2] filled {int(unlabeled_fg.sum())} unlabeled fg px -> torso")

    iio.imwrite(out_dir / "semantic_overlay.png", overlay(rgb, label_map))
    Image.fromarray(draw_points(rgb, PART_ANCHORS)).save(out_dir / "anchor_points.png")
    torch.save(torch.from_numpy(label_map), out_dir / "label_map_2d.pt")
    np.savez_compressed(
        out_dir / "part_masks.npz",
        **{k: v.astype(np.uint8) for k, v in part_masks.items()},
    )
    (out_dir / "parts.json").write_text(
        json.dumps({"parts": list(part_masks.keys()), "priority": PART_PRIORITY,
                    "summary": summary}, indent=2)
    )
    print(f"[sam2] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
