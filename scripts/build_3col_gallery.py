"""3-column gallery: clean ref || SV4D GT || our part-rigid LBS.

Picks the 25 test frames of scene00_split_t. For each:
  * left   = clean_ref render at same (cam,t)  [runs_aux/clean_gt_at_sv4d_cams/renders/r_NNNNN.png]
  * middle = SV4D GT                            [extracted from eval png left half]
  * right  = part-rigid LBS prediction          [extracted from eval png right half]

Outputs PNG strip and an animated GIF over (view 0, t in test set).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", default=REPO / "runs_aux/clean_gt_at_sv4d_cams/renders")
    p.add_argument("--eval_dir",  default=REPO / "runs_aux/partrigid_eval_sv4d/lbs_photo1_scene00_split_t")
    p.add_argument("--test_json", default=REPO / "data/custom/scene00_split_t/transforms_test.json")
    p.add_argument("--out_dir",   default=REPO / "runs_aux/gallery_3col")
    p.add_argument("--T_full", type=int, default=21, help="frames per view in full scene00")
    args = p.parse_args()

    test_meta = json.loads(Path(args.test_json).read_text())
    frames = test_meta["frames"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    H = 576
    pad = 8
    label_h = 28
    n = len(frames)
    print(f"[gallery] processing {n} test frames")

    # Build per-frame 3-col tile (clean | gt | pred)
    tiles = []
    for i, f in enumerate(frames):
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        flat = v * args.T_full + t

        clean_path = Path(args.clean_dir) / f"r_{flat:05d}.png"
        eval_path  = Path(args.eval_dir)  / f"{i:03d}.png"
        if not clean_path.exists() or not eval_path.exists():
            print(f"[gallery] skip {i} ({clean_path.name} / {eval_path.name})")
            continue

        clean = np.asarray(Image.open(clean_path).convert("RGB"))
        # clean might be 576x576 with alpha pre-comp; ensure HxWx3
        if clean.shape[0] != H:
            clean = np.asarray(Image.fromarray(clean).resize((H, H)))

        eval_img = np.asarray(Image.open(eval_path).convert("RGB"))
        Heval, Weval, _ = eval_img.shape
        # eval is GT|PRED concat horizontally — split at midpoint
        half = Weval // 2
        gt = eval_img[:, :half]
        pred = eval_img[:, half:]
        if gt.shape[0] != H:
            gt = np.asarray(Image.fromarray(gt).resize((H, H)))
            pred = np.asarray(Image.fromarray(pred).resize((H, H)))

        # White spacer
        sep = np.full((H, pad, 3), 255, dtype=np.uint8)
        tile = np.concatenate([clean, sep, gt, sep, pred], axis=1)

        # Add a small header strip with labels and (v, t)
        tile_pil = Image.fromarray(tile)
        header = Image.new("RGB", (tile_pil.width, label_h), (255, 255, 255))
        draw = ImageDraw.Draw(header)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        col_w = H + pad
        draw.text((H // 2 - 60, 6), "Clean ref (D-NeRF GT, 4D-GS)", fill="black", font=font)
        draw.text((col_w + H // 2 - 50, 6), "SV4D GT (VGM, our train)", fill="black", font=font)
        draw.text((2 * col_w + H // 2 - 50, 6), "Ours (Part-rigid LBS)", fill="black", font=font)
        # right-align (v,t)
        draw.text((tile_pil.width - 80, 6), f"v={v}  t={t}", fill="black", font=font)
        full = Image.new("RGB", (tile_pil.width, tile_pil.height + label_h), (255, 255, 255))
        full.paste(header, (0, 0))
        full.paste(tile_pil, (0, label_h))
        tiles.append((v, t, full))

        full.save(out_dir / f"tile_{i:03d}_v{v}_t{t:02d}.png")

    # Build a per-view GIF (view 0 only — that's t=3,7,11,15,19)
    for view in range(5):
        gif_frames = [im for v, t, im in tiles if v == view]
        if not gif_frames:
            continue
        gif_path = out_dir / f"gallery_v{view}.gif"
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=600, loop=0)
        print(f"[gallery] view {view}: {len(gif_frames)} frames -> {gif_path.name}")

    # Vertical contact sheet: all 25 frames stacked
    if tiles:
        widths = [im.width for _, _, im in tiles]
        max_w = max(widths)
        # Resize all to common width
        resized = []
        for v, t, im in tiles:
            if im.width != max_w:
                ratio = max_w / im.width
                im2 = im.resize((max_w, int(im.height * ratio)))
            else:
                im2 = im
            resized.append(im2)
        total_h = sum(im.height for im in resized) + (len(resized) - 1) * 4
        sheet = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for im in resized:
            sheet.paste(im, (0, y))
            y += im.height + 4
        sheet.save(out_dir / "contact_sheet.png")
        print(f"[gallery] contact_sheet.png ({sheet.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
