"""2x3 poster grid showing vanilla SC-GS's failure in our VGM-supervised setting.

Layout (matches meetTW_checkpoint_0601/poster_assets/sample.png):
  top-left  = clean d-3dgs GT (reference)
  other 5   = vanilla SC-GS rendered from various novel angles (black-spike explosion)

Tight grid, thin white borders, label top-left of each tile (white text + black halo).
"""

from __future__ import annotations

import argparse
import math
import sys
import json
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
from viz_vanilla_noprior import (  # noqa: E402
    load_model, render_at, c2w_to_RT, pose_spherical, load_rgba_white, H, W,
)


def tile(img, text, font, scale=1.0):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    if scale != 1.0:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(pil)
    d.text((8, 6), text, fill=(255, 255, 255), font=font,
           stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(pil, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--d3dgs_dir", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--t", type=int, default=10)
    p.add_argument("--gt_view", type=int, default=0)
    p.add_argument("--elev", type=float, default=-15.0)
    p.add_argument("--azis", type=str, default="0,72,144,216,288")
    p.add_argument("--tile_px", type=int, default=420)
    args = p.parse_args()

    scene_dir = Path(args.scene_dir); d3dgs_dir = Path(args.d3dgs_dir)
    out = REPO / "runs_aux/vanilla_noprior_viz" / args.scene
    out.mkdir(parents=True, exist_ok=True)

    meta_tr = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = meta_tr["frames"] + meta_te["frames"]
    fov_x = meta_tr["camera_angle_x"]; fov_y = focal2fov(fov2focal(fov_x, W), H)
    T_full = max(int(f["frame_idx"]) for f in frames) + 1
    radius = float(np.linalg.norm(np.array(frames[0]["transform_matrix"])[:3, 3]))
    fid = args.t / max(T_full - 1, 1)

    g, deform, it = load_model(args.model)
    _pp = _A(); pipe = PipelineParams(_pp).extract(_pp.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # top-left: clean d-3dgs GT at a training view (we only have GT at train cams)
    gt_flat = args.gt_view * T_full + args.t
    gt = load_rgba_white(d3dgs_dir / f"{gt_flat:05d}.png")

    # 5 vanilla renders at novel azimuths
    azis = [float(x) for x in args.azis.split(",")]
    vanilla_imgs = []
    for az in azis:
        R, T = c2w_to_RT(pose_spherical(az, args.elev, radius))
        vanilla_imgs.append(render_at(g, deform, pipe, bg, R, T, fov_x, fov_y, fid))

    sc = args.tile_px / float(W)
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)

    cells = [tile(gt, "GT  (clean d-3dgs)", big, sc)]
    for az, im in zip(azis, vanilla_imgs):
        cells.append(tile(im, f"vanilla SC-GS\naz={az:.0f}°", sm, sc))

    th, tw = cells[0].shape[:2]
    b = 4  # white border
    grid = np.full((2 * th + 3 * b, 3 * tw + 4 * b, 3), 255, dtype=np.uint8)
    for idx, c in enumerate(cells):
        r, col = divmod(idx, 3)
        y = b + r * (th + b); x = b + col * (tw + b)
        grid[y:y + th, x:x + tw] = c
    Image.fromarray(grid).save(out / "vanilla_failure_grid.png")
    print(f"[grid] {out / 'vanilla_failure_grid.png'}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    raise SystemExit(main())
