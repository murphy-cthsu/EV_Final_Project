"""Novel-view comparison: GT (d-3dgs) | vanilla SC-GS | ours.

Both models were trained on lego_v2 (5 views: elev0 az 0/60/120/180/240).
We render them at lego_v3 camera poses NOT in that set (same world frame,
verified centers match exactly), against the clean d-3dgs reference.

Outputs (runs_aux/novel_view_compare/):
  novel_az<az>_elev<el>.gif   3-panel GIF over 21 frames, per novel view
  novel_keyframe_grid.png     rows = novel views, cols = GT|vanilla|ours @ t=10
Per-view full-frame PSNR vs GT printed on each panel.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))

import torch  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

from eval_region_psnr import make_deformmlp_renderer, make_partrigid_renderer  # noqa: E402

# novel views to show (lego_v3 view_idx): interpolated ring + elevated
NOVEL_TAGS = ["elev_0_az_150", "elev_0_az_210", "elev_0_az_270", "elev_0_az_330",
              "elev_20_az_180", "elev_30_az_120"]
KEY_T = 10
T_FULL = 21


def label_panel(img, text, color=(0, 0, 0)):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 8), text, fill=color, font=font)
    return np.asarray(pil, dtype=np.float32) / 255.0


def psnr(a, b):
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    out_dir = REPO / "runs_aux/novel_view_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    m3 = json.loads((REPO / "data/custom/lego_v3/transforms_train.json").read_text())
    m3t = json.loads((REPO / "data/custom/lego_v3/transforms_test.json").read_text())
    all3 = m3["frames"] + m3t["frames"]
    fov_x = m3["camera_angle_x"]
    d3dir = REPO / "outputs/custom/lego_v3_d3dgs_ref/renders"

    # (view_tag, t) -> frame dict
    by_key = {(f["view_tag"], int(f["frame_idx"])): f for f in all3}
    vidx = {f["view_tag"]: int(f["view_idx"]) for f in all3}

    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, 576), 576)  # unused directly; renderers compute their own

    parser_pipe = _A(); pp = PipelineParams(parser_pipe); pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    print("[novel-view] loading models...")
    vanilla = make_deformmlp_renderer(REPO / "outputs/custom/lego_v2_vanilla_sam_node", pipe, bg)
    ours = make_partrigid_renderer("lego_v2_A1_leakfree_B", pipe, bg, d_rot_zero=True)

    key_rows = []
    for tag in NOVEL_TAGS:
        if (tag, 0) not in by_key and (tag, 1) not in by_key:
            print(f"  !! view {tag} not found, skipping"); continue
        v = vidx[tag]
        frames_gif = []
        ps_v, ps_o = [], []
        key_row = None
        for t in range(T_FULL):
            f = by_key.get((tag, t))
            if f is None:
                continue
            gt = np.asarray(iio.imread(d3dir / f"{v * T_FULL + t:05d}.png"), dtype=np.float32) / 255.0
            gt = gt[..., :3]
            iv = vanilla(f, fov_x, FovY, T_FULL)
            io_ = ours(f, fov_x, FovY, T_FULL)
            ps_v.append(psnr(iv, gt)); ps_o.append(psnr(io_, gt))
            row = np.concatenate([
                label_panel(gt, f"clean GT  {tag}  t={t:02d}"),
                np.ones((gt.shape[0], 4, 3)),
                label_panel(iv, f"vanilla SC-GS  {ps_v[-1]:.1f} dB", (180, 0, 0)),
                np.ones((gt.shape[0], 4, 3)),
                label_panel(io_, f"ours  {ps_o[-1]:.1f} dB", (0, 100, 0)),
            ], axis=1)
            frames_gif.append((row * 255).astype(np.uint8))
            if t == KEY_T:
                key_row = row
        gif_path = out_dir / f"novel_{tag}.gif"
        iio.imwrite(gif_path, frames_gif, duration=120, loop=0)
        print(f"  {tag}: vanilla {np.mean(ps_v):.2f} dB | ours {np.mean(ps_o):.2f} dB -> {gif_path.name}")
        if key_row is not None:
            key_rows.append(key_row)

    grid = np.concatenate([r for kr in key_rows for r in (kr, np.ones((6, kr.shape[1], 3)))][:-1], axis=0)
    grid_path = out_dir / "novel_keyframe_grid.png"
    Image.fromarray((grid * 255).astype(np.uint8)).save(grid_path)
    print(f"[novel-view] wrote {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
