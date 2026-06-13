"""Show that vanilla SC-GS on 57-view hellwarrior does NOT explode: it faithfully
reproduces its (noisy) SV4D supervision, incl. on HELD-OUT timesteps (25.5 dB).

Panel per (view, held-out t): SV4D supervision | vanilla render, PSNR vs SV4D.
A coherent figure here = geometry is intact (the model interpolates held-out
time well); the low vs-CLEAN score elsewhere is overfitting noise, not collapse.

Run (scgs env):
  CUDA_VISIBLE_DEVICES=1 python scripts/render_hw_vanilla_vs_sv4d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))

from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
from eval_region_psnr import make_deformmlp_renderer  # noqa: E402

SCENE = "hellwarrior"
VANILLA = REPO / "outputs/custom/hellwarrior_vanilla_sam_node"
VIEWS = [0, 7, 14, 28, 42]
HELDOUT_T = [0, 8, 16]   # subset of test timesteps
T_FULL = 21
H = W = 576

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
except Exception:
    FONT = ImageFont.load_default()


def psnr(a, b):
    return -10 * np.log10(max(float(((a - b) ** 2).mean()), 1e-12))


def lbl(arr, text, color=(0, 0, 0)):
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil); d.rectangle([0, 0, pil.width, 26], fill=(255, 255, 255))
    d.text((6, 3), text, fill=color, font=FONT)
    return np.asarray(pil, np.float32) / 255.0


def main():
    scene_dir = REPO / "data/custom" / SCENE
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = meta["frames"] + meta_te["frames"]
    by_vt = {(int(f["view_idx"]), int(f["frame_idx"])): f for f in frames}
    test_t = set(int(f["frame_idx"]) for f in meta_te["frames"])
    fov_x = meta["camera_angle_x"]
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    r_van = make_deformmlp_renderer(VANILLA, pipe, bg)

    def sv4d(f):
        rgba = np.asarray(Image.open(scene_dir / (f["file_path"].lstrip("./") + ".png")),
                          np.float32) / 255.0
        a = rgba[..., 3:4]; return rgba[..., :3] * a + (1 - a)

    # mean vs-SV4D on held-out timesteps (should reproduce ~25 dB, no explosion)
    held, allf = [], []
    for f in frames:
        v, t = int(f["view_idx"]), int(f["frame_idx"])
        p = psnr(r_van(f, fov_x, FovY, T_FULL), sv4d(f))
        allf.append(p)
        if t in test_t:
            held.append(p)
    print(f"[van-vs-sv4d] vanilla vs SV4D: held-out-time mean = {np.mean(held):.2f} dB "
          f"| all frames = {np.mean(allf):.2f} dB  (high = coherent fit, no explosion)")

    out = REPO / "runs_aux/hellwarrior_vanilla_compare"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VIEWS:
        cells = []
        for t in HELDOUT_T:
            f = by_vt[(v, t)]
            gt = sv4d(f); pred = r_van(f, fov_x, FovY, T_FULL)
            az = float(f.get("azimuth_deg", 0))
            cells.append(lbl(gt, f"SV4D v{v} az{az:.0f} t{t}(held-out)"))
            cells.append(lbl(pred, f"vanilla  {psnr(pred, gt):.1f} dB vs SV4D", (0, 90, 160)))
        rows.append(np.concatenate(cells, axis=1))
    Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(
        out / "vanilla_fits_sv4d_heldout.png")
    print(f"[van-vs-sv4d] wrote {out}/vanilla_fits_sv4d_heldout.png")


if __name__ == "__main__":
    main()
