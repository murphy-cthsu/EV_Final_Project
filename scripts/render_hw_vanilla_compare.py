"""Hellwarrior: 4-column comparison SV4D | clean GT | vanilla SC-GS | ours.

Mirrors the lego head-to-head (vanilla geometry-explosion vs ours). Prints
mean PSNR vs clean for vanilla and ours over all frames, and renders a keyframe
grid + per-view GIFs.

Run (scgs env):
  CUDA_VISIBLE_DEVICES=1 python scripts/render_hw_vanilla_compare.py
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
from eval_region_psnr import make_deformmlp_renderer, make_partrigid_renderer  # noqa: E402

SCENE = "hellwarrior"
VANILLA = REPO / "outputs/custom/hellwarrior_vanilla_sam_node"
OURS = "hellwarrior_cleancanon_ctrl_rotfix"
VIEWS = [0, 7, 14, 28, 42]
KEY_T = 10
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
    d3_dir = REPO / "outputs/custom" / f"{SCENE}_d3dgs_ref" / "renders"
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = meta["frames"] + meta_te["frames"]
    by_vt = {(int(f["view_idx"]), int(f["frame_idx"])): f for f in frames}
    fov_x = meta["camera_angle_x"]
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    r_vanilla = make_deformmlp_renderer(VANILLA, pipe, bg)
    r_ours = make_partrigid_renderer(OURS, pipe, bg, d_rot_zero=True)

    def sv4d(f):
        rgba = np.asarray(Image.open(scene_dir / (f["file_path"].lstrip("./") + ".png")),
                          np.float32) / 255.0
        a = rgba[..., 3:4]; return rgba[..., :3] * a + (1 - a)

    def clean(v, t):
        a = np.asarray(iio.imread(d3_dir / f"{v * T_FULL + t:05d}.png"), np.float32) / 255.0
        if a.shape[-1] == 4:
            al = a[..., 3:4]; return a[..., :3] * al + (1 - al)
        return a[..., :3]

    # PSNR over all frames
    pv, po = [], []
    for f in frames:
        v, t = int(f["view_idx"]), int(f["frame_idx"])
        gt = clean(v, t)
        pv.append(psnr(r_vanilla(f, fov_x, FovY, T_FULL), gt))
        po.append(psnr(r_ours(f, fov_x, FovY, T_FULL), gt))
    print(f"[hw-vanilla] vanilla SC-GS vs clean: {np.mean(pv):.2f} dB")
    print(f"[hw-vanilla] ours          vs clean: {np.mean(po):.2f} dB")
    print(f"[hw-vanilla] ours - vanilla = {np.mean(po) - np.mean(pv):+.2f} dB")

    out = REPO / "runs_aux/hellwarrior_vanilla_compare"
    out.mkdir(parents=True, exist_ok=True)

    # keyframe grid
    rows = []
    for v in VIEWS:
        f = by_vt[(v, KEY_T)]
        az = float(f.get("azimuth_deg", 0))
        gt = clean(v, KEY_T)
        van = r_vanilla(f, fov_x, FovY, T_FULL)
        ou = r_ours(f, fov_x, FovY, T_FULL)
        rows.append(np.concatenate([
            lbl(sv4d(f), f"SV4D supervision  v{v} az{az:.0f}"),
            lbl(gt, "clean d-3dgs GT"),
            lbl(van, f"vanilla SC-GS  {psnr(van, gt):.1f} dB", (170, 0, 0)),
            lbl(ou, f"ours  {psnr(ou, gt):.1f} dB", (0, 110, 0)),
        ], axis=1))
    Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(
        out / "vanilla_vs_ours_keyframes.png")

    # one head-to-head GIF (view 0)
    gif = []
    for t in range(T_FULL):
        f = by_vt[(0, t)]
        gt = clean(0, t)
        van = r_vanilla(f, fov_x, FovY, T_FULL)
        ou = r_ours(f, fov_x, FovY, T_FULL)
        gif.append((np.concatenate([
            lbl(sv4d(f), f"SV4D t{t:02d}"),
            lbl(van, f"vanilla SC-GS {psnr(van,gt):.1f}", (170, 0, 0)),
            lbl(ou, f"ours {psnr(ou,gt):.1f}", (0, 110, 0)),
        ], axis=1) * 255).astype(np.uint8))
    iio.imwrite(out / "vanilla_vs_ours_v0.gif", gif, duration=140, loop=0)
    np.savez(out / "psnr.npz", vanilla=np.array(pv), ours=np.array(po))
    print(f"[hw-vanilla] wrote {out}")


if __name__ == "__main__":
    main()
