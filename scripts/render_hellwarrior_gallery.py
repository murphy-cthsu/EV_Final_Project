"""Hellwarrior qualitative visualizations: 3-col gallery + novel-pose orbit.

Outputs (runs_aux/hellwarrior_gallery/):
  gallery_v{v}.gif        SV4D | clean d-3dgs | ours, 21 frames, per grid view
  gallery_keyframes.png   rows = views, cols = SV4D | clean | ours @ t=10
  orbit_novel.gif         camera slerped along the elev-0 ring BETWEEN grid
                          azimuths (novel poses) while motion plays;
                          left = floor model (clean-supervised oracle),
                          right = ours (SV4D-supervised) -> the 9.2 dB
                          oracle gap, visually.

Run (scgs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_hellwarrior_gallery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))

import torch  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402

from eval_region_psnr import make_partrigid_renderer  # noqa: E402

SCENE = "hellwarrior"
OURS_LABEL = "hellwarrior_cleancanon_ctrl_rotfix"
FLOOR_LABEL = "hellwarrior_cleancanon_floor"
VIEWS = [0, 7, 14, 28]
T_FULL = 21
H = W = 576

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    FONT = ImageFont.load_default()


def label_img(arr01, text, color=(0, 0, 0)):
    pil = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, pil.width, 28], fill=(255, 255, 255))
    d.text((8, 4), text, fill=color, font=FONT)
    return np.asarray(pil, np.float32) / 255.0


def psnr(a, b):
    return -10 * np.log10(max(float(((a - b) ** 2).mean()), 1e-12))


def main():
    scene_dir = REPO / "data/custom" / SCENE
    d3_dir = REPO / "outputs/custom" / f"{SCENE}_d3dgs_ref" / "renders"
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = {}
    ring = {}  # az -> frame (elev 0)
    for f in meta["frames"] + meta_te["frames"]:
        frames[(int(f["view_idx"]), int(f["frame_idx"]))] = f
        if float(f.get("elevation_deg", 99)) == 0.0:
            ring[float(f["azimuth_deg"])] = f
    fov_x = meta["camera_angle_x"]
    from utils.graphics_utils import focal2fov, fov2focal
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    out = REPO / "runs_aux/hellwarrior_gallery"
    out.mkdir(parents=True, exist_ok=True)

    render_ours = make_partrigid_renderer(OURS_LABEL, pipe, bg, d_rot_zero=True)
    render_floor = make_partrigid_renderer(FLOOR_LABEL, pipe, bg, d_rot_zero=True)

    def load_sv4d(v, t):
        f = frames[(v, t)]
        rgba = np.asarray(Image.open(scene_dir / (f["file_path"].lstrip("./") + ".png")),
                          np.float32) / 255.0
        a = rgba[..., 3:4]
        return rgba[..., :3] * a + (1 - a)

    def load_clean(v, t):
        arr = np.asarray(iio.imread(d3_dir / f"{v * T_FULL + t:05d}.png"), np.float32) / 255.0
        if arr.shape[-1] == 4:
            a = arr[..., 3:4]
            return arr[..., :3] * a + (1 - a)
        return arr[..., :3]

    # ---------- 3-col gallery per grid view ----------
    keyrows = []
    for v in VIEWS:
        gif_frames = []
        ps_sv, ps_cl = [], []
        az = frames[(v, 0)].get("azimuth_deg", "?")
        el = frames[(v, 0)].get("elevation_deg", "?")
        for t in range(T_FULL):
            ours = render_ours(frames[(v, t)], fov_x, FovY, T_FULL)
            sv = load_sv4d(v, t)
            cl = load_clean(v, t)
            ps_sv.append(psnr(ours, sv)); ps_cl.append(psnr(ours, cl))
            row = np.concatenate([
                label_img(sv, f"SV4D (supervision)  v{v} az{az} el{el} t{t:02d}"),
                label_img(cl, "clean d-3dgs (GT)"),
                label_img(ours, f"ours  {ps_cl[-1]:.1f} dB vs GT", (180, 0, 0)),
            ], axis=1)
            gif_frames.append((row * 255).astype(np.uint8))
            if t == 10:
                keyrows.append(row)
        iio.imwrite(out / f"gallery_v{v:02d}.gif", gif_frames, duration=120, loop=0)
        print(f"  gallery v{v}: ours vs clean {np.mean(ps_cl):.2f} dB / vs SV4D {np.mean(ps_sv):.2f} dB")

    Image.fromarray((np.concatenate(keyrows, axis=0) * 255).astype(np.uint8)).save(
        out / "gallery_keyframes.png")

    # ---------- novel-pose orbit: slerp between elev-0 grid cameras ----------
    azs = sorted(ring.keys())
    mats = {a: np.asarray(ring[a]["transform_matrix"], np.float64) for a in azs}
    orbit = []
    for t in range(T_FULL):
        az_t = 360.0 * t / T_FULL + 15.0  # offset so poses fall BETWEEN grid azimuths
        az_t = az_t % 360.0
        lo = max([a for a in azs if a <= az_t], default=azs[-1])
        hi = min([a for a in azs if a > az_t], default=azs[0] + 360.0)
        hi_key = hi % 360.0
        w = (az_t - lo) / max(hi - lo, 1e-6)
        m0, m1 = mats[lo], mats[hi_key]
        sl = Slerp([0, 1], Rotation.from_matrix(np.stack([m0[:3, :3], m1[:3, :3]])))
        R_t = sl([w]).as_matrix()[0]
        # positions live on a circle; lerp + renormalize radius to stay on the arc
        c0, c1 = m0[:3, 3], m1[:3, 3]
        c_t = (1 - w) * c0 + w * c1
        r_avg = (np.linalg.norm(c0[:2]) + np.linalg.norm(c1[:2])) / 2
        c_t[:2] *= r_avg / max(np.linalg.norm(c_t[:2]), 1e-9)
        c2w = np.eye(4); c2w[:3, :3] = R_t; c2w[:3, 3] = c_t
        f_t = {"frame_idx": t, "transform_matrix": c2w.tolist()}
        fl = render_floor(f_t, fov_x, FovY, T_FULL)
        ou = render_ours(f_t, fov_x, FovY, T_FULL)
        row = np.concatenate([
            label_img(fl, f"oracle/floor (clean-supervised, 22.7 dB)  az~{az_t:.0f} t{t:02d}"),
            label_img(ou, "ours (SV4D-supervised, 13.5 dB)", (180, 0, 0)),
        ], axis=1)
        orbit.append((row * 255).astype(np.uint8))
    iio.imwrite(out / "orbit_novel.gif", orbit, duration=160, loop=0)
    print(f"[gallery] wrote {out}")


if __name__ == "__main__":
    main()
