"""Per-scene videos: (1) ours-vs-vanilla side-by-side GIF over 21 frames,
(2) novel-view orbit GIF (camera slerped BETWEEN grid azimuths) with vanilla|ours.

Run (scgs env), e.g.:
  CUDA_VISIBLE_DEVICES=2 python scripts/render_scene_videos.py \
     --scene jumpingjacks --ours_label jumpingjacks_ours \
     --vanilla outputs/custom/jumpingjacks_vanilla_node \
     --views 0 --d3dgs outputs/custom/jumpingjacks_d3dgs_ref/renders
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal
from eval_region_psnr import make_partrigid_renderer, make_deformmlp_renderer

H = W = 576; T_FULL = 21
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception: FONT = ImageFont.load_default()


def lbl(a, t, c=(0, 0, 0)):
    p = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(p)
    d.rectangle([0, 0, p.width, 24], fill=(255, 255, 255)); d.text((5, 3), t, fill=c, font=FONT)
    return np.asarray(p, np.float32) / 255.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--ours_label", required=True)
    p.add_argument("--vanilla", required=True, help="vanilla node model path")
    p.add_argument("--views", type=int, nargs="+", default=[0])
    p.add_argument("--d3dgs", default=None, help="clean ref renders dir (optional)")
    args = p.parse_args()

    sd = REPO / "data/custom" / args.scene
    meta = json.loads((sd / "transforms_train.json").read_text())
    mte = json.loads((sd / "transforms_test.json").read_text())
    frames = meta["frames"] + mte["frames"]
    byvt = {(int(f["view_idx"]), int(f["frame_idx"])): f for f in frames}
    fov_x = meta["camera_angle_x"]; FovY = focal2fov(fov2focal(fov_x, W), H)
    par = _A(); pp = PipelineParams(par); pipe = pp.extract(par.parse_args([]))
    bg = torch.tensor([1, 1, 1.], device="cuda")
    ro = make_partrigid_renderer(args.ours_label, pipe, bg, d_rot_zero=True)
    rv = make_deformmlp_renderer(args.vanilla, pipe, bg)

    def sv(f):
        rgba = np.asarray(Image.open(sd / (f["file_path"].lstrip("./") + ".png")), np.float32) / 255.
        a = rgba[..., 3:4]; return rgba[..., :3] * a + (1 - a)
    out = REPO / "runs_aux/scene_videos"; out.mkdir(parents=True, exist_ok=True)

    # (1) ours-vs-vanilla GIF
    for v in args.views:
        az = byvt[(v, 0)].get("azimuth_deg", byvt[(v, 0)].get("azimuth_offset_deg", v))
        gif = []
        for t in range(T_FULL):
            f = byvt[(v, t)]
            cells = [lbl(sv(f), f"SV4D az{az:.0f} t{t:02d}"),
                     lbl(rv(f, fov_x, FovY, T_FULL), "vanilla SC-GS", (170, 0, 0)),
                     lbl(ro(f, fov_x, FovY, T_FULL), "ours", (0, 110, 0))]
            gif.append((np.concatenate(cells, 1) * 255).astype(np.uint8))
        imageio.mimwrite(out / f"{args.scene}_compare_v{v}.gif", gif, duration=120, loop=0)
        print(f"wrote {out}/{args.scene}_compare_v{v}.gif")

    # (2) novel-view orbit: slerp between elev-0 grid cameras (non-grid azimuths)
    ring = {}
    for f in frames:
        el = f.get("elevation_deg", f.get("elevation_offset_deg", 0))
        if abs(el) < 1e-3:
            az = float(f.get("azimuth_deg", f.get("azimuth_offset_deg", f["view_idx"])))
            ring[az] = f["transform_matrix"]
    azs = sorted(ring)
    if len(azs) >= 3:
        mats = {a: np.asarray(ring[a], np.float64) for a in azs}
        orbit = []
        for t in range(T_FULL):
            az_t = (360.0 * t / T_FULL + 18.0) % 360.0  # offset so poses fall BETWEEN grid az
            lo = max([a for a in azs if a <= az_t], default=azs[-1])
            hi = min([a for a in azs if a > az_t], default=azs[0] + 360.0)
            w = (az_t - lo) / max(hi - lo, 1e-6)
            m0, m1 = mats[lo], mats[hi % 360.0]
            R_t = Slerp([0, 1], Rotation.from_matrix(np.stack([m0[:3, :3], m1[:3, :3]])))([w]).as_matrix()[0]
            c0, c1 = m0[:3, 3], m1[:3, 3]; c_t = (1 - w) * c0 + w * c1
            r = (np.linalg.norm(c0[:2]) + np.linalg.norm(c1[:2])) / 2
            c_t[:2] *= r / max(np.linalg.norm(c_t[:2]), 1e-9)
            c2w = np.eye(4); c2w[:3, :3] = R_t; c2w[:3, 3] = c_t
            f_t = {"frame_idx": t, "transform_matrix": c2w.tolist()}
            cells = [lbl(rv(f_t, fov_x, FovY, T_FULL), f"vanilla (novel az~{az_t:.0f})", (170, 0, 0)),
                     lbl(ro(f_t, fov_x, FovY, T_FULL), "ours (novel view)", (0, 110, 0))]
            orbit.append((np.concatenate(cells, 1) * 255).astype(np.uint8))
        imageio.mimwrite(out / f"{args.scene}_novel.gif", orbit, duration=140, loop=0)
        print(f"wrote {out}/{args.scene}_novel.gif")
    else:
        print(f"[skip novel] only {len(azs)} elev-0 azimuths")


if __name__ == "__main__":
    main()
