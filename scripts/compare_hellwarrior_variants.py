"""Compare hellwarrior model variants by PER-AZIMUTH PSNR (not just the mean).

The mean PSNR-vs-clean averages over all 57 views including the noisy far-back
cone, which can mask improvements in the reliable near-input region (which is
what 'looks like lego' visually). This breaks the comparison down by azimuth
distance from the input view and renders a side-by-side gallery.

Run (scgs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/compare_hellwarrior_variants.py \
      --labels hellwarrior_cleancanon_ctrl_rotfix hw_relw_b1p5 hw_relw_b3 \
      --names control beta1.5 beta3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from eval_region_psnr import make_partrigid_renderer  # noqa: E402

SCENE = "hellwarrior"
CANON = "/mnt/HDD_1/cthsu/EV_Final_Project/outputs/hellwarrior_scgs_default_node/point_cloud/iteration_30000/point_cloud.ply"
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
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--gallery_views", type=int, nargs="+", default=[0, 7, 14, 28])
    p.add_argument("--gallery_t", type=int, default=10)
    p.add_argument("--eval_times", type=int, nargs="+", default=None,
                   help="subsample timesteps for the per-azimuth PSNR (default: all). "
                        "CPU-side deformation is the bottleneck, so e.g. 0 5 10 15 20 "
                        "is ~4x faster with near-identical per-azimuth means.")
    args = p.parse_args()
    assert len(args.labels) == len(args.names)

    scene_dir = REPO / "data/custom" / SCENE
    d3_dir = REPO / "outputs/custom" / f"{SCENE}_d3dgs_ref" / "renders"
    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    frames = meta["frames"] + meta_te["frames"]
    fov_x = meta["camera_angle_x"]
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    renders = {n: make_partrigid_renderer(l, pipe, bg, d_rot_zero=True)
               for n, l in zip(args.names, args.labels)}

    def clean(v, t):
        a = np.asarray(iio.imread(d3_dir / f"{v * T_FULL + t:05d}.png"), np.float32) / 255.0
        if a.shape[-1] == 4:
            al = a[..., 3:4]; return a[..., :3] * al + (1 - al)
        return a[..., :3]

    # per-azimuth-distance PSNR
    binned = {n: defaultdict(list) for n in args.names}
    overall = {n: [] for n in args.names}
    for f in frames:
        v, t = int(f["view_idx"]), int(f["frame_idx"])
        if args.eval_times is not None and t not in args.eval_times:
            continue
        az = float(f.get("azimuth_deg", 0)) % 360
        azd = min(az, 360 - az)
        gt = clean(v, t)
        for n in args.names:
            ps = psnr(renders[n](f, fov_x, FovY, T_FULL), gt)
            binned[n][azd].append(ps); overall[n].append(ps)

    azds = sorted(binned[args.names[0]].keys())
    print("\nPSNR vs clean by azimuth distance from input view:")
    hdr = "az_dist  " + "  ".join(f"{n:>10}" for n in args.names)
    print(hdr); print("-" * len(hdr))
    for azd in azds:
        row = f"{azd:6.0f}   " + "  ".join(
            f"{np.mean(binned[n][azd]):10.2f}" for n in args.names)
        print(row)
    print("-" * len(hdr))
    print("mean     " + "  ".join(f"{np.mean(overall[n]):10.2f}" for n in args.names))
    # reliable cone (az_dist <= 60) vs far (>= 150)
    print("\nreliable cone (az_dist<=60) mean:")
    for n in args.names:
        rel = [p for azd in azds if azd <= 60 for p in binned[n][azd]]
        far = [p for azd in azds if azd >= 150 for p in binned[n][azd]]
        print(f"  {n:>10}: reliable {np.mean(rel):.2f}  far {np.mean(far):.2f}")

    # gallery: rows = views, cols = clean GT | each variant @ t
    out = REPO / "runs_aux/hellwarrior_variant_compare"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.gallery_views:
        f = next(fr for fr in frames if int(fr["view_idx"]) == v
                 and int(fr["frame_idx"]) == args.gallery_t)
        az = float(f.get("azimuth_deg", 0))
        gt = clean(v, args.gallery_t)
        cells = [lbl(gt, f"clean GT  v{v} az{az:.0f}")]
        for n in args.names:
            pred = renders[n](f, fov_x, FovY, T_FULL)
            cells.append(lbl(pred, f"{n}  {psnr(pred, gt):.1f} dB",
                             (180, 0, 0) if n != args.names[0] else (0, 0, 0)))
        rows.append(np.concatenate(cells, axis=1))
    Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(
        out / "variant_keyframes.png")
    print(f"\n[compare] wrote {out}/variant_keyframes.png")

    # per-azimuth curve plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for n in args.names:
        ax.plot(azds, [np.mean(binned[n][a]) for a in azds], "o-", label=n)
    ax.set_xlabel("azimuth distance from input view (deg)")
    ax.set_ylabel("PSNR vs clean (dB)")
    ax.set_title("hellwarrior — reliability-weighted supervision by view")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "per_azimuth_psnr.png", dpi=130)
    print(f"[compare] wrote {out}/per_azimuth_psnr.png")


if __name__ == "__main__":
    main()
