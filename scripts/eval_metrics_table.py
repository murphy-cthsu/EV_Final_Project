"""Tier-1 metric tables: PSNR / SSIM / LPIPS vs BOTH GTs (clean d-3dgs and
SV4D supervision) for all headline models of a scene.

Dual reporting closes the "PSNR rewards blur" hole and gives the perceptual
version of the oracle gap + the overfit gap (vs-SV4D minus vs-clean).

Per-model d_rot convention follows how the model was TRAINED:
rotfix-era models (ctrl_rotfix, L1b) use d_rot_zero=True; legacy models
(oracle/floor, alpha16-era) use the legacy -(1,0,0,0) bias.

Run (scgs env):
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_metrics_table.py --scene lego_v2
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_metrics_table.py --scene hellwarrior
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))

from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

from eval_region_psnr import make_deformmlp_renderer, make_partrigid_renderer  # noqa: E402

H = W = 576

# (name, kind, label_or_path, d_rot_zero)
PRESETS = {
    "lego_v2": [
        ("vanilla",      "mlp",  "lego_v2_vanilla_sam_node",          None),
        ("F1_warmstart", "mlp",  "lego_v2_F1_vanilla_warmstart_node", None),
        ("F2_frozen",    "mlp",  "F2_scgs_frozen",                    None),
        ("ours",         "hier", "lego_v2_ctrl_rotfix",               True),
        ("ours_noStageD", "hier", "lego_v2_L1b_zeroinit",             True),
        ("oracle",       "hier", "lego_v2_d3dgs_sup_ceiling",         False),
    ],
    "hellwarrior": [
        ("ours",   "hier", "hellwarrior_cleancanon_ctrl_rotfix", True),
        ("oracle", "hier", "hellwarrior_cleancanon_floor",       False),
    ],
}


def psnr_fn(a, b):
    return -10 * math.log10(max(float(((a - b) ** 2).mean()), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = p.parse_args()

    from skimage.metrics import structural_similarity
    import lpips
    lpips_model = lpips.LPIPS(net="alex").cuda()

    scene_dir = REPO / "data/custom" / args.scene
    d3_dir = REPO / "outputs/custom" / f"{args.scene}_d3dgs_ref" / "renders"
    meta_t = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    all_frames = meta_t["frames"] + meta_te["frames"]
    fov_x = meta_t["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in all_frames) + 1
    FovY = focal2fov(fov2focal(fov_x, W), H)
    parser_pipe = _A(); pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    renderers = {}
    for name, kind, label, drz in PRESETS[args.scene]:
        if kind == "mlp":
            renderers[name] = make_deformmlp_renderer(REPO / "outputs/custom" / label, pipe, bg)
        else:
            renderers[name] = make_partrigid_renderer(label, pipe, bg, d_rot_zero=bool(drz))

    def load_sv4d(f):
        rgba = np.asarray(Image.open(scene_dir / (f["file_path"].lstrip("./") + ".png")),
                          np.float32) / 255.0
        a = rgba[..., 3:4]
        return rgba[..., :3] * a + (1 - a)

    def load_clean(v, t):
        arr = np.asarray(iio.imread(d3_dir / f"{v * T_full + t:05d}.png"), np.float32) / 255.0
        if arr.shape[-1] == 4:
            a = arr[..., 3:4]
            return arr[..., :3] * a + (1 - a)
        return arr[..., :3]

    def lpips_fn(a, b):
        at = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() * 2 - 1
        bt = torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda() * 2 - 1
        with torch.no_grad():
            return float(lpips_model(at, bt).item())

    frames = all_frames if args.max_frames <= 0 else all_frames[:args.max_frames]
    acc = {n: {gt: {m: [] for m in ("psnr", "ssim", "lpips")} for gt in ("clean", "sv4d")}
           for n, *_ in PRESETS[args.scene]}

    for i, f in enumerate(frames):
        v, t = int(f["view_idx"]), int(f["frame_idx"])
        gt_clean = load_clean(v, t)
        gt_sv4d = load_sv4d(f)
        for name in renderers:
            pred = renderers[name](f, fov_x, FovY, T_full)
            for gt_name, gt in (("clean", gt_clean), ("sv4d", gt_sv4d)):
                acc[name][gt_name]["psnr"].append(psnr_fn(pred, gt))
                acc[name][gt_name]["ssim"].append(
                    structural_similarity(pred, gt, channel_axis=-1, data_range=1.0))
                acc[name][gt_name]["lpips"].append(lpips_fn(pred, gt))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)} frames")

    out = REPO / "runs_aux" / f"metrics_table_{args.scene}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "raw.npz", **{
        f"{n}_{g}_{m}": np.array(acc[n][g][m])
        for n in acc for g in acc[n] for m in acc[n][g]})

    lines = [f"# Metric table — {args.scene} ({len(frames)} frames)",
             "",
             "| model | PSNR vs clean | SSIM vs clean | LPIPS vs clean | PSNR vs SV4D | SSIM vs SV4D | LPIPS vs SV4D | overfit gap (dB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, *_ in PRESETS[args.scene]:
        c, s = acc[name]["clean"], acc[name]["sv4d"]
        gap = np.mean(c["psnr"]) - np.mean(s["psnr"])
        lines.append(
            f"| {name} | {np.mean(c['psnr']):.2f} | {np.mean(c['ssim']):.3f} | "
            f"{np.mean(c['lpips']):.3f} | {np.mean(s['psnr']):.2f} | "
            f"{np.mean(s['ssim']):.3f} | {np.mean(s['lpips']):.3f} | {gap:+.2f} |")
    md = "\n".join(lines) + "\n\n(LPIPS low = good; overfit gap = vs-clean − vs-SV4D, positive = denoising past supervision.)\n"
    (out / "table.md").write_text(md)
    print(md)
    print(f"[metrics] wrote {out}")


if __name__ == "__main__":
    main()
