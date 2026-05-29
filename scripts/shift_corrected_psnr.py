"""Apply per-view (dy, dx) shift correction + bbox-scale to clean-ref (nobase),
then recompute PSNR vs SV4D.

This is the final spatial fix attempt: after baseplate removal,
we apply the per-frame detected center offset to align centers, then re-PSNR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def load_rgba(p: Path):
    return np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0


def sv4d_to_rgb(rgba):
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a), a[..., 0]


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return -10 * math.log10(max(float(mse), 1e-12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sv4d_dir",  default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--sv4d_meta", default=REPO / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--clean_dir", default=REPO / "runs_aux/clean_gt_fine_nobase/renders")
    p.add_argument("--matching_map", default=REPO / "runs_aux/alignment_A_nobase/matching_map.json")
    p.add_argument("--align_stats",  default=REPO / "runs_aux/clean_ref_aligned_nobase/align_stats.json")
    p.add_argument("--out_dir",      default=REPO / "runs_aux/clean_ref_shift_corrected")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)

    meta = json.loads(Path(args.sv4d_meta).read_text())
    match = json.loads(Path(args.matching_map).read_text())
    stats = json.loads(Path(args.align_stats).read_text())
    per_frame_lookup = {(d["v"], d["t"]): d for d in stats["per_frame"]}

    psnrs_shift = []
    psnrs_full = []
    print(f"[shift] processing {len(meta['frames'])} frames")
    for f in meta["frames"]:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        key = f"v{v}_t{t}"
        if key not in match or (v, t) not in per_frame_lookup: continue
        fi = match[key]["best_fid_idx"]
        sv4d_path = Path(args.sv4d_dir) / f"{Path(f['file_path']).name}.png"
        clean_path = Path(args.clean_dir) / f"r_v{v}_f{fi:03d}.png"
        sv4d_rgba = load_rgba(sv4d_path)
        sv4d_rgb, sv4d_alpha = sv4d_to_rgb(sv4d_rgba)
        clean_rgba = load_rgba(clean_path)
        clean_rgb = clean_rgba[..., :3] * clean_rgba[..., 3:4] + 1.0 * (1 - clean_rgba[..., 3:4])
        H, W = sv4d_rgb.shape[:2]
        if clean_rgb.shape[:2] != (H, W):
            clean_rgb = np.asarray(Image.fromarray((clean_rgb*255).astype(np.uint8)).resize((W, H)),
                                    dtype=np.float32)/255

        # Shift clean by (dy, dx)
        info = per_frame_lookup[(v, t)]
        dy = int(round(info["center_dy"]))
        dx = int(round(info["center_dx"]))
        shifted = np.ones_like(clean_rgb)
        ys = max(0, dy); ye = min(H, H + dy)
        xs = max(0, dx); xe = min(W, W + dx)
        sys = max(0, -dy); sye = sys + (ye - ys)
        sxs = max(0, -dx); sxe = sxs + (xe - xs)
        shifted[ys:ye, xs:xe] = clean_rgb[sys:sye, sxs:sxe]
        psnrs_shift.append(psnr(shifted, sv4d_rgb))
        psnrs_full.append(psnr(clean_rgb, sv4d_rgb))

        # Save view 0 viz
        if v == 0 and t in [0, 5, 10, 15, 20]:
            trio = np.concatenate([sv4d_rgb, np.ones((H, 4, 3)), clean_rgb,
                                    np.ones((H, 4, 3)), shifted], axis=1)
            Image.fromarray((trio * 255).astype(np.uint8)).save(out_dir / "vis" / f"shift_v0_t{t:02d}.png")

    print()
    print(f"[shift] === Final spatial-corrected PSNR ===")
    print(f"[shift] No shift (nobase)    mean = {np.mean(psnrs_full):.3f}  median = {np.median(psnrs_full):.3f}")
    print(f"[shift] With (dy,dx) shift   mean = {np.mean(psnrs_shift):.3f}  median = {np.median(psnrs_shift):.3f}")
    print(f"[shift] Improvement          {np.mean(psnrs_shift)-np.mean(psnrs_full):+.3f} dB")

    (out_dir / "summary.json").write_text(json.dumps({
        "psnr_no_shift_mean": float(np.mean(psnrs_full)),
        "psnr_with_shift_mean": float(np.mean(psnrs_shift)),
        "uplift": float(np.mean(psnrs_shift) - np.mean(psnrs_full)),
        "n_frames": len(psnrs_shift),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
