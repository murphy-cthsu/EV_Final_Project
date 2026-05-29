"""Diagnose + fix clean-ref vs SV4D spatial misalignment.

Reads matching_map.json (from script A) and for each SV4D frame (v, t):
  1. Compute foreground bbox of SV4D and matched clean-ref
  2. Compute (scale, translation) needed to register clean-ref → SV4D
  3. Apply that 2D transform → re-evaluate PSNR
  4. Report: per-view scale + center offset statistics
  5. Visualize before/after on view 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent


def load_rgba(p: Path):
    return np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0


def sv4d_to_rgb(rgba):
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a), a[..., 0]


def fg_bbox(alpha, thresh=0.1):
    """Return (y0, y1, x0, x1) bbox of pixels with alpha > thresh; None if empty."""
    m = alpha > thresh
    if not m.any():
        return None
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def cleanref_alpha_from_rgb(rgb):
    """Clean ref bg is white. FG = pixels that differ from white."""
    return (np.abs(rgb - 1).sum(axis=-1) > 0.05).astype(np.float32)


def crop_resize(img: np.ndarray, bbox, target_hw):
    y0, y1, x0, x1 = bbox
    crop = img[y0:y1, x0:x1]
    pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((target_hw[1], target_hw[0]))
    return np.asarray(pil, dtype=np.float32) / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return -10 * math.log10(max(float(mse), 1e-12))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sv4d_dir",  default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--sv4d_meta", default=REPO / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--clean_dir", default=REPO / "runs_aux/clean_gt_fine/renders")
    p.add_argument("--matching_map", default=REPO / "runs_aux/alignment_A/matching_map.json")
    p.add_argument("--out_dir",   default=REPO / "runs_aux/clean_ref_aligned")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)
    (out_dir / "aligned_renders").mkdir(exist_ok=True)

    meta = json.loads(Path(args.sv4d_meta).read_text())
    match = json.loads(Path(args.matching_map).read_text())
    T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
    print(f"[align] {len(meta['frames'])} SV4D frames, T_full={T_full}")

    per_frame = []
    psnrs_naive = []        # SV4D vs raw clean ref (no align)
    psnrs_bbox = []         # SV4D vs bbox-aligned clean ref
    scale_ratios = []
    center_offsets = []

    for f in meta["frames"]:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        key = f"v{v}_t{t}"
        if key not in match: continue
        fi = match[key]["best_fid_idx"]

        # Load both
        sv4d_path = Path(args.sv4d_dir) / f"{Path(f['file_path']).name}.png"
        clean_path = Path(args.clean_dir) / f"r_v{v}_f{fi:03d}.png"
        sv4d_rgba = load_rgba(sv4d_path)
        sv4d_rgb, sv4d_alpha = sv4d_to_rgb(sv4d_rgba)
        clean_rgba = load_rgba(clean_path)
        clean_rgb = clean_rgba[..., :3] * clean_rgba[..., 3:4] + 1.0 * (1 - clean_rgba[..., 3:4])
        clean_alpha = cleanref_alpha_from_rgb(clean_rgb)

        # Resize clean to match SV4D resolution if differs
        H, W = sv4d_rgb.shape[:2]
        if clean_rgb.shape[:2] != (H, W):
            clean_rgb = np.asarray(Image.fromarray((clean_rgb*255).astype(np.uint8)).resize((W, H)),
                                    dtype=np.float32)/255
            clean_alpha = cleanref_alpha_from_rgb(clean_rgb)

        # Naive PSNR
        psnrs_naive.append(psnr(clean_rgb, sv4d_rgb))

        sv4d_bb = fg_bbox(sv4d_alpha)
        clean_bb = fg_bbox(clean_alpha)
        if sv4d_bb is None or clean_bb is None: continue

        sv4d_h = sv4d_bb[1] - sv4d_bb[0]; sv4d_w = sv4d_bb[3] - sv4d_bb[2]
        clean_h = clean_bb[1] - clean_bb[0]; clean_w = clean_bb[3] - clean_bb[2]
        scale_ratio_h = sv4d_h / max(clean_h, 1)
        scale_ratio_w = sv4d_w / max(clean_w, 1)
        scale_ratio = (scale_ratio_h + scale_ratio_w) / 2
        sv4d_cy = (sv4d_bb[0] + sv4d_bb[1]) / 2; sv4d_cx = (sv4d_bb[2] + sv4d_bb[3]) / 2
        clean_cy = (clean_bb[0] + clean_bb[1]) / 2; clean_cx = (clean_bb[2] + clean_bb[3]) / 2
        scale_ratios.append((v, t, scale_ratio_h, scale_ratio_w, scale_ratio))
        center_offsets.append((v, t, sv4d_cy - clean_cy, sv4d_cx - clean_cx))

        # bbox-aligned PSNR: crop+resize both to common 256x256
        target = (256, 256)
        sv4d_crop = crop_resize(sv4d_rgb, sv4d_bb, target)
        clean_crop = crop_resize(clean_rgb, clean_bb, target)
        ps_bbox = psnr(clean_crop, sv4d_crop)
        psnrs_bbox.append(ps_bbox)

        # Also: register clean to SV4D inside the full-image canvas
        # Place clean's bbox content at sv4d's bbox location, same size
        canvas = np.ones_like(sv4d_rgb)
        clean_in_bbox = clean_rgb[clean_bb[0]:clean_bb[1], clean_bb[2]:clean_bb[3]]
        clean_resized = np.asarray(Image.fromarray((clean_in_bbox*255).astype(np.uint8)).resize(
                                    (sv4d_w, sv4d_h)), dtype=np.float32) / 255
        canvas[sv4d_bb[0]:sv4d_bb[1], sv4d_bb[2]:sv4d_bb[3]] = clean_resized
        ps_canvas = psnr(canvas, sv4d_rgb)

        per_frame.append({
            "v": v, "t": t, "fi": fi,
            "psnr_naive": psnrs_naive[-1],
            "psnr_bbox_aligned": ps_bbox,
            "psnr_canvas_aligned": ps_canvas,
            "scale_ratio_h": scale_ratio_h,
            "scale_ratio_w": scale_ratio_w,
            "center_dy": sv4d_cy - clean_cy,
            "center_dx": sv4d_cx - clean_cx,
            "sv4d_bb": sv4d_bb, "clean_bb": clean_bb,
        })
        # Save aligned canvas as the registered clean ref
        Image.fromarray((canvas * 255).astype(np.uint8)).save(
            out_dir / "aligned_renders" / f"r_v{v}_t{t:02d}.png")

        # Vis: side-by-side w/ bbox overlay, for view 0 only
        if v == 0:
            vis = np.concatenate([sv4d_rgb, np.ones((H, 8, 3)), clean_rgb,
                                  np.ones((H, 8, 3)), canvas], axis=1)
            vis_pil = Image.fromarray((vis * 255).astype(np.uint8))
            d = ImageDraw.Draw(vis_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            offsets = [0, W + 8, 2 * (W + 8)]
            labels = [f"SV4D v=0 t={t}", f"clean ref @ fid={fi/99:.3f}",
                      f"bbox-aligned (PSNR {ps_canvas:.2f})"]
            for off, lbl in zip(offsets, labels):
                d.text((off + 8, 8), lbl, fill="red", font=font)
            # Draw bbox
            for bb, off in [(sv4d_bb, offsets[0]), (clean_bb, offsets[1]), (sv4d_bb, offsets[2])]:
                d.rectangle([off + bb[2], bb[0], off + bb[3], bb[1]], outline="lime", width=2)
            vis_pil.save(out_dir / "vis" / f"diag_v0_t{t:02d}.png")

    # Per-view stats
    print()
    print(f"[align] === Stats ===")
    print(f"[align] PSNR naive (no align)    mean = {np.mean(psnrs_naive):.3f}  median = {np.median(psnrs_naive):.3f}")
    print(f"[align] PSNR bbox-aligned crops  mean = {np.mean(psnrs_bbox):.3f}  median = {np.median(psnrs_bbox):.3f}")
    psnr_canvas_all = [r["psnr_canvas_aligned"] for r in per_frame]
    print(f"[align] PSNR canvas-aligned      mean = {np.mean(psnr_canvas_all):.3f}  median = {np.median(psnr_canvas_all):.3f}")
    print()
    # Per-view scale
    sr = np.array(scale_ratios)
    for v in range(5):
        mask = sr[:, 0] == v
        if mask.any():
            print(f"[align] view {v}: scale_h mean={sr[mask, 2].mean():.3f}±{sr[mask, 2].std():.3f}"
                  f"  scale_w mean={sr[mask, 3].mean():.3f}±{sr[mask, 3].std():.3f}"
                  f"  avg_scale={sr[mask, 4].mean():.3f}")
    print()
    co = np.array(center_offsets)
    for v in range(5):
        mask = co[:, 0] == v
        if mask.any():
            print(f"[align] view {v}: center_dy mean={co[mask, 2].mean():+.1f}px  center_dx mean={co[mask, 3].mean():+.1f}px")

    # Save table
    (out_dir / "align_stats.json").write_text(json.dumps({
        "psnr_naive_mean": float(np.mean(psnrs_naive)),
        "psnr_bbox_aligned_mean": float(np.mean(psnrs_bbox)),
        "psnr_canvas_aligned_mean": float(np.mean(psnr_canvas_all)),
        "per_view_scale_avg": {str(v): float(sr[sr[:, 0] == v, 4].mean()) if (sr[:, 0] == v).any() else None
                                for v in range(5)},
        "per_view_center_dy_mean": {str(v): float(co[co[:, 0] == v, 2].mean()) if (co[:, 0] == v).any() else None
                                     for v in range(5)},
        "per_view_center_dx_mean": {str(v): float(co[co[:, 0] == v, 3].mean()) if (co[:, 0] == v).any() else None
                                     for v in range(5)},
        "per_frame": per_frame,
    }, indent=2))

    # Build a quick GIF for view 0
    vis_paths = sorted((out_dir / "vis").glob("diag_v0_t*.png"))
    if vis_paths:
        gif_frames = [Image.open(p) for p in vis_paths]
        gif_frames[0].save(out_dir / "diag_v0.gif", save_all=True,
                           append_images=gif_frames[1:], duration=500, loop=0)
        print(f"\n[align] wrote diag_v0.gif ({len(gif_frames)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
