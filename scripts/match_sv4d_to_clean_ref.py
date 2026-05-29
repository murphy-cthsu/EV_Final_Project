"""A) Image-similarity temporal alignment.

For each SV4D frame (v, t), search clean_ref renders at the SAME view across
the fid grid, pick the fid with min L1 distance. Outputs:
  - matching_map.json     (v, t) -> {best_fid_idx, best_fid_val, l1, mse, psnr}
  - matched_curve.png     plot of matched_fid vs SV4D t, per view
  - alignment_strip_v{V}.gif  side-by-side SV4D | clean_ref@matched_fid

Run AFTER render_clean_ref_fine_grid.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def load_rgb(path: Path, size=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size)
    return np.asarray(im, dtype=np.float32) / 255.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sv4d_dir",  default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--sv4d_meta", default=REPO / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--clean_dir", default=REPO / "runs_aux/clean_gt_fine/renders")
    p.add_argument("--out_dir",   default=REPO / "runs_aux/alignment_A")
    p.add_argument("--n_fid", type=int, default=100)
    p.add_argument("--n_views", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(args.sv4d_meta).read_text())
    frames = meta["frames"]
    T_full = max(int(f["frame_idx"]) for f in frames) + 1

    # Pre-load clean ref bank per view: shape [N_fid, H, W, 3]
    print(f"[match] preloading clean ref bank ({args.n_views} views x {args.n_fid} fids)")
    bank = {}
    for v in range(args.n_views):
        imgs = []
        for fi in range(args.n_fid):
            p_img = Path(args.clean_dir) / f"r_v{v}_f{fi:03d}.png"
            imgs.append(load_rgb(p_img))
        bank[v] = np.stack(imgs, axis=0)
        print(f"[match] view {v}: {bank[v].shape}")

    # For each SV4D frame, match to best fid
    matching = {}
    matched_curves = {v: [] for v in range(args.n_views)}
    for f in frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        sv4d_path = Path(args.sv4d_dir) / f"{Path(f['file_path']).name}.png"
        # SV4D is RGBA; alpha-composite onto white to match clean ref bg
        rgba = np.asarray(Image.open(sv4d_path).convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba[..., 3:4]
        sv4d_rgb = rgba[..., :3] * alpha + 1.0 * (1 - alpha)
        # Resize to clean ref size if needed
        if sv4d_rgb.shape[:2] != bank[v].shape[1:3]:
            sv4d_img = Image.fromarray((sv4d_rgb * 255).astype(np.uint8)).resize(
                (bank[v].shape[2], bank[v].shape[1]))
            sv4d_rgb = np.asarray(sv4d_img, dtype=np.float32) / 255.0

        # L1 over all fids
        diffs = np.abs(bank[v] - sv4d_rgb[None]).mean(axis=(1, 2, 3))
        best_fi = int(diffs.argmin())
        best_l1 = float(diffs[best_fi])
        # PSNR at best
        mse = ((bank[v][best_fi] - sv4d_rgb) ** 2).mean()
        psnr = -10 * math.log10(max(float(mse), 1e-12))
        fid_val = best_fi / max(args.n_fid - 1, 1)
        matching[f"v{v}_t{t}"] = {
            "view": v, "t_sv4d": t,
            "best_fid_idx": best_fi, "best_fid_val": fid_val,
            "l1": best_l1, "psnr_aligned": psnr,
        }
        matched_curves[v].append((t, fid_val, psnr))

    (out_dir / "matching_map.json").write_text(json.dumps(matching, indent=2))

    # Plot matched fid curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for v in range(args.n_views):
        pts = sorted(matched_curves[v])
        ts, fids, psnrs = zip(*pts)
        axes[0].plot(ts, fids, "o-", color=colors[v % 5], label=f"view {v}")
        axes[1].plot(ts, psnrs, "o-", color=colors[v % 5], label=f"view {v}")
    axes[0].set_xlabel("SV4D frame t")
    axes[0].set_ylabel("matched clean-ref fid")
    axes[0].set_title("Temporal alignment SV4D → D-NeRF")
    axes[0].plot([0, T_full - 1], [0, 1], "k--", alpha=0.3, label="identity (no warp)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("SV4D frame t")
    axes[1].set_ylabel("PSNR vs matched clean-ref")
    axes[1].set_title("Best-match residual (full-frame)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "matched_curves.png", dpi=130)
    plt.close()

    # Build alignment strip GIF per view (sv4d | clean_ref @ matched)
    for v in range(args.n_views):
        pts = sorted([(t, matching[f"v{v}_t{t}"]) for t in range(T_full)
                      if f"v{v}_t{t}" in matching])
        gif_frames = []
        H = W = 576
        for t, info in pts:
            fi = info["best_fid_idx"]
            sv4d_p = Path(args.sv4d_dir) / f"r_{v*T_full+t:05d}.png"
            clean_p = Path(args.clean_dir) / f"r_v{v}_f{fi:03d}.png"
            rgba = np.asarray(Image.open(sv4d_p).convert("RGBA"), dtype=np.float32) / 255.0
            a = rgba[..., 3:4]
            sv4d = (rgba[..., :3] * a + 1 * (1 - a)) * 255
            sv4d = Image.fromarray(sv4d.astype(np.uint8)).resize((W, H))
            clean = Image.open(clean_p).convert("RGB").resize((W, H))
            strip = Image.new("RGB", (W * 2 + 8, H + 24), (255, 255, 255))
            strip.paste(sv4d, (0, 24))
            strip.paste(clean, (W + 8, 24))
            from PIL import ImageDraw, ImageFont
            d = ImageDraw.Draw(strip)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
            d.text((10, 4), f"SV4D v{v} t={t}", fill="black", font=font)
            d.text((W + 18, 4), f"clean ref @ fid={info['best_fid_val']:.3f}  (PSNR {info['psnr_aligned']:.2f})",
                   fill="black", font=font)
            gif_frames.append(strip)
        if gif_frames:
            gif_path = out_dir / f"alignment_v{v}.gif"
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                               duration=400, loop=0)
            print(f"[match] view {v}: {len(gif_frames)} aligned frames -> {gif_path.name}")

    # Summary
    psnrs_aligned = [v["psnr_aligned"] for v in matching.values()]
    l1s = [v["l1"] for v in matching.values()]
    print()
    print(f"[match] === Alignment summary ===")
    print(f"[match] aligned PSNR  mean={np.mean(psnrs_aligned):.3f}  median={np.median(psnrs_aligned):.3f}  std={np.std(psnrs_aligned):.3f}")
    print(f"[match] L1 residual   mean={np.mean(l1s):.4f}  median={np.median(l1s):.4f}")
    print(f"[match] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
