"""Q2 measurement: is VGM (SV4D) noise appearance or geometry?

Compares SV4D supervision frames vs d-3dgs clean GT at the SAME (cam, t), with
NO training and NO rendering — just existing PNG pairs. Separates two axes:

  - silhouette IoU (alpha masks)  -> GEOMETRY/pose fidelity, texture-independent
  - in-mask PSNR (RGB inside the agreed silhouette) -> APPEARANCE fidelity
  - full-frame PSNR -> the conflated number the benchmark currently reports

Broken down by view: v0 = SV4D input pose (clean), v1-4 = SV4D-generated novel
views. If v1-4 have HIGH IoU but LOW in-mask PSNR -> noise is appearance.
If v1-4 have LOW IoU -> geometry is genuinely polluted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCENE = REPO / "data/custom/lego_v2"
D3DGS = REPO / "outputs/custom/lego_v2_d3dgs_ref/renders"


def over_white(rgba):
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a), a[..., 0]


def psnr(a, b, mask=None):
    if mask is not None:
        m = mask > 0.5
        if m.sum() == 0:
            return float("nan")
        d = ((a[m] - b[m]) ** 2).mean()
    else:
        d = ((a - b) ** 2).mean()
    return -10 * math.log10(max(d, 1e-12))


def load_d3dgs(flat):
    """d-3dgs renders are RGB on white (no alpha). Derive silhouette from
    non-white pixels."""
    p = D3DGS / f"{flat:05d}.png"
    rgba = np.asarray(iio.imread(p), dtype=np.float32) / 255.0
    if rgba.shape[-1] == 4:
        rgb, a = over_white(rgba)
        return rgb, a
    rgb = rgba[..., :3]
    a = (rgb.mean(-1) < 0.98).astype(np.float32)  # non-white = object
    return rgb, a


def main():
    meta_tr = json.loads((SCENE / "transforms_train.json").read_text())
    meta_te = json.loads((SCENE / "transforms_test.json").read_text())
    frames = meta_tr["frames"] + meta_te["frames"]

    per_view = {}  # v -> list of (iou, inmask_psnr, full_psnr)
    for f in frames:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        name = f"{Path(f['file_path']).name}.png"
        sv4d_path = None
        for split in ("train", "test"):
            c = SCENE / split / name
            if c.exists():
                sv4d_path = c; break
        if sv4d_path is None:
            continue
        sv4d_rgba = np.asarray(iio.imread(sv4d_path), dtype=np.float32) / 255.0
        sv4d_rgb, sv4d_a = over_white(sv4d_rgba)
        d3_rgb, d3_a = load_d3dgs(v * 21 + t)

        m_sv = sv4d_a > 0.5
        m_d3 = d3_a > 0.5
        inter = (m_sv & m_d3).sum()
        union = (m_sv | m_d3).sum()
        iou = inter / max(union, 1)
        inter_mask = (m_sv & m_d3).astype(np.float32)
        per_view.setdefault(v, []).append(
            (iou, psnr(sv4d_rgb, d3_rgb, inter_mask), psnr(sv4d_rgb, d3_rgb))
        )

    print(f"{'view':<6}{'role':<14}{'silh IoU':>10}{'in-mask PSNR':>14}{'full PSNR':>12}")
    print("-" * 56)
    all_iou, all_in, all_full = [], [], []
    for v in sorted(per_view):
        arr = np.array(per_view[v])
        role = "input (clean)" if v == 0 else "generated"
        print(f"{v:<6}{role:<14}{arr[:,0].mean():>10.3f}{arr[:,1].mean():>14.2f}{arr[:,2].mean():>12.2f}")
        all_iou += list(arr[:, 0]); all_in += list(arr[:, 1]); all_full += list(arr[:, 2])
    print("-" * 56)
    gen = [v for v in per_view if v != 0]
    ga = np.array([r for v in gen for r in per_view[v]])
    print(f"{'gen':<6}{'v1-4 mean':<14}{ga[:,0].mean():>10.3f}{ga[:,1].mean():>14.2f}{ga[:,2].mean():>12.2f}")
    print()
    print("Reading guide:")
    print("  high IoU + low in-mask PSNR on generated views -> noise is APPEARANCE (texture)")
    print("  low IoU on generated views                     -> geometry/pose is POLLUTED")
    print("  gap(in-mask PSNR - full PSNR)                   -> how much silhouette mismatch costs")


if __name__ == "__main__":
    main()
