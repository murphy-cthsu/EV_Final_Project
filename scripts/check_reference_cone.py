"""GAP-2 check: does the CLEAN REFERENCE itself have a view-dependent cone?

Compares two INDEPENDENT clean models at t=0 (same pose by construction —
both derive from the clean D-NeRF sequence):
  A = frozen-canonical static render  (SC-GS static / d-3dgs canonical)
  B = d-3dgs reference render         (the GT used by every D-series diagnostic)

If PSNR(A, B) is flat across azimuth/elevation, neither clean model carries a
view-dependent artifact — so the 18 dB reliability cone measured against B
belongs to SV4D, not to the reference. Comparison is restricted to the
canonical's own foreground (excludes the baseplate confound on lego).

Run: python scripts/check_reference_cone.py --scene lego_v3
     python scripts/check_reference_cone.py --scene hellwarrior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
T_FULL = 21

CANON_RENDER = {
    "lego_v3": "lego_v3_canon_static_render",
    "hellwarrior": "hellwarrior_cleancanon_static_render",
    "lego_v2": "lego_v2_canonical_static_render",
}


def load(path):
    arr = np.asarray(iio.imread(path), np.float32) / 255.0
    if arr.shape[-1] == 4:
        a = arr[..., 3:4]
        return arr[..., :3] * a + (1 - a)
    return arr[..., :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    args = p.parse_args()

    scene_dir = REPO / "data/custom" / args.scene
    canon_dir = REPO / "outputs/custom" / CANON_RENDER[args.scene]
    d3_dir = REPO / "outputs/custom" / f"{args.scene}_d3dgs_ref" / "renders"

    meta = json.loads((scene_dir / "transforms_train.json").read_text())
    meta_te = json.loads((scene_dir / "transforms_test.json").read_text())
    view_info = {}
    for f in meta["frames"] + meta_te["frames"]:
        v = int(f["view_idx"])
        if v not in view_info:
            view_info[v] = (float(f.get("azimuth_deg", v)), float(f.get("elevation_deg", 0)))

    rows = []
    for v in sorted(view_info):
        flat = v * T_FULL + 0  # t=0: canonical pose == d-3dgs pose
        a = load(canon_dir / f"{flat:05d}.png")
        b = load(d3_dir / f"{flat:05d}.png")
        fg = (np.abs(a - 1).sum(-1) > 0.05)  # canonical's own foreground
        if fg.sum() < 500:
            continue
        mse = float(((a - b) ** 2)[fg].mean())
        psnr = -10 * np.log10(max(mse, 1e-12))
        az, el = view_info[v]
        rows.append((v, az, el, psnr))

    azs = np.array([r[1] for r in rows]); els = np.array([r[2] for r in rows])
    ps = np.array([r[3] for r in rows])

    # azimuthal distance from the input view (az=0), same x-axis as the cone
    az_dist = np.minimum(azs % 360, 360 - azs % 360)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(az_dist, ps)
    e0 = els == 0
    print(f"[ref-cone {args.scene}] views={len(rows)}  PSNR(A,B) mean={ps.mean():.2f} "
          f"std={ps.std():.2f}  range=[{ps.min():.2f}, {ps.max():.2f}]")
    print(f"  spearman(az-dist, psnr) = {rho:.3f} (p={pval:.3f})  "
          f"elev-slope = {np.polyfit(els, ps, 1)[0]:.3f} dB/deg")
    print(f"  SV4D cone for comparison: 37.5 -> 19.4 dB (18 dB drop), rho strongly negative")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(az_dist[~e0], ps[~e0], s=18, c="silver", label="other elevations")
    if e0.any():
        o = np.argsort(az_dist[e0])
        ax.plot(az_dist[e0][o], ps[e0][o], "o-", color="seagreen", label="elev = 0 ring")
    ax.set_xlabel("azimuth distance from input view (deg)")
    ax.set_ylabel("PSNR(canonical render, d-3dgs) @ t=0, canonical FG")
    ax.set_title(f"{args.scene} — reference self-consistency vs view "
                 f"(rho={rho:.2f}; SV4D cone rho<<0, 18 dB drop)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPO / "runs_aux" / f"reference_cone_check_{args.scene}.png"
    fig.savefig(out, dpi=130)
    np.savez(REPO / "runs_aux" / f"reference_cone_check_{args.scene}.npz",
             az=azs, el=els, az_dist=az_dist, psnr=ps, rho=rho, p=pval)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
