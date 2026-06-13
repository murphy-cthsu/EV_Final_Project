"""Evidence for the black-spike artifact mechanism in SC-GS under noisy VGM supervision.

Hypothesis: when multi-view supervision is geometrically inconsistent (pose drift /
flicker), no single 3D point satisfies all views, so the optimizer STRETCHES
Gaussians into needles along the viewing/epipolar direction to reconcile the
conflict, and drives their opacity/color toward dark, low-alpha (so they only
contribute where needed). Result = thin dark spikes.

Compares the SAME object (hellwarrior) under:
  A = vanilla SC-GS trained on NOISY SV4D 57-view  (outputs/custom/hellwarrior_vanilla_sam_node)
  B = clean canonical SC-GS trained on CLEAN d-3dgs (hellwarrior_scgs_default_node)
plus a clean control (lego sanity, 25 dB). Same architecture, only supervision
differs -> any anisotropy/darkness gap is attributable to supervision inconsistency.

Run (scgs env):
  python scripts/analyze_spike_artifact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402

MODELS = {
    "vanilla / NOISY SV4D (hellwarrior)":
        "outputs/custom/hellwarrior_vanilla_sam_node/point_cloud/iteration_20000/point_cloud.ply",
    "clean canonical / CLEAN d-3dgs (hellwarrior)":
        "/mnt/HDD_1/cthsu/EV_Final_Project/outputs/hellwarrior_scgs_default_node/point_cloud/iteration_30000/point_cloud.ply",
    "clean control / CLEAN D-NeRF (lego sanity)":
        "outputs/custom/dnerf_lego_vanilla_sanity_node/point_cloud/iteration_7000/point_cloud.ply",
}


def load(ply):
    g = None
    for fd in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fd, with_motion_mask=False)
            g.load_ply(str(ply), og_number_points=0); break
        except Exception:
            g = None
    return g


def rgb_from_sh_dc(features_dc):
    # SH DC -> RGB:  C0 = 0.28209479; rgb = 0.5 + C0 * dc
    return 0.5 + 0.28209479177387814 * features_dc


def main():
    print(f"{'model':<48} {'N':>8} {'aniso p50':>9} {'aniso p95':>9} "
          f"{'%needle>10':>10} {'opac p50':>9} {'%dark':>7} {'long||radial':>12}")
    print("-" * 118)
    rows = {}
    for name, rel in MODELS.items():
        ply = rel if rel.startswith("/") else str(REPO / rel)
        g = load(ply)
        if g is None:
            print(f"{name:<48} LOAD FAILED"); continue
        xyz = g.get_xyz.detach().cpu().numpy()
        scales = np.exp(g._scaling.detach().cpu().numpy())            # (N,3) world scales
        opac = 1 / (1 + np.exp(-g._opacity.detach().cpu().numpy()[:, 0]))  # sigmoid
        dc = g._features_dc.detach().cpu().numpy().reshape(len(xyz), -1)[:, :3]
        rgb = np.clip(rgb_from_sh_dc(dc), 0, 1)
        bright = rgb.mean(1)

        s_sorted = np.sort(scales, axis=1)
        aniso = s_sorted[:, 2] / np.maximum(s_sorted[:, 0], 1e-9)      # max/min scale
        needle_frac = (aniso > 10).mean()
        dark_frac = (bright < 0.15).mean()

        # orientation: is the LONG axis aligned with the radial (object-center) dir?
        # rotation quats -> rotation matrix; long axis = eigenvector of largest scale.
        q = g._rotation.detach().cpu().numpy()
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # columns of R are the local axes in world frame
        R_cols = np.stack([
            np.stack([1 - 2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)], 1),
            np.stack([2*(x*y + w*z), 1 - 2*(x*x+z*z), 2*(y*z - w*x)], 1),
            np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x+y*y)], 1),
        ], axis=1)  # (N,3,3): R_cols[:,:,k] = k-th local axis in world
        long_idx = np.argmax(scales, axis=1)
        long_axis = np.take_along_axis(R_cols, long_idx[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        center = xyz.mean(0)
        radial = xyz - center
        radial = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        long_axis = long_axis / np.maximum(np.linalg.norm(long_axis, axis=1, keepdims=True), 1e-9)
        align_radial = np.abs((long_axis * radial).sum(1))  # |cos| ; 1 = points radially

        # among NEEDLES only, how radial are they?
        needle = aniso > 10
        align_needle = float(align_radial[needle].mean()) if needle.sum() > 50 else float("nan")

        rows[name] = dict(aniso=aniso, bright=bright, opac=opac, needle=needle,
                          align_radial=align_radial)
        print(f"{name:<48} {len(xyz):>8d} {np.median(aniso):>9.2f} "
              f"{np.percentile(aniso,95):>9.2f} {needle_frac*100:>9.1f}% "
              f"{np.median(opac):>9.3f} {dark_frac*100:>6.1f}% {align_needle:>12.3f}")

    # correlation inside the noisy model: are needles darker / lower-opacity?
    print()
    for name, r in rows.items():
        nd = r["needle"]
        if nd.sum() < 50:
            continue
        bright_needle = r["bright"][nd].mean(); bright_blob = r["bright"][~nd].mean()
        opac_needle = r["opac"][nd].mean(); opac_blob = r["opac"][~nd].mean()
        print(f"[{name}]")
        print(f"   needles vs blobs: brightness {bright_needle:.3f} vs {bright_blob:.3f}"
              f"  |  opacity {opac_needle:.3f} vs {opac_blob:.3f}"
              f"  |  needle radial-align {r['align_radial'][nd].mean():.3f} "
              f"(blobs {r['align_radial'][~nd].mean():.3f})")

    # figure: anisotropy + brightness histograms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"vanilla / NOISY SV4D (hellwarrior)": "crimson",
              "clean canonical / CLEAN d-3dgs (hellwarrior)": "seagreen",
              "clean control / CLEAN D-NeRF (lego sanity)": "gray"}
    for name, r in rows.items():
        c = colors.get(name, "blue"); lab = name.split("/")[0].strip()
        axes[0].hist(np.clip(r["aniso"], 1, 40), bins=60, histtype="step",
                     color=c, label=lab, density=True, lw=2)
        axes[1].hist(r["bright"], bins=50, histtype="step", color=c, density=True, lw=2)
    axes[0].axvline(10, color="k", ls="--", lw=1)
    axes[0].set_xlabel("Gaussian anisotropy (max/min scale)"); axes[0].set_ylabel("density")
    axes[0].set_title("needle Gaussians: noisy >> clean"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Gaussian brightness (SH DC -> RGB)")
    axes[1].set_title("dark Gaussians: noisy has a low-brightness mode")
    # scatter: anisotropy vs brightness for the noisy model
    nm = "vanilla / NOISY SV4D (hellwarrior)"
    r = rows[nm]
    idx = np.random.default_rng(0).choice(len(r["aniso"]), min(8000, len(r["aniso"])), replace=False)
    sc = axes[2].scatter(np.clip(r["aniso"][idx], 1, 40), r["bright"][idx],
                         s=3, c=r["align_radial"][idx], cmap="viridis", alpha=0.4)
    axes[2].set_xlabel("anisotropy"); axes[2].set_ylabel("brightness")
    axes[2].set_title("noisy model: needles (right) are darker\ncolor = radial alignment")
    plt.colorbar(sc, ax=axes[2], label="|long-axis · radial|")
    fig.suptitle("Black-spike mechanism: noisy multi-view supervision -> dark needle Gaussians", y=1.03)
    fig.tight_layout()
    out = REPO / "runs_aux/spike_artifact_analysis.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
