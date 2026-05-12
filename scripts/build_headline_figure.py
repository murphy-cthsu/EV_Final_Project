"""Assemble the headline composite figure for the W1 group meeting.

Reads the per-scene artifacts already on disk:
  - test/ours_<iter>/{gt,renders}/<worst_frame>.png   (from SC-GS render)
  - test/interpolate_<iter>/renders/*.png             (from SC-GS render --mode time)
  - inspection/per_frame_psnr.json                    (from inspect_scgs_failure.py)
  - cross-scene spatial_error_summary.json            (from spatial_error_analysis.py)

Produces a single PNG with:
  - row per scene: [GT worst | pred | per-pixel error heatmap | temporal-std heatmap]
  - bottom panel: cross-scene radial-error profile (the discriminating chart)

Output: outputs/_cross_scene_failure/HEADLINE.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES = ["jumpingjacks", "hellwarrior", "bouncingballs", "standup"]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def temporal_std(interp_dir: Path, step: int = 10) -> np.ndarray:
    frames = sorted(interp_dir.glob("*.png"))
    sampled = frames[::step]
    stack = np.stack([load_rgb(p) for p in sampled], axis=0)
    return stack.std(axis=0).sum(axis=-1)


def build(iteration: int, out_path: Path, strobe_step: int) -> None:
    summary_path = REPO_ROOT / "outputs" / "_cross_scene_failure" / "spatial_error_summary.json"
    summary = json.loads(summary_path.read_text())
    by_scene = {s["scene"]: s for s in summary["scenes"]}

    fig = plt.figure(figsize=(15, 17))
    gs = gridspec.GridSpec(
        nrows=len(SCENES) + 1,
        ncols=4,
        height_ratios=[1] * len(SCENES) + [0.85],
        hspace=0.18,
        wspace=0.05,
        figure=fig,
    )

    fig.suptitle(
        "SC-GS-default on D-NeRF: where reconstruction fails, and how scene complexity scales it",
        fontsize=14, y=0.995, fontweight="bold",
    )

    # find global error scale across scenes for honest comparison
    err_scenes = []
    tstd_scenes = []
    for s in SCENES:
        run_dir = REPO_ROOT / "outputs" / f"{s}_scgs_default_node"
        pf = json.loads((run_dir / "inspection" / "per_frame_psnr.json").read_text())
        worst = pf[0]
        gt = load_rgb(run_dir / "test" / f"ours_{iteration}" / "gt" / f"{worst['frame']}.png")
        pred = load_rgb(run_dir / "test" / f"ours_{iteration}" / "renders" / f"{worst['frame']}.png")
        err = np.sqrt(((gt - pred) ** 2).sum(axis=-1))
        tstd = temporal_std(run_dir / "test" / f"interpolate_{iteration}" / "renders",
                            step=strobe_step)
        err_scenes.append((s, worst, gt, pred, err, tstd))
        tstd_scenes.append(tstd)

    err_vmax = float(np.percentile(np.concatenate([e[4].ravel() for e in err_scenes]), 99))
    tstd_vmax = float(np.percentile(np.concatenate([t.ravel() for t in tstd_scenes]), 99))

    for i, (s, worst, gt, pred, err, tstd) in enumerate(err_scenes):
        scene_stats = by_scene[f"{s}_scgs_default_node"]
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_pr = fig.add_subplot(gs[i, 1])
        ax_er = fig.add_subplot(gs[i, 2])
        ax_ts = fig.add_subplot(gs[i, 3])

        ax_gt.imshow(gt)
        ax_pr.imshow(pred)
        ax_er.imshow(np.clip(err / err_vmax, 0, 1), cmap="inferno", vmin=0, vmax=1)
        ax_ts.imshow(np.clip(tstd / tstd_vmax, 0, 1), cmap="inferno", vmin=0, vmax=1)

        for ax in (ax_gt, ax_pr, ax_er, ax_ts):
            ax.set_xticks([])
            ax.set_yticks([])

        if i == 0:
            ax_gt.set_title("GT (worst test frame)", fontsize=10)
            ax_pr.set_title("SC-GS-default pred", fontsize=10)
            ax_er.set_title(f"|GT − pred| (vmax={err_vmax:.3f})", fontsize=10)
            ax_ts.set_title(f"temporal-std on time-interp (vmax={tstd_vmax:.3f})", fontsize=10)

        ratio = scene_stats["periphery_over_core"]
        ax_gt.set_ylabel(
            f"{s}\nworst frame {worst['frame']} | PSNR {worst['psnr']:.2f}\n"
            f"periphery / core = ×{ratio:.2f}",
            fontsize=10, rotation=0, ha="right", va="center", labelpad=80,
        )

    # bottom row: radial profile, full width
    ax_rad = fig.add_subplot(gs[len(SCENES), 0:3])
    colors = {"jumpingjacks": "#4477AA", "hellwarrior": "#228833",
              "bouncingballs": "#EE6677", "standup": "#CCBB44"}
    for scene_block in summary["scenes"]:
        edges = np.asarray(scene_block["radial_edges"])
        mid = 0.5 * (edges[:-1] + edges[1:])
        mean = np.asarray(scene_block["radial_mean"])
        std = np.asarray(scene_block["radial_std"])
        scene_short = scene_block["scene"].replace("_scgs_default_node", "")
        color = colors.get(scene_short, "#888888")
        ax_rad.plot(mid, mean, color=color, lw=2.2, marker="o", ms=5, label=scene_short)
        ax_rad.fill_between(mid, np.clip(mean - std, 0, None), mean + std,
                             color=color, alpha=0.15)
    ax_rad.set_xlabel("normalized distance from object centroid  (0 = center, 1 ≈ bbox edge)",
                      fontsize=10)
    ax_rad.set_ylabel("mean per-pixel error |GT − pred|", fontsize=10)
    ax_rad.set_title(
        "Radial error profile across scenes — error growth with distance scales with articulation complexity",
        fontsize=11)
    ax_rad.grid(alpha=0.3)
    ax_rad.legend(fontsize=10, loc="upper left")

    # bottom-right: takeaway text box
    ax_txt = fig.add_subplot(gs[len(SCENES), 3])
    ax_txt.axis("off")
    txt = (
        "Headline read\n"
        "──────────────\n"
        "• Same SC-GS config, four D-NeRF\n"
        "  scenes. Periphery / core error ratio\n"
        "  tracks articulation complexity:\n"
        "  2.28 (jumpingjacks) →\n"
        "  1.35 (hellwarrior)  →\n"
        "  1.14 (standup)      →\n"
        "  1.03 (bouncingballs).\n"
        "\n"
        "• Non-articulated control sits at 1.03\n"
        "  — periphery and core indistinguishable.\n"
        "\n"
        "• The radial-error curve has a 5-6× ramp\n"
        "  on jumpingjacks and is flat on\n"
        "  bouncingballs. This is the W1 evidence\n"
        "  that uniform-rigidity ARAP under-fits\n"
        "  articulated kinematic chains."
    )
    ax_txt.text(0.0, 1.0, txt, fontsize=9.5, va="top", ha="left", family="monospace")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[headline] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--strobe_step", type=int, default=10)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "_cross_scene_failure" / "HEADLINE.png")
    args = p.parse_args()
    build(args.iteration, args.out, args.strobe_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
