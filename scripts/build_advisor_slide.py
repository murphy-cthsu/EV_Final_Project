"""Single-slide advisor progress update (16:9, ~1920x1080).

Layout:
  +-----------------------------------------------+
  | title strip                                   |
  +---------------------------------+-------------+
  | 4 error heatmaps (W1 evidence)  | pipeline    |
  +---------------------------------+ status      |
  | radial profile                  |             |
  +---------------------------------+-------------+
  | decisions needed (full width)                 |
  +-----------------------------------------------+
  | risk strip (full width)                       |
  +-----------------------------------------------+

Reads existing artifacts on disk; no training required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES = ["jumpingjacks", "hellwarrior", "bouncingballs", "standup"]

SCENE_COLORS = {
    "jumpingjacks":  "#4477AA",
    "hellwarrior":   "#228833",
    "bouncingballs": "#EE6677",
    "standup":       "#CCBB44",
}

STATUS_COLOR = {
    "done": "#2ca02c",  # green
    "plan": "#7f7f7f",  # gray
    "risk": "#ff7f0e",  # orange
    "todo": "#d62728",  # red
}

PIPELINE_STATUS = [
    ("section", "Tier 1 — multi-view D-NeRF (W1)"),
    ("done",    "4 scenes × 30K iters trained"),
    ("done",    "Cross-scene failure characterized"),
    ("done",    "Headline figure built"),
    ("plan",    "W2 ablation matrix (5 days)"),
    ("blank",   ""),
    ("section", "Tier 2 — SV4D-supervised (W3)"),
    ("done",    "SV4D2 adapter (subprocess)"),
    ("done",    "SV4D → D-NeRF converter"),
    ("done",    "End-to-end pipeline driver"),
    ("done",    "CPU smoke 117/117 pass"),
    ("risk",    "1st GPU run untested"),
    ("blank",   ""),
    ("section", "Tier 3 — DyCheck real (W3)"),
    ("todo",    "Data prep not started"),
]

DECISIONS = [
    ("1", "Framing", "3-component story, or fold C3 into a unified \"drift-protection\" module?"),
    ("2", "W2 scope", "Add RigGS / SK-GS discovery-stage-on-SV4D mini-experiment (2 days)?"),
    ("3", "SV4D variant", "4-view (120° gap) vs 8-view (30° gap) for Tier 2 supervision?"),
]

RISK_LINE = (
    "CRITICAL PATH:  SV4D 1st GPU run (~$5 RunPod) is unverified → gates all of Tier 2.   "
    "Plan: derisk this weekend, 1 scene end-to-end."
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def draw_title(ax) -> None:
    ax.axis("off")
    ax.text(
        0.0, 0.78,
        "MotionPrior-4DGS · advisor update · 2026-05-13",
        fontsize=18, fontweight="bold", va="center", ha="left", color="#111111",
    )
    ax.text(
        0.0, 0.25,
        "W1 closed · Tier 2 pipeline ready · awaiting 1st GPU run",
        fontsize=12, va="center", ha="left", color="#555555", style="italic",
    )


def draw_status_panel(ax, items) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(items))
    ax.invert_yaxis()
    ax.axis("off")

    ax.add_patch(mpatches.Rectangle(
        (0.0, 0.0), 1.0, len(items),
        facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.8, zorder=0,
    ))

    ax.text(0.5, -0.6, "PIPELINE STATUS",
            fontsize=11, fontweight="bold", ha="center", va="bottom",
            color="#333333")

    for i, (kind, text) in enumerate(items):
        y = i + 0.5
        if kind == "section":
            ax.text(0.04, y, text, fontsize=10, fontweight="bold",
                    va="center", ha="left", color="#222222")
        elif kind == "blank":
            continue
        else:
            ax.scatter([0.08], [y], s=80, c=STATUS_COLOR[kind],
                       marker="s", zorder=2, edgecolors="white", linewidths=0.5)
            ax.text(0.16, y, text, fontsize=9.5, va="center", ha="left",
                    color="#222222")


def draw_decisions_panel(ax, items) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(mpatches.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        facecolor="#fff8e7", edgecolor="#e0c060", lw=1.2, zorder=0,
    ))

    ax.text(0.01, 0.86, "DECISIONS I NEED FROM YOU (this week)",
            fontsize=12, fontweight="bold", va="center", ha="left",
            color="#7a5a00")

    n = len(items)
    for i, (num, header, body) in enumerate(items):
        y = 0.62 - i * 0.22
        ax.text(0.015, y, f"{num}", fontsize=14, fontweight="bold",
                va="center", ha="left", color="#7a5a00")
        ax.text(0.045, y, header, fontsize=11, fontweight="bold",
                va="center", ha="left", color="#222222")
        ax.text(0.165, y, body, fontsize=10.5,
                va="center", ha="left", color="#333333")


def draw_risk_strip(ax, text: str) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        facecolor="#fce8e8", edgecolor="#cc5555", lw=1.2, zorder=0,
    ))
    ax.text(0.015, 0.5, text, fontsize=11, va="center", ha="left",
            color="#7a1c1c", fontweight="bold")


def draw_heatmap_row(fig, gs_heat, err_scenes, err_vmax, by_scene):
    for i, (scene, worst, gt, pred, err) in enumerate(err_scenes):
        ax = fig.add_subplot(gs_heat[0, i])
        ax.imshow(np.clip(err / err_vmax, 0, 1), cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

        ratio = by_scene[f"{scene}_scgs_default_node"]["periphery_over_core"]
        ctrl_tag = "  (control)" if scene == "bouncingballs" else ""

        # Scene name only on top — short and bold
        ax.set_title(f"{scene}{ctrl_tag}",
                     fontsize=11.5, color=SCENE_COLORS[scene],
                     fontweight="bold", pad=4)

        # Metadata as bottom xlabel — compact, 2 numbers only
        ax.set_xlabel(
            f"PSNR {worst['psnr']:.2f}   ·   ×{ratio:.2f}",
            fontsize=10.5, color="#222222", labelpad=4, fontweight="bold",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(SCENE_COLORS[scene])
            spine.set_linewidth(1.8)


def draw_radial(ax, summary) -> None:
    for block in summary["scenes"]:
        edges = np.asarray(block["radial_edges"])
        mid = 0.5 * (edges[:-1] + edges[1:])
        mean = np.asarray(block["radial_mean"])
        std = np.asarray(block["radial_std"])
        scene = block["scene"].replace("_scgs_default_node", "")
        c = SCENE_COLORS[scene]
        ax.plot(mid, mean, color=c, lw=2.2, marker="o", ms=5, label=scene)
        ax.fill_between(mid, np.clip(mean - std, 0, None), mean + std,
                        color=c, alpha=0.13)
    ax.set_xlabel("normalized distance from object centroid  (0 = center, 1 ≈ bbox edge)",
                  fontsize=10)
    ax.set_ylabel("mean per-pixel error |GT − pred|", fontsize=10)
    ax.set_title("Radial error profile — jumpingjacks ramps 6× from core to edge; "
                 "bouncingballs control stays flat",
                 fontsize=10.5, color="#222222")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=4, frameon=True)


def build(iteration: int, out_path: Path) -> None:
    summary = json.loads(
        (REPO_ROOT / "outputs" / "_cross_scene_failure" / "spatial_error_summary.json").read_text()
    )
    by_scene = {s["scene"]: s for s in summary["scenes"]}

    err_scenes = []
    for s in SCENES:
        run = REPO_ROOT / "outputs" / f"{s}_scgs_default_node"
        pf = json.loads((run / "inspection" / "per_frame_psnr.json").read_text())
        worst = pf[0]
        gt = load_rgb(run / "test" / f"ours_{iteration}" / "gt" / f"{worst['frame']}.png")
        pred = load_rgb(run / "test" / f"ours_{iteration}" / "renders" / f"{worst['frame']}.png")
        err = np.sqrt(((gt - pred) ** 2).sum(axis=-1))
        err_scenes.append((s, worst, gt, pred, err))

    err_vmax = float(np.percentile(np.concatenate([e[4].ravel() for e in err_scenes]), 99))

    fig = plt.figure(figsize=(17, 10), dpi=130)
    outer = gridspec.GridSpec(
        nrows=5, ncols=2,
        height_ratios=[0.55, 2.7, 2.5, 1.5, 0.45],
        width_ratios=[3.3, 1.25],
        hspace=0.70, wspace=0.10,
        left=0.025, right=0.985, top=0.965, bottom=0.030,
        figure=fig,
    )

    # Row 0: title (spans both cols)
    ax_title = fig.add_subplot(outer[0, :])
    draw_title(ax_title)

    # Row 1: 4 heatmaps on left (nested gridspec), status panel on right (rows 1-2)
    gs_heat = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1, 0], wspace=0.08,
    )
    draw_heatmap_row(fig, gs_heat, err_scenes, err_vmax, by_scene)

    ax_status = fig.add_subplot(outer[1:3, 1])
    draw_status_panel(ax_status, PIPELINE_STATUS)

    # Row 2: radial profile (left)
    ax_radial = fig.add_subplot(outer[2, 0])
    draw_radial(ax_radial, summary)

    # Row 3: decisions panel (full width)
    ax_dec = fig.add_subplot(outer[3, :])
    draw_decisions_panel(ax_dec, DECISIONS)

    # Row 4: risk strip (full width)
    ax_risk = fig.add_subplot(outer[4, :])
    draw_risk_strip(ax_risk, RISK_LINE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"[advisor-slide] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, default=30000)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "outputs" / "advisor_slides" / "advisor_slide_2026-05-13.png")
    args = p.parse_args()
    build(args.iteration, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
