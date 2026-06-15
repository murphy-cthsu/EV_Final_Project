#!/usr/bin/env python3
"""Visualize 3D Gaussian Splatting point clouds as interactive 3D scatter plots."""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def load_ply_points(path: Path, max_points: int | None = None):
    ply = PlyData.read(str(path))
    v = ply.elements[0]

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    sh_dc = np.stack(
        [np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1
    )
    rgb = np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)

    opacity = None
    if "opacity" in v.data.dtype.names:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))

    if opacity is not None:
        keep = opacity > 0.05
        xyz, rgb = xyz[keep], rgb[keep]
        opacity = opacity[keep]

    n = xyz.shape[0]
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        if opacity is not None:
            opacity = opacity[idx]

    rgb_u8 = (rgb * 255).astype(np.uint8)
    return xyz, rgb_u8, opacity


def make_plotly_html(clouds: list[dict], out_path: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=len(clouds),
        specs=[[{"type": "scatter3d"}] * len(clouds)],
        subplot_titles=[c["title"] for c in clouds],
        horizontal_spacing=0.02,
    )

    for i, cloud in enumerate(clouds, start=1):
        xyz, rgb = cloud["xyz"], cloud["rgb"]
        color_hex = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color_hex, opacity=0.85),
                name=cloud["name"],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
        scene_name = "scene" if i == 1 else f"scene{i}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                    bgcolor="white",
                    xaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                    yaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                    zaxis=dict(backgroundcolor="white", gridcolor="rgb(200,200,200)"),
                )
            }
        )

    fig.update_layout(
        title_text="3D Gaussian Point Clouds — Lego vs Jumping Jack",
        title_x=0.5,
        paper_bgcolor="white",
        font=dict(color="black"),
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote interactive HTML: {out_path}")


def make_matplotlib_png(clouds: list[dict], out_path: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6), facecolor="white")
    for i, cloud in enumerate(clouds, start=1):
        ax = fig.add_subplot(1, len(clouds), i, projection="3d", facecolor="white")
        xyz, rgb = cloud["xyz"], cloud["rgb"]
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rgb / 255.0,
            s=0.3,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(cloud["title"], color="black", fontsize=12, pad=8)
        ax.set_xlabel("X", color="black")
        ax.set_ylabel("Y", color="black")
        ax.set_zlabel("Z", color="black")
        ax.tick_params(colors="black", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, color="#cccccc", alpha=0.6)

    fig.suptitle(
        "3D Gaussian Point Clouds — Lego vs Jumping Jack",
        color="black",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote static PNG: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lego", type=Path, required=True)
    parser.add_argument("--jumpingjack", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=None, help="Downsample cap (default: no limit)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lego_xyz, lego_rgb, _ = load_ply_points(args.lego, args.max_points)
    jj_xyz, jj_rgb, _ = load_ply_points(args.jumpingjack, args.max_points)

    clouds = [
        {
            "name": "lego",
            "title": f"Lego ({lego_xyz.shape[0]:,} pts)",
            "xyz": lego_xyz,
            "rgb": lego_rgb,
        },
        {
            "name": "jumpingjack",
            "title": f"Jumping Jack ({jj_xyz.shape[0]:,} pts)",
            "xyz": jj_xyz,
            "rgb": jj_rgb,
        },
    ]

    make_matplotlib_png(clouds, args.out_dir / "point_clouds_3d.png")
    try:
        make_plotly_html(clouds, args.out_dir / "point_clouds_3d.html")
    except ImportError:
        print("plotly not installed; skipping interactive HTML output")


if __name__ == "__main__":
    main()
