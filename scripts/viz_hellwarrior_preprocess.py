#!/usr/bin/env python3
"""Visualize Stages B–D preprocessing (1–3 sample views). PIL-only, no matplotlib."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def project_to_view(xyz, c2w, fov_x, H, W):
    w2c = np.linalg.inv(c2w)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c = flip @ w2c
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    valid = z > 0
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return np.stack([u, v], axis=-1), valid


def load_rgba(data_dir: Path, view_idx: int, t: int, n_frames: int) -> np.ndarray:
    flat = view_idx * n_frames + t
    for split in ("train", "test"):
        p = data_dir / split / f"r_{flat:05d}.png"
        if p.exists():
            return np.asarray(Image.open(p), dtype=np.float32) / 255.0
    raise FileNotFoundError(f"r_{flat:05d}")


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[-1] == 4:
        a = rgb[..., 3:4]
        rgb = rgb[..., :3] * a + (1 - a)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def overlay(rgb_u8: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 110):
    out = rgb_u8.astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (255 - alpha) / 255 * out[..., c][m] + alpha / 255 * color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def heatmap_std(std: np.ndarray) -> np.ndarray:
    s = std.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    r = np.clip(1.5 * s, 0, 1)
    g = np.clip(1.5 * np.abs(s - 0.5), 0, 1)
    b = np.clip(1.5 * (1 - s), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def label_panel(img: np.ndarray, text: str, bar_h: int = 28) -> np.ndarray:
    pil = Image.fromarray(img)
    out = Image.new("RGB", (pil.width, pil.height + bar_h), (255, 255, 255))
    out.paste(pil, (0, bar_h))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    d.text((6, 4), text, fill=(0, 0, 0), font=font)
    return np.asarray(out)


def hstack(panels: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    x = 0
    for p in panels:
        canvas[: p.shape[0], x : x + p.shape[1]] = p[:, :, :3] if p.ndim == 3 else p
        x += p.shape[1] + gap
    return canvas


def project_parts(rgb_u8, uv, valid, part_id, colors):
    H, W = rgb_u8.shape[:2]
    canvas = rgb_u8.copy()
    ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
    vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
    for pid, col in enumerate(colors):
        sel = valid & (part_id == pid)
        if sel.any():
            canvas[vi[sel], ui[sel]] = col
    return ((0.5 * canvas) + (0.5 * rgb_u8)).astype(np.uint8)


def draw_trail(rgb_u8, motion_mask, centroids, conf):
    pil = Image.fromarray(rgb_u8.copy())
    base = overlay(rgb_u8, motion_mask, (255, 80, 80), 80)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    pts = [(float(centroids[t, 0]), float(centroids[t, 1])) for t in range(len(centroids)) if conf[t] > 0.05]
    if len(pts) > 1:
        draw.line(pts, fill=(255, 220, 0), width=2)
    for t, (x, y) in enumerate(centroids):
        if conf[t] > 0.05:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 200, 0), outline=(0, 0, 0))
    return np.asarray(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hellwarrior")
    ap.add_argument("--canon_ply", default="/root/hellwarrior/iteration_40000/point_cloud.ply")
    ap.add_argument("--src_video_root", default="/root/hellwarrior/hellwarrior_r32_train_iter40000")
    ap.add_argument("--views", default="0,28,42")
    ap.add_argument("--n_frames", type=int, default=21)
    args = ap.parse_args()

    from plyfile import PlyData

    def load_xyz_from_ply(ply_path):
        v = PlyData.read(str(ply_path)).elements[0]
        return np.stack((np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])), axis=1)

    data_dir = REPO / "data/custom" / args.dataset
    motion_dir = REPO / "runs_aux" / f"parts_motion_{args.dataset}"
    part_dir = REPO / "runs_aux" / f"part_assignment_{args.dataset}"
    out_dir = REPO / "runs_aux" / f"{args.dataset}_preprocess_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_tr = json.loads((data_dir / "transforms_train.json").read_text())
    meta_te = json.loads((data_dir / "transforms_test.json").read_text())
    all_f = meta_tr["frames"] + meta_te["frames"]
    fov_x = meta_tr["camera_angle_x"]
    H = W = 576
    view_tags, cams = {}, {}
    for f in all_f:
        v = int(f["view_idx"])
        view_tags[v] = f.get("view_tag", f"view{v}")
        cams[v] = np.asarray(f["transform_matrix"], dtype=np.float64)

    xyz = load_xyz_from_ply(args.canon_ply)
    part_id = np.load(part_dir / "part_id.npy")
    arm_w = np.load(part_dir / "gaussian_arm_weights.npy")
    centroid_3d = np.load(part_dir / "part_centroid_3d.npy")
    conf_3d = np.load(part_dir / "part_centroid_confidence.npy")

    colors = [(242, 50, 50), (50, 190, 90), (140, 140, 140)]
    times = [0, 10, 20]

    # --- Stage C on anchor view 0 ---
    rgb0 = to_uint8(load_rgba(data_dir, 0, 0, args.n_frames))
    uv0, valid0 = project_to_view(xyz, cams[0], fov_x, H, W)
    voted = project_parts(rgb0, uv0, valid0, part_id, colors)
    heat = np.zeros((H, W), np.float32)
    ui = np.clip(uv0[:, 0].astype(int), 0, W - 1)
    vi = np.clip(uv0[:, 1].astype(int), 0, H - 1)
    heat[vi[valid0], ui[valid0]] = np.maximum(heat[vi[valid0], ui[valid0]], arm_w[valid0])
    heat_u8 = (np.stack([heat * 255, heat * 80, heat * 80], axis=-1)).astype(np.uint8)
    heat_blend = ((0.45 * rgb0) + (0.55 * heat_u8)).astype(np.uint8)
    row_c = hstack([
        label_panel(voted, "C: multi-view vote (red=arm, green=body, gray=unassigned)"),
        label_panel(heat_blend, "C: arm vote strength on canonical projection"),
    ])
    iio.imwrite(out_dir / "stage_C_part_voting.png", row_c)
    print("saved stage_C_part_voting.png")

    # --- Stage D summary strip ---
    arm = centroid_3d[:, 0, :]
    ok = conf_3d[:, 0] > 0.01
    plot_h, plot_w = 320, 480
    traj_img = Image.new("RGB", (plot_w, plot_h), (250, 250, 250))
    draw = ImageDraw.Draw(traj_img)
    if ok.sum() > 1:
        pts = arm[ok]
        xs = pts[:, 0]; ys = pts[:, 2]
        xmin, xmax = xs.min(), xs.max()
        zmin, zmax = ys.min(), ys.max()
        scale = 0.85 * min(plot_w / (xmax - xmin + 1e-6), plot_h / (zmax - zmin + 1e-6))
        cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2
        px = [int(plot_w / 2 + (x - cx) * scale) for x in xs]
        pz = [int(plot_h / 2 + (z - cz) * scale) for z in ys]
        for i in range(len(px) - 1):
            draw.line([(px[i], pz[i]), (px[i + 1], pz[i + 1])], fill=(220, 40, 40), width=2)
        for i, (x, z) in enumerate(zip(px, pz)):
            draw.ellipse((x - 3, z - 3, x + 3, z + 3), fill=(255, 180, 0))
    draw.text((8, 8), "D: arm centroid X–Z (DLT 3D)", fill=(0, 0, 0))
    conf_strip = Image.new("RGB", (plot_w, 80), (255, 255, 255))
    dc = ImageDraw.Draw(conf_strip)
    t_axis = np.arange(len(conf_3d))
    for t in range(1, len(t_axis)):
        x0 = int(t_axis[t - 1] / (len(t_axis) - 1) * (plot_w - 20)) + 10
        x1 = int(t_axis[t] / (len(t_axis) - 1) * (plot_w - 20)) + 10
        y0 = 70 - int(conf_3d[t - 1, 0] * 60)
        y1 = 70 - int(conf_3d[t, 0] * 60)
        dc.line([(x0, y0), (x1, y1)], fill=(200, 50, 50), width=2)
    dc.text((8, 2), "D: triangulation confidence vs t", fill=(0, 0, 0))
    stage_d = np.vstack([np.asarray(traj_img), np.asarray(conf_strip)])
    iio.imwrite(out_dir / "stage_D_3d_trajectory.png", stage_d)
    print("saved stage_D_3d_trajectory.png")

    for vi in [int(x) for x in args.views.split(",")]:
        tag = view_tags[vi]
        masks = np.load(motion_dir / f"view{vi}_part_masks.npy")
        moving, static = masks[0].astype(bool), masks[1].astype(bool)
        vid = iio.imread(Path(args.src_video_root) / f"{tag}.mp4").astype(np.float32) / 255.0
        std_map = vid.mean(-1).std(axis=0)
        panels = [
            label_panel(heatmap_std(std_map), f"B: temporal σ  view {vi} ({tag})"),
            label_panel(overlay(to_uint8(load_rgba(data_dir, vi, 10, args.n_frames)), moving, (255, 60, 60)), "B: moving mask"),
            label_panel(overlay(to_uint8(load_rgba(data_dir, vi, 10, args.n_frames)), static, (60, 120, 255)), "B: static mask"),
        ]
        for t in times:
            panels.append(label_panel(to_uint8(load_rgba(data_dir, vi, t, args.n_frames)), f"SV4D t={t}"))
        row1 = hstack(panels)

        gray = vid.mean(-1)
        med = np.median(gray, axis=0)
        c2d = np.zeros((args.n_frames, 2), np.float32)
        cf = np.zeros(args.n_frames, np.float32)
        for t in range(args.n_frames):
            diff = np.abs(gray[t] - med) * moving.astype(np.float32)
            if diff.sum() < 10:
                continue
            ys, xs = np.where(diff > 0)
            ws = diff[ys, xs]
            c2d[t] = [ws @ xs / ws.sum(), ws @ ys / ws.sum()]
            cf[t] = min(1.0, ws.sum() / 50000.0)
        rgb_mid = to_uint8(load_rgba(data_dir, vi, 10, args.n_frames))
        uv, valid = project_to_view(xyz, cams[vi], fov_x, H, W)
        pid_v = np.full(len(xyz), 2, dtype=np.int32)
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vpx = np.clip(uv[:, 1].astype(int), 0, H - 1)
        pid_v[moving[vpx, ui] & valid] = 0
        pid_v[static[vpx, ui] & valid] = 1
        row2 = hstack([
            label_panel(draw_trail(rgb_mid, moving, c2d, cf), "D: 2D arm centroid trail"),
            label_panel(project_parts(rgb_mid, uv, valid, pid_v, colors), "C: single-view mask vote (diagnostic)"),
            label_panel(rgb_mid, "canonical RGB mid-frame"),
        ])
        if row2.shape[1] < row1.shape[1]:
            pad = np.ones((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8) * 255
            row2 = np.hstack([row2, pad])
        sheet = np.vstack([row1, np.ones((6, row1.shape[1], 3), dtype=np.uint8) * 255, row2])
        path = out_dir / f"view{vi:02d}_{tag}_stages_B_D.png"
        iio.imwrite(path, sheet)
        print(f"saved {path.name}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
