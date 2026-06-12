"""Prototype: 3D-projection-prompted SAM-2 part masks.

Idea (L3 direction, identity-from-3D): part identity is defined ONCE in 3D on
the clean canonical (3D k-means on moving Gaussians — left/right limbs are
naturally separate in 3D, so no cross-view identity swap is possible). Each
part is projected into every view with a z-buffer visibility test; the visible
projected points become positive point prompts for SAM-2 (other parts'
points = negatives). SAM-2 only refines pixel boundaries — it never decides
identity, so the classic left/right swap failure mode is structurally ruled out.

Inputs (prepared by an scgs-env dump, see runs_aux/sam3dprompt_<scene>/):
    xyz.npy     (N, 3)  canonical Gaussian centers
    part3d.npy  (N,)    int part id in {-1 (body/static), 0..P-1}

Run from the motionprior env (has sam2; no scipy/matplotlib needed):
    /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_prompt_from_3d.py \
        --scene hellwarrior --views 0 14 28 42 --times 0 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "sam2"))

import torch  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

CKPT = REPO / "checkpoints" / "sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"

PART_COLORS = [(255, 60, 60), (60, 200, 60), (80, 120, 255), (255, 200, 0),
               (200, 60, 255), (0, 220, 220)]


def project(xyz, c2w, fov_x, H, W):
    w2c = np.linalg.inv(c2w)
    w2c = np.diag([1.0, -1.0, -1.0, 1.0]) @ w2c
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1))], -1)
    cam = (w2c @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return np.stack([u, v], -1), z, z > 0


def visible_mask(uv, z, valid, part, cell=6, H=576, W=576):
    """Z-buffer at cell resolution: a Gaussian is 'visible' if it is within a
    small margin of the nearest Gaussian in its cell. Returns bool (N,)."""
    gw, gh = W // cell, H // cell
    ci = np.clip((uv[:, 0] / cell).astype(int), 0, gw - 1)
    cj = np.clip((uv[:, 1] / cell).astype(int), 0, gh - 1)
    flat = cj * gw + ci
    zbuf = np.full(gw * gh, np.inf, np.float32)
    ok = valid & (part >= -1)
    np.minimum.at(zbuf, flat[ok], z[ok])
    margin = 0.05 * np.median(z[ok])
    return ok & (z <= zbuf[flat] + margin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="hellwarrior")
    p.add_argument("--views", type=int, nargs="+", default=[0, 14, 28, 42])
    p.add_argument("--times", type=int, nargs="+", default=[0, 10])
    p.add_argument("--n_pos", type=int, default=4, help="positive points per part")
    p.add_argument("--T", type=int, default=21)
    args = p.parse_args()

    aux = REPO / "runs_aux" / f"sam3dprompt_{args.scene}"
    xyz = np.load(aux / "xyz.npy")
    part3d = np.load(aux / "part3d.npy")
    P = int(part3d.max()) + 1
    data_dir = REPO / "data/custom" / args.scene

    meta_tr = json.loads((data_dir / "transforms_train.json").read_text())
    meta_te = json.loads((data_dir / "transforms_test.json").read_text())
    fov_x = meta_tr["camera_angle_x"]
    cams = {}
    for f in meta_tr["frames"] + meta_te["frames"]:
        v = int(f["view_idx"])
        if v not in cams:
            cams[v] = np.asarray(f["transform_matrix"], np.float64)
    H = W = 576

    def load_frame(v, t):
        flat = v * args.T + t
        for split in ("train", "test"):
            fp = data_dir / split / f"r_{flat:05d}.png"
            if fp.exists():
                rgba = np.asarray(Image.open(fp).convert("RGBA"))
                rgb = rgba[..., :3].astype(np.float32)
                a = rgba[..., 3:4].astype(np.float32) / 255.0
                return (rgb * a + 255.0 * (1 - a)).astype(np.uint8), a[..., 0] > 0.5
        raise FileNotFoundError(f"view {v} t {t}")

    print("[sam3d] loading SAM-2")
    sam = build_sam2(CFG, str(CKPT), device="cuda")
    predictor = SAM2ImagePredictor(sam)

    iou_rows = []
    for v in args.views:
        uv, z, valid = project(xyz, cams[v], fov_x, H, W)
        vis = visible_mask(uv, z, valid, part3d, H=H, W=W)
        for t in args.times:
            img, fg = load_frame(v, t)
            predictor.set_image(img)
            canvas = Image.fromarray(img.copy())
            draw = ImageDraw.Draw(canvas, "RGBA")
            raw_canvas = Image.fromarray(img.copy())

            # prompts per part: visible projected points nearest the part's
            # visible 2D centroid (guaranteed on-part, occlusion-safe)
            pts_by_part = {}
            for pi in range(P):
                m = (part3d == pi) & vis
                if m.sum() < 10:
                    print(f"  view {v} t {t} part {pi}: too few visible pts, skip")
                    continue
                pu = uv[m]
                # Pose-misalignment guard: canonical pose != generated pose, so
                # projections can land in the background — SAM then grabs the
                # whole background. Keep only points inside the frame's fg alpha.
                ui = np.clip(pu[:, 0].astype(int), 0, W - 1)
                vi = np.clip(pu[:, 1].astype(int), 0, H - 1)
                on_fg = fg[vi, ui]
                if on_fg.sum() < 5:
                    print(f"  view {v} t {t} part {pi}: projections off-foreground "
                          f"({on_fg.sum()}/{len(pu)} on fg), skip")
                    continue
                pu = pu[on_fg]
                c = pu.mean(0)
                order = np.argsort(np.linalg.norm(pu - c, axis=1))
                picks = pu[order[:: max(1, len(order) // args.n_pos)][: args.n_pos]]
                pts_by_part[pi] = picks

            for pi, pos in pts_by_part.items():
                neg = np.concatenate([q[:2] for pj, q in pts_by_part.items() if pj != pi]) \
                    if len(pts_by_part) > 1 else np.zeros((0, 2))
                coords = np.concatenate([pos, neg]).astype(np.float32)
                labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=False)
                sam_mask = masks[0].astype(bool)

                # raw 3D projection mask (dilated splat) for comparison
                raw = np.zeros((H, W), np.uint8)
                m = (part3d == pi) & vis
                ui = np.clip(uv[m, 0].astype(int), 0, W - 1)
                vi = np.clip(uv[m, 1].astype(int), 0, H - 1)
                raw[vi, ui] = 255
                raw = np.asarray(Image.fromarray(raw).filter(ImageFilter.MaxFilter(9))) > 0
                inter = (sam_mask & raw).sum()
                union = (sam_mask | raw).sum()
                iou = inter / max(union, 1)
                iou_rows.append((v, t, pi, float(iou), float(scores[0])))

                col = PART_COLORS[pi % len(PART_COLORS)]
                overlay = np.zeros((H, W, 4), np.uint8)
                overlay[sam_mask] = (*col, 110)
                canvas.paste(Image.fromarray(overlay), (0, 0), Image.fromarray(overlay))
                draw = ImageDraw.Draw(canvas, "RGBA")
                for (x, y) in pos:
                    draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(*col, 255),
                                 outline=(0, 0, 0, 255))
                ov2 = np.zeros((H, W, 4), np.uint8)
                ov2[raw] = (*col, 110)
                raw_canvas.paste(Image.fromarray(ov2), (0, 0), Image.fromarray(ov2))

            side = Image.new("RGB", (W * 2 + 8, H), "white")
            side.paste(raw_canvas, (0, 0))
            side.paste(canvas, (W + 8, 0))
            outp = aux / f"overlay_v{v:02d}_t{t:02d}.png"
            side.save(outp)
            print(f"  view {v} t {t}: saved {outp.name} (left=raw 3D proj, right=SAM refined)")

    print("\n[sam3d] raw-projection vs SAM-refined IoU (identity sanity):")
    for v, t, pi, iou, sc in iou_rows:
        print(f"  v{v:02d} t{t:02d} part{pi}: IoU={iou:.2f} sam_score={sc:.2f}")
    arr = np.array([r[3] for r in iou_rows])
    print(f"  mean IoU = {arr.mean():.3f}  (low IoU = SAM grabbed a different region "
          f"than the 3D part — inspect overlay)")


if __name__ == "__main__":
    main()
