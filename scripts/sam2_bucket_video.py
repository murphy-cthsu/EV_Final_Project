"""Run SAM-2 video predictor to track BUCKET region across 21 frames per view.

Output: runs_aux/sam_bucket/
    view{V}/mask_t{T:02d}.png        binary mask 576x576 uint8
    view{V}/overlay_t{T:02d}.png     debug overlay (red = bucket)
    summary.json                      mask areas per (v, t)

Manual click points per view (positive on bucket, negative away from bucket).
Tuned for scene00_masked frame 0 of each view. Iterate if masks look wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SAM2_PATH = REPO / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_PATH))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEFAULT_CKPT = REPO / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_CFG = "configs/sam2/sam2_hiera_l.yaml"

# Per-view bucket click prompts (positive points marking bucket scoop region)
# Coordinates in (x, y) on 576×576 frame at t=0. Tuned by visual inspection.
BUCKET_PROMPTS = {
    # Bucket = the orange-ish scoop body at the END of the arm linkage.
    # Negative points anchor cabin (yellow with red top) + treads (dark gray).
    0: {"pos": [(440, 180)], "neg": [(220, 220), (250, 350), (140, 180)]},
    1: {"pos": [(440, 160), (480, 180)], "neg": [(160, 220), (160, 350)]},
    # view 2: bucket = dark scoop at upper-LEFT (not right). Digger faces forward.
    2: {"pos": [(180, 110), (220, 110)], "neg": [(320, 200), (300, 350), (380, 230)]},
    3: {"pos": [(120, 140), (90, 180)], "neg": [(330, 260), (300, 380), (350, 200)]},
    # view 4: bucket = scoop at upper-LEFT. Need strong negatives on body.
    4: {"pos": [(180, 110), (140, 130)], "neg": [(380, 200), (350, 400), (350, 250), (300, 180)]},
}


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.55):
    out = rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out.clip(0, 255).astype(np.uint8)


def draw_points(rgb, pos, neg):
    out = rgb.copy()
    H, W = out.shape[:2]
    import PIL.ImageDraw as D
    pil = Image.fromarray(out)
    d = D.Draw(pil)
    for (x, y) in pos:
        d.ellipse([x-6, y-6, x+6, y+6], fill=(0, 255, 0), outline=(0, 0, 0))
    for (x, y) in neg:
        d.ellipse([x-6, y-6, x+6, y+6], fill=(255, 0, 0), outline=(0, 0, 0))
    return np.asarray(pil)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--out_dir", default=REPO / "runs_aux/sam_bucket")
    p.add_argument("--n_views", type=int, default=5)
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--cfg",  default=DEFAULT_CFG)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"SAM-2 ckpt missing: {args.ckpt}")

    print(f"[sam-bucket] loading SAM-2 from {args.ckpt}")
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=args.device)

    summary = {}
    with tempfile.TemporaryDirectory(prefix="sam2_bucket_") as tmproot:
        tmproot = Path(tmproot)
        for v in range(args.n_views):
            view_tmp = tmproot / f"view_{v}"
            view_tmp.mkdir()
            # Symlink (or copy) 21 frames into tmp, named 00000.jpg ... (SAM-2 wants JPG)
            for t in range(args.T):
                src = Path(args.src_dir) / f"r_{v*args.T + t:05d}.png"
                rgba = np.asarray(iio.imread(src))
                if rgba.shape[-1] == 4:
                    a = rgba[..., 3:4].astype(np.float32) / 255.0
                    rgb = (rgba[..., :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
                else:
                    rgb = rgba[..., :3]
                Image.fromarray(rgb).save(view_tmp / f"{t:05d}.jpg")

            print(f"[sam-bucket] view {v}: init_state on {view_tmp}")
            state = predictor.init_state(str(view_tmp))
            prompt = BUCKET_PROMPTS[v]
            pos = np.asarray(prompt["pos"], dtype=np.float32)
            neg = np.asarray(prompt["neg"], dtype=np.float32) if prompt["neg"] else np.zeros((0, 2), dtype=np.float32)
            pts = np.concatenate([pos, neg], axis=0)
            labels = np.concatenate([
                np.ones(len(pos), dtype=np.int32),
                np.zeros(len(neg), dtype=np.int32),
            ])
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=pts, labels=labels, clear_old_points=True,
            )

            view_out = out_dir / f"view{v}"
            view_out.mkdir(exist_ok=True)
            (view_out / "_debug").mkdir(exist_ok=True)
            masks = {}
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                m = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                masks[int(frame_idx)] = m

            areas = []
            H = W = 576
            for t in range(args.T):
                m = masks[t]
                if m.shape != (H, W):
                    import torch.nn.functional as F
                    m_t = torch.from_numpy(m)[None, None].float()
                    m = F.interpolate(m_t, (H, W), mode="nearest")[0, 0].numpy().astype(np.uint8)
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    view_out / f"mask_t{t:02d}.png")
                # Overlay viz
                rgb = np.asarray(Image.open(view_tmp / f"{t:05d}.jpg"))
                if rgb.shape[:2] != (H, W):
                    rgb = np.asarray(Image.fromarray(rgb).resize((W, H)))
                ov = overlay_mask(rgb, m)
                if t == 0:
                    ov = draw_points(ov, prompt["pos"], prompt["neg"])
                Image.fromarray(ov).save(view_out / "_debug" / f"overlay_t{t:02d}.png")
                areas.append(int(m.sum()))
            summary[f"view{v}"] = {
                "prompt": prompt,
                "areas": areas,
                "mean_area": float(np.mean(areas)),
            }
            print(f"[sam-bucket] view {v}: bucket mask area mean={np.mean(areas):.0f} "
                  f"min={min(areas)} max={max(areas)} px")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sam-bucket] saved 5×{args.T} masks + debug overlays to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
