"""Apply SAM-2 video predictor masks to the lego_v2 dataset, replacing the
white-bg-diff alpha (which includes baseplate + VGM-noisy grain) with a tight
digger-only mask. This mirrors the SAM-2 step used on the OLD scene00 dataset
(which gave PSNR 25.75 with vanilla SC-GS).

For each of the 5 views:
  - decode the 21 frames from /mnt/HDD_1/cthsu/lego/sv4d2/000001_v00{V}.mp4
  - SAM-2 video predictor: positive click at image center on frame 0 → propagate
  - Replace alpha channel of data/custom/lego_v2{,_4v}/{train,test}/r_*.png

Run from the motionprior conda env (which has hydra + sam2):
  /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_mask_lego_v2.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import imageio
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SAM2_PATH = REPO / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_PATH))
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

DEFAULT_CKPT = REPO / "checkpoints" / "sam2_hiera_large.pt"
DEFAULT_CFG = "configs/sam2/sam2_hiera_l.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_videos", default=Path("/mnt/HDD_1/cthsu/lego/sv4d2"))
    p.add_argument("--data_dirs", nargs="+",
                   default=["data/custom/lego_v2", "data/custom/lego_v2_4v"],
                   help="Dataset directories to update (each must have train/ and possibly test/)")
    p.add_argument("--n_views", type=int, default=5)
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--prompt", default=None,
                   help="Per-view click points 'v=x,y;v=x,y'; default = image center")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"[sam-mask] loading SAM-2 from {DEFAULT_CKPT}")
    predictor = build_sam2_video_predictor(DEFAULT_CFG, str(DEFAULT_CKPT), device=args.device)

    # Default prompt = image center, override if user passes
    prompts = {v: (288, 288) for v in range(args.n_views)}
    if args.prompt:
        for part in args.prompt.split(";"):
            v_s, xy = part.split("=")
            x, y = map(int, xy.split(","))
            prompts[int(v_s)] = (x, y)
    print(f"[sam-mask] prompts: {prompts}")

    # Cache SAM-2 masks per view (5 × 21 = 105 binary masks at 576×576)
    all_masks = {}
    with tempfile.TemporaryDirectory(prefix="sam2_lego_") as tmproot:
        tmproot = Path(tmproot)
        for v in range(args.n_views):
            video_path = Path(args.src_videos) / f"000001_v00{v}.mp4"
            view_tmp = tmproot / f"view_{v}"
            view_tmp.mkdir()

            r = imageio.get_reader(str(video_path))
            for t in range(args.n_frames):
                fr = r.get_data(t)
                Image.fromarray(fr).save(view_tmp / f"{t:05d}.jpg")

            state = predictor.init_state(str(view_tmp))
            cx, cy = prompts[v]
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=np.asarray([[cx, cy]], dtype=np.float32),
                labels=np.asarray([1], dtype=np.int32),
                clear_old_points=True,
            )

            masks_per_t = {}
            for fi, _objs, logits in predictor.propagate_in_video(state):
                m = (logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                masks_per_t[int(fi)] = m

            # Ensure 576x576 dim
            H = W = 576
            areas = []
            for t in range(args.n_frames):
                m = masks_per_t[t]
                if m.shape != (H, W):
                    import torch.nn.functional as F
                    mt = torch.from_numpy(m)[None, None].float()
                    m = F.interpolate(mt, (H, W), mode="nearest")[0, 0].numpy().astype(np.uint8)
                all_masks[(v, t)] = m
                areas.append(int(m.sum()))
            print(f"[sam-mask] view {v}: prompt=({cx},{cy})  mask area mean={np.mean(areas):.0f} "
                  f"range=[{min(areas)}, {max(areas)}]  fg_frac={np.mean(areas)/H/W*100:.1f}%")

    # Re-write the alpha channel in every PNG of every data dir
    n_updated = 0
    for ddir in args.data_dirs:
        ddir = REPO / ddir if not Path(ddir).is_absolute() else Path(ddir)
        if not ddir.exists():
            print(f"[sam-mask] skip (not found): {ddir}")
            continue
        for split in ("train", "test"):
            split_dir = ddir / split
            if not split_dir.exists(): continue
            for png in sorted(split_dir.glob("r_*.png")):
                idx = int(png.stem.split("_")[-1])
                v, t = idx // args.n_frames, idx % args.n_frames
                if (v, t) not in all_masks: continue
                rgba = np.asarray(Image.open(png)).copy()
                # Replace alpha channel with SAM-2 mask
                rgba[..., 3] = all_masks[(v, t)] * 255
                Image.fromarray(rgba).save(png)
                n_updated += 1
            print(f"[sam-mask] {ddir.name}/{split}: updated {len(list(split_dir.glob('r_*.png')))} PNGs")
    print(f"[sam-mask] total {n_updated} PNGs re-masked with SAM-2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
