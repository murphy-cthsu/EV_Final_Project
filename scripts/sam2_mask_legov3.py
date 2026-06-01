"""Apply SAM-2 video predictor to mask digger only (no baseplate) for lego_v3.
Replaces alpha in each PNG with SAM-2 mask.

For 57 views × 21 frames. Uses center prompt (288, 288) on the lego digger.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import imageio
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lego_v3")
    p.add_argument("--prompt_xy", type=int, nargs=2, default=[288, 288],
                   help="Center click on digger (image coords)")
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    src_base = Path(f"/mnt/HDD_1/cthsu/{args.dataset}")
    sub = "lego_r7_train" if args.dataset == "lego_v3" else "hellwarrior_r32_train"
    sv4d_iter = "lego_r7_train_iter30000" if args.dataset == "lego_v3" else "hellwarrior_r32_train_iter40000"
    src_meta = json.loads((src_base / f"camera_estimation_math_{sub}" / "transforms_sv4d2_math.json").read_text())
    sv4d_dir = src_base / "sv4d2" / sv4d_iter

    data_dir = REPO / "data/custom" / args.dataset

    print(f"[sam-mask-{args.dataset}] loading SAM-2 ckpt")
    predictor = build_sam2_video_predictor(DEFAULT_CFG, str(DEFAULT_CKPT), device=args.device)

    with tempfile.TemporaryDirectory(prefix=f"sam2_{args.dataset}_") as tmproot:
        tmproot = Path(tmproot)
        for v_idx, f in enumerate(src_meta["frames"]):
            tag = f["offset_tag"]
            mp4 = sv4d_dir / f"{tag}.mp4"
            if not mp4.exists():
                continue
            view_tmp = tmproot / f"view_{v_idx}"
            view_tmp.mkdir(exist_ok=True)
            r = imageio.get_reader(str(mp4))
            for t in range(args.n_frames):
                fr = r.get_data(t)
                Image.fromarray(fr).save(view_tmp / f"{t:05d}.jpg")

            state = predictor.init_state(str(view_tmp))
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=np.asarray([args.prompt_xy], dtype=np.float32),
                labels=np.asarray([1], dtype=np.int32),
                clear_old_points=True,
            )

            H = W = 576
            masks = {}
            for fi, _objs, logits in predictor.propagate_in_video(state):
                m = (logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                if m.shape != (H, W):
                    import torch.nn.functional as F
                    mt = torch.from_numpy(m)[None, None].float()
                    m = F.interpolate(mt, (H, W), mode="nearest")[0, 0].numpy().astype(np.uint8)
                masks[int(fi)] = m

            # Replace alpha in existing PNGs
            for t in range(args.n_frames):
                flat = v_idx * args.n_frames + t
                png_name = f"r_{flat:05d}.png"
                png_path = None
                for split in ("train", "test"):
                    cand = data_dir / split / png_name
                    if cand.exists():
                        png_path = cand
                        break
                if png_path is None:
                    continue
                rgba = np.asarray(Image.open(png_path)).copy()
                rgba[..., 3] = masks[t] * 255
                Image.fromarray(rgba).save(png_path)

            if (v_idx + 1) % 10 == 0:
                print(f"[sam-mask-{args.dataset}] view {v_idx+1}/{len(src_meta['frames'])} masked")

    print(f"[sam-mask-{args.dataset}] done")


if __name__ == "__main__":
    raise SystemExit(main())
