"""Build a lego_v2-style benchmark dataset from the NEW 57-view SV4D/d-3dgs format.

The new datasets (lego_v3, hellwarrior) under /mnt/HDD_1/cthsu/<scene>/ use a
different layout than the old 5-view lego_v2:

  <src>/
    camera_estimation_math_<scene>_rN_train/transforms_sv4d2_math.json   (57 view cams)
    sv4d2/<scene>_rN_train_iterM/elev_X_az_Y.mp4                          (57 noisy views, 21f)
    d-3dgs_video/elev_X_az_Y/frames/00000..00020.png                     (57 clean views, 21f)

This builder produces the exact artifacts the partrigid train/eval expect:

  data/custom/<scene>/transforms_{train,test}.json
  data/custom/<scene>/{train,test}/r_{view*21+t:05d}.png   (SV4D, RGBA via non-white alpha)
  outputs/custom/<scene>_d3dgs_ref/renders/{view*21+t:05d}.png  (d-3dgs clean GT, RGB)

Views are indexed 0..V-1 in the order they appear in transforms_sv4d2_math.json
(view 0 = input_pose_video, the clean anchor). Temporal split: every 4th frame
(t % 4 == 0) -> test, rest -> train. SV4D background is already clean white, so
foreground alpha is derived from non-white pixels (no SAM-2 needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NONWHITE_THRESH = 0.95  # pixel mean below this = foreground


def find_one(base: Path, pattern: str) -> Path:
    hits = sorted(base.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern} under {base}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="e.g. hellwarrior, lego_v3")
    ap.add_argument("--src", required=True, help="dataset root, e.g. /mnt/HDD_1/cthsu/hellwarrior")
    ap.add_argument("--test_every", type=int, default=4, help="t %% test_every == 0 -> test")
    ap.add_argument("--n_frames", type=int, default=21)
    ap.add_argument("--max_views", type=int, default=0, help="0 = all views; else cap for a quick build")
    args = ap.parse_args()

    src = Path(args.src)
    cam_dir = find_one(src, "camera_estimation_math_*").parent if (src / "camera_estimation_math").exists() else None
    cam_json = find_one(src, "camera_estimation_math_*/transforms_sv4d2_math.json")
    sv4d_dir = find_one(src, "sv4d2/*").parent if (src / "sv4d2").is_dir() else None
    sv4d_iter_dir = find_one(src, "sv4d2/*")  # the iter subdir holding the mp4s
    d3_root = src / "d-3dgs_video"

    meta = json.loads(cam_json.read_text())
    cam_x = meta["camera_angle_x"]
    frames_meta = meta["frames"]
    if args.max_views > 0:
        frames_meta = frames_meta[: args.max_views]
    V = len(frames_meta)
    T = args.n_frames
    print(f"[build] scene={args.scene}  views={V}  frames/view={T}  cam_x={cam_x:.4f}")
    print(f"[build] sv4d mp4 dir: {sv4d_iter_dir}")
    print(f"[build] d-3dgs root:  {d3_root}")

    out_data = REPO / "data" / "custom" / args.scene
    out_train = out_data / "train"; out_test = out_data / "test"
    out_d3 = REPO / "outputs" / "custom" / f"{args.scene}_d3dgs_ref" / "renders"
    for d in (out_train, out_test, out_d3):
        d.mkdir(parents=True, exist_ok=True)

    train_frames, test_frames = [], []
    missing = 0
    for vi, fm in enumerate(frames_meta):
        tag = fm["video"]
        tmat = fm["transform_matrix"]
        az = fm.get("azimuth_offset_deg", 0.0)
        # load sv4d mp4 (21,H,W,3) once per view
        mp4 = sv4d_iter_dir / f"{tag}.mp4"
        vid = iio.imread(mp4).astype(np.float32) / 255.0  # (T,H,W,3)
        for t in range(T):
            flat = vi * T + t
            is_test = (t % args.test_every == 0)
            # --- SV4D supervision frame -> RGBA via non-white alpha ---
            rgb = vid[min(t, vid.shape[0] - 1)]
            alpha = (rgb.mean(-1) < NONWHITE_THRESH).astype(np.float32)
            rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
            split_dir = out_test if is_test else out_train
            iio.imwrite(split_dir / f"r_{flat:05d}.png", (rgba * 255).astype(np.uint8))
            # --- d-3dgs clean GT render (flat-indexed) ---
            d3_png = d3_root / tag / "frames" / f"{t:05d}.png"
            if d3_png.exists():
                iio.imwrite(out_d3 / f"{flat:05d}.png", iio.imread(d3_png))
            else:
                missing += 1
            entry = {
                "file_path": f"./{'test' if is_test else 'train'}/r_{flat:05d}",
                "view_idx": vi, "frame_idx": t, "azimuth_offset_deg": az,
                "transform_matrix": tmat, "time": round(t / (T - 1), 4),
            }
            (test_frames if is_test else train_frames).append(entry)
        if (vi + 1) % 10 == 0:
            print(f"[build]   {vi+1}/{V} views done")

    common = {"camera_angle_x": cam_x, "n_views": V, "n_frames": T,
              "split_mode": "temporal", "test_every": args.test_every}
    (out_data / "transforms_train.json").write_text(
        json.dumps({**common, "frames": train_frames}, indent=1))
    (out_data / "transforms_test.json").write_text(
        json.dumps({**common, "frames": test_frames}, indent=1))

    print(f"[build] DONE. train={len(train_frames)} test={len(test_frames)} "
          f"d3dgs_missing={missing}")
    print(f"[build] data:   {out_data}")
    print(f"[build] d3dgs:  {out_d3}")


if __name__ == "__main__":
    main()
