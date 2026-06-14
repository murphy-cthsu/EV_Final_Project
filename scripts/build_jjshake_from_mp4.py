"""CORRECT supervision builder for jumpingjacks_shake_lefthand.

Uses the bundle's OWN render mp4s (only the left hand shakes) + the estimated
orbit poses in camera_estimation_math/transforms_sv4d2_math.json. This replaces
the earlier (wrong) approach of re-rendering the deform over a full fid sweep,
which produced full jumping-jacks (legs opening) instead of just the hand shake.

Alpha is extracted from the gray render background (dist-from-bg threshold).

Writes data/custom/jumpingjacks_shake_lefthand/{train,test}/r_{flat:05d}.png (RGBA),
transforms_{train,test}.json, and outputs/custom/jjshake_selfref/renders/{flat}.png.
"""
from __future__ import annotations
import json, re, shutil
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "outputs/custom/jumpingjacks_shake_lefthand"
VIDDIR = BUNDLE / "jumpingjacks_shake_leftthand_r10_train_iter40000"
XFORM = BUNDLE / "camera_estimation_math/transforms_sv4d2_math.json"
SCENE = REPO / "data/custom/jumpingjacks_shake_lefthand"
SELFREF = REPO / "outputs/custom/jjshake_selfref/renders"
T = 21
ALPHA_TH = 0.08


def extract(frame):
    f = frame.astype(np.float32) / 255.0
    corners = np.concatenate([f[:20, :20].reshape(-1, 3), f[:20, -20:].reshape(-1, 3),
                              f[-20:, :20].reshape(-1, 3), f[-20:, -20:].reshape(-1, 3)], 0)
    bg = corners.mean(0)
    a = np.linalg.norm(f - bg, axis=-1) > ALPHA_TH
    a = binary_closing(binary_opening(a, iterations=2), iterations=3)
    rgb = f * a[..., None] + (1 - a[..., None])  # composite to white
    return rgb, a.astype(np.float32)


def main():
    if SCENE.exists():
        shutil.rmtree(SCENE)
    for d in (SCENE / "train", SCENE / "test", SELFREF):
        d.mkdir(parents=True, exist_ok=True)

    meta = json.loads(XFORM.read_text())
    fov_x = meta["camera_angle_x"]
    views = sorted(meta["frames"], key=lambda f: int(re.search(r"az_(\d+)", f["offset_tag"]).group(1)))
    print(f"[build] {len(views)} views, T={T}")

    train_frames, test_frames = [], []
    for vi, vf in enumerate(views):
        tag = vf["offset_tag"]
        c2w = vf["transform_matrix"]
        vid = iio.imread(VIDDIR / f"{tag}.mp4")
        for t in range(T):
            rgb, a = extract(vid[t])
            flat = vi * T + t
            rgba = (np.clip(np.concatenate([rgb, a[..., None]], -1), 0, 1) * 255).astype(np.uint8)
            is_test = (t % 4 == 0)
            split = "test" if is_test else "train"
            Image.fromarray(rgba).save(SCENE / split / f"r_{flat:05d}.png")
            Image.fromarray((rgb * 255).astype(np.uint8)).save(SELFREF / f"{flat:05d}.png")
            fm = {"file_path": f"./{split}/r_{flat:05d}", "view_idx": vi, "frame_idx": t,
                  "view_tag": tag, "time": t / (T - 1), "transform_matrix": c2w}
            (test_frames if is_test else train_frames).append(fm)
        print(f"  view {vi:2d} {tag}")

    for split, frames in (("train", train_frames), ("test", test_frames)):
        (SCENE / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": fov_x, "frames": frames}, indent=2))
    print(f"[build] train={len(train_frames)} test={len(test_frames)} -> {SCENE}")


if __name__ == "__main__":
    raise SystemExit(main())
