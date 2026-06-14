"""Generic self-supervised scene builder for a bundle (canonical + render mp4s +
camera_estimation_math/transforms_sv4d2_math.json), e.g. jumpingjacks_shake_lefthand
or jumpingjacks_splitting.

Supervision = the bundle's own render mp4s (gray bg → alpha by dist-from-bg).
Poses from transforms_sv4d2_math.json. Writes a D-NeRF scene + self-ref renders.

Usage:
  python scripts/build_bundle_from_mp4.py --bundle jumpingjacks_splitting \
      --viddir jumpingjacks_splitting_r10_train --scene jumpingjacks_splitting
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
import numpy as np, imageio.v3 as iio
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening

REPO = Path(__file__).resolve().parent.parent
T = 21
ALPHA_TH = 0.08


def extract(frame):
    f = frame.astype(np.float32) / 255.0
    corners = np.concatenate([f[:20, :20].reshape(-1, 3), f[:20, -20:].reshape(-1, 3),
                              f[-20:, :20].reshape(-1, 3), f[-20:, -20:].reshape(-1, 3)], 0)
    bg = corners.mean(0)
    a = np.linalg.norm(f - bg, axis=-1) > ALPHA_TH
    a = binary_closing(binary_opening(a, iterations=2), iterations=3)
    rgb = f * a[..., None] + (1 - a[..., None])
    return rgb, a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="bundle dir under repo root")
    ap.add_argument("--viddir", required=True, help="mp4 subdir name inside bundle")
    ap.add_argument("--scene", required=True, help="output scene name under data/custom")
    args = ap.parse_args()

    BUNDLE = REPO / args.bundle
    VIDDIR = BUNDLE / args.viddir
    XFORM = BUNDLE / "camera_estimation_math/transforms_sv4d2_math.json"
    SCENE = REPO / "data/custom" / args.scene
    SELFREF = REPO / "outputs/custom" / f"{args.scene}_selfref" / "renders"

    if SCENE.exists():
        shutil.rmtree(SCENE)
    for d in (SCENE / "train", SCENE / "test", SELFREF):
        d.mkdir(parents=True, exist_ok=True)

    meta = json.loads(XFORM.read_text())
    fov_x = meta["camera_angle_x"]
    views = sorted(meta["frames"], key=lambda f: int(re.search(r"az_(\d+)", f["offset_tag"]).group(1)))
    print(f"[build:{args.scene}] {len(views)} views, T={T}")

    tr, te = [], []
    for vi, vf in enumerate(views):
        tag = vf["offset_tag"]; c2w = vf["transform_matrix"]
        vid = iio.imread(VIDDIR / f"{tag}.mp4")
        for t in range(T):
            rgb, a = extract(vid[t]); flat = vi * T + t
            rgba = (np.clip(np.concatenate([rgb, a[..., None]], -1), 0, 1) * 255).astype(np.uint8)
            split = "test" if (t % 4 == 0) else "train"
            Image.fromarray(rgba).save(SCENE / split / f"r_{flat:05d}.png")
            Image.fromarray((rgb * 255).astype(np.uint8)).save(SELFREF / f"{flat:05d}.png")
            fm = {"file_path": f"./{split}/r_{flat:05d}", "view_idx": vi, "frame_idx": t,
                  "view_tag": tag, "time": t / (T - 1), "transform_matrix": c2w}
            (te if split == "test" else tr).append(fm)
        print(f"  view {vi:2d} {tag}")

    for split, frames in (("train", tr), ("test", te)):
        (SCENE / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": fov_x, "frames": frames}, indent=2))
    print(f"[build:{args.scene}] train={len(tr)} test={len(te)} -> {SCENE}")


if __name__ == "__main__":
    raise SystemExit(main())
