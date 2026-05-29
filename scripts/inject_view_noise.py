"""Inject controlled multi-view inconsistency into a D-NeRF-style scene.

Use this as a knob-tunable VGM-failure simulator: starting from a clean (or
SAM-2-masked) multi-view scene, apply a per-view RGB tint that is constant
across time within each view but independent across views. This mirrors the
specific failure mode of multi-view VGMs (e.g. SV4D 2.0): each view is
temporally coherent within itself but the views disagree on the underlying
3D appearance.

Output mirrors the input scene_dir layout (transforms_train.json,
transforms_test.json, train/*.png), with noisy copies of the train PNGs and
the test PNGs left UNTOUCHED (test = clean GT, so held-out PSNR measures
the model's ability to recover the consistent underlying scene despite
inconsistent supervision).

Noise model (apply per pixel):
    rgb' = clamp(rgb + tint_view, 0, 255)
    alpha unchanged

`tint_view` is a per-view constant 3-vector drawn from
Uniform[-noise_level, +noise_level] independently per view. The HELD-OUT
view (if specified) gets ZERO tint -- it represents the clean target.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/inject_view_noise.py \\
        --src_scene_dir data/custom/scene00_split \\
        --out_dir       data/custom/scene00_noisy_mild \\
        --noise_level   10 \\
        --skip_held_out_view 2 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_scene_dir", type=Path, required=True,
                   help="Scene dir with transforms_train.json + transforms_test.json + train/*.png")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--noise_level", type=int, required=True,
                   help="Half-range of per-view RGB tint (e.g. 10 = +/- 10 intensity)")
    p.add_argument("--skip_held_out_view", type=int, default=-1,
                   help="If >=0, this view_idx is NOT tinted (left clean)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = args.src_scene_dir.resolve()
    out = args.out_dir.resolve()

    train_json = json.loads((src / "transforms_train.json").read_text())
    test_json_path = src / "transforms_test.json"
    have_test = test_json_path.is_file()
    test_json = json.loads(test_json_path.read_text()) if have_test else None

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)
    if have_test:
        (out / "test").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # One tint per view_idx seen in the union of train+test.
    all_views = {int(f["view_idx"]) for f in train_json["frames"]}
    if have_test:
        all_views |= {int(f["view_idx"]) for f in test_json["frames"]}
    tints: dict[int, np.ndarray] = {}
    for v in sorted(all_views):
        if v == args.skip_held_out_view:
            tints[v] = np.zeros(3, dtype=np.float32)
        else:
            tints[v] = rng.uniform(-args.noise_level, args.noise_level, size=3).astype(np.float32)

    print(f"[noise] tints (per view RGB shift):")
    for v in sorted(tints):
        flag = " (held-out, clean)" if v == args.skip_held_out_view else ""
        print(f"  view {v}: {tints[v].round(1).tolist()}{flag}")

    def _tint_image(src_png: Path, dst_png: Path, tint: np.ndarray):
        img = np.asarray(iio.imread(src_png))
        rgb = img[..., :3].astype(np.float32)
        alpha = img[..., 3:] if img.shape[-1] == 4 else None
        rgb = np.clip(rgb + tint[None, None, :], 0, 255).astype(np.uint8)
        if alpha is not None:
            out_img = np.concatenate([rgb, alpha], axis=-1)
            Image.fromarray(out_img, mode="RGBA").save(dst_png)
        else:
            Image.fromarray(rgb, mode="RGB").save(dst_png)

    # Tint train PNGs in place. Keep transforms_train.json identical.
    for f in train_json["frames"]:
        v = int(f["view_idx"])
        stem = Path(f["file_path"]).name
        src_png = src / "train" / f"{stem}.png"
        dst_png = out / "train" / f"{stem}.png"
        _tint_image(src_png, dst_png, tints[v])

    # Test PNGs: copy as-is (clean GT). Adapt transforms_test.json if it
    # references ./train/* (the splitter we wrote keeps test entries in
    # train/ because of flat indexing); preserve that.
    if have_test:
        for f in test_json["frames"]:
            stem = Path(f["file_path"]).name
            # Try test/ first, fall back to train/ (our splitter's convention).
            src_png = src / "train" / f"{stem}.png"
            if not src_png.is_file():
                src_png = src / "test" / f"{stem}.png"
            # Mirror to out/train (preserves file_path links in transforms_test.json)
            dst_png = out / "train" / f"{stem}.png"
            if not dst_png.exists():
                _tint_image(src_png, dst_png, tints[int(f["view_idx"])])

    (out / "transforms_train.json").write_text(json.dumps(train_json, indent=2))
    if have_test:
        (out / "transforms_test.json").write_text(json.dumps(test_json, indent=2))

    meta = {
        "source": "inject_view_noise",
        "src_scene_dir": str(src),
        "noise_level": int(args.noise_level),
        "skip_held_out_view": int(args.skip_held_out_view),
        "seed": int(args.seed),
        "tints_per_view": {int(k): tints[k].tolist() for k in sorted(tints)},
        "n_train_frames": len(train_json["frames"]),
        "n_test_frames": len(test_json["frames"]) if have_test else 0,
    }
    (out / "noise_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[noise] wrote noisy scene to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
