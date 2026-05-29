"""Split an existing D-NeRF-format scene by HELD-OUT FRAMES (temporal interpolation).

Unlike split_train_test.py (which holds out a whole view), this script holds
out every-k-th frame across all views. Tests the model's ability to
*interpolate* between training timesteps rather than extrapolate to a novel
viewpoint -- a much more direct probe of what SC-GS is designed for.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/split_temporal.py \\
        --src_scene_dir data/custom/scene00_masked \\
        --out_dir       data/custom/scene00_split_t \\
        --hold_every    4   # hold out frames where frame_idx % 4 == 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_scene_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--hold_every", type=int, default=4,
                   help="Hold out frames where (frame_idx %% hold_every) == (hold_every - 1). "
                        "e.g. 4 -> hold out frame_idx in {3, 7, 11, 15, 19}")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = args.src_scene_dir.resolve()
    out = args.out_dir.resolve()
    src_json = src / "transforms_train.json"
    if not src_json.is_file():
        raise FileNotFoundError(src_json)

    data = json.loads(src_json.read_text())
    cax = data["camera_angle_x"]
    frames = data["frames"]
    k = args.hold_every
    if k <= 1:
        raise ValueError(f"hold_every must be >= 2, got {k}")

    train: list[dict] = []
    test: list[dict] = []
    for f in frames:
        t = int(f["frame_idx"])
        if t % k == k - 1:
            test.append(f)
        else:
            train.append(f)
    if not test:
        raise ValueError("no test frames produced; check hold_every")
    if not train:
        raise ValueError("no train frames produced")

    # Hold-out per view summary
    held_by_view: dict[int, list[int]] = {}
    for f in test:
        held_by_view.setdefault(int(f["view_idx"]), []).append(int(f["frame_idx"]))
    print(f"[split_t] kept {len(train)} train / {len(test)} test "
          f"(hold_every={k}, held times per view = "
          f"{sorted({tuple(sorted(v)) for v in held_by_view.values()})})")

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)

    for f in frames:
        stem = Path(f["file_path"]).name
        src_png = src / "train" / f"{stem}.png"
        dst_png = out / "train" / f"{stem}.png"
        if not dst_png.exists():
            os.symlink(src_png, dst_png)

    (out / "transforms_train.json").write_text(
        json.dumps({"camera_angle_x": cax, "frames": train}, indent=2)
    )
    (out / "transforms_test.json").write_text(
        json.dumps({"camera_angle_x": cax, "frames": test}, indent=2)
    )

    src_meta = src / "custom_metadata.json"
    if src_meta.is_file():
        meta = json.loads(src_meta.read_text())
        meta["hold_every"] = int(k)
        meta["n_train_frames_split"] = len(train)
        meta["n_test_frames_split"] = len(test)
        meta["split_mode"] = "temporal_interpolation"
        (out / "custom_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[split_t] wrote scene to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
