"""Split an existing D-NeRF-format multi-view scene into train/test by view.

Reads <src>/transforms_train.json (which has flat-indexed entries with per-frame
view_idx metadata), and produces <out>/transforms_train.json + transforms_test.json
where one view's 21 frames are held out as test. PNGs are symlinked (not copied)
to avoid duplicating data.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/split_train_test.py \\
        --src_scene_dir data/custom/scene00_masked \\
        --out_dir       data/custom/scene00_split \\
        --test_view     2
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
    p.add_argument("--test_view", type=int, required=True,
                   help="view_idx to hold out")
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
    train = [f for f in frames if int(f["view_idx"]) != args.test_view]
    test  = [f for f in frames if int(f["view_idx"]) == args.test_view]
    if not test:
        raise ValueError(f"no frames with view_idx={args.test_view}")
    print(f"[split] kept {len(train)} train / {len(test)} test "
          f"(holding out view {args.test_view})")

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train").mkdir(exist_ok=True)

    # Symlink every train PNG (we keep using flat-index naming so the SC-GS sort
    # key still works; same PNGs are referenced by both transforms files).
    for f in frames:
        # file_path is e.g. "./train/r_00042" — resolve to the actual png
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
    # Carry over provenance if present
    src_meta = src / "custom_metadata.json"
    if src_meta.is_file():
        meta = json.loads(src_meta.read_text())
        meta["test_view_idx"] = int(args.test_view)
        meta["n_train_frames_split"] = len(train)
        meta["n_test_frames_split"] = len(test)
        (out / "custom_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[split] wrote scene to {out}")
    print(f"[split]   transforms_train.json ({len(train)} entries)")
    print(f"[split]   transforms_test.json  ({len(test)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
