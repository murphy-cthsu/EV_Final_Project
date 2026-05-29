"""Build a frame-0-only subset scene for static canonical training.

Takes scene00_masked (5 views x 21 frames) and produces scene00_frame0/ with
only the 5 frame-0 RGBA images + a transforms.json matching SC-GS's static
scene loader expectations.

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/build_frame0_subset.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "custom" / "scene00_masked"
OUT = REPO_ROOT / "data" / "custom" / "scene00_frame0"


def main() -> int:
    src_json = SRC / "transforms_train.json"
    data = json.loads(src_json.read_text())
    frame0 = [f for f in data["frames"] if int(f["frame_idx"]) == 0]
    print(f"[frame0] selected {len(frame0)} frames at t=0")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "train").mkdir()

    for f in frame0:
        stem = Path(f["file_path"]).name
        src_png = SRC / "train" / f"{stem}.png"
        dst_png = OUT / "train" / f"{stem}.png"
        if not dst_png.exists():
            os.symlink(src_png, dst_png)

    # Static: time field set to 0 (all frames at same time).
    # We keep view_idx for downstream tools but SC-GS doesn't need it.
    out_frames = []
    for f in frame0:
        out_frames.append({
            "file_path": f["file_path"],
            "rotation": f.get("rotation", 0.0),
            "time": 0.0,
            "transform_matrix": f["transform_matrix"],
            "view_idx": f["view_idx"],
            "frame_idx": 0,
        })
    out_json = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": out_frames,
    }
    (OUT / "transforms_train.json").write_text(json.dumps(out_json, indent=2))

    src_meta = SRC / "custom_metadata.json"
    if src_meta.is_file():
        meta = json.loads(src_meta.read_text())
        meta["subset"] = "frame0_static"
        meta["n_train_frames_split"] = len(frame0)
        (OUT / "custom_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[frame0] wrote {OUT}")
    print(f"[frame0]   transforms_train.json: {len(frame0)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
