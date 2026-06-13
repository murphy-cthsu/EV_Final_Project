"""Assemble a partrigid-format dataset from self-generated SV4D multi-ring output,
so the fit-residual probe can run on a scene with NO clean 4D reference.

Source: /mnt/HDD_1/cthsu/sv4d_p1_out/<scene>_5v_elev{E}/sv4d2/000000_{process_input,v001..v004}.mp4
  input -> az 0 ; v001..v004 -> az 60/120/180/240 (per diagnose_sed_selfgen).
Cameras: reused from lego_v3 transforms by (elev,az) view_tag (orbit geometry is
  scene-independent; absolute misregistration is absorbed by the probe's fitting).
Alpha: non-white threshold (SV4D background is clean white).

Writes data/custom/<scene>_selfgen/{train,test}/r_*.png + transforms_{train,test}.json
(temporal split: t%4==0 -> test).

Run: python scripts/build_selfgen_dataset.py --scene jumpingjacks
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SRC = Path("/mnt/HDD_1/cthsu/sv4d_p1_out")
AZ_FOR_V = {"process_input": 0, "v001": 60, "v002": 120, "v003": 180, "v004": 240}
NONWHITE = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--elevs", default="0,5,10")
    ap.add_argument("--T", type=int, default=21)
    args = ap.parse_args()
    elevs = [int(x) for x in args.elevs.split(",")]

    lego = json.loads((REPO / "data/custom/lego_v3/transforms_train.json").read_text())
    lego2 = json.loads((REPO / "data/custom/lego_v3/transforms_test.json").read_text())
    cam_by_tag, fov_x = {}, lego["camera_angle_x"]
    for f in lego["frames"] + lego2["frames"]:
        cam_by_tag.setdefault(f.get("view_tag"), f["transform_matrix"])

    out = REPO / "data/custom" / f"{args.scene}_selfgen"
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    # enumerate (elev, az, mp4) views
    views = []
    for e in elevs:
        ring = SRC / f"{args.scene}_5v_elev{e}" / "sv4d2"
        for stem, az in AZ_FOR_V.items():
            tag = f"elev_{e}_az_{az}"
            mp4 = ring / f"000000_{stem}.mp4"
            if not mp4.exists() or tag not in cam_by_tag:
                continue
            views.append((e, az, tag, mp4))
    print(f"[{args.scene}] {len(views)} views: "
          + ", ".join(f"e{e}a{az}" for e, az, _, _ in views))

    tr_frames, te_frames = [], []
    for vi, (e, az, tag, mp4) in enumerate(views):
        rd = imageio.get_reader(str(mp4))
        for t in range(args.T):
            rgb = rd.get_data(t).astype(np.float32) / 255.0
            if rgb.shape[:2] != (576, 576):
                rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8))
                                 .resize((576, 576))).astype(np.float32) / 255.0
            fg = rgb.mean(-1) < NONWHITE
            rgba = np.concatenate([rgb, fg[..., None].astype(np.float32)], -1)
            flat = vi * args.T + t
            split = "test" if t % 4 == 0 else "train"
            Image.fromarray((rgba * 255).astype(np.uint8)).save(out / split / f"r_{flat:05d}.png")
            fr = dict(file_path=f"./{split}/r_{flat:05d}", view_idx=vi, view_tag=tag,
                      elevation_deg=float(e), azimuth_deg=float(az), frame_idx=t,
                      time=round(t / (args.T - 1), 4), rotation=0,
                      transform_matrix=cam_by_tag[tag])
            (te_frames if split == "test" else tr_frames).append(fr)

    base = dict(camera_angle_x=fov_x, n_views=len(views), n_frames=args.T,
                split_mode="temporal", test_every=4)
    (out / "transforms_train.json").write_text(json.dumps({**base, "frames": tr_frames}, indent=1))
    (out / "transforms_test.json").write_text(json.dumps({**base, "frames": te_frames}, indent=1))
    print(f"[{args.scene}] wrote {out}  (train {len(tr_frames)} / test {len(te_frames)} frames)")


if __name__ == "__main__":
    main()
