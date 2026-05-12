"""Convert SV4D 2.0 output into a D-NeRF-format scene directory SC-GS can train on.

Pipeline:
    SV4D output ((V, T, H, W, 3) uint8 + azimuth list)
    -> per-frame PNGs under <out_dir>/train/r_<view><frame>.png
    -> synthesized transforms_train.json with object-centric orbit cameras
    -> reuse the original scene's transforms_test.json + test/ images for eval
       (lets us evaluate the SV4D-supervised SC-GS run against the same D-NeRF
        ground truth as the multi-view baseline)

Output layout matches third_party/SC-GS/scene/dataset_readers.py:readNerfSyntheticInfo:

    <out_dir>/
        train/
            r_v0_t000.png ... r_v0_t020.png       # 5 views × 21 frames = 105 imgs (sv4d2)
            r_v1_t000.png ...
            ...
        test/                                      # symlinked from original scene
            r_000.png ...
        transforms_train.json                      # synthesized
        transforms_test.json                       # copied from original scene
        transforms_val.json                        # copied from original scene
        sv4d_metadata.json                         # provenance

Usage:
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/sv4d_to_dnerf.py \\
        --sv4d_output_dir outputs/sv4d_supervised/jumpingjacks/sv4d_run \\
        --orig_scene_dir  data/dnerf/jumpingjacks \\
        --out_dir         data/dnerf_sv4d/jumpingjacks \\
        --orbit_radius    4.0

The converter is also importable: see ``convert_sv4d_to_dnerf`` for the API.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motionprior.integration.vgm import SV4DResult  # noqa: E402


def look_at_blender(camera_pos: np.ndarray, target: np.ndarray = None,
                    up: np.ndarray = None) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in Blender convention.

    Blender camera looks down -Z, with +X right and +Y up. D-NeRF's
    transforms_*.json files use this same convention (see how the C2W is
    composed in third_party/SC-GS/scene/dataset_readers.py:readCamerasFromTransforms
    -> R = transpose(c2w[:3,:3]); T = -R @ c2w[:3,3]).

    Args:
        camera_pos: (3,) world position.
        target: (3,) point to look at. Defaults to origin.
        up: (3,) up direction. Defaults to +Y.

    Returns:
        (4, 4) camera-to-world transform.
    """
    if target is None:
        target = np.zeros(3, dtype=np.float64)
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward) + 1e-12
    # blender: -Z is forward, so the cam's local Z = -forward
    cam_z = -forward
    cam_x = np.cross(up, cam_z)
    cam_x /= np.linalg.norm(cam_x) + 1e-12
    cam_y = np.cross(cam_z, cam_x)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = cam_x
    c2w[:3, 1] = cam_y
    c2w[:3, 2] = cam_z
    c2w[:3, 3] = camera_pos
    return c2w


def orbit_camera(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    """Object-centric orbit camera in Blender convention.

    Azimuth 0 = camera on +Z looking at origin; positive azimuth rotates CCW
    around +Y. Elevation 0 = horizon; positive elevation = camera above horizon.
    This matches the convention SV4D 2.0 documents in
    simple_video_sample_4d2.py:159 (polars_rad = pi/2 - elev_rad).
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = radius * np.cos(el) * np.sin(az)
    y = radius * np.sin(el)
    z = radius * np.cos(el) * np.cos(az)
    return look_at_blender(np.array([x, y, z]))


def camera_angle_x_for_size(img_size: int, focal_length_mm: float = 50.0,
                            sensor_width_mm: float = 36.0) -> float:
    """Estimate horizontal FoV in radians.

    D-NeRF synthetic uses 0.6911112 rad (~39.6°) which corresponds to a 50mm
    lens on a 36mm sensor — Blender default. Match this so the cameras
    we synthesize have the same intrinsics as the original D-NeRF scene
    cameras (modulo resolution, which SC-GS handles).
    """
    return 2.0 * np.arctan(0.5 * sensor_width_mm / focal_length_mm)


def convert_sv4d_to_dnerf(
    sv4d_result: SV4DResult,
    orig_scene_dir: Path,
    out_dir: Path,
    orbit_radius: float = 4.0,
    camera_angle_x: float | None = None,
    copy_test_split: bool = True,
    overwrite: bool = False,
) -> dict:
    """Write D-NeRF-shaped scene dir derived from SV4D output.

    Args:
        sv4d_result: parsed SV4D output (V views × T frames + azimuths).
        orig_scene_dir: the original D-NeRF scene dir (we copy its test split
            for evaluation against the same GT as the baseline).
        out_dir: where to write the new scene. Created.
        orbit_radius: world-space orbit radius. D-NeRF synthetic uses ~4.0.
        camera_angle_x: horizontal FoV in radians. If None, defaults to D-NeRF's
            standard 0.6911 (Blender 50mm on 36mm sensor).
        copy_test_split: if True, symlink/copy the original scene's test split
            so that SC-GS eval points at the real D-NeRF GT.
        overwrite: if True, clobber any existing out_dir.

    Returns:
        A dict summarizing what was written.
    """
    out_dir = Path(out_dir).resolve()
    orig_scene_dir = Path(orig_scene_dir).resolve()
    if not orig_scene_dir.is_dir():
        raise FileNotFoundError(orig_scene_dir)
    if camera_angle_x is None:
        camera_angle_x = 0.6911112070083618  # D-NeRF default

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "train"
    train_dir.mkdir(exist_ok=True)

    V, T, H, W, _ = sv4d_result.frames.shape
    if len(sv4d_result.azimuths_deg) != V:
        raise ValueError(
            f"azimuths_deg length {len(sv4d_result.azimuths_deg)} != V={V}"
        )

    # SC-GS's sort key (dataset_readers.py:292) parses int from the LAST
    # underscore-separated token of the filename basename. We therefore use a
    # flat integer index as the trailing component and keep view/time
    # metadata in the JSON fields, not in the filename. flat_idx = v * T + t.
    frames_meta: list[dict] = []
    n_written = 0
    for v, (az, el) in enumerate(
        zip(sv4d_result.azimuths_deg, sv4d_result.elevations_deg)
    ):
        c2w = orbit_camera(az, el, orbit_radius)
        for t in range(T):
            flat_idx = v * T + t
            fname = f"r_{flat_idx:05d}"
            png_path = train_dir / f"{fname}.png"
            Image.fromarray(sv4d_result.frames[v, t]).save(png_path)
            n_written += 1
            frames_meta.append({
                "file_path": f"./train/{fname}",
                "rotation": 0.0,
                "time": float(t) / max(T - 1, 1),
                "transform_matrix": c2w.tolist(),
                "azimuth_deg": float(az),
                "elevation_deg": float(el),
                "view_idx": int(v),
                "frame_idx": int(t),
            })

    transforms_train = {
        "camera_angle_x": camera_angle_x,
        "frames": frames_meta,
    }
    (out_dir / "transforms_train.json").write_text(json.dumps(transforms_train, indent=2))

    if copy_test_split:
        for split in ("test", "val"):
            src_split = orig_scene_dir / split
            if src_split.is_dir():
                dst_split = out_dir / split
                if dst_split.exists():
                    shutil.rmtree(dst_split)
                shutil.copytree(src_split, dst_split, symlinks=True)
            src_json = orig_scene_dir / f"transforms_{split}.json"
            if src_json.is_file():
                shutil.copy2(src_json, out_dir / f"transforms_{split}.json")
        # also copy point cloud init if present
        src_ply = orig_scene_dir / "points3d.ply"
        if src_ply.is_file():
            shutil.copy2(src_ply, out_dir / "points3d.ply")

    metadata = {
        "source": "sv4d2",
        "sv4d_variant": sv4d_result.variant,
        "sv4d_seed": sv4d_result.seed,
        "sv4d_img_size": sv4d_result.img_size,
        "sv4d_output_dir": str(sv4d_result.output_dir),
        "orig_scene_dir": str(orig_scene_dir),
        "n_views": int(V),
        "n_frames_per_view": int(T),
        "orbit_radius": orbit_radius,
        "camera_angle_x": camera_angle_x,
        "azimuths_deg": list(sv4d_result.azimuths_deg),
        "elevations_deg": list(sv4d_result.elevations_deg),
        "n_train_images_written": int(n_written),
    }
    (out_dir / "sv4d_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------- #
# CLI: can be called either with a real SV4D output dir, or in "from-disk"
# mode where the user already has a (V, T, H, W, 3) numpy array on disk.
# ---------------------------------------------------------------------- #
def parse_sv4d_output_dir(sv4d_dir: Path, variant: str = "sv4d2") -> SV4DResult:
    """Reconstruct SV4DResult from an SV4D output directory created by our
    adapter. Reads sv4d_result.json + demuxes the mp4s. Used when the user
    runs SV4D separately and just wants to convert."""
    sv4d_dir = Path(sv4d_dir).resolve()
    meta_json = sv4d_dir / "sv4d_result.json"
    if meta_json.is_file():
        meta = json.loads(meta_json.read_text())
    else:
        meta = None

    sub = sv4d_dir / variant
    if not sub.is_dir():
        raise FileNotFoundError(f"{sub} not found")

    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio  # type: ignore

    # Find base count
    v001 = sorted(sub.glob("*_v001.mp4"))
    if not v001:
        raise FileNotFoundError(f"no *_v001.mp4 in {sub}")
    base = v001[-1].stem.split("_")[0]

    # Preprocessed input
    inp_candidates = [p for p in sub.glob("*.mp4") if "_v" not in p.stem]
    if not inp_candidates:
        inp_candidates = [p for p in sv4d_dir.glob("*.mp4") if "_v" not in p.stem]
    if not inp_candidates:
        raise FileNotFoundError(f"no preprocessed input mp4 in {sub} or {sv4d_dir}")
    per_view = [np.asarray(iio.imread(inp_candidates[0]))]

    # Novel views
    if meta is not None:
        V = meta["n_views"]
        azimuths = meta["azimuths_deg"]
        elevations = meta["elevations_deg"]
        n_frames = meta["n_frames"]
        img_size = meta["img_size"]
        seed = meta["seed"]
    else:
        # Fall back to defaults for the variant
        from motionprior.integration.vgm import SV4D2_AZIMUTHS_DEG
        azimuths = list(SV4D2_AZIMUTHS_DEG[variant])
        V = len(azimuths)
        elevations = [0.0] * V
        n_frames = per_view[0].shape[0]
        img_size = per_view[0].shape[1]
        seed = -1

    for v in range(1, V):
        mp4 = sub / f"{base}_v{v:03d}.mp4"
        if not mp4.is_file():
            raise FileNotFoundError(mp4)
        per_view.append(np.asarray(iio.imread(mp4)))

    T = min(v.shape[0] for v in per_view)
    per_view = [v[:T] for v in per_view]
    frames = np.stack(per_view, axis=0)
    return SV4DResult(
        frames=frames,
        azimuths_deg=azimuths,
        elevations_deg=elevations,
        variant=variant,
        seed=seed,
        img_size=img_size,
        n_frames=T,
        output_dir=sv4d_dir,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sv4d_output_dir", type=Path, required=True,
                   help="Directory containing SV4D 2.0 output (e.g. outputs/sv4d_supervised/<scene>/sv4d_run)")
    p.add_argument("--orig_scene_dir", type=Path, required=True,
                   help="Original D-NeRF scene dir (e.g. data/dnerf/jumpingjacks)")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output dir for the new D-NeRF-format scene")
    p.add_argument("--variant", default="sv4d2", choices=["sv4d2", "sv4d2_8views"])
    p.add_argument("--orbit_radius", type=float, default=4.0)
    p.add_argument("--camera_angle_x", type=float, default=None,
                   help="Horizontal FoV in radians (default = D-NeRF's 0.6911)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print(f"[convert] parsing SV4D output from {args.sv4d_output_dir}")
    result = parse_sv4d_output_dir(args.sv4d_output_dir, variant=args.variant)
    print(f"[convert]   frames shape: {result.frames.shape}")
    print(f"[convert]   azimuths:     {result.azimuths_deg}")

    print(f"[convert] writing D-NeRF scene to {args.out_dir}")
    meta = convert_sv4d_to_dnerf(
        result,
        orig_scene_dir=args.orig_scene_dir,
        out_dir=args.out_dir,
        orbit_radius=args.orbit_radius,
        camera_angle_x=args.camera_angle_x,
        overwrite=args.overwrite,
    )
    print(f"[convert]   wrote {meta['n_train_images_written']} train images, "
          f"{meta['n_views']} views × {meta['n_frames_per_view']} frames")
    print(f"[convert] done. point SC-GS --source_path at {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
