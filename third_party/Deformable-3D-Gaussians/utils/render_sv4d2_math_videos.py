#!/usr/bin/env python3
"""Render full SV4D2-view videos using math-estimated camera poses.

Reads transforms from camera_estimation_math/transforms_sv4d2_math.json, sweeps
deformation time 0..1 over num_frames at each fixed camera, and writes mp4 files.

Example:
  conda activate d-3dgs
  cd /root/Deformable-3D-Gaussians
  python utils/render_sv4d2_math_videos.py
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from argparse import Namespace
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from scene.deform_model import DeformModel
from utils.general_utils import PILtoTorch, safe_state
from utils.graphics_utils import focal2fov, fov2focal

_INTERRUPTED = False


def _handle_signal(signum, _frame):
    global _INTERRUPTED
    if not _INTERRUPTED:
        _INTERRUPTED = True
        name = "SIGINT" if signum == signal.SIGINT else f"signal {signum}"
        print(
            f"\n[{name}] Stop requested — waiting for the current frame to finish "
            f"(CUDA cannot be interrupted mid-kernel). Press Ctrl+C again to force quit.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print("\nForce quitting.", file=sys.stderr, flush=True)
        raise KeyboardInterrupt


def load_cfg(model_path: Path) -> Namespace:
    return eval((model_path / "cfg_args").read_text())


def c2w_to_RT(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.linalg.inv(c2w)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return R.astype(np.float32), T.astype(np.float32)


def make_camera(
    R: np.ndarray,
    T: np.ndarray,
    camera_angle_x: float,
    width: int,
    height: int,
    white_background: bool,
    fid: float,
    uid: int = 0,
) -> Camera:
    bg = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    dummy = PILtoTorch(
        Image.new("RGB", (width, height), tuple(int(255 * c) for c in bg)), (width, height)
    )
    fovx = camera_angle_x
    fovy = focal2fov(fov2focal(fovx, width), height)
    return Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=dummy[:3],
        gt_alpha_mask=None,
        image_name="dummy",
        uid=uid,
        data_device="cuda",
        fid=fid,
    )


def load_view_poses(transforms_path: Path) -> tuple[float, list[dict]]:
    with open(transforms_path) as f:
        data = json.load(f)
    frames = sorted(data["frames"], key=lambda x: x.get("video", ""))
    return float(data["camera_angle_x"]), frames


def save_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def render_view_videos(args: argparse.Namespace) -> None:
    global _INTERRUPTED
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    model_path = args.model_path.resolve()
    cfg = load_cfg(model_path)
    transforms_path = args.transforms.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_angle_x, view_frames = load_view_poses(transforms_path)
    render_size = args.render_size
    num_frames = args.num_frames
    times = [i / (num_frames - 1) for i in range(num_frames)]

    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(cfg.sh_degree)
    gaussians.load_ply(
        model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    )
    deform = DeformModel(is_blender=cfg.is_blender, is_6dof=cfg.is_6dof)
    deform.load_weights(str(model_path), iteration=args.iteration)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        done = {v["video_id"] for v in manifest.get("videos", [])}
    else:
        manifest = {
            "model_path": str(model_path),
            "iteration": args.iteration,
            "transforms": str(transforms_path),
            "render_size": render_size,
            "num_frames": num_frames,
            "fps": args.fps,
            "videos": [],
        }
        done = set()

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    for entry in view_frames:
        if _INTERRUPTED:
            break

        video_id = entry["video"]
        video_path = out_dir / video_id / f"{video_id}.mp4"
        if args.skip_existing and video_path.exists():
            print(f"Skipping {video_id} (already exists)", flush=True)
            if video_id not in done:
                manifest["videos"].append(
                    {
                        "video_id": video_id,
                        "elevation_offset_deg": entry.get("elevation_offset_deg"),
                        "azimuth_offset_deg": entry.get("azimuth_offset_deg"),
                        "offset_tag": entry.get("offset_tag"),
                        "pose_source": entry.get("pose_source"),
                        "output_mp4": str(video_path),
                        "frames_dir": str(out_dir / video_id / "frames"),
                    }
                )
                save_manifest(out_dir, manifest)
                done.add(video_id)
            continue

        c2w = np.array(entry["transform_matrix"], dtype=np.float64)
        R, T = c2w_to_RT(c2w)

        view_out = out_dir / video_id
        frames_dir = view_out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== {video_id} "
            f"(elev +{entry.get('elevation_offset_deg', 0):g}°, az +{entry.get('azimuth_offset_deg', 0):g}°) ===",
            flush=True,
        )
        renderings = []

        with torch.no_grad():
            frame_iter = tqdm(
                enumerate(times),
                total=len(times),
                desc=f"Rendering {video_id}",
                leave=False,
                file=sys.stderr,
            )
            for idx, t in frame_iter:
                if _INTERRUPTED:
                    break

                cam = make_camera(
                    R,
                    T,
                    camera_angle_x,
                    render_size,
                    render_size,
                    args.white_background,
                    fid=float(t),
                    uid=idx,
                )
                xyz = gaussians.get_xyz
                time_input = torch.full((xyz.shape[0], 1), t, device=xyz.device, dtype=xyz.dtype)
                d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                result = render(
                    cam,
                    gaussians,
                    pipeline,
                    background,
                    d_xyz,
                    d_rotation,
                    d_scaling,
                    cfg.is_6dof,
                )
                frame = to8b(result["render"].cpu().numpy())
                renderings.append(frame.transpose(1, 2, 0))
                imageio.imwrite(frames_dir / f"{idx:05d}.png", frame.transpose(1, 2, 0))

        if _INTERRUPTED:
            print(f"Interrupted during {video_id} — partial frames kept in {frames_dir}", flush=True)
            break

        imageio.mimwrite(video_path, renderings, fps=args.fps, quality=8)
        print(f"Saved {video_path}", flush=True)

        manifest["videos"] = [v for v in manifest["videos"] if v["video_id"] != video_id]
        manifest["videos"].append(
            {
                "video_id": video_id,
                "elevation_offset_deg": entry.get("elevation_offset_deg"),
                "azimuth_offset_deg": entry.get("azimuth_offset_deg"),
                "offset_tag": entry.get("offset_tag"),
                "pose_source": entry.get("pose_source"),
                "output_mp4": str(video_path),
                "frames_dir": str(frames_dir),
            }
        )
        save_manifest(out_dir, manifest)

    save_manifest(out_dir, manifest)
    n_done = len(manifest["videos"])
    if _INTERRUPTED:
        print(f"\nStopped early. {n_done} videos in manifest at {out_dir}", flush=True)
        sys.exit(130)
    print(f"\nSaved {n_done} videos to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    lego_fixed = REPO_ROOT / "output" / "lego_fixed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=lego_fixed)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--transforms",
        type=Path,
        default=lego_fixed / "camera_estimation_math_lego_r7_train" / "transforms_sv4d2_math.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=lego_fixed / "d-3dgs_video",
    )
    parser.add_argument("--render_size", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--white_background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip videos whose .mp4 already exists (default: true)",
    )
    parser.add_argument(
        "--no-skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Re-render all videos even if output exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    safe_state(True)
    render_view_videos(parse_args())
