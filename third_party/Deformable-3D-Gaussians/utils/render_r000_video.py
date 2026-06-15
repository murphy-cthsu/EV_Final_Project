#!/usr/bin/env python3
"""Render deformation video from a fixed train camera in transforms_train.json."""

from __future__ import annotations

import argparse
import json
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
    image_name: str = "train",
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
        image_name=image_name,
        uid=uid,
        data_device="cuda",
        fid=fid,
    )


def load_frame_pose(transforms_path: Path, frame_id: str) -> tuple[float, np.ndarray, dict]:
    with open(transforms_path) as f:
        data = json.load(f)
    needle = frame_id if frame_id.startswith("r_") else f"r_{frame_id}"
    frame = next(fr for fr in data["frames"] if needle in fr["file_path"])
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    return float(data["camera_angle_x"]), c2w, frame


def render_video(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    cfg = load_cfg(model_path)
    transforms_path = args.transforms.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id = args.frame
    camera_angle_x, c2w, frame_meta = load_frame_pose(transforms_path, frame_id)
    R, T = c2w_to_RT(c2w)
    cam_center = c2w[:3, 3].tolist()

    num_frames = args.num_frames
    times = np.linspace(0.0, 1.0, num_frames).tolist()

    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(cfg.sh_degree)
    gaussians.load_ply(
        model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    )
    deform = DeformModel(is_blender=cfg.is_blender, is_6dof=cfg.is_6dof)
    deform.load_weights(str(model_path), iteration=args.iteration)

    render_size = args.render_size
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = []

    print(f"Camera center (world): {cam_center}")
    print(f"frame time in transforms: {frame_meta.get('time')}")
    print(f"Rendering {num_frames} frames at {render_size}x{render_size}")

    with torch.no_grad():
        for idx, t in enumerate(tqdm(times, desc=f"Rendering {frame_id}")):
            cam = make_camera(
                R,
                T,
                camera_angle_x,
                render_size,
                render_size,
                args.white_background,
                fid=float(t),
                uid=idx,
                image_name=frame_id,
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
            frame = to8b(result["render"].cpu().numpy()).transpose(1, 2, 0)
            renderings.append(frame)

    stem = f"{frame_id}_iter{args.iteration}"
    mp4_path = out_dir / f"{stem}.mp4"
    gif_path = out_dir / f"{stem}.gif"
    imageio.mimwrite(mp4_path, renderings, fps=args.fps, quality=8)
    imageio.mimwrite(gif_path, renderings, fps=args.fps)
    print(f"Saved {mp4_path}")
    print(f"Saved {gif_path}")

    meta = {
        "camera_center_world": cam_center,
        "transform_matrix_c2w": c2w.tolist(),
        "camera_angle_x": camera_angle_x,
        "frame_meta": {k: v for k, v in frame_meta.items() if k != "transform_matrix"},
        "model_path": str(model_path),
        "iteration": args.iteration,
        "num_frames": num_frames,
        "fps": args.fps,
        "mp4": str(mp4_path),
        "gif": str(gif_path),
    }
    (out_dir / f"{frame_id}_render_meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    hellwarrior = REPO_ROOT / "output" / "hellwarrior"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=hellwarrior)
    parser.add_argument("--frame", type=str, default="r_032", help="Train frame id, e.g. r_032 or 032")
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument(
        "--transforms",
        type=Path,
        default=Path("/root/data_fixed/hellwarrior/transforms_train.json"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--render_size", type=int, default=800)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
        help="Number of output frames; timesteps are sampled evenly in [0, 1]",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--white_background",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = hellwarrior / f"{args.frame}_video"
    return args


if __name__ == "__main__":
    safe_state(True)
    render_video(parse_args())
