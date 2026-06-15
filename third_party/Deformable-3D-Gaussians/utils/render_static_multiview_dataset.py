#!/usr/bin/env python3
"""Render a static multi-view image dataset from a trained D-3DGS model at t=0.

Reads camera poses from a NeRF-style transforms JSON, renders one RGB frame per
camera with deformation fixed at time=0, and writes a Blender/NeRF synthetic
dataset (transforms_train.json, transforms_test.json, train/*.png, test/*.png).

Example:
  conda activate d-3dgs
  cd /root/Deformable-3D-Gaussians
  python utils/render_static_multiview_dataset.py \\
    --model_path output/jumpingjacks \\
    --poses /root/data_fixed/jumpingjacks/transforms_train.json \\
    --output_dir /root/data_fixed/jumpingjacks_static_t0
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

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

TIME_T0 = 0.0


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
    uid: int = 0,
) -> Camera:
    bg = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    dummy = PILtoTorch(
        Image.new("RGB", (width, height), tuple(int(255 * c) for c in bg)),
        (width, height),
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
        fid=float(TIME_T0),
    )


def load_pose_frames(poses_path: Path) -> tuple[float, list[dict]]:
    with open(poses_path) as f:
        data = json.load(f)
    frames = data["frames"]
    return float(data["camera_angle_x"]), frames


def split_train_test(frames: list[dict], test_every: int) -> tuple[list[dict], list[dict]]:
    if test_every <= 0:
        return frames, []
    train, test = [], []
    for idx, frame in enumerate(frames):
        (test if idx % test_every == 0 else train).append(frame)
    return train, test


def write_transforms(path: Path, camera_angle_x: float, frames_meta: list[dict]) -> None:
    payload = {
        "camera_angle_x": camera_angle_x,
        "frames": frames_meta,
    }
    path.write_text(json.dumps(payload, indent=4))


def render_static_dataset(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    poses_path = args.poses.resolve()
    out_dir = args.output_dir.resolve()
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(model_path)
    camera_angle_x, train_poses = load_pose_frames(poses_path)
    if args.poses_test is not None:
        test_camera_angle_x, test_poses = load_pose_frames(args.poses_test.resolve())
        if abs(test_camera_angle_x - camera_angle_x) > 1e-6:
            raise ValueError(
                f"camera_angle_x mismatch: train={camera_angle_x}, test={test_camera_angle_x}"
            )
    else:
        if args.max_views is not None:
            train_poses = train_poses[: args.max_views]
        train_poses, test_poses = split_train_test(train_poses, args.test_every)

    if args.max_views is not None and args.poses_test is None:
        pass
    elif args.max_views is not None:
        train_poses = train_poses[: args.max_views]
        test_poses = test_poses[: args.max_views]
    all_jobs: list[tuple[str, dict, Path, str]] = []
    for split_name, split_poses, split_dir in (
        ("train", train_poses, train_dir),
        ("test", test_poses, test_dir),
    ):
        for idx, frame in enumerate(split_poses):
            stem = Path(frame["file_path"]).name
            rel_path = f"./{split_name}/{stem}"
            out_png = split_dir / f"{stem}.png"
            all_jobs.append((split_name, frame, out_png, rel_path))

    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(cfg.sh_degree)
    gaussians.load_ply(
        model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    )
    deform = DeformModel(is_blender=cfg.is_blender, is_6dof=cfg.is_6dof)
    deform.load_weights(str(model_path), iteration=args.iteration)

    train_meta: list[dict] = []
    test_meta: list[dict] = []
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    print(
        f"Rendering {len(all_jobs)} views at t=0 "
        f"({len(train_poses)} train, {len(test_poses)} test), "
        f"size={args.render_size}",
        flush=True,
    )

    with torch.no_grad():
        for uid, (split_name, frame, out_png, rel_path) in enumerate(
            tqdm(all_jobs, desc="Rendering static views")
        ):
            if args.skip_existing and out_png.exists():
                meta = {
                    "file_path": rel_path,
                    "rotation": frame.get("rotation", 0.0),
                    "time": TIME_T0,
                    "transform_matrix": frame["transform_matrix"],
                }
            else:
                c2w = np.array(frame["transform_matrix"], dtype=np.float64)
                R, T = c2w_to_RT(c2w)
                cam = make_camera(
                    R,
                    T,
                    camera_angle_x,
                    args.render_size,
                    args.render_size,
                    args.white_background,
                    uid=uid,
                )
                xyz = gaussians.get_xyz
                time_input = torch.full(
                    (xyz.shape[0], 1), TIME_T0, device=xyz.device, dtype=xyz.dtype
                )
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
                frame_rgb = to8b(result["render"].cpu().numpy()).transpose(1, 2, 0)
                Image.fromarray(frame_rgb).save(out_png)
                meta = {
                    "file_path": rel_path,
                    "rotation": frame.get("rotation", 0.0),
                    "time": TIME_T0,
                    "transform_matrix": frame["transform_matrix"],
                }

            if split_name == "train":
                train_meta.append(meta)
            else:
                test_meta.append(meta)

    write_transforms(out_dir / "transforms_train.json", camera_angle_x, train_meta)
    write_transforms(out_dir / "transforms_test.json", camera_angle_x, test_meta)

    manifest = {
        "model_path": str(model_path),
        "iteration": args.iteration,
        "poses_source": str(poses_path),
        "time": TIME_T0,
        "render_size": args.render_size,
        "num_train": len(train_meta),
        "num_test": len(test_meta),
        "test_every": args.test_every,
        "white_background": args.white_background,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved static dataset to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    jumpingjacks = REPO_ROOT / "output" / "jumpingjacks"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=jumpingjacks)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("/root/data_fixed/jumpingjacks/transforms_train.json"),
        help="NeRF transforms JSON listing train camera poses to render",
    )
    parser.add_argument(
        "--poses_test",
        type=Path,
        default=None,
        help="Optional transforms JSON for test views (uses explicit train/test split)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/root/data_fixed/jumpingjacks_static_t0"),
    )
    parser.add_argument("--render_size", type=int, default=800)
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Every Nth view goes to transforms_test.json (0 = all train)",
    )
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument(
        "--white_background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip_existing", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    safe_state(True)
    render_static_dataset(parse_args())
