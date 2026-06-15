#!/usr/bin/env python3
"""Estimate SV4D2 camera poses analytically from the input view + SV3D orbit offsets.

Step 1: estimate input view (elev_0_az_0 or v000) via coarse training-camera search.
Step 2: derive other views by applying relative azimuth and elevation offsets from the
        input pose, following the SV3D/SV4D2 spherical camera convention.

Orbit convention (generative-models/scripts/demo/sv3d_helpers.py plot_3D,
simple_video_sample_4d2.py):
  polar = pi/2 - elevation_deg,  elev = pi/2 - polar = radians(elevation_deg)
  direction: x = cos(elev)*cos(az), y = cos(elev)*sin(az), z = sin(elev)
  elevations_deg are relative to the input view; azimuths_deg are absolute values
  whose offset from the input azimuth is applied after subtracting the input view.

Saves RGB comparisons and grayscale subtraction maps (target_gray, render_gray, diff).

Example:
  cd /root/Deformable-3D-Gaussians
  python utils/estimate_sv4d2_cameras_math.py \\
    --video_dir assets/sv4d2/lego_r7_train
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as tf
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
from utils.image_utils import psnr
from utils.loss_utils import ssim

import lpips

RENDER_SIZE = 576
TIME_T0 = 0.0

# Default SV4D2 4-view model: view index -> azimuth offset (degrees) from input.
SV4D2_AZIMUTH_OFFSETS_4VIEW: dict[int, float] = {0: 0.0, 1: 60.0, 2: 120.0, 3: 180.0, 4: 240.0}

# SV4D2 8-view model offsets (view index -> degrees).
SV4D2_AZIMUTH_OFFSETS_8VIEW: dict[int, float] = {
    0: 0.0,
    1: 30.0,
    2: 75.0,
    3: 120.0,
    4: 165.0,
    5: 210.0,
    6: 255.0,
    7: 300.0,
    8: 330.0,
}


@dataclass
class CameraCandidate:
    name: str
    file_path: str
    c2w: np.ndarray
    R: np.ndarray
    T: np.ndarray


@dataclass
class MetricBundle:
    psnr: float
    ssim: float
    lpips: float
    combined: float


def load_cfg(model_path: Path) -> Namespace:
    return eval((model_path / "cfg_args").read_text())


def c2w_to_RT(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.linalg.inv(c2w)
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return R.astype(np.float32), T.astype(np.float32)


def z_rotation_matrix(delta_rad: float) -> np.ndarray:
    """Rotation around +Z (SV3D azimuth axis)."""
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def sv3d_spherical_from_position(rel: np.ndarray) -> tuple[float, float, float]:
    """Decompose position into SV3D spherical coords (radius, azimuth, elevation).

    Matches generative-models/scripts/demo/sv3d_helpers.py plot_3D and
    simple_video_sample_4d2.py (polar = pi/2 - elevation_deg):
      x = cos(elev)*cos(az), y = cos(elev)*sin(az), z = sin(elev)
    """
    r = float(np.linalg.norm(rel))
    if r < 1e-12:
        return 0.0, 0.0, 0.0
    elev = float(np.arcsin(np.clip(rel[2] / r, -1.0, 1.0)))
    az = float(np.arctan2(rel[1], rel[0]))
    return r, az, elev


def sv3d_position_from_spherical(r: float, az: float, elev: float) -> np.ndarray:
    """Reconstruct a position from SV3D spherical coords."""
    return r * np.array(
        [np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)],
        dtype=np.float64,
    )


def build_c2w_look_at(
    cam_pos: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
) -> np.ndarray:
    """NeRF/Blender c2w rotation (columns: right, up, back); camera looks down local -Z."""
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_hint)
    n = np.linalg.norm(right)
    if n < 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_hint)
        n = np.linalg.norm(right)
    right = right / n
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    R = np.zeros((3, 3), dtype=np.float64)
    R[:, 0] = right
    R[:, 1] = up
    R[:, 2] = -forward
    return R


def sv3d_orbit_camera_c2w(
    c2w_input: np.ndarray,
    delta_azimuth_deg: float,
    delta_elevation_deg: float = 0.0,
    orbit_center: np.ndarray | None = None,
    orbit_sign: float = 1.0,
) -> np.ndarray:
    """Orbit camera relative to the input view (elev_0_az_0 basis).

    The input c2w (e.g. transforms_fixed_cam_r7.json) is the reference pose where
    elevation_offset=0 and azimuth_offset=0.  SV4D2 offsets from
    simple_video_sample_4d2.py are additive in SV3D spherical coordinates:
      polar = pi/2 - elevation_deg,  elev = pi/2 - polar
      direction: [cos(elev)cos(az), cos(elev)sin(az), sin(elev)]

    Azimuth offset: rotate around +Z through orbit_center (proven for elev=0).
    Elevation offset: move along the SV3D spherical meridian, then re-aim at
    orbit_center (R_el @ R_az does not match the new position on the sphere).
    """
    if orbit_center is None:
        orbit_center = np.zeros(3, dtype=np.float64)

    c2w = c2w_input.astype(np.float64).copy()
    delta_az = np.deg2rad(delta_azimuth_deg * orbit_sign)
    delta_el = np.deg2rad(delta_elevation_deg)

    rel0 = c2w[:3, 3] - orbit_center
    r, az0, elev0 = sv3d_spherical_from_position(rel0)
    if r < 1e-12:
        return c2w

    new_rel = sv3d_position_from_spherical(r, az0 + delta_az, elev0 + delta_el)
    new_pos = orbit_center + new_rel
    c2w[:3, 3] = new_pos

    R_az = z_rotation_matrix(delta_az)
    if abs(delta_el) < 1e-12:
        c2w[:3, :3] = R_az @ c2w[:3, :3]
    else:
        up_hint = R_az @ c2w[:3, :3] @ np.array([0.0, 1.0, 0.0])
        c2w[:3, :3] = build_c2w_look_at(new_pos, orbit_center, up_hint)
    return c2w


def orbit_camera_c2w(
    c2w_input: np.ndarray,
    delta_azimuth_deg: float,
    orbit_center: np.ndarray | None = None,
    orbit_sign: float = 1.0,
) -> np.ndarray:
    """Azimuth-only orbit (backward-compatible wrapper)."""
    return sv3d_orbit_camera_c2w(
        c2w_input,
        delta_azimuth_deg,
        delta_elevation_deg=0.0,
        orbit_center=orbit_center,
        orbit_sign=orbit_sign,
    )


def parse_view_index(video_stem: str) -> int | None:
    match = re.search(r"_v(\d+)$", video_stem)
    return int(match.group(1)) if match else None


def parse_elev_az_stem(video_stem: str) -> tuple[float, float] | None:
    """Parse elev_{E}_az_{A} naming used by lego_r7_train assets."""
    match = re.fullmatch(r"elev_(\d+(?:\.\d+)?)_az_(\d+(?:\.\d+)?)", video_stem)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def parse_video_offsets(video_stem: str, azimuth_offsets: dict[int, float]) -> tuple[float, float, str] | None:
    """Return (delta_elevation_deg, delta_azimuth_deg, source_tag) for a video stem."""
    elev_az = parse_elev_az_stem(video_stem)
    if elev_az is not None:
        elev_deg, az_deg = elev_az
        return elev_deg, az_deg, f"elev_{elev_deg:g}_az_{az_deg:g}"

    view_idx = parse_view_index(video_stem)
    if view_idx is None:
        return None
    offset_deg = azimuth_offsets.get(view_idx)
    if offset_deg is None:
        return None
    return 0.0, offset_deg, f"v{view_idx:03d}"


def get_azimuth_offsets(model: str) -> dict[int, float]:
    if model == "sv4d2_8views":
        return SV4D2_AZIMUTH_OFFSETS_8VIEW
    return SV4D2_AZIMUTH_OFFSETS_4VIEW


def load_training_camera_candidates(transforms_path: Path) -> list[CameraCandidate]:
    with open(transforms_path) as f:
        data = json.load(f)
    candidates = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        R, T = c2w_to_RT(c2w)
        name = Path(frame["file_path"]).name
        candidates.append(
            CameraCandidate(name=name, file_path=frame["file_path"], c2w=c2w, R=R, T=T)
        )
    return candidates


def load_r7_ground_truth(transforms_path: Path) -> CameraCandidate:
    with open(transforms_path) as f:
        data = json.load(f)
    frame = data["frames"][0]
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    R, T = c2w_to_RT(c2w)
    return CameraCandidate(name="r_7_gt", file_path=frame["file_path"], c2w=c2w, R=R, T=T)


def load_camera_by_name(transforms_path: Path, camera_name: str) -> CameraCandidate:
    with open(transforms_path) as f:
        data = json.load(f)
    for frame in data["frames"]:
        name = Path(frame["file_path"]).name
        if name == camera_name:
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            R, T = c2w_to_RT(c2w)
            return CameraCandidate(
                name=camera_name, file_path=frame["file_path"], c2w=c2w, R=R, T=T
            )
    raise ValueError(f"Camera {camera_name!r} not found in {transforms_path}")


def load_input_camera(args: argparse.Namespace, transforms_train: Path) -> CameraCandidate:
    if args.input_camera:
        return load_camera_by_name(transforms_train, args.input_camera)
    return load_r7_ground_truth(args.r7_transforms.resolve())


def load_video_first_frame(video_path: Path, size: int = RENDER_SIZE) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    reader.close()
    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return tf.to_tensor(img).unsqueeze(0).cuda()


def make_camera(
    R: np.ndarray,
    T: np.ndarray,
    camera_angle_x: float,
    width: int,
    height: int,
    white_background: bool,
    fid: float = TIME_T0,
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


class SceneRenderer:
    def __init__(
        self,
        model_path: Path,
        iteration: int,
        camera_angle_x: float,
        white_background: bool,
        is_blender: bool,
        is_6dof: bool,
        sh_degree: int,
    ):
        self.camera_angle_x = camera_angle_x
        self.white_background = white_background
        self.pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        bg = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.load_ply(
            model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        )
        self.deform = DeformModel(is_blender=is_blender, is_6dof=is_6dof)
        self.deform.load_weights(str(model_path), iteration=iteration)
        self.is_6dof = is_6dof
        self._precompute_deformation()

    def _precompute_deformation(self) -> None:
        xyz = self.gaussians.get_xyz
        time_input = torch.full((xyz.shape[0], 1), TIME_T0, device=xyz.device, dtype=xyz.dtype)
        with torch.no_grad():
            self._d_xyz, self._d_rotation, self._d_scaling = self.deform.step(xyz.detach(), time_input)

    @torch.no_grad()
    def render_c2w(self, c2w: np.ndarray) -> torch.Tensor:
        R, T = c2w_to_RT(c2w)
        return self.render_numpy(R, T)

    @torch.no_grad()
    def render_numpy(self, R: np.ndarray, T: np.ndarray) -> torch.Tensor:
        cam = make_camera(
            R, T, self.camera_angle_x, RENDER_SIZE, RENDER_SIZE, self.white_background, fid=TIME_T0
        )
        out = render(
            cam,
            self.gaussians,
            self.pipeline,
            self.background,
            self._d_xyz,
            self._d_rotation,
            self._d_scaling,
            self.is_6dof,
        )
        return out["render"].clamp(0.0, 1.0).unsqueeze(0)


class MetricEvaluator:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()

    @torch.no_grad()
    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> MetricBundle:
        p = pred.clamp(0.0, 1.0)
        t = target.clamp(0.0, 1.0)
        psnr_val = psnr(p, t).item()
        ssim_val = ssim(p, t).item()
        lpips_val = self.lpips_fn(p, t).item()
        combined = (psnr_val / 40.0) + ssim_val + (1.0 - lpips_val)
        return MetricBundle(psnr_val, ssim_val, lpips_val, combined)


def coarse_search(
    target: torch.Tensor,
    candidates: list[CameraCandidate],
    renderer: SceneRenderer,
    metrics: MetricEvaluator,
) -> tuple[CameraCandidate, MetricBundle]:
    best_candidate = None
    best_metrics = None
    for cand in tqdm(candidates, desc="Coarse search (v000)", leave=False):
        pred = renderer.render_numpy(cand.R, cand.T)
        m = metrics.evaluate(pred, target)
        if best_metrics is None or m.combined > best_metrics.combined:
            best_candidate = cand
            best_metrics = m
    return best_candidate, best_metrics


def rgb_to_gray(tensor: torch.Tensor) -> torch.Tensor:
    """RGB [1,3,H,W] in [0,1] -> grayscale [1,1,H,W]."""
    weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (tensor * weights).sum(dim=1, keepdim=True)


def save_rgb_comparison(path: Path, target: torch.Tensor, pred: torch.Tensor, title: str) -> None:
    import torchvision

    grid = torch.cat([target, pred], dim=0)
    torchvision.utils.save_image(grid, path, nrow=2)
    path.with_suffix(".txt").write_text(title)


def save_gray_subtraction_comparison(
    path: Path, target: torch.Tensor, pred: torch.Tensor, title: str
) -> None:
    """Save target gray | render gray | signed gray difference (zero = mid-gray)."""
    import torchvision

    target_gray = rgb_to_gray(target.clamp(0.0, 1.0))
    pred_gray = rgb_to_gray(pred.clamp(0.0, 1.0))
    diff = target_gray - pred_gray
    diff_vis = (0.5 + 0.5 * diff).clamp(0.0, 1.0)
    diff_vis_rgb = diff_vis.repeat(1, 3, 1, 1)

    grid = torch.cat([target_gray.repeat(1, 3, 1, 1), pred_gray.repeat(1, 3, 1, 1), diff_vis_rgb], dim=0)
    torchvision.utils.save_image(grid, path, nrow=3)

    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    path.with_suffix(".txt").write_text(
        f"{title}\n"
        f"gray MAE={mae:.6f}  gray RMSE={rmse:.6f}\n"
        f"diff panel: 0.5 + 0.5 * (target_gray - render_gray); mid-gray = zero difference"
    )


def pose_error(gt: CameraCandidate, est_c2w: np.ndarray) -> dict:
    gt_center = np.linalg.inv(gt.c2w)[:3, 3]
    est_center = np.linalg.inv(est_c2w)[:3, 3]
    center_dist = float(np.linalg.norm(gt_center - est_center))
    R_rel = est_c2w[:3, :3].T @ gt.c2w[:3, :3]
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace)))
    return {
        "camera_center_distance": center_dist,
        "rotation_error_deg": rot_deg,
        "gt_camera_center": gt_center.tolist(),
        "est_camera_center": est_center.tolist(),
    }


def estimate_input_pose(
    target: torch.Tensor,
    candidates: list[CameraCandidate],
    renderer: SceneRenderer,
    metrics: MetricEvaluator,
    use_input_camera: bool,
    input_gt: CameraCandidate,
) -> tuple[np.ndarray, str, MetricBundle, str]:
    if use_input_camera:
        pred = renderer.render_c2w(input_gt.c2w)
        m = metrics.evaluate(pred, target)
        return input_gt.c2w, input_gt.name, m, f"known_{input_gt.name}"

    coarse, coarse_m = coarse_search(target, candidates, renderer, metrics)
    return coarse.c2w, coarse.name, coarse_m, "coarse_search"


def run_estimation(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    cfg = load_cfg(model_path)
    data_path = Path(args.data_path).resolve()
    transforms_train = data_path / "transforms_train.json"
    video_dir = args.video_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(transforms_train) as f:
        camera_angle_x = json.load(f)["camera_angle_x"]

    candidates = load_training_camera_candidates(transforms_train)
    input_gt = load_input_camera(args, transforms_train)
    azimuth_offsets = get_azimuth_offsets(args.sv4d2_model)

    renderer = SceneRenderer(
        model_path=model_path,
        iteration=args.iteration,
        camera_angle_x=camera_angle_x,
        white_background=args.white_background,
        is_blender=cfg.is_blender,
        is_6dof=cfg.is_6dof,
        sh_degree=cfg.sh_degree,
    )
    metrics = MetricEvaluator()

    video_paths = sorted(video_dir.glob("*.mp4"))
    if args.videos:
        names = set(args.videos)
        video_paths = [p for p in video_paths if p.name in names or p.stem in names]

    orbit_center = np.array(args.orbit_center, dtype=np.float64)

    results = {
        "settings": {
            "method": "math_orbit",
            "model_path": str(model_path),
            "iteration": args.iteration,
            "render_size": RENDER_SIZE,
            "time": TIME_T0,
            "sv4d2_model": args.sv4d2_model,
            "azimuth_offsets_deg": azimuth_offsets,
            "orbit_sign": args.orbit_sign,
            "orbit_center": orbit_center.tolist(),
            "use_input_camera": args.use_input_camera,
            "input_camera": args.input_camera,
            "convention": "SV3D spherical: elev relative to input, az absolute offset from input az=0",
        },
        "input_pose": {},
        "videos": {},
    }

    input_path = next(
        (p for p in video_paths if parse_elev_az_stem(p.stem) == (0.0, 0.0)),
        None,
    )
    if input_path is None:
        input_path = next((p for p in video_paths if parse_view_index(p.stem) == 0), None)
    if input_path is None:
        raise FileNotFoundError(
            f"No input view found (elev_0_az_0 or v000) in {video_dir}"
        )

    print(f"Estimating input pose from {input_path.name}")
    v000_target = load_video_first_frame(input_path)
    input_c2w, input_coarse_name, input_metrics, input_method = estimate_input_pose(
        v000_target,
        candidates,
        renderer,
        metrics,
        args.use_input_camera,
        input_gt,
    )

    results["input_pose"] = {
        "video": str(input_path),
        "method": input_method,
        "coarse_camera": input_coarse_name,
        "transform_matrix": input_c2w.tolist(),
        "metrics": input_metrics.__dict__,
        "validation_vs_input": pose_error(input_gt, input_c2w),
    }
    print(
        f"  input pose: {input_method} via {input_coarse_name}  "
        f"PSNR={input_metrics.psnr:.2f} SSIM={input_metrics.ssim:.4f}"
    )
    print(
        f"  vs {input_gt.name}: rot_err={results['input_pose']['validation_vs_input']['rotation_error_deg']:.4f} deg"
    )

    for video_path in video_paths:
        video_id = video_path.stem
        offsets = parse_video_offsets(video_id, azimuth_offsets)
        if offsets is None:
            print(f"Skipping {video_id}: cannot parse elevation/azimuth offsets")
            continue

        delta_elev_deg, delta_az_deg, offset_tag = offsets
        is_input = parse_elev_az_stem(video_id) == (0.0, 0.0) or parse_view_index(video_id) == 0

        print(
            f"\n=== {video_id} (elev +{delta_elev_deg:g}°, az +{delta_az_deg:g}°) ==="
        )
        target = load_video_first_frame(video_path)

        if is_input:
            est_c2w = input_c2w
            pose_source = "input_estimated"
        else:
            est_c2w = sv3d_orbit_camera_c2w(
                input_c2w,
                delta_azimuth_deg=delta_az_deg,
                delta_elevation_deg=delta_elev_deg,
                orbit_center=orbit_center,
                orbit_sign=args.orbit_sign,
            )
            pose_source = f"math_orbit_{offset_tag}"

        pred = renderer.render_c2w(est_c2w)
        m = metrics.evaluate(pred, target)

        vid_out = out_dir / video_id
        vid_out.mkdir(parents=True, exist_ok=True)

        metric_str = f"PSNR={m.psnr:.3f} SSIM={m.ssim:.4f} LPIPS={m.lpips:.4f}"
        save_rgb_comparison(
            vid_out / "compare_render.png",
            target,
            pred,
            f"{video_id} [{pose_source}] {metric_str}",
        )
        save_gray_subtraction_comparison(
            vid_out / "compare_gray_diff.png",
            target,
            pred,
            f"{video_id} [{pose_source}] {metric_str}",
        )

        entry = {
            "video": str(video_path),
            "elevation_offset_deg": delta_elev_deg,
            "azimuth_offset_deg": delta_az_deg,
            "offset_tag": offset_tag,
            "pose_source": pose_source,
            "transform_matrix": est_c2w.tolist(),
            "metrics": m.__dict__,
        }
        if is_input:
            entry["validation_vs_input"] = pose_error(input_gt, est_c2w)

        print(f"  {pose_source}: PSNR={m.psnr:.2f} SSIM={m.ssim:.4f} LPIPS={m.lpips:.4f}")
        results["videos"][video_id] = entry

    transforms_out = {
        "camera_angle_x": camera_angle_x,
        "render_size": RENDER_SIZE,
        "time": TIME_T0,
        "method": "math_orbit_from_input_view",
        "input_pose_video": input_path.stem,
        "frames": [
            {
                "video": vid,
                "elevation_offset_deg": entry["elevation_offset_deg"],
                "azimuth_offset_deg": entry["azimuth_offset_deg"],
                "offset_tag": entry["offset_tag"],
                "pose_source": entry["pose_source"],
                "transform_matrix": entry["transform_matrix"],
                "metrics": entry["metrics"],
            }
            for vid, entry in sorted(results["videos"].items())
        ],
    }

    (out_dir / "results_math.json").write_text(json.dumps(results, indent=2))
    (out_dir / "transforms_sv4d2_math.json").write_text(json.dumps(transforms_out, indent=2))
    print(f"\nSaved results to {out_dir}")


def parse_args() -> argparse.Namespace:
    lego_fixed = REPO_ROOT / "output" / "lego_fixed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=lego_fixed)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--data_path", type=Path, default=Path("/workspace/data_fixed/lego"))
    parser.add_argument(
        "--r7_transforms",
        type=Path,
        default=lego_fixed / "transforms_fixed_cam_r7.json",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=REPO_ROOT / "assets" / "sv4d2" / "lego_r7_train",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=lego_fixed / "camera_estimation_math_lego_r7_train",
    )
    parser.add_argument(
        "--sv4d2_model",
        choices=["sv4d2", "sv4d2_8views"],
        default="sv4d2",
        help="Which SV4D2 model azimuth schedule to use",
    )
    parser.add_argument(
        "--orbit_sign",
        type=float,
        default=1.0,
        help="Multiply azimuth offsets by this sign (+1 or -1) to flip orbit direction",
    )
    parser.add_argument(
        "--orbit_center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Orbit center in world coordinates",
    )
    parser.add_argument(
        "--input_camera",
        type=str,
        default=None,
        help="Training camera name for input view (e.g. r_032). Loaded from transforms_train.json.",
    )
    parser.add_argument(
        "--use_input_camera",
        action="store_true",
        help="Use ground-truth c2w for the input view instead of coarse search",
    )
    parser.add_argument(
        "--use_r7_input",
        action="store_true",
        help="Alias for --use_input_camera (lego r_7 workflow)",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional subset of video filenames/stems",
    )
    parser.add_argument(
        "--white_background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    safe_state(True)
    args = parse_args()
    if args.use_r7_input:
        args.use_input_camera = True
    run_estimation(args)
