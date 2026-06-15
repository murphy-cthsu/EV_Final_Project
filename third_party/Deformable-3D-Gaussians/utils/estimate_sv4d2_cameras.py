#!/usr/bin/env python3
"""Estimate SV4D2 camera poses via coarse training-camera search + refinement.

Compares each video's first frame to lego_fixed renders at t=0 (576x576).
Validates v000 against r_7 from transforms_fixed_cam_r7.json.

Example:
  cd /root/Deformable-3D-Gaussians
  python utils/estimate_sv4d2_cameras.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
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
from scipy.optimize import minimize


RENDER_SIZE = 576
TIME_T0 = 0.0


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


def pose_params_to_c2w(base_c2w: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Apply axis-angle rotation and translation delta to a base c2w pose."""
    omega = torch.tensor(params[:3], dtype=torch.float64)
    delta_t = torch.tensor(params[3:], dtype=torch.float64)
    base = torch.tensor(base_c2w, dtype=torch.float64)
    dR = axis_angle_to_matrix(omega.double()).double()
    c2w = base.clone()
    c2w[:3, :3] = dR @ base[:3, :3]
    c2w[:3, 3] = base[:3, 3] + delta_t
    return c2w.numpy().astype(np.float64)


def RT_to_c2w(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=np.float64)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    c2w = np.linalg.inv(Rt)
    return c2w.astype(np.float64)


def load_training_camera_candidates(
    transforms_path: Path,
) -> list[CameraCandidate]:
    with open(transforms_path) as f:
        data = json.load(f)

    candidates = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        R, T = c2w_to_RT(c2w)
        name = Path(frame["file_path"]).name
        candidates.append(
            CameraCandidate(
                name=name,
                file_path=frame["file_path"],
                c2w=c2w,
                R=R,
                T=T,
            )
        )
    return candidates


def load_r7_ground_truth(transforms_path: Path) -> CameraCandidate:
    with open(transforms_path) as f:
        data = json.load(f)
    frame = data["frames"][0]
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    R, T = c2w_to_RT(c2w)
    return CameraCandidate(
        name="r_7_gt",
        file_path=frame["file_path"],
        c2w=c2w,
        R=R,
        T=T,
    )


def load_video_first_frame(video_path: Path, size: int = RENDER_SIZE) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    reader.close()
    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    tensor = tf.to_tensor(img).unsqueeze(0).cuda()
    return tensor


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
    dummy = PILtoTorch(Image.new("RGB", (width, height), tuple(int(255 * c) for c in bg)), (width, height))
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


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle)
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    axis = axis_angle / angle
    x, y, z = axis[0], axis[1], axis[2]
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        device=axis_angle.device,
        dtype=axis_angle.dtype,
    )
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


class PoseRefiner:
    """6-DOF pose refinement around a base c2w."""

    def __init__(self, base_c2w: np.ndarray, params: np.ndarray | None = None):
        self.base_c2w = base_c2w.astype(np.float64)
        self.params = np.zeros(6, dtype=np.float64) if params is None else params.astype(np.float64)

    def c2w(self) -> np.ndarray:
        return pose_params_to_c2w(self.base_c2w, self.params)

    def RT(self) -> tuple[np.ndarray, np.ndarray]:
        return c2w_to_RT(self.c2w())


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

        self._d_xyz = None
        self._d_rotation = None
        self._d_scaling = None
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
            R,
            T,
            self.camera_angle_x,
            RENDER_SIZE,
            RENDER_SIZE,
            self.white_background,
            fid=TIME_T0,
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

        combined = (psnr_val / 40.0) + ssim_val + (1.0 - lpips_val)
        return MetricBundle(psnr_val, ssim_val, lpips_val, combined)


def evaluate_pose_params(
    base_c2w: np.ndarray,
    params: np.ndarray,
    renderer: SceneRenderer,
    target: torch.Tensor,
    metrics: MetricEvaluator,
) -> tuple[MetricBundle, np.ndarray]:
    c2w = pose_params_to_c2w(base_c2w, params)
    pred = renderer.render_c2w(c2w)
    return metrics.evaluate(pred, target), c2w


def local_grid_refine(
    base_c2w: np.ndarray,
    start_params: np.ndarray,
    renderer: SceneRenderer,
    target: torch.Tensor,
    metrics: MetricEvaluator,
    rot_step_deg: float,
    trans_step: float,
    rot_range_deg: float,
    trans_range: float,
) -> tuple[np.ndarray, MetricBundle, list[dict]]:
    best_params = start_params.copy()
    best_metrics, _ = evaluate_pose_params(base_c2w, best_params, renderer, target, metrics)
    history = [{"stage": "grid_start", "params": best_params.tolist(), **best_metrics.__dict__}]

    rot_step = np.deg2rad(rot_step_deg)
    rot_range = np.deg2rad(rot_range_deg)
    rot_offsets = np.arange(-rot_range, rot_range + rot_step * 0.5, rot_step)
    trans_offsets = np.arange(-trans_range, trans_range + trans_step * 0.5, trans_step)

    for axis in range(3):
        for delta in rot_offsets:
            if abs(delta) < 1e-12:
                continue
            trial = best_params.copy()
            trial[axis] += delta
            trial_metrics, _ = evaluate_pose_params(base_c2w, trial, renderer, target, metrics)
            if trial_metrics.combined > best_metrics.combined:
                best_params = trial
                best_metrics = trial_metrics

    for axis in range(3):
        for delta in trans_offsets:
            if abs(delta) < 1e-12:
                continue
            trial = best_params.copy()
            trial[3 + axis] += delta
            trial_metrics, _ = evaluate_pose_params(base_c2w, trial, renderer, target, metrics)
            if trial_metrics.combined > best_metrics.combined:
                best_params = trial
                best_metrics = trial_metrics

    history.append({"stage": "grid_end", "params": best_params.tolist(), **best_metrics.__dict__})
    return best_params, best_metrics, history


def refine_pose(
    target: torch.Tensor,
    coarse: CameraCandidate,
    renderer: SceneRenderer,
    metrics: MetricEvaluator,
    maxiter: int,
    ftol: float,
    xtol: float,
    grid_rot_step_deg: float,
    grid_trans_step: float,
    grid_rot_range_deg: float,
    grid_trans_range: float,
) -> tuple[CameraCandidate, MetricBundle, list[dict]]:
    """Refine pose with Powell optimization, then a local grid polish.

    The CUDA Gaussian rasterizer does not backpropagate through the camera matrix, so
    refinement uses derivative-free optimization on the same combined PSNR/SSIM/LPIPS score.
    """
    refiner = PoseRefiner(coarse.c2w)
    history: list[dict] = []
    eval_count = 0

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        m, _ = evaluate_pose_params(refiner.base_c2w, params, renderer, target, metrics)
        if eval_count == 1 or eval_count % max(1, maxiter // 5) == 0:
            history.append(
                {
                    "stage": "powell",
                    "eval": eval_count,
                    "params": params.tolist(),
                    **m.__dict__,
                }
            )
        return -m.combined

    result = minimize(
        objective,
        refiner.params,
        method="Powell",
        options={"maxiter": maxiter, "xtol": xtol, "ftol": ftol},
    )

    powell_params = result.x.astype(np.float64)
    powell_metrics, powell_c2w = evaluate_pose_params(
        refiner.base_c2w, powell_params, renderer, target, metrics
    )
    history.append(
        {
            "stage": "powell_final",
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "params": powell_params.tolist(),
            **powell_metrics.__dict__,
        }
    )

    grid_params, grid_metrics, grid_history = local_grid_refine(
        refiner.base_c2w,
        powell_params,
        renderer,
        target,
        metrics,
        rot_step_deg=grid_rot_step_deg,
        trans_step=grid_trans_step,
        rot_range_deg=grid_rot_range_deg,
        trans_range=grid_trans_range,
    )
    history.extend(grid_history)

    if grid_metrics.combined >= powell_metrics.combined:
        best_params = grid_params
        best_metrics = grid_metrics
    else:
        best_params = powell_params
        best_metrics = powell_metrics

    best_c2w = pose_params_to_c2w(refiner.base_c2w, best_params)
    R_final, T_final = c2w_to_RT(best_c2w)
    refined = CameraCandidate(
        name=f"{coarse.name}_refined",
        file_path=coarse.file_path,
        c2w=best_c2w,
        R=R_final,
        T=T_final,
    )
    return refined, best_metrics, history


def coarse_search(
    target: torch.Tensor,
    candidates: list[CameraCandidate],
    renderer: SceneRenderer,
    metrics: MetricEvaluator,
) -> tuple[CameraCandidate, MetricBundle, list[dict]]:
    best_candidate = None
    best_metrics = None
    all_scores = []

    for cand in tqdm(candidates, desc="Coarse camera search", leave=False):
        pred = renderer.render_numpy(cand.R, cand.T)
        m = metrics.evaluate(pred, target)
        row = {
            "name": cand.name,
            "file_path": cand.file_path,
            "psnr": m.psnr,
            "ssim": m.ssim,
            "lpips": m.lpips,
            "combined": m.combined,
        }
        all_scores.append(row)
        if best_metrics is None or m.combined > best_metrics.combined:
            best_candidate = cand
            best_metrics = m

    all_scores.sort(key=lambda x: x["combined"], reverse=True)
    return best_candidate, best_metrics, all_scores


def pose_error(gt: CameraCandidate, est: CameraCandidate) -> dict:
    gt_center = np.linalg.inv(gt.c2w)[:3, 3]
    est_center = np.linalg.inv(est.c2w)[:3, 3]
    center_dist = float(np.linalg.norm(gt_center - est_center))
    R_rel = est.c2w[:3, :3].T @ gt.c2w[:3, :3]
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace)))
    return {
        "camera_center_distance": center_dist,
        "rotation_error_deg": rot_deg,
        "gt_camera_center": gt_center.tolist(),
        "est_camera_center": est_center.tolist(),
    }


def save_comparison_image(path: Path, target: torch.Tensor, pred: torch.Tensor, title: str) -> None:
    import torchvision

    grid = torch.cat([target, pred], dim=0)
    torchvision.utils.save_image(grid, path, nrow=2)
    with open(path.with_suffix(".txt"), "w") as f:
        f.write(title)


def run_estimation(args: argparse.Namespace) -> None:
    model_path = args.model_path.resolve()
    cfg = load_cfg(model_path)
    data_path = Path(args.data_path).resolve()
    transforms_train = data_path / "transforms_train.json"
    r7_json = args.r7_transforms.resolve()
    video_dir = args.video_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(transforms_train) as f:
        camera_angle_x = json.load(f)["camera_angle_x"]

    candidates = load_training_camera_candidates(transforms_train)
    r7_gt = load_r7_ground_truth(r7_json)

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

    results = {
        "settings": {
            "model_path": str(model_path),
            "iteration": args.iteration,
            "render_size": RENDER_SIZE,
            "time": TIME_T0,
            "num_training_cameras": len(candidates),
            "refine_maxiter": args.refine_maxiter,
            "refine_ftol": args.refine_ftol,
            "refine_xtol": args.refine_xtol,
            "grid_rot_step_deg": args.grid_rot_step_deg,
            "grid_trans_step": args.grid_trans_step,
            "grid_rot_range_deg": args.grid_rot_range_deg,
            "grid_trans_range": args.grid_trans_range,
        },
        "videos": {},
        "validation": {},
    }

    print(f"Loaded {len(candidates)} training camera candidates")
    print(f"Processing {len(video_paths)} videos from {video_dir}")

    for video_path in video_paths:
        video_id = video_path.stem
        print(f"\n=== {video_id} ===")
        target = load_video_first_frame(video_path)

        coarse_cam, coarse_m, coarse_all = coarse_search(target, candidates, renderer, metrics)
        refined_cam, refined_m, refine_hist = refine_pose(
            target,
            coarse_cam,
            renderer,
            metrics,
            maxiter=args.refine_maxiter,
            ftol=args.refine_ftol,
            xtol=args.refine_xtol,
            grid_rot_step_deg=args.grid_rot_step_deg,
            grid_trans_step=args.grid_trans_step,
            grid_rot_range_deg=args.grid_rot_range_deg,
            grid_trans_range=args.grid_trans_range,
        )

        pred_coarse = renderer.render_numpy(coarse_cam.R, coarse_cam.T)
        pred_refined = renderer.render_numpy(refined_cam.R, refined_cam.T)

        vid_out = out_dir / video_id
        vid_out.mkdir(parents=True, exist_ok=True)
        save_comparison_image(
            vid_out / "compare_refined.png",
            target,
            pred_refined,
            f"{video_id} refined PSNR={refined_m.psnr:.3f} SSIM={refined_m.ssim:.4f} LPIPS={refined_m.lpips:.4f}",
        )
        save_comparison_image(
            vid_out / "compare_coarse.png",
            target,
            pred_coarse,
            f"{video_id} coarse({coarse_cam.name}) PSNR={coarse_m.psnr:.3f} SSIM={coarse_m.ssim:.4f}",
        )

        entry = {
            "video": str(video_path),
            "coarse": {
                "camera_name": coarse_cam.name,
                "file_path": coarse_cam.file_path,
                "transform_matrix": refined_cam.c2w.tolist()
                if False
                else coarse_cam.c2w.tolist(),
                "metrics": coarse_m.__dict__,
                "top5": coarse_all[:5],
            },
            "refined": {
                "camera_name": refined_cam.name,
                "transform_matrix": refined_cam.c2w.tolist(),
                "metrics": refined_m.__dict__,
            },
            "refine_history": refine_hist,
        }
        entry["coarse"]["transform_matrix"] = coarse_cam.c2w.tolist()

        if video_id.endswith("v000") or "v000" in video_id:
            entry["ground_truth_r7"] = {
                "camera_name": r7_gt.name,
                "transform_matrix": r7_gt.c2w.tolist(),
            }
            entry["validation"] = {
                "coarse_vs_r7": pose_error(r7_gt, coarse_cam),
                "refined_vs_r7": pose_error(r7_gt, refined_cam),
                "coarse_is_r7": coarse_cam.name == "r_7",
                "metrics_at_r7_pose": metrics.evaluate(
                    renderer.render_numpy(r7_gt.R, r7_gt.T), target
                ).__dict__,
            }
            results["validation"][video_id] = entry["validation"]
            print(
                f"  v000 validation: coarse={coarse_cam.name} (is r_7? {coarse_cam.name == 'r_7'})"
            )
            print(
                f"    refined rot err: {entry['validation']['refined_vs_r7']['rotation_error_deg']:.3f} deg, "
                f"center dist: {entry['validation']['refined_vs_r7']['camera_center_distance']:.5f}"
            )

        print(
            f"  coarse: {coarse_cam.name}  PSNR={coarse_m.psnr:.2f} SSIM={coarse_m.ssim:.4f} LPIPS={coarse_m.lpips:.4f}"
        )
        print(
            f"  refined: PSNR={refined_m.psnr:.2f} SSIM={refined_m.ssim:.4f} LPIPS={refined_m.lpips:.4f}"
        )
        results["videos"][video_id] = entry

    transforms_out = {
        "camera_angle_x": camera_angle_x,
        "render_size": RENDER_SIZE,
        "time": TIME_T0,
        "frames": [],
    }
    for video_id, entry in results["videos"].items():
        transforms_out["frames"].append(
            {
                "video": video_id,
                "coarse_camera": entry["coarse"]["camera_name"],
                "transform_matrix": entry["refined"]["transform_matrix"],
                "metrics": entry["refined"]["metrics"],
            }
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "transforms_sv4d2_estimated.json").write_text(json.dumps(transforms_out, indent=2))
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
        default=REPO_ROOT / "assets" / "sv4d2",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=lego_fixed / "camera_estimation",
    )
    parser.add_argument("--refine_maxiter", type=int, default=200)
    parser.add_argument("--refine_ftol", type=float, default=1e-4)
    parser.add_argument("--refine_xtol", type=float, default=1e-4)
    parser.add_argument("--grid_rot_step_deg", type=float, default=0.5)
    parser.add_argument("--grid_trans_step", type=float, default=0.02)
    parser.add_argument("--grid_rot_range_deg", type=float, default=3.0)
    parser.add_argument("--grid_trans_range", type=float, default=0.12)
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
    run_estimation(parse_args())
