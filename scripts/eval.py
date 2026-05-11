"""Evaluation for a trained MotionPrior-4DGS run.

Computes PSNR/SSIM/LPIPS on held-out novel-view + novel-time renderings, plus
our new metric: inter-part angular consistency. Emits a single JSON file per
run for downstream ablation-table aggregation.

Architecture:
  * Image metrics (PSNR/SSIM/LPIPS) require rendering, which needs GPU + SC-GS
    rasterizer. Stubbed here; runs on the pod.
  * Inter-part angular consistency is pure tensor math; runs on CPU.
  * Floater count requires COLMAP/sparse-points -- delegated to a separate
    helper; this script consumes a precomputed value if present.

Usage:
    python scripts/eval.py \\
        --run_dir outputs/dnerf_jumpingjacks_ours \\
        --scene_root data/dnerf/jumpingjacks \\
        --part_labels_path runs_aux/jumpingjacks_part_labels.pt \\
        --gaussian_trajectory_path outputs/dnerf_jumpingjacks_ours/positions.pt \\
        --out_json outputs/dnerf_jumpingjacks_ours/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval a MotionPrior-4DGS run.")
    p.add_argument("--run_dir", required=True, help="Output dir from training.")
    p.add_argument("--scene_root", required=True, help="Dataset root for the scene.")
    p.add_argument(
        "--part_labels_path", default=None,
        help="Per-Gaussian part labels (.pt). Needed for inter-part angular consistency.",
    )
    p.add_argument(
        "--gaussian_trajectory_path", default=None,
        help="Per-frame Gaussian positions (T, N, 3) saved during training. Needed for "
             "inter-part angular consistency and for the sim-bridge URDF emission.",
    )
    p.add_argument(
        "--inter_part_pair", nargs=2, type=int, default=None,
        metavar=("K1", "K2"),
        help="Which part pair to report angular consistency on. Default: every pair.",
    )
    p.add_argument(
        "--out_json", default=None,
        help="Where to write metrics JSON. Default: <run_dir>/eval.json.",
    )
    p.add_argument(
        "--render_metrics", action="store_true",
        help="Compute PSNR/SSIM/LPIPS by re-rendering on GPU (requires SC-GS env). "
             "Without this flag, image metrics are read from the run_dir if a "
             "saved render-metrics file exists.",
    )
    p.add_argument(
        "--emit_urdf", action="store_true",
        help="Also emit URDF + Genesis YAML to <run_dir>/sim/ for downstream demos.",
    )
    return p.parse_args()


def compute_inter_part_metrics(
    positions_path: Path,
    part_labels_path: Path,
    static_label: int = -1,
    pair: tuple[int, int] | None = None,
) -> dict:
    """Compute inter-part angular consistency."""
    import torch
    from motionprior.metrics.articulation import (
        inter_part_angle_trajectory,
        angular_consistency_score,
    )

    positions = torch.load(positions_path).float()
    parts = torch.load(part_labels_path).long()

    if positions.dim() != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape (T, N, 3); got {tuple(positions.shape)}"
        )
    if parts.shape[0] != positions.shape[1]:
        raise ValueError(
            f"part_labels length {parts.shape[0]} != positions N {positions.shape[1]}"
        )

    unique = sorted(int(p) for p in set(parts.tolist()) if int(p) != static_label)
    num_parts = max(unique) + 1 if unique else 0

    if not unique:
        return {
            "inter_part_angular_consistency_by_pair": [],
            "inter_part_angular_consistency_mean": float("nan"),
            "num_parts": 0,
        }

    pairs = [(pair[0], pair[1])] if pair else [
        (a, b) for i, a in enumerate(unique) for b in unique[i + 1:]
    ]
    by_pair = []
    for (k_a, k_b) in pairs:
        traj = inter_part_angle_trajectory(
            positions, parts, num_parts=num_parts,
            pair=(k_a, k_b), static_label=static_label,
        )
        score = angular_consistency_score(traj)
        by_pair.append(
            {
                "pair": [int(k_a), int(k_b)],
                "consistency": float(score.item()),
                "angle_range_deg": [
                    float(traj.min().item() * 180.0 / math.pi),
                    float(traj.max().item() * 180.0 / math.pi),
                ],
            }
        )
    mean_score = (
        sum(p["consistency"] for p in by_pair) / len(by_pair)
        if by_pair else float("nan")
    )
    return {
        "inter_part_angular_consistency_by_pair": by_pair,
        "inter_part_angular_consistency_mean": mean_score,
        "num_parts": len(unique),
    }


def maybe_compute_render_metrics(run_dir: Path, scene_root: Path) -> dict:
    metrics_file = run_dir / "render_metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return {
        "psnr": float("nan"),
        "ssim": float("nan"),
        "lpips": float("nan"),
        "note": "render metrics not present; run with --render_metrics on GPU",
    }


def maybe_emit_urdf(run_dir: Path, positions_path: Path, part_labels_path: Path) -> dict:
    import torch
    from motionprior.integration import (
        extract_parts_from_4dgs,
        emit_urdf,
        emit_genesis_yaml,
    )
    positions = torch.load(positions_path).float()
    parts = torch.load(part_labels_path).long()
    parts_list = extract_parts_from_4dgs(positions, parts)
    sim_dir = run_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = emit_urdf(parts_list, sim_dir / "scene.urdf")
    yaml_path = emit_genesis_yaml(parts_list, sim_dir / "scene.yaml")
    return {
        "urdf_path": str(urdf_path),
        "genesis_yaml_path": str(yaml_path),
        "num_articulated_parts": len(parts_list),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    scene_root = Path(args.scene_root)
    out_json = Path(args.out_json) if args.out_json else run_dir / "eval.json"

    report: dict = {
        "run_dir": str(run_dir),
        "scene_root": str(scene_root),
    }

    report["image_metrics"] = maybe_compute_render_metrics(run_dir, scene_root)

    if args.gaussian_trajectory_path and args.part_labels_path:
        positions_path = Path(args.gaussian_trajectory_path)
        part_labels_path = Path(args.part_labels_path)
        if positions_path.exists() and part_labels_path.exists():
            pair = tuple(args.inter_part_pair) if args.inter_part_pair else None
            report["articulation_metrics"] = compute_inter_part_metrics(
                positions_path, part_labels_path, pair=pair,
            )
            if args.emit_urdf:
                report["sim_bridge"] = maybe_emit_urdf(
                    run_dir, positions_path, part_labels_path,
                )
        else:
            report["articulation_metrics"] = {
                "note": f"missing positions ({positions_path.exists()}) or "
                        f"part labels ({part_labels_path.exists()})",
            }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
