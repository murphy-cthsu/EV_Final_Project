"""Decompose existing arm Gaussians into BUCKET vs ARM-SHAFT using motion variance
+ 3D position clustering. No SAM dependency.

The intuition: bucket Gaussians move the FURTHEST (more rotation arc), while arm
shaft Gaussians (closer to the cabin pivot) move LESS. K-means in
(motion_amplitude × distance_from_pivot) 2D feature space gives clean
separation.

Output:
    runs_aux/part_assignment/bucket_mask.npy   bool (N,) — True if bucket
    runs_aux/part_assignment/sub_assignment.npy int (N,)   0=body, 1=arm-shaft, 2=bucket
    runs_aux/part_assignment/decomp_viz.png    3D scatter + cluster overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SCGS = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS))
from scene.gaussian_model import GaussianModel  # noqa: E402

CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"
PART_DIR = REPO / "runs_aux/part_assignment"


def kmeans_2cluster(x: np.ndarray, n_iter: int = 50, seed: int = 0):
    """Simple K=2 K-means. x: (N, D). Returns labels (N,) in {0, 1}."""
    rng = np.random.default_rng(seed)
    N, D = x.shape
    # Init: pick farthest pair to seed
    idx_a = rng.integers(N)
    dists = ((x - x[idx_a]) ** 2).sum(-1)
    idx_b = int(np.argmax(dists))
    centers = np.stack([x[idx_a], x[idx_b]], axis=0)  # (2, D)
    for _ in range(n_iter):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (N, 2)
        labels = d.argmin(axis=1)
        for k in range(2):
            mk = labels == k
            if mk.sum() > 0:
                centers[k] = x[mk].mean(axis=0)
    return labels, centers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm_thresh", type=float, default=0.5,
                   help="LBS arm weight > this = arm Gaussian (rest = body)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Load canonical Gaussians + global arm weights
    g = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    g.load_ply(str(CANON), og_number_points=0)
    xyz = g.get_xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    arm_w = np.load(PART_DIR / "gaussian_arm_weights.npy")
    arm_mask = arm_w > args.arm_thresh
    n_arm = int(arm_mask.sum())
    print(f"[bucket-decomp] N={N}, arm Gaussians (w>{args.arm_thresh}): {n_arm}")

    # Pivot = mean of body Gaussians' position (the static centroid)
    body_mask = ~arm_mask
    pivot = xyz[body_mask].mean(axis=0)
    print(f"[bucket-decomp] body pivot = {pivot}")

    # Per-arm-Gaussian distance to pivot
    arm_xyz = xyz[arm_mask]
    d_pivot = np.linalg.norm(arm_xyz - pivot, axis=1)  # (n_arm,)
    print(f"[bucket-decomp] arm distance to pivot: min={d_pivot.min():.3f} "
          f"median={np.median(d_pivot):.3f} max={d_pivot.max():.3f}")

    # K-means on (x, y, z, d_pivot) — 4D feature, gives spatial cluster
    feat = np.concatenate([arm_xyz, d_pivot[:, None] * 3.0], axis=-1)  # scale distance up
    labels_local, centers_local = kmeans_2cluster(feat, seed=args.seed)
    print(f"[bucket-decomp] cluster sizes: {[int((labels_local==k).sum()) for k in range(2)]}")
    print(f"[bucket-decomp] cluster centers: {centers_local}")

    # Bucket = cluster with HIGHER mean distance from pivot
    mean_d_per_cluster = [d_pivot[labels_local == k].mean() for k in range(2)]
    bucket_local_id = int(np.argmax(mean_d_per_cluster))
    print(f"[bucket-decomp] bucket cluster = {bucket_local_id} (mean d_pivot = "
          f"{mean_d_per_cluster[bucket_local_id]:.3f} vs "
          f"{mean_d_per_cluster[1-bucket_local_id]:.3f})")

    bucket_mask = np.zeros(N, dtype=bool)
    arm_idx = np.where(arm_mask)[0]
    bucket_mask[arm_idx[labels_local == bucket_local_id]] = True

    # 3-part assignment: 0=body, 1=arm-shaft, 2=bucket
    sub_assign = np.zeros(N, dtype=np.int32)
    sub_assign[arm_mask] = 1
    sub_assign[bucket_mask] = 2

    n_body = int((sub_assign == 0).sum())
    n_shaft = int((sub_assign == 1).sum())
    n_bucket = int((sub_assign == 2).sum())
    print(f"[bucket-decomp] FINAL: body={n_body} ({n_body/N*100:.1f}%)  "
          f"arm-shaft={n_shaft} ({n_shaft/N*100:.1f}%)  "
          f"bucket={n_bucket} ({n_bucket/N*100:.1f}%)")

    np.save(PART_DIR / "bucket_mask.npy", bucket_mask)
    np.save(PART_DIR / "sub_assignment.npy", sub_assign)
    print(f"[bucket-decomp] saved bucket_mask.npy + sub_assignment.npy")

    # Quick 3D viz: project to (x, z) plane
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["x-y view", "x-z view", "y-z view"]
    for ax, (i, j), tt in zip(axes, [(0, 1), (0, 2), (1, 2)], titles):
        ax.scatter(xyz[sub_assign == 0, i], xyz[sub_assign == 0, j],
                   c="#aaaaaa", s=0.5, alpha=0.3, label=f"body ({n_body})")
        ax.scatter(xyz[sub_assign == 1, i], xyz[sub_assign == 1, j],
                   c="#1f77b4", s=1.0, alpha=0.6, label=f"arm-shaft ({n_shaft})")
        ax.scatter(xyz[sub_assign == 2, i], xyz[sub_assign == 2, j],
                   c="#d62728", s=1.5, alpha=0.8, label=f"bucket ({n_bucket})")
        ax.scatter(*pivot[[i, j]], c="green", s=200, marker="*", label="pivot")
        ax.set_xlabel(["x", "y", "z"][i])
        ax.set_ylabel(["x", "y", "z"][j])
        ax.set_title(tt)
        ax.legend(fontsize=7); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(PART_DIR / "decomp_viz.png", dpi=130)
    plt.close()
    print(f"[bucket-decomp] viz -> {PART_DIR / 'decomp_viz.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
