"""Filter the lego_v2 canonical to remove baseplate Gaussians.

Cut z > z_min (default -0.15) → keeps digger, removes lego baseplate slab.
This matches SAM-2-masked SV4D supervision (which excludes baseplate).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402


def filter_gaussians_by_z(g: GaussianModel, z_min: float):
    xyz = g.get_xyz.detach()
    keep = xyz[:, 2] > z_min
    n_before = xyz.shape[0]
    n_keep = int(keep.sum())
    print(f"[filter] z > {z_min}: keep {n_keep}/{n_before} ({100*n_keep/n_before:.1f}%)")
    g._xyz = nn.Parameter(g._xyz[keep])
    g._features_dc = nn.Parameter(g._features_dc[keep])
    g._features_rest = nn.Parameter(g._features_rest[keep])
    g._scaling = nn.Parameter(g._scaling[keep])
    g._rotation = nn.Parameter(g._rotation[keep])
    g._opacity = nn.Parameter(g._opacity[keep])
    if hasattr(g, "feature") and isinstance(g.feature, nn.Parameter) and g.feature.shape[0] == n_before:
        g.feature = nn.Parameter(g.feature[keep])
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=REPO / "outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--dst", default=REPO / "outputs/custom/lego_v2_canonical_nobase/point_cloud/iteration_0/point_cloud.ply")
    p.add_argument("--z_min", type=float, default=-0.15)
    args = p.parse_args()

    g = None
    for fdim in (8, 2, 0):
        try:
            g = GaussianModel(3, fea_dim=fdim, with_motion_mask=False)
            g.load_ply(str(args.src), og_number_points=0)
            print(f"[filter] loaded src with fea_dim={fdim}, N={g.get_xyz.shape[0]}")
            break
        except Exception:
            g = None
    if g is None:
        raise RuntimeError(f"can't load {args.src}")

    filter_gaussians_by_z(g, args.z_min)

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    g.save_ply(str(args.dst))
    print(f"[filter] saved filtered canonical to {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
