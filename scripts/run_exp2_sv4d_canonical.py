"""Exp 2 runner: rebuild part assignment + train Phase 2 with SV4D-derived canonical.

Steps:
  1. Wait for canonical training (assumed done)
  2. Symlink/copy sv4d canonical to standard location
  3. Run build_part_assignments_lego_v2.py with new canonical
  4. Run train_partrigid_hier.py with new canonical + part dir
  5. Eval against both SV4D + d-3dgs clean GT
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SV4D_CANON_SRC = REPO / "outputs/custom/lego_v2_canonical_sv4d_node/point_cloud/iteration_5000/point_cloud.ply"
SV4D_CANON_DST = REPO / "outputs/custom/lego_v2_canonical_sv4d/point_cloud/iteration_0/point_cloud.ply"


def run(cmd: list[str]) -> None:
    print(f"\n[exp2] $ {' '.join(map(str, cmd))}")
    proc = subprocess.run(cmd, check=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    if not SV4D_CANON_SRC.exists():
        print(f"[exp2] FATAL: canonical not found at {SV4D_CANON_SRC}")
        print(f"[exp2] (training may still be running, or failed)")
        sys.exit(1)

    # Stage in standardized path
    SV4D_CANON_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SV4D_CANON_SRC, SV4D_CANON_DST)
    print(f"[exp2] canonical staged to {SV4D_CANON_DST}")

    # Build part assignment with new canonical (modify path in script via env-like override)
    # Easiest: import and run programmatically with overridden CANON path.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bpa", REPO / "scripts/build_part_assignments_lego_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Monkey-patch AFTER exec_module (otherwise module-level assigns overwrite)
    mod.CANON = SV4D_CANON_DST
    mod.OUT = REPO / "runs_aux/part_assignment_lego_v2_sv4d"
    mod.OUT.mkdir(parents=True, exist_ok=True)
    mod.main()

    # Train Phase 2 with the new canonical + part dir
    run([
        "/home/cthsu/miniconda3/envs/scgs/bin/python",
        "scripts/train_partrigid_hier.py",
        "--label", "lego_v2_K100_sv4dcanon",
        "--canon_ply", str(SV4D_CANON_DST),
        "--part_dir", "runs_aux/part_assignment_lego_v2_sv4d",
        "--scene_dir", "data/custom/lego_v2",
        "--v5_render_dir", "outputs/custom/lego_v2_d3dgs_ref/renders",  # smart filter still uses d-3dgs
        "--use_test_too",
        "--k_arm", "100", "--lbs_K", "6", "--lam_arap", "1.0",
        "--lam_photo_smart", "3.0", "--use_per_time_scale",
        "--iterations", "8000",
    ])

    # Eval
    run([
        "/home/cthsu/miniconda3/envs/scgs/bin/python",
        "scripts/eval_lego_v2_hier.py",
        "--label", "lego_v2_K100_sv4dcanon",
        "--save_renders",
    ])
    print(f"[exp2] DONE")


if __name__ == "__main__":
    raise SystemExit(main())
