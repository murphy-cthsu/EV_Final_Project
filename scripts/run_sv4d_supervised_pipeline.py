"""End-to-end driver: D-NeRF scene → SV4D-2.0-supervised SC-GS training scene.

Stages (each idempotent; --skip_<stage> lets you resume):
    1. PREPARE: sample 21 frames from the original D-NeRF train split (camera 0
       trajectory through time) and composite RGBA→white-RGB at the SV4D input
       resolution (576x576). Output: <scratch>/sv4d_input/frame_NNNN.png.
    2. SV4D:    run SV4D 2.0 inference on the prepared input video.
                Output: <scratch>/sv4d_output/sv4d2/{base}_v{N}.mp4 + sidecar json.
    3. CONVERT: produce a D-NeRF-format scene from the SV4D output.
                Output: data/dnerf_sv4d/<scene>/{transforms_*.json, train/, test/}
    4. TRAIN:   emit (or run) the SC-GS training command for that scene.
                Output: outputs/<scene>_scgs_sv4d_default/

The SV4D stage requires a GPU with enough VRAM. The PREPARE and CONVERT stages
are CPU-only and tested by the unit tests in tests/test_sv4d_pipeline.py.

Usage (single scene, full pipeline):
    /home/cthsu/miniconda3/envs/scgs/bin/python scripts/run_sv4d_supervised_pipeline.py \\
        --scene jumpingjacks \\
        --dnerf_root data/dnerf \\
        --scratch_root outputs/sv4d_supervised \\
        --out_data_root data/dnerf_sv4d \\
        --sv4d_python /home/cthsu/miniconda3/envs/motionprior/bin/python

Skip the heavy GPU stage but rebuild the converter output:
    ... --skip_sv4d        (assumes <scratch>/sv4d_output exists)

CPU smoke test (no GPU at all):
    ... --skip_sv4d --skip_train --fake_sv4d
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motionprior.integration.vgm import (  # noqa: E402
    SV4D2Adapter, SV4D2_AZIMUTHS_DEG, SV4DResult,
)
from scripts.sv4d_to_dnerf import convert_sv4d_to_dnerf, parse_sv4d_output_dir  # noqa: E402


# --------------------------------------------------------------------- #
# Stage 1: PREPARE — D-NeRF train cam 0 video → SV4D input dir
# --------------------------------------------------------------------- #
def prepare_input_video(
    dnerf_scene_dir: Path,
    out_dir: Path,
    n_frames: int = 21,
    img_size: int = 576,
    cam_indices: list[int] | None = None,
) -> dict:
    """Subsample the original D-NeRF train split into a 21-frame video at the
    requested resolution, composited to white background (SV4D's preferred input).

    Strategy: read transforms_train.json, group frames by their (cam idx).
    For each scene the original D-NeRF dataset has multiple cameras × many
    timesteps. We use camera 0 (the first appearance of each timestamp) by
    default, sampled to 21 frames evenly spaced in time.
    """
    transforms = json.loads((dnerf_scene_dir / "transforms_train.json").read_text())
    frames = transforms["frames"]
    # Sort by time, then take the first occurrence of each unique time.
    by_time: dict[float, dict] = {}
    for f in frames:
        t = f["time"]
        # Take the camera with the smallest file_path index per timestamp
        if t not in by_time or f["file_path"] < by_time[t]["file_path"]:
            by_time[t] = f
    times_sorted = sorted(by_time.keys())
    if len(times_sorted) < n_frames:
        raise ValueError(
            f"{dnerf_scene_dir.name} has only {len(times_sorted)} unique timestamps; "
            f"need >= {n_frames}"
        )
    # Evenly sample n_frames timestamps
    idxs = np.linspace(0, len(times_sorted) - 1, n_frames).round().astype(int)
    sampled = [by_time[times_sorted[i]] for i in idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, frame_meta in enumerate(sampled):
        rel = frame_meta["file_path"]
        # D-NeRF format: file_path is `./train/r_NNN` without extension; PNG is added
        img_path = dnerf_scene_dir / f"{rel.lstrip('./')}.png"
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("RGBA")
        # Composite onto white
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img_rgb = bg.resize((img_size, img_size), Image.LANCZOS)
        out_path = out_dir / f"frame_{i:04d}.png"
        img_rgb.save(out_path)
        written.append(out_path)

    sidecar = {
        "scene_dir": str(dnerf_scene_dir),
        "n_frames": n_frames,
        "img_size": img_size,
        "sampled_timestamps": [float(times_sorted[i]) for i in idxs],
        "source_file_paths": [f["file_path"] for f in sampled],
    }
    (out_dir / "prepare_metadata.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


# --------------------------------------------------------------------- #
# Stage 2: SV4D inference (subprocess) or FakeSV4D for CPU smoke tests
# --------------------------------------------------------------------- #
def run_sv4d(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    sv4d_python: str,
    variant: str = "sv4d2",
    img_size: int = 576,
    n_frames: int = 21,
    seed: int = 23,
    encoding_t: int = 1,
    decoding_t: int = 1,
    device: str = "cuda",
) -> SV4DResult:
    adapter = SV4D2Adapter(
        checkpoint=checkpoint,
        variant=variant,
        img_size=img_size,
        n_frames=n_frames,
        seed=seed,
        encoding_t=encoding_t,
        decoding_t=decoding_t,
        device=device,
        python_exe=sv4d_python,
    )
    return adapter.run(input_video_dir=input_dir, output_dir=output_dir)


def fake_sv4d_result(input_dir: Path, output_dir: Path, variant: str = "sv4d2",
                     img_size: int = 576, n_frames: int = 21) -> SV4DResult:
    """For CPU smoke tests: construct a fake SV4DResult that mimics what
    SV4D would produce. Uses the prepared input frames for all views (so the
    downstream converter sees a plausibly-shaped tensor)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    azimuths = list(SV4D2_AZIMUTHS_DEG[variant])
    V = len(azimuths)
    input_frames = sorted(input_dir.glob("*.png"))[:n_frames]
    if len(input_frames) < n_frames:
        raise RuntimeError(f"prepared input has {len(input_frames)} frames < {n_frames}")
    stack = []
    for p in input_frames:
        arr = np.asarray(Image.open(p).convert("RGB"))
        if arr.shape[:2] != (img_size, img_size):
            arr = np.asarray(Image.fromarray(arr).resize((img_size, img_size), Image.LANCZOS))
        stack.append(arr)
    one_view = np.stack(stack, axis=0)  # (T, H, W, 3)
    frames = np.broadcast_to(one_view[None], (V, n_frames, img_size, img_size, 3)).copy()
    result = SV4DResult(
        frames=frames,
        azimuths_deg=azimuths,
        elevations_deg=[0.0] * V,
        variant=variant,
        seed=-1,
        img_size=img_size,
        n_frames=n_frames,
        output_dir=output_dir,
        metadata={"fake": True},
    )
    (output_dir / "FAKE_SV4D.json").write_text(json.dumps({
        "warning": "These frames are NOT a real SV4D output. Generated by "
        "fake_sv4d_result for CPU smoke testing the downstream converter.",
        "input_dir": str(input_dir),
        "shape": list(frames.shape),
    }, indent=2))
    return result


# --------------------------------------------------------------------- #
# Stage 4: emit SC-GS training command
# --------------------------------------------------------------------- #
def emit_scgs_command(sv4d_scene_dir: Path, scgs_output_dir: Path,
                      iterations: int = 30000, resolution: int = 1,
                      img_size: int = 576) -> list[str]:
    """The SC-GS train_gui.py invocation that would consume this scene.
    Returns the argv as a list — caller can `subprocess.run` it or just print.
    """
    cmd = [
        "python",
        "third_party/SC-GS/train_gui.py",
        "--source_path", str(sv4d_scene_dir.resolve()),
        "--model_path", str(scgs_output_dir.resolve()),
        "--deform_type", "node",
        "--node_num", "512",
        "--hyper_dim", "8",
        "--is_blender",
        "--eval",
        "--gt_alpha_mask_as_scene_mask",
        "--local_frame",
        "--resolution", str(resolution),
        "--W", str(img_size),
        "--H", str(img_size),
        "--iterations", str(iterations),
    ]
    return cmd


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True,
                   help="Scene name under --dnerf_root (e.g. jumpingjacks)")
    p.add_argument("--dnerf_root", type=Path, default=REPO_ROOT / "data" / "dnerf")
    p.add_argument("--scratch_root", type=Path,
                   default=REPO_ROOT / "outputs" / "sv4d_supervised",
                   help="Intermediate dir for SV4D input/output mp4s")
    p.add_argument("--out_data_root", type=Path,
                   default=REPO_ROOT / "data" / "dnerf_sv4d",
                   help="Where the SV4D-supervised D-NeRF-shaped scenes are written")
    p.add_argument("--scgs_output_root", type=Path,
                   default=REPO_ROOT / "outputs",
                   help="Where SC-GS would write its run output")
    p.add_argument("--sv4d_python",
                   default="/home/cthsu/miniconda3/envs/motionprior/bin/python",
                   help="Python interpreter for the SV4D subprocess")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "sv4d2.safetensors")
    p.add_argument("--variant", default="sv4d2", choices=["sv4d2", "sv4d2_8views"])
    p.add_argument("--n_frames", type=int, default=21)
    p.add_argument("--img_size", type=int, default=576)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--encoding_t", type=int, default=1, help="SV4D VRAM knob (1 = lowest)")
    p.add_argument("--decoding_t", type=int, default=1, help="SV4D VRAM knob (1 = lowest)")
    p.add_argument("--orbit_radius", type=float, default=4.0,
                   help="Object orbit radius used to synthesize transforms_train.json")
    p.add_argument("--iterations", type=int, default=30000,
                   help="SC-GS iters (only used in the emitted command)")

    p.add_argument("--skip_prepare", action="store_true")
    p.add_argument("--skip_sv4d", action="store_true")
    p.add_argument("--skip_convert", action="store_true")
    p.add_argument("--skip_train", action="store_true",
                   help="Only emit the SC-GS training command, don't run it")
    p.add_argument("--run_train", action="store_true",
                   help="Actually launch SC-GS training (the default just prints the command)")
    p.add_argument("--fake_sv4d", action="store_true",
                   help="Skip real SV4D inference; fabricate a fake SV4DResult to "
                        "exercise the downstream converter (CPU smoke test).")
    p.add_argument("--overwrite", action="store_true",
                   help="Clobber existing output dirs.")
    args = p.parse_args()

    scene_dir = args.dnerf_root / args.scene
    if not scene_dir.is_dir():
        print(f"[pipeline] FATAL: scene dir {scene_dir} not found")
        return 2

    scratch = args.scratch_root / args.scene
    sv4d_input_dir = scratch / "sv4d_input"
    sv4d_output_dir = scratch / "sv4d_output"
    converted_scene_dir = args.out_data_root / args.scene
    scgs_output_dir = args.scgs_output_root / f"{args.scene}_scgs_sv4d_default"

    # Stage 1
    if args.skip_prepare and sv4d_input_dir.exists():
        print(f"[pipeline] [1/4] skip-prepare; reusing {sv4d_input_dir}")
    else:
        print(f"[pipeline] [1/4] PREPARE: {scene_dir} -> {sv4d_input_dir}")
        meta = prepare_input_video(scene_dir, sv4d_input_dir,
                                    n_frames=args.n_frames, img_size=args.img_size)
        print(f"[pipeline]   wrote {args.n_frames} frames @ {args.img_size}x{args.img_size}")

    # Stage 2
    if args.skip_sv4d:
        if args.fake_sv4d:
            print(f"[pipeline] [2/4] FAKE_SV4D: fabricating SV4DResult for smoke test")
            sv4d_result = fake_sv4d_result(
                sv4d_input_dir, sv4d_output_dir, variant=args.variant,
                img_size=args.img_size, n_frames=args.n_frames,
            )
        else:
            print(f"[pipeline] [2/4] skip-sv4d; parsing existing {sv4d_output_dir}")
            sv4d_result = parse_sv4d_output_dir(sv4d_output_dir, variant=args.variant)
    else:
        if not args.checkpoint.is_file():
            print(f"[pipeline] FATAL: SV4D checkpoint missing at {args.checkpoint}")
            return 2
        print(f"[pipeline] [2/4] SV4D: {sv4d_input_dir} -> {sv4d_output_dir}")
        sv4d_result = run_sv4d(
            sv4d_input_dir, sv4d_output_dir,
            checkpoint=args.checkpoint,
            sv4d_python=args.sv4d_python,
            variant=args.variant,
            img_size=args.img_size,
            n_frames=args.n_frames,
            seed=args.seed,
            encoding_t=args.encoding_t,
            decoding_t=args.decoding_t,
        )
    print(f"[pipeline]   frames shape: {sv4d_result.frames.shape}")

    # Stage 3
    if args.skip_convert and converted_scene_dir.exists():
        print(f"[pipeline] [3/4] skip-convert; reusing {converted_scene_dir}")
    else:
        print(f"[pipeline] [3/4] CONVERT: {sv4d_result.output_dir} -> {converted_scene_dir}")
        meta = convert_sv4d_to_dnerf(
            sv4d_result,
            orig_scene_dir=scene_dir,
            out_dir=converted_scene_dir,
            orbit_radius=args.orbit_radius,
            overwrite=args.overwrite,
        )
        print(f"[pipeline]   wrote {meta['n_train_images_written']} train PNGs")

    # Stage 4
    train_cmd = emit_scgs_command(
        converted_scene_dir, scgs_output_dir,
        iterations=args.iterations, img_size=args.img_size,
    )
    print(f"[pipeline] [4/4] SC-GS command:")
    print("    " + " \\\n    ".join(train_cmd))
    if args.run_train and not args.skip_train:
        print(f"[pipeline] launching SC-GS training in {scgs_output_dir}")
        scgs_output_dir.mkdir(parents=True, exist_ok=True)
        log = scgs_output_dir / "train.log"
        with log.open("w") as f:
            f.write(f"# cmd: {' '.join(train_cmd)}\n")
            f.flush()
            proc = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT,
                                  cwd=str(REPO_ROOT))
        print(f"[pipeline]   exit {proc.returncode}; log {log}")
        return proc.returncode

    print("[pipeline] done. To launch training: re-run with --run_train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
