"""SAM-2 video-predictor foreground masks for multi-view videos -> RGBA dataset.

Builds a parallel D-NeRF-format scene dir where each train PNG is RGBA, with
alpha = SAM-2 foreground mask. This unlocks SC-GS's dynamic-mask path
(--gt_alpha_mask_as_dynamic_mask --gs_with_motion_mask), which is the
established fix when an unmasked SC-GS run fails to learn motion because
background reconstruction dominates the loss.

Pipeline per video:
    decode 21 frames -> tmp/<view>/000.jpg ... 020.jpg
    SAM-2.init_state(tmp/<view>)
    seed center positive point on frame 0
    propagate forward -> per-frame mask
    write <out_dir>/train/r_<flat_idx>.png as RGBA (alpha = mask*255)

Notes:
    * Default prompt = positive point at image center on frame 0. For your
      object-centered orbit captures (cameras at radius ~4 looking at origin),
      this hits the foreground reliably. If a view fails, pass a different
      (x,y) via --prompt v=x,y,... .
    * Writes alongside the existing transforms_train.json so the camera
      poses + FoV stay identical. SC-GS's readNerfSyntheticInfo reads the
      alpha channel directly when the PNG is RGBA.

Usage:
    /home/cthsu/miniconda3/envs/motionprior/bin/python scripts/sam2_seg_multiview.py \\
        --src_dir /mnt/HDD_1/cthsu/multiview_videos \\
        --orig_scene_dir data/custom/scene00 \\
        --out_dir data/custom/scene00_masked \\
        --checkpoint checkpoints/sam2_hiera_large.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM2_PATH = REPO_ROOT / "third_party" / "sam2"
if str(SAM2_PATH) not in sys.path:
    sys.path.insert(0, str(SAM2_PATH))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def parse_prompts(spec: str | None, n_views: int) -> dict[int, list[tuple[int, int]]]:
    """Parse --prompt 'v=x,y;...' into {view_idx: [(x, y), ...]}.

    Missing views default to a single image-center point at run time.
    """
    out: dict[int, list[tuple[int, int]]] = {}
    if not spec:
        return out
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        v_str, coords = chunk.split("=", 1)
        v = int(v_str.strip())
        pts: list[tuple[int, int]] = []
        for pair in coords.split("|"):
            x, y = pair.split(",")
            pts.append((int(x), int(y)))
        out[v] = pts
    return out


def extract_video_frames(video_path: Path, out_dir: Path) -> tuple[int, int, int]:
    """Decode mp4 -> 000.jpg ... NNN.jpg. Returns (n_frames, H, W)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(iio.imread(video_path))
    if arr.ndim != 4:
        raise ValueError(f"unexpected video shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    T, H, W, _ = arr.shape
    for t in range(T):
        Image.fromarray(arr[t]).save(out_dir / f"{t:03d}.jpg", quality=95)
    return T, H, W


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", type=Path, required=True,
                   help="Directory with the source mp4s + camera_pos.json")
    p.add_argument("--orig_scene_dir", type=Path, required=True,
                   help="Existing D-NeRF-format scene to mirror (uses its transforms_train.json)")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output scene dir (will be created)")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "checkpoints" / "sam2_hiera_large.pt")
    p.add_argument("--model_cfg", default="configs/sam2/sam2_hiera_l.yaml")
    p.add_argument("--prompt", type=str, default=None,
                   help="Per-view prompts: '0=x,y;1=x,y|x,y;...'. "
                        "Default: image center for every view.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src_dir = args.src_dir.resolve()
    orig_scene_dir = args.orig_scene_dir.resolve()
    out_dir = args.out_dir.resolve()

    cam_json = src_dir / "camera_pos.json"
    if not cam_json.is_file():
        raise FileNotFoundError(cam_json)
    views = sorted(json.loads(cam_json.read_text())["views"],
                   key=lambda v: int(v["view_index"]))
    V = len(views)

    orig_transforms = orig_scene_dir / "transforms_train.json"
    if not orig_transforms.is_file():
        raise FileNotFoundError(orig_transforms)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "train"
    train_dir.mkdir(exist_ok=True)
    debug_dir = out_dir / "_mask_debug"
    debug_dir.mkdir(exist_ok=True)

    print(f"[sam2] checkpoint: {args.checkpoint}")
    print(f"[sam2] device:     {args.device}")
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint),
                                           device=args.device)

    prompts = parse_prompts(args.prompt, V)
    print(f"[sam2] per-view prompts (custom): {prompts if prompts else 'none — using image center'}")

    summary: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmproot:
        tmproot = Path(tmproot)
        for view_obj in views:
            v_idx = int(view_obj["view_index"])
            video_path = src_dir / view_obj["video"]
            if not video_path.is_file():
                raise FileNotFoundError(video_path)

            view_tmp = tmproot / f"view_{v_idx}"
            T, H, W = extract_video_frames(video_path, view_tmp)
            print(f"[sam2] view {v_idx}: {video_path.name} -> {T} frames ({W}x{H})")

            state = predictor.init_state(str(view_tmp))
            pts = prompts.get(v_idx, [(W // 2, H // 2)])
            points = np.asarray(pts, dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)  # all positive
            print(f"[sam2]   seed points (frame 0, +1): {pts}")
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=points, labels=labels, clear_old_points=True,
            )

            # Propagate forward through all frames
            masks_per_frame: dict[int, np.ndarray] = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                # mask_logits shape: (n_obj, 1, H, W) — we have 1 obj
                m = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                masks_per_frame[int(frame_idx)] = m

            if len(masks_per_frame) != T:
                raise RuntimeError(
                    f"view {v_idx}: got {len(masks_per_frame)} masks for {T} frames"
                )

            # Read original RGB frames and write RGBA PNGs at the SC-GS flat-index path
            rgb = np.asarray(iio.imread(video_path))
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            assert rgb.shape == (T, H, W, 3)

            fg_frac_per_t = []
            for t in range(T):
                m = masks_per_frame[t]
                if m.shape != (H, W):
                    # SAM-2 returns mask at model resolution; resize to original
                    import torch.nn.functional as F
                    m_t = torch.from_numpy(m)[None, None].float()
                    m = F.interpolate(m_t, (H, W), mode="nearest")[0, 0].numpy().astype(np.uint8)
                alpha = (m * 255).astype(np.uint8)
                rgba = np.concatenate([rgb[t], alpha[..., None]], axis=-1)
                flat_idx = v_idx * T + t
                Image.fromarray(rgba, mode="RGBA").save(train_dir / f"r_{flat_idx:05d}.png")
                fg_frac_per_t.append(float(m.mean()))

            # Debug overlay for frame 0 + frame T//2
            for t in (0, T // 2, T - 1):
                m = masks_per_frame[t]
                if m.shape != (H, W):
                    import torch.nn.functional as F
                    m_t = torch.from_numpy(m)[None, None].float()
                    m = F.interpolate(m_t, (H, W), mode="nearest")[0, 0].numpy().astype(np.uint8)
                overlay = rgb[t].copy()
                overlay[m == 1] = (0.4 * overlay[m == 1].astype(np.float32)
                                   + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
                Image.fromarray(overlay).save(debug_dir / f"view{v_idx}_t{t:02d}_overlay.png")

            fg_min, fg_max = min(fg_frac_per_t), max(fg_frac_per_t)
            print(f"[sam2]   fg fraction across frames: [{fg_min:.3f}, {fg_max:.3f}]")
            summary.append({
                "view_idx": v_idx,
                "n_frames": T,
                "fg_frac_min": fg_min,
                "fg_frac_max": fg_max,
                "prompt": pts,
            })

    # Reuse the original transforms_train.json wholesale (filenames are identical)
    shutil.copy2(orig_transforms, out_dir / "transforms_train.json")

    meta = {
        "source": "sam2_seg_multiview",
        "src_dir": str(src_dir),
        "orig_scene_dir": str(orig_scene_dir),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "views": summary,
    }
    (out_dir / "sam2_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[sam2] done. wrote masked scene to {out_dir}")
    print(f"[sam2]      debug overlays in {debug_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
