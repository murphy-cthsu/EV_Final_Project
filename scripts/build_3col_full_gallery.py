"""3-column visual gallery (full 5 cams × 21 times = 105 frames).

Columns per frame:
  [clean ref nobase @ matched fid]  |  [SV4D GT]  |  [part-rigid LBS render]

Outputs:
  runs_aux/gallery_3col_full/
    ├ tiles/r_v{V}_t{T:02d}.png   (105 PNG)
    ├ gallery_v{V}.gif             (5 GIFs, one per view, 21 frames each)
    ├ contact_sheet_t0.png         (5 views @ t=0 side-by-side)
    └ all_views_animation.gif      (one panel-of-5 GIF, 21 frames)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SCGS_ROOT = REPO / "third_party" / "SC-GS"
sys.path.insert(0, str(SCGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera as SCGSCamera  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402

from PIL import Image, ImageDraw, ImageFont


CANON = REPO / "outputs/custom/canonical_static_node/point_cloud/iteration_5000/point_cloud.ply"


def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / theta
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    th = theta[..., None]
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def render_partrigid(gaussians, xyz_canon, arm_trans, arm_aa, arm_pivot, arm_weights,
                     v_R, v_T, fov_x, FovY, H, W, t_idx, T_train, pipe, bg, N):
    t_lo = min(t_idx, T_train - 1)
    R_t = axis_angle_to_matrix_np(arm_aa[t_lo:t_lo+1])[0]
    rel = xyz_canon - arm_pivot
    rotated = rel @ R_t.T
    arm_xyz = rotated + arm_pivot + arm_trans[t_lo]
    new_xyz = arm_weights[:, None] * arm_xyz + (1 - arm_weights[:, None]) * xyz_canon
    d_xyz_np = new_xyz - xyz_canon

    d_xyz_t = torch.from_numpy(d_xyz_np).float().cuda()
    d_rotation = torch.zeros(N, 4, device="cuda") - torch.tensor([1, 0, 0, 0], device="cuda")
    d_scaling = torch.zeros(N, 3, device="cuda")
    dummy = torch.zeros(3, H, W, dtype=torch.float32)
    cam = SCGSCamera(colmap_id=0, R=v_R, T=v_T, FoVx=fov_x, FoVy=FovY,
                     image=dummy, gt_alpha_mask=None,
                     image_name="x", uid=0, fid=torch.tensor(0.0).float())
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg,
                     d_xyz=d_xyz_t, d_rotation=d_rotation, d_scaling=d_scaling,
                     d_rot_as_res=True)
    return torch.clamp(pkg["render"], 0, 1).cpu().numpy().transpose(1, 2, 0)


def load_sv4d_rgb(p: Path):
    rgba = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + 1.0 * (1 - a)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--partrigid_label", default="lbs_photo1")
    p.add_argument("--sv4d_meta", default=REPO / "data/custom/scene00_masked/transforms_train.json")
    p.add_argument("--sv4d_dir",  default=REPO / "data/custom/scene00_masked/train")
    p.add_argument("--clean_dir", default=REPO / "runs_aux/clean_gt_fine_nobase/renders")
    p.add_argument("--matching_map", default=REPO / "runs_aux/alignment_A_nobase/matching_map.json")
    p.add_argument("--out_dir",   default=REPO / "runs_aux/gallery_3col_full")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "tiles").mkdir(exist_ok=True)

    # Load partrigid state
    state = np.load(REPO / f"outputs/custom/partrigid_{args.partrigid_label}/partrigid_state.npz",
                    allow_pickle=True)
    arm_trans = state["arm_trans"]; arm_aa = state["arm_aa"]
    arm_pivot = state["arm_pivot"]
    if "arm_weights" in state.files:
        arm_weights = state["arm_weights"]
        kind = "LBS"
    else:
        arm_weights = (state["part_id"] == 0).astype(np.float32)
        kind = "hard"
    T_train = arm_trans.shape[0]

    gaussians = GaussianModel(3, fea_dim=2, with_motion_mask=False)
    gaussians.load_ply(str(CANON), og_number_points=0)
    xyz_canon = gaussians.get_xyz.detach().cpu().numpy()
    N = xyz_canon.shape[0]
    print(f"[3col] partrigid {args.partrigid_label} ({kind}), canon N={N}")

    meta = json.loads(Path(args.sv4d_meta).read_text())
    match = json.loads(Path(args.matching_map).read_text())
    fov_x = meta["camera_angle_x"]
    T_full = max(int(f["frame_idx"]) for f in meta["frames"]) + 1
    H = W = 576
    FovY = focal2fov(fov2focal(fov_x, W), H)

    cams_by_view = {}
    for f in meta["frames"]:
        v = int(f["view_idx"])
        if v in cams_by_view: continue
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        M = np.linalg.inv(c2w)
        R = -np.transpose(M[:3, :3]); R[:, 0] = -R[:, 0]
        Tr = -M[:3, 3]
        cams_by_view[v] = (R, Tr)
    V = len(cams_by_view)

    parser_pipe = _A()
    pp = PipelineParams(parser_pipe)
    pipe = pp.extract(parser_pipe.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Build tiles per (v, t)
    tiles_by_view = {v: [] for v in range(V)}
    pad = 6
    label_h = 30
    print(f"[3col] rendering 3-col tiles for {V * T_full} (v, t) pairs…")
    for f in meta["frames"]:
        v = int(f["view_idx"]); t = int(f["frame_idx"])
        Rv, Tv = cams_by_view[v]

        # Column 1: clean ref nobase @ matched fid
        key = f"v{v}_t{t}"
        fi = match[key]["best_fid_idx"] if key in match else 0
        fid_val = match[key]["best_fid_val"] if key in match else 0
        clean_p = Path(args.clean_dir) / f"r_v{v}_f{fi:03d}.png"
        clean = np.asarray(Image.open(clean_p).convert("RGB"), dtype=np.float32) / 255.0
        if clean.shape[:2] != (H, W):
            clean = np.asarray(Image.fromarray((clean * 255).astype(np.uint8)).resize((W, H)),
                                dtype=np.float32) / 255

        # Column 2: SV4D GT
        gt_p = Path(args.sv4d_dir) / f"{Path(f['file_path']).name}.png"
        gt = load_sv4d_rgb(gt_p)
        if gt.shape[:2] != (H, W):
            gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((W, H)),
                             dtype=np.float32) / 255

        # Column 3: part-rigid LBS render
        pred = render_partrigid(gaussians, xyz_canon, arm_trans, arm_aa, arm_pivot,
                                 arm_weights, Rv, Tv, fov_x, FovY, H, W, t, T_train,
                                 pipe, bg, N)

        # Concat
        sep = np.ones((H, pad, 3))
        tile = np.concatenate([clean, sep, gt, sep, pred], axis=1)
        tile_pil = Image.fromarray((tile * 255).astype(np.uint8))
        full = Image.new("RGB", (tile_pil.width, tile_pil.height + label_h), (255, 255, 255))
        d = ImageDraw.Draw(full)
        col_w = W + pad
        d.text((W // 2 - 130, 6), f"Clean ref (D-NeRF) @ fid={fid_val:.2f}", fill="black", font=font)
        d.text((col_w + W // 2 - 50, 6), "SV4D GT (VGM)", fill="black", font=font)
        d.text((2 * col_w + W // 2 - 110, 6), f"Ours (Part-rigid {kind})", fill="black", font=font)
        d.text((full.width - 100, 6), f"v={v}  t={t:02d}", fill="black", font=font)
        full.paste(tile_pil, (0, label_h))
        full.save(out / "tiles" / f"r_v{v}_t{t:02d}.png")
        tiles_by_view[v].append(full)
    print(f"[3col] tiles done")

    # Per-view GIFs
    for v in range(V):
        gif_p = out / f"gallery_v{v}.gif"
        tiles_by_view[v][0].save(gif_p, save_all=True, append_images=tiles_by_view[v][1:],
                                  duration=350, loop=0)
        print(f"[3col] view {v}: {len(tiles_by_view[v])} frames -> {gif_p.name}")

    # Contact sheet @ t=0 (5 views stacked vertically)
    t0_tiles = [tiles_by_view[v][0] for v in range(V)]
    cs_w = t0_tiles[0].width
    cs_h = sum(t.height for t in t0_tiles) + (V - 1) * 4
    cs = Image.new("RGB", (cs_w, cs_h), (255, 255, 255))
    y = 0
    for t in t0_tiles:
        cs.paste(t, (0, y))
        y += t.height + 4
    cs.save(out / "contact_sheet_t0.png")
    print(f"[3col] contact_sheet_t0.png ({cs.size})")

    # All-views animation: at each t, stack v=0..4 vertically; 21 frames total
    all_frames = []
    for t in range(T_full):
        per_view_at_t = [tiles_by_view[v][t] for v in range(V)]
        h_each = per_view_at_t[0].height
        canvas = Image.new("RGB", (per_view_at_t[0].width,
                                    V * h_each + (V - 1) * 4), (255, 255, 255))
        y = 0
        for im in per_view_at_t:
            canvas.paste(im, (0, y))
            y += h_each + 4
        # Downscale for GIF size sanity
        scale = 0.4
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)))
        all_frames.append(canvas)
    all_frames[0].save(out / "all_views_animation.gif", save_all=True,
                       append_images=all_frames[1:], duration=350, loop=0)
    print(f"[3col] all_views_animation.gif ({all_frames[0].size}, {len(all_frames)} frames)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
