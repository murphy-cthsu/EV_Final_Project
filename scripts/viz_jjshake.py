"""Visualize part-rigid reconstruction of jumpingjacks_shake_lefthand (self-supervised).

Compares GT (model render = supervision) vs ours under two motion classifications:
  tuned = provided 14% (arms + shaking hand)   |   auto = motion_parts_generic 60% (incl. legs)

Outputs runs_aux/jjshake_viz/:
  keyframes.png      rows=t, cols=[GT | ours-tuned | ours-auto]
  train_view{V}.gif  3-col animation over 21 frames
  novel_orbit.gif    turntable (ours-auto) novel view + time
"""
from __future__ import annotations
import sys, math, json
from pathlib import Path
import numpy as np, torch, imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
from eval_region_psnr import make_partrigid_renderer  # noqa: E402
from viz_vanilla_noprior import pose_spherical  # noqa: E402

H = W = 576; T_FULL = 21; RADIUS = 4.03
SCENE = REPO / "data/custom/jumpingjacks_shake_lefthand"
OUT = REPO / "runs_aux/jjshake_viz"; OUT.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)


def psnr(a, b): return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def lab(img, text):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    top = Image.new("RGB", (pil.width, pil.height + 24), (255, 255, 255)); top.paste(pil, (0, 24))
    ImageDraw.Draw(top).text((4, 3), text, fill=(0, 0, 0), font=FONT)
    return np.asarray(top, np.float32) / 255


def gt_frame(flat):
    for sp in ("train", "test"):
        p = SCENE / sp / f"r_{flat:05d}.png"
        if p.exists():
            a = np.asarray(iio.imread(p), np.float32) / 255
            al = a[..., 3:4]; return a[..., :3] * al + (1 - al)
    raise FileNotFoundError(flat)


def main():
    _pp = _A(); pipe = PipelineParams(_pp).extract(_pp.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    rf_t = make_partrigid_renderer("jjshake_tuned", pipe, bg, d_rot_zero=True)
    rf_a = make_partrigid_renderer("jjshake_auto", pipe, bg, d_rot_zero=True)
    meta = json.loads((SCENE / "transforms_train.json").read_text())
    meta_te = json.loads((SCENE / "transforms_test.json").read_text())
    fov_x = meta["camera_angle_x"]; FovY = focal2fov(fov2focal(fov_x, W), H)

    # held-out PSNR
    for name, rf in (("tuned", rf_t), ("auto", rf_a)):
        ps = []
        for f in meta_te["frames"]:
            img = rf(f, fov_x, FovY, T_FULL)
            ps.append(psnr(img, gt_frame(f["view_idx"] * T_FULL + f["frame_idx"])))
        print(f"[viz] held-out PSNR ours-{name}: mean={np.mean(ps):.2f} median={np.median(ps):.2f}")

    allf = {(f["view_idx"], f["frame_idx"]): f for f in meta["frames"] + meta_te["frames"]}

    # keyframes grid: rows=t, cols=[GT|tuned|auto], view 8 (front)
    V = 8
    rows = []
    for t in (0, 5, 10, 15, 20):
        f = allf[(V, t)]; flat = V * T_FULL + t
        gt = gt_frame(flat); it = rf_t(f, fov_x, FovY, T_FULL); ia = rf_a(f, fov_x, FovY, T_FULL)
        sep = np.ones((H + 24, 5, 3))
        rows.append((np.clip(np.concatenate([
            lab(gt, f"GT (supervision) t={t}"), sep, lab(it, "ours: tuned mask (arms only)"),
            sep, lab(ia, "ours: full mask (incl legs)")], 1), 0, 1) * 255).astype(np.uint8))
    hs = (np.ones((5, rows[0].shape[1], 3)) * 255).astype(np.uint8)
    grid = rows[0]
    for r in rows[1:]: grid = np.concatenate([grid, hs, r], 0)
    Image.fromarray(grid).save(OUT / "keyframes.png"); print("[viz] keyframes.png")

    # train-view animation (3-col) for views 8 and 0
    for V in (8, 0):
        g = []
        for t in range(T_FULL):
            f = allf[(V, t)]; flat = V * T_FULL + t
            gt = gt_frame(flat); it = rf_t(f, fov_x, FovY, T_FULL); ia = rf_a(f, fov_x, FovY, T_FULL)
            sep = np.ones((H + 24, 5, 3))
            g.append((np.clip(np.concatenate([
                lab(gt, f"GT t={t:02d}"), sep, lab(it, "ours tuned(arms)"), sep,
                lab(ia, "ours full(+legs)")], 1), 0, 1) * 255).astype(np.uint8))
        iio.imwrite(OUT / f"train_view{V}.gif", np.stack(g), duration=120, loop=0)
        print(f"[viz] train_view{V}.gif")

    # novel-view turntable (ours-auto)
    N = 48; azis = np.linspace(0, 360, N, endpoint=False); tmap = np.round(np.linspace(0, T_FULL - 1, N)).astype(int)
    g = []
    for az, t in zip(azis, tmap):
        f = {"transform_matrix": pose_spherical(float(az), 0.0, RADIUS).tolist(),
             "frame_idx": int(t), "view_idx": 0}
        img = rf_a(f, fov_x, FovY, T_FULL)
        g.append((np.clip(lab(img, f"ours novel view  az={az:.0f} t={t:02d}"), 0, 1) * 255).astype(np.uint8))
    iio.imwrite(OUT / "novel_orbit.gif", np.stack(g), duration=120, loop=0)
    print("[viz] novel_orbit.gif")


if __name__ == "__main__":
    raise SystemExit(main())
