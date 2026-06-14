"""Stage B/C motion-mask visualization (2 rows x N views), matching stageBC_<scene>.png.

Row 1: Stage B motion mask (red = moving) over the SV4D frame.
Row 2: Stage C per-Gaussian arm(red)/body(grey) projected to each view.

Reads Stage B masks from runs_aux/parts_motion_<motion_tag>/view*_part_masks.npy
and Stage C weights from runs_aux/part_assignment_<part_tag>/gaussian_arm_weights.npy.

Run (scgs env):
  python scripts/viz_stage_bc.py --scene standup --canon /mnt/HDD_1/.../point_cloud.ply \
      --part_tag standup --motion_tag standup
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
from scene.gaussian_model import GaussianModel  # noqa: E402

try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception:
    F = ImageFont.load_default()


def project(xyz, c2w, fov_x, H, W):
    w2c = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(c2w)
    cam = (w2c @ np.concatenate([xyz, np.ones((len(xyz), 1))], -1).T).T[:, :3]
    z = cam[:, 2]
    fx = (W / 2) / np.tan(fov_x / 2)
    u = fx * cam[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = fx * cam[:, 1] / np.maximum(z, 1e-6) + H / 2
    return np.stack([u, v], -1), z


def lab(arr, t):
    p = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(p); d.rectangle([0, 0, p.width, 22], fill=(255, 255, 255))
    d.text((4, 3), t, fill=(0, 0, 0), font=F)
    return np.asarray(p, np.float32) / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--canon", required=True)
    p.add_argument("--part_tag", required=True)
    p.add_argument("--motion_tag", required=True)
    p.add_argument("--n_views", type=int, default=5)
    p.add_argument("--t", type=int, default=10)
    args = p.parse_args()
    H = W = 576

    sd = REPO / "data/custom" / args.scene
    meta = json.loads((sd / "transforms_train.json").read_text())
    meta2 = json.loads((sd / "transforms_test.json").read_text())
    fov_x = meta["camera_angle_x"]
    cam_by_v, az_by_v = {}, {}
    for f in meta["frames"] + meta2["frames"]:
        v = int(f["view_idx"])
        cam_by_v.setdefault(v, np.asarray(f["transform_matrix"]))
        az_by_v.setdefault(v, float(f.get("azimuth_deg", v)))
    nv = max(cam_by_v) + 1
    views = list(range(0, nv, max(1, nv // args.n_views)))[: args.n_views]

    g = GaussianModel(3, fea_dim=8, with_motion_mask=False)
    g.load_ply(args.canon, og_number_points=0)
    xyz = g.get_xyz.detach().cpu().numpy()
    arm_w = np.load(REPO / f"runs_aux/part_assignment_{args.part_tag}/gaussian_arm_weights.npy")
    arm = arm_w > 0.5

    def sv4d(v, t):
        for split in ("train", "test"):
            for f in (meta["frames"] if split == "train" else meta2["frames"]):
                if int(f["view_idx"]) == v and int(f["frame_idx"]) == t:
                    rgba = np.asarray(Image.open(sd / (f["file_path"].lstrip("./") + ".png")),
                                      np.float32) / 255
                    a = rgba[..., 3:4] if rgba.shape[-1] == 4 else np.ones_like(rgba[..., :1])
                    return rgba[..., :3] * a + (1 - a)
        return np.ones((H, W, 3), np.float32)

    row_b, row_c = [], []
    for v in views:
        # Stage B: motion mask overlay
        base = sv4d(v, args.t).copy()
        mp = REPO / f"runs_aux/parts_motion_{args.motion_tag}/view{v}_part_masks.npy"
        if mp.exists():
            mov = np.load(mp)[0] > 0  # channel 0 = moving
            base[mov] = base[mov] * 0.45 + np.array([1, 0.1, 0.1]) * 0.55
        row_b.append(lab(base, f"Stage B mask v{v} az{az_by_v[v]:.0f} (red=moving)"))

        # Stage C: per-Gaussian arm/body projection
        pg = np.ones((H, W, 3), np.float32)
        uv, z = project(xyz, cam_by_v[v], fov_x, H, W)
        order = np.argsort(-z)
        ui = np.clip(uv[order, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[order, 1].astype(int), 0, H - 1)
        col = np.where(arm[order, None], np.array([0.85, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]))
        pg[vi, ui] = col
        row_c.append(lab(pg, f"Stage C v{v} (red=arm grey=body)"))

    grid = np.concatenate([np.concatenate(row_b, 1), np.concatenate(row_c, 1)], 0)
    out = REPO / f"runs_aux/stageBC_{args.scene}.png"
    Image.fromarray((grid * 255).astype(np.uint8)).save(out)
    arm_frac = arm.mean()
    print(f"[stageBC {args.scene}] views={views} arm Gaussians={arm.sum()} ({arm_frac*100:.1f}%)  wrote {out}")


if __name__ == "__main__":
    main()
