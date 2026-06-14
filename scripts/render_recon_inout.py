"""Render the VGM-supervised 4D recon (deform-MLP SC-GS fit to SV4D) at the
trained INPUT azimuth (sharp) vs OFF-AXIS novel views (fuzzy), for block ① of
the architecture diagram. Single model, same object (lego digger), correct
in-view/off-axis axis. Saves raw PNGs to runs_aux/recon_inout/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party" / "SC-GS"))
import torch  # noqa: E402
from arguments import PipelineParams  # noqa: E402
from argparse import ArgumentParser as _A  # noqa: E402
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402
from eval_region_psnr import make_deformmlp_renderer  # noqa: E402

SHARP = "elev_0_az_0"                       # trained input azimuth
OFFAXIS = ["elev_0_az_150", "elev_0_az_210", "elev_0_az_270",
           "elev_20_az_180", "elev_30_az_120", "elev_30_az_270"]
T = 10


def main():
    out = REPO / "runs_aux/recon_inout"; out.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPO / "data/custom/lego_v3/transforms_train.json").read_text())
    mt = json.loads((REPO / "data/custom/lego_v3/transforms_test.json").read_text())
    allf = m["frames"] + mt["frames"]
    fov_x = m["camera_angle_x"]
    FovY = focal2fov(fov2focal(fov_x, 576), 576)
    by_key = {(f["view_tag"], int(f["frame_idx"])): f for f in allf}

    _pq = _A(); pp = PipelineParams(_pq); pipe = pp.extract(_pq.parse_args([]))
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    print("[recon-inout] loading VGM-supervised deform-MLP recon...")
    recon = make_deformmlp_renderer(REPO / "outputs/custom/lego_v2_vanilla_sam_node", pipe, bg)

    def save(tag):
        f = by_key.get((tag, T)) or by_key.get((tag, 1)) or by_key.get((tag, 0))
        if f is None:
            print("  !! missing", tag); return
        img = recon(f, fov_x, FovY, 21)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f"{tag}.png")
        print("  saved", tag)

    save(SHARP)
    for tag in OFFAXIS:
        save(tag)
    print("[recon-inout] done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
