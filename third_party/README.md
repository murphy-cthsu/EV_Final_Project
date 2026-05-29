# Vendored upstream code

This directory is **gitignored**. It is populated by
`../scripts/bootstrap_third_party.sh` at fixed SHAs:

| Upstream | Repo | Pinned SHA |
|---|---|---|
| SC-GS | https://github.com/yihua7/SC-GS | `3a9d2ad4e4fc058b0763d446ae9e6b1be120b872` |
| AnySplat | https://github.com/OpenRobotLab/AnySplat | `5f5e208a7dd57d52e43ea0d553a95eab526e8775` |
| **SV4D 2.0** (primary VGM front-end) | HuggingFace `stabilityai/sv4d2.0` | weights only — commercial license |
| **Wan-2.2-I2V** (fallback VGM front-end) | HuggingFace `Wan-AI/Wan2.2-I2V-A14B` | weights only |
| SAM 2 (part segmentation) | HuggingFace `facebook/sam2-hiera-large` | weights only |

## Pipeline role of each upstream

- **SV4D 2.0** (primary) / **Wan-2.2 I2V** (fallback) — the *video generative model* front-end. Takes the static input image; produces a generated video for supervision. SV4D 2.0 is multi-view-aware by construction (preferred); Wan-2.2 is monocular I2V (fallback if SV4D 2.0 license is blocked).
- **AnySplat** — feed-forward canonical 3DGS + camera poses from the generated video frames. Bypasses COLMAP / static 3DGS optimization. Static-trained, so it operates on dense temporal frames where inter-frame parallax approximates multi-view; works under SV4D 2.0's multi-view output, fragile under Wan-2.2's near-monocular output.
- **SC-GS** — deformable backbone. Provides the deformation MLP, sparse control points, and ARAP regularizer. We patch its `train_gui.py` with three single-line hooks (see `docs/design/scgs_hook_design.md`).
- **SAM 2** — part-level segmentation on the static input image. Produces hierarchical part masks that we project through the canonical 3DGS depth to label Gaussians by part.

## Environment isolation

SC-GS, AnySplat, SV4D 2.0, and Wan-2.2 have different (and probably conflicting) Python / CUDA requirements. Each lives in its own conda env on the RunPod box:

- `scgs`: from `third_party/SC-GS/environment.yml`; **motionprior installs into this env** because the SC-GS hook lives here
- `anysplat`: from `third_party/AnySplat/requirements.txt` over CUDA 12.1 torch
- `sv4d2`: a fresh env for SV4D 2.0 inference (transformers, diffusers, accelerate)
- `wan22`: fallback env if SV4D 2.0 is unusable (transformers >= 4.40, accelerate)
- `sam2`: SAM 2 inference (lightweight; can share with motionprior env)

A driver script (`scripts/run_pipeline.sh`, to be written W1 on the pod) activates each env in sequence: SV4D 2.0 → save generated video → AnySplat → save canonical 3DGS + poses → SAM 2 → save part masks → SC-GS w/ hook → train + save 4DGS.

## Why not git submodules?

Submodules add coordination cost for two collaborators. A bootstrap script
that pins SHAs is equivalently reproducible, easier to extend, and lets us
modify upstream code in-place during debugging without touching this repo.
