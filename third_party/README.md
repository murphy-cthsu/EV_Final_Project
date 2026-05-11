# Vendored upstream code

This directory is **gitignored**. It is populated by
`../scripts/bootstrap_third_party.sh` at fixed SHAs:

| Upstream | Repo | Pinned SHA |
|---|---|---|
| SC-GS | https://github.com/yihua7/SC-GS | `3a9d2ad4e4fc058b0763d446ae9e6b1be120b872` |
| AnySplat | https://github.com/OpenRobotLab/AnySplat | `5f5e208a7dd57d52e43ea0d553a95eab526e8775` |
| Wan-2.2-I2V | HuggingFace `Wan-AI/Wan2.2-I2V-A14B` | weights only |

## Environment isolation

SC-GS and AnySplat have different (and probably conflicting) Python / CUDA
requirements. Each lives in its own conda env on the RunPod box:

- `scgs`: from `third_party/SC-GS/environment.yml`
- `anysplat`: from `third_party/AnySplat/requirements.txt` over CUDA 12.1 torch
- `wan22`: a fresh env with `transformers >= 4.40` and `accelerate`

Our `motionprior` package installs into the `scgs` env (we hook into SC-GS's
training loop, so it must be importable there).

## Why not git submodules?

Submodules add coordination cost for two collaborators. A bootstrap script
that pins SHAs is equivalently reproducible, easier to extend, and lets us
modify upstream code in-place during debugging without touching this repo.
