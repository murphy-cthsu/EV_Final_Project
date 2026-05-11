# Ownership Map

Two collaborators. Boundaries are by file path; if you need to cross, ping in chat first.

## Member A — Murphy (pipeline + novel components)

Owns:
- `motionprior/losses/` — gating, articulated ARAP, rest-state
- `motionprior/curriculum/` — frequency curriculum
- `motionprior/geometry/` — ARAP-prior precomputation
- `motionprior/segmentation/` — SAM 2 wrapper + part labeling
- `motionprior/configs/` — config schema + scene-specific YAMLs
- `scripts/bootstrap_third_party.sh` — upstream pinning

## Member B — (teammate) (front-end pipeline + baselines + eval)

Owns:
- `third_party/SC-GS` integration glue (the file(s) that wire our losses into SC-GS's training loop — created in a follow-up plan)
- `third_party/AnySplat` integration glue (script that runs AnySplat on the static input image and produces canonical 3DGS + camera poses)
- `third_party/Wan2.2` integration glue (Wan-2.2 I2V inference wrapper that takes a static image and produces the supervision video)
- `scripts/setup_runpod.sh` — H100 env bootstrap
- `scripts/run_training.py` — end-to-end training entrypoint (next plan)
- Baseline implementations (SC-GS default, CAT4D-style, GaussVideoDreamer-style masking)
- Evaluation pipeline (PSNR/SSIM/LPIPS, inter-part angular consistency, ablation table generation)

## Cross-cutting

- `docs/`, `README.md` — anyone edits, no conflict expected (small files)
- `tests/` — anyone adds tests for the modules they touch

## Branch hygiene

- `main`: stable, every commit passes `pytest -q`
- Feature branches optional; for fast iteration in this 1-month window, commit straight to `main` with frequent small commits
- Push after every commit so the other person sees state immediately
