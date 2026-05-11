# MotionPrior-4DGS

**Articulation-aware video-diffusion-supervised 4D Gaussian Splatting from a single image — a perception module for visual world models.**

A robot in the wild has one photograph, not a captured video, not a multi-view rig. To reason about articulated objects (cabinet doors, robot arms, hinged tools, limbs) it needs a 4D scene representation that preserves piecewise-rigid kinematic structure. Existing single-image-input 4D methods (ViDAR, 4D-Fly, DIFF4SPLAT, CAT4D) smear joints into elastic bends; existing articulation-aware 4D methods (RigGS, VideoArtGS, Part2GS) require real captured video. We close that gap. See [`docs/vwm_framing.md`](docs/vwm_framing.md) for the full positioning.

> Status: bootstrap complete (2026-05-11). Articulation novelty audit + Option C repositioning (VGM back, single-image input, VWM-native) completed 2026-05-12. Components implemented and CPU-tested: gating, frequency curriculum, articulation-aware ARAP, rest-state L2, ARAP-prior precomputation, part-label assignment, SC-GS hook, training entry. Next: VGM front-end (SV4D 2.0) + SC-GS reproduction on H100.

## Pipeline

```
Static input image
   ↓
[SV4D 2.0  or  Wan-2.2 I2V]    -- video generative model (VGM) front-end
   ↓
[AnySplat]                     -- feed-forward canonical 3DGS + camera poses
   ↓
[SAM 2 hierarchical masks]     -- part-level segmentation on static input
   ↓
[SC-GS deformable backbone + MotionPriorHook]
   ├─ articulation-aware ARAP   (per-edge λ_intra / λ_inter from SAM 2 part labels)
   ├─ supervision gating        (per-frame weight from offline ARAP-prior energy)
   ├─ frequency curriculum      (Fourier-band mask on temporal PE)
   └─ rest-state L2 anchor      (small, fixed weight)
   ↓
Output: 4DGS with kinematically-structured deformation field
   ↓
Downstream VWM consumer:        physics simulator import (Genesis / PyBullet / MuJoCo)
                                for IK, contact, articulated manipulation policy
```

## Repository layout

- `motionprior/` — our novel losses, curriculum, segmentation, and prior-precomputation utilities. Pure-tensor; CPU-testable.
- `third_party/` — gitignored. Populated by `scripts/bootstrap_third_party.sh`. Contains pinned forks of SC-GS and AnySplat.
- `tests/` — pytest suite for `motionprior`. Runs on CPU.
- `scripts/` — bootstrap scripts for upstream code and the RunPod training environment.
- `docs/` — design docs, ownership map, plan files.

## Quickstart (local dev, CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Quickstart (RunPod, H100)

```bash
bash scripts/setup_runpod.sh        # installs conda env + cuda torch
bash scripts/bootstrap_third_party.sh
# follow third_party/README.md for per-upstream env activation
```

## Ownership

See [`docs/ownership.md`](docs/ownership.md).

## License

MIT — see [`LICENSE`](LICENSE).
