# MotionPrior-4DGS

Physics-constrained deformable 4D Gaussian Splatting from a single static image, supervised by a video generative model. Selectively trusts the video prior — gates physically implausible frames before they enter the photometric gradient, applies a frequency-domain curriculum during deformation MLP training, and uses an articulation-aware ARAP regularizer for piecewise-rigid structure.

> Status: under active development. Target submission: ICCV 2026 / 3DV 2026.

## Pipeline

```
Static Image
   -> Wan-2.2 I2V (image-to-video diffusion)
   -> AnySplat (feed-forward 3DGS + camera pose)
   -> SC-GS deformation MLP, trained with:
       * physically-gated photometric loss
       * frequency-domain curriculum
       * articulation-aware ARAP
       * rest-state L2 anchor
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
