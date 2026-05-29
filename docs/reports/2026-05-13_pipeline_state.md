# Pipeline state — current snapshot

> Last verified: 2026-05-13 (post-commit `923adb5`).
>
> Companion docs:
> - `docs/reports/2026-05-12_progress.md` — what's done + plan + asks
> - `docs/planning/2026-05-13_slide_plan.md` — group-meeting slides
> - `docs/design/sv4d2_api.md` — SV4D 2.0 inference API reference
> - `docs/runbooks/sv4d_runbook.md` — operational runbook (CPU / lab / RunPod)
> - `docs/runbooks/demo_runbook.md` — copy-paste reproduction commands

Two pipelines coexist on disk — one fully run (Pipeline 1, W1 failure-analysis), one implemented and CPU-tested but awaiting first GPU run (Pipeline 2, W3 SV4D-supervised training).

---

## Pipeline 1 — Failure-analysis (W1, fully run, all artifacts on disk)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  data/dnerf/<scene>/                                                    │
│  ├ transforms_{train,test,val}.json                                     │
│  └ {train,test,val}/r_NNN.png                                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ SC-GS-default training (third_party/SC-GS/train_gui.py)          │   │
│  │   --deform_type node --node_num 512 --hyper_dim 8                │   │
│  │   --iterations 30000   (lab A4500, ~14 min/scene)                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ├─→ point_cloud/iteration_30000/   (.ply checkpoint)            │
│         ├─→ test/ours_30000/{gt,renders,depth}/   (per-frame PNG)       │
│         ├─→ test/interpolate_30000/{renders,depth}/                     │
│         └─→ train.log                                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Inspection + analysis (CPU, ~30 s/scene)                         │   │
│  │   inspect_scgs_failure.py    → metrics_history.json + curves     │   │
│  │   qualitative_inspect_scgs.py → contact sheet, joint zoom,       │   │
│  │                                 temporal strobe                  │   │
│  │   spatial_error_analysis.py  → core-vs-periphery + radial        │   │
│  │   build_headline_figure.py   → composite HEADLINE.png            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         └─→ outputs/<scene>/inspection/qualitative/*.png                │
│             outputs/_cross_scene_failure/{HEADLINE,radial,core}.png     │
└─────────────────────────────────────────────────────────────────────────┘
```

**State:** fully run on jumpingjacks / hellwarrior / bouncingballs / standup.

---

## Pipeline 2 — SV4D-supervised training with MotionPrior hook (W3, awaiting first GPU run)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  data/dnerf/<scene>/  ← single camera trajectory (21 sampled frames)         │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 1: PREPARE                                          [CPU, ~5s]│    │
│  │   run_sv4d_supervised_pipeline.py::prepare_input_video              │    │
│  │   RGBA → composite onto white → resize 800→576                      │    │
│  │   Output: <scratch>/sv4d_input/frame_NNNN.png × 21                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 2: SV4D 2.0 INFERENCE                  [GPU 48GB, ~10-15 min] │    │
│  │   motionprior.integration.vgm.SV4D2Adapter.run                      │    │
│  │   subprocess → third_party/generative-models/                       │    │
│  │                 scripts/sampling/simple_video_sample_4d2.py         │    │
│  │   Output: <scratch>/sv4d_output/sv4d2/{000000_v1..v4}.mp4           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 3: CONVERT                                          [CPU, ~5s]│    │
│  │   scripts/sv4d_to_dnerf.py::convert_sv4d_to_dnerf                   │    │
│  │   Demux mp4s → (V, T, H, W, 3) → synthesize orbit cameras           │    │
│  │   at hardcoded SV4D azimuths [240°, 0°, 60°, 120°, 180°]            │    │
│  │   Copy original D-NeRF test/ + transforms_test.json (EVAL ORACLE)   │    │
│  │   Output: data/dnerf_sv4d/<scene>/                                  │    │
│  │   ├ train/r_NNNNN.png × 105 (= 5 views × 21 frames)                 │    │
│  │   ├ test/r_NNN.png      ← unchanged from original D-NeRF            │    │
│  │   ├ transforms_train.json (synthesized 105 cams)                    │    │
│  │   └ transforms_{test,val}.json (copied from original)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stage 4: SC-GS + MotionPrior hook       [lab A4500, ~15 min/scene]  │    │
│  │   third_party/SC-GS/train_gui.py + scripts/patches/scgs_hook.patch  │    │
│  │   Source: data/dnerf_sv4d/<scene>  (SV4D-supervised training)       │    │
│  │   Eval:   transforms_test.json     (original D-NeRF GT)             │    │
│  │   Output: outputs/<scene>_scgs_sv4d_<ablation>/                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Inspection + analysis (same scripts as Pipeline 1)                  │    │
│  │   Periphery/core ratio + radial profile numbers directly            │    │
│  │   comparable to W1 baseline                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**State:** all 4 stages implemented + CPU-tested (117/117 pytest cases pass). First real GPU run pending; cost ~$5–10 for the 4-scene sweep on a 48 GB RunPod card.

---

## What's inside Stage 4 — the MotionPrior hook in detail

The hook lives in `motionprior/integration/scgs_hook.py:MotionPriorHook` and patches SC-GS at 3 sites via `scripts/patches/scgs_hook.patch`:

```
                third_party/SC-GS/train_gui.py
                ┌────────────────────────────────────────┐
                │   Trainer.training_step():             │
                │                                        │
   Hook A  ────►│  time_input + ast_noise                │
   curriculum   │    = hook.gate_temporal_encoding(...)  │  ← Component 3
                │      (frequency-band mask on PE)       │
                │                                        │
                │   ... [forward pass through deform MLP]│
                │                                        │
   Hook B  ────►│   Ll1 = l1_loss(image, gt_image)       │
   gating       │       * hook.photometric_gating(...)   │  ← Component 2
                │      (exp(-α(t)·E_t) from offline      │
                │       ARAP energy table)               │
                │                                        │
   Hook C  ────►│   loss += hook.extra_losses(d_xyz, it) │
   rest-state   │      (small L2 anchor)                 │
                │                                        │
                │   ... [+ ControlNodeWarp.arap_loss()]  │
                └────────────────────────────────────────┘
                         ▲
                         │ monkey-patched at init time by
                         │ scgs_arap_adapter.install():
                         │   builds anisotropic λ matrix
   Adapter ──────────────┘   from SAM-2 part labels      ← Component 1
   (DIRECT)               and replaces the uniform-λ
                          ControlNodeWarp.arap_loss
```

### Module surface (all exist as real code, not stubs)

| Module | Purpose | Component |
|---|---|---|
| `motionprior.integration.scgs_hook.MotionPriorHook` | Top-level hook object; orchestrates A/B/C | (all) |
| `motionprior.integration.scgs_arap_adapter.install` | Monkey-patches `ControlNodeWarp.arap_loss` with anisotropic λ matrix | **1** |
| `motionprior.losses.arap_articulated.articulated_edge_weights` | λ_intra / λ_inter computation from SAM-2 part-id pairs | **1** |
| `motionprior.losses.gating.AdaptiveAlpha + compute_gating_weights` | α(t) schedule + `exp(−α(t)·E_t)` per-frame weight | **2** |
| `motionprior.geometry.arap_prior.compute_arap_prior_energy` | Offline precompute of `E_t` from lifted sparse trajectories | **2** |
| `motionprior.geometry.flow_lifting` | Trajectory lifting (optical flow → sparse 3D tracks) feeding into the energy precompute | **2** (input pipeline) |
| `motionprior.curriculum.frequency.FrequencyCurriculum + frequency_band_mask` | Temporal-PE band mask + unlock schedule | **3** |
| `motionprior.losses.rest_state.rest_state_l2` | Small L2 anchor on `d_xyz` | (regularizer) |
| `motionprior.segmentation.parts.SAM2Segmenter` | SAM-2 inference wrapper | **1** (input pipeline) |
| `motionprior.metrics.articulation.angular_consistency_score` | Inter-part angular consistency metric for eval | (eval) |

Plus the eval and orchestration scripts: `scripts/{eval,train,aggregate_results}.py`.

---

## Readiness summary

| Component | Code | Unit tests | End-to-end run on a real SC-GS training |
|---|---|---|---|
| Pipeline 1 (W1 failure analysis) | ✓ | n/a | **✓ 4 scenes complete** |
| `SV4D2Adapter` (subprocess wrapper) | ✓ | ✓ (mocked) | ⏳ awaits first GPU run |
| SV4D-output → D-NeRF converter | ✓ | ✓ (SC-GS reader parses output) | ✓ verified via fake-SV4D path |
| Pipeline 2 driver | ✓ | ✓ (CPU smoke) | ⏳ |
| Hook A (frequency curriculum / Component 3) | ✓ | ✓ unit (`test_scgs_hook.py`) | ⏳ never run through SC-GS training |
| Hook B (photometric gating / Component 2) | ✓ | ✓ unit | ⏳ never run through SC-GS training |
| Hook C (rest-state L2) | ✓ | ✓ unit | ⏳ |
| ARAP adapter (Component 1) | ✓ | ✓ unit | ⏳ never installed into a real SC-GS training |
| Patch file (3 hook sites) | ✓ | `git apply --check` only | ⏳ never applied + trained |
| Inter-part angular consistency metric | ✓ | ✓ unit | ⏳ never run on real outputs |

---

## What this means in plain English

- **W1 (Tier 1) runs are real and the data they produced is real.** The SC-GS-default trainings, the periphery/core numbers, the radial profile — all genuinely measured.
- **The full hook is wired but not yet exercised.** Hooks A/B/C and the ARAP adapter exist as code with unit tests, but no end-to-end run has fired the patch into SC-GS and trained a model with it. That's the Tier 1 ablation matrix Day 1–2 work.
- **The SV4D pipeline is wired but waits on GPU.** Stages 1–4 are implemented, the converter output passes SC-GS's own dataset reader on CPU, but no real SV4D inference has been run yet. That's the first item on the "tomorrow + next 2 days" list.
- **Inspection / analysis is reusable across pipelines.** The same `spatial_error_analysis.py` and `qualitative_inspect_scgs.py` scripts that produced the W1 headline figure will produce directly comparable numbers for the SV4D-supervised runs — no new analysis code needed.

---

## Where each dataset tier fits

| Tier | Pipeline | Scenes | Active components | Status |
|---|---|---|---|---|
| **1** — D-NeRF multi-view | Pipeline 1 + Pipeline 1 with hook A/B/C inactive | jumpingjacks, hellwarrior, bouncingballs, standup | Component 1 only | W1 baseline done; ablation matrix pending |
| **2** — D-NeRF + SV4D-supervised | Pipeline 2 (full) | same 4 scenes (Tier 1's scenes via SV4D-supervised retraining) | Components 1 + 2 + 3 jointly | Implemented + CPU-tested; awaits GPU |
| **3** — DyCheck real monocular | Pipeline 1 + hook A/B/C (data adapter needed for Nerfies format) | apple, hand, pillow, paper-windmill, spin, teddy, wheel | Components 1 + 2 + 3 jointly | Not started; needs data-prep adapter |

---

## Next concrete actions, in order

1. **Run the Tier 1 ablation matrix on jumpingjacks** (Pipeline 1 with the hook patch applied). 2 days lab time; produces the discriminating Component 1 result that turns the W1 observation into a causal claim.
2. **Rent a 48 GB RunPod card and execute the first SV4D-supervised scene end-to-end** (Pipeline 2, real GPU). ~$2 for one scene. Verifies the stage 2 + 3 + 4 interfaces work on real SV4D output.
3. **Sweep Pipeline 2 across all 4 D-NeRF scenes × 4 hook ablations** (rows 1, 5, 8, 9 of the experiments matrix). The Tier 2 headline numbers.
4. **DyCheck preprocessing adapter** in parallel with (3), then Tier 3 existence-proof runs.
