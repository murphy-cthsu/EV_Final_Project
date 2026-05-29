# Visual World Model Framing — MotionPrior-4DGS

> Source of truth for the paper introduction. Cites the audited 2025–2026 landscape and the VWM literature explicitly. Wiki counterpart: `wiki/research/MotionPrior4DGS.md` (in the notes repo).

## What is a visual world model

A **visual world model** is a system that, given visual observations of an environment, predicts how the environment will physically evolve, enabling an embodied agent to reason about consequences of its actions before taking them. Three layers, following the DreamerV3 / Cosmos / V-JEPA 2 / GAIA decomposition:

1. **Perception** — observations → structured scene state.
2. **Dynamics** — current state + action → next state.
3. **Policy** — predicted future → action.

We contribute to the **perception layer**, specifically the single-static-image deployment regime: an agent in the wild has one photograph of an unknown scene, no multi-view rig, no captured video, no time to wait. The perception layer must lift that single observation into a 4D representation that the dynamics layer can act on.

## Why single-image input is the right framing for embodied VWM

A multi-view rig or pre-captured video is a deployment-time assumption that breaks in every realistic embodied setting:

| Setting | Why pre-captured video fails |
|---|---|
| Household robot encountering a new room | No prior visit; one camera; the agent must act now |
| Search-and-rescue drone over debris | Scene is novel; multi-view capture is unsafe and slow |
| Industrial inspection of a one-of-a-kind asset | Each asset is unique; building a video corpus per asset is infeasible |
| Assistive robot reading a hand-held photo for context | The photo *is* the input modality |
| Open-vocabulary manipulation in dialogue | "Pick up the thing in this picture" is the natural interface |

For these settings the perception layer must operate from a single observation. Methods that require captured video (RigGS, VideoArtGS, MoSca, Shape of Motion, 4DGT) cannot run. Methods that operate from a single image (DIFF4SPLAT, CAT4D, 4Diffusion, ViDAR with single-frame DreamBooth) can run but produce 4D scenes that lose the articulated kinematic structure the agent needs.

## Why articulation specifically is load-bearing for downstream VWM use

The dynamics and policy layers of a VWM, when consuming a 4D scene representation, perform three operations on articulated content:

| Operation | What it needs from the 4D scene |
|---|---|
| **Inverse kinematics** for end-effector reach | Identifiable joint axes — decision variables for the IK solver |
| **Contact-point prediction** on a rotating rigid part | Geometry that stays rigid under part motion — collision queries stable over time |
| **Articulated-object manipulation policy** (push a hinge, pull a drawer) | Per-part rigid-body simulation — policy gradients computable on simulator output |

Elastic-bend reconstructions (uniform-ARAP, motion-scaffold, smooth deformation MLP) fail all three operations. The 4DGS produced by ViDAR / 4D-Fly / DIFF4SPLAT looks visually fine in novel-view renderings, but the joint-region Gaussians have learned a smooth deformation that has no IK structure and no stable collision geometry. The downstream dynamics layer cannot use it.

Articulation-aware piecewise-rigid 4DGS fixes this: each part rotates as a near-rigid body, joints localize at boundaries, collision geometry on a part is preserved.

## The unique contribution

The intersection of "video-diffusion-supervised" + "articulation-aware" + "single-image input" is unoccupied in the May-2026 landscape (verified against 25+ papers in the survey). Concretely:

| Cluster | Sample papers | What they don't do |
|---|---|---|
| **Diffusion-supervised 4D** | ViDAR, 4D-Fly, DIFF4SPLAT, CAT4D, 4Diffusion | No articulated kinematic structure |
| **Articulated 4D reconstruction** | RigGS, VideoArtGS, Part2GS, GaussianArt | Require real captured video; no diffusion supervision |
| **Feed-forward 4D** | 4DGT, DIFF4SPLAT | No articulation; no per-scene control |
| **MPM-based physics** | PhysDreamer, DreamPhysics, Physics3D | Require material specification; object-centric |

MotionPrior-4DGS sits at the intersection. Three components, each designed for the cross-setting:

1. **SAM-2-grounded piecewise-rigid ARAP** — anisotropic per-edge λ_intra/λ_inter weighting derived from SAM 2 hierarchical part masks on the static input image. Independent of motion-clustering (which fails under generative supervision noise — VideoArtGS's stated limitation) and independent of skeleton extraction (which requires clean 2D skeletons across frames — RigGS's stated limitation).
2. **ARAP-energy-based supervision gating** — per-frame photometric loss weighted by exp(−α(t) · E_t), where E_t is the ARAP energy computed offline from optical-flow-lifted sparse trajectories of the generated video. Removes geometrically implausible supervision frames. Orthogonal to ViDAR's appearance-region mask and USplat4D's observability uncertainty.
3. **Frequency-domain motion curriculum** — Fourier-band schedule on the deformation MLP's temporal positional encoding. Locks out high-frequency channels during early training so the network can only express slow macro-motions; unlocks higher frequencies progressively. Targets the high-frequency hallucinations characteristic of generative video supervision.

## Pipeline

```
Static input image
   ↓
[SV4D 2.0 / Wan-2.2-I2V] — video generative model (VGM) front-end
   ↓
[AnySplat] — feed-forward canonical 3DGS + camera poses
   ↓
[SC-GS deformable backbone + MotionPriorHook]
   ├─ articulation-aware ARAP (SAM 2 part labels → per-edge λ)
   ├─ supervision gating (precomputed ARAP-energy → per-frame weight)
   ├─ frequency curriculum (band mask on temporal PE)
   └─ rest-state L2 anchor (small, fixed weight)
   ↓
Output: 4DGS with kinematically-structured deformation field
   ↓
Downstream consumer: physics simulator (Genesis / PyBullet) for IK / contact / policy
```

The pipeline is single-image-input; the VGM step generates the supervision video internally. The output is a per-scene 4D representation; downstream VWM dynamics and policy layers consume it.

## Related VWM literature to cite

| Paper | Why it's relevant |
|---|---|
| **DreamerV3** (Hafner et al. 2023) | Canonical decomposition of perception + dynamics + policy; predicts in latent space, not pixel; foundational for VWM framing |
| **GAIA-1 / GAIA-2** (Wayve, 2023–2024) | Large-scale predictive world model for driving; pixel-level prediction; multi-modal input |
| **Cosmos** (NVIDIA, 2024) | Foundation world model; explicit modular perception/dynamics interface; the consumer-side of our output |
| **V-JEPA 2** (Meta, 2025) | Self-supervised predictive perception; learns dynamics-aware representations; argues representation quality precedes prediction quality |
| **Genie 2** (DeepMind, 2024) | Action-conditioned video generation; world-model-as-imagination |
| **GenRL / VLM-Action models** (various 2024–2025) | Image + language → action; perception inputs are images, not video |

Our contribution slots into the perception layer ahead of any of these dynamics models. We do not claim to compete with them; we claim to provide the kind of structurally-rich 4D scene representation they currently lack at the perception input.

## What we explicitly do NOT claim

- We do not predict future states conditioned on actions — that's the dynamics layer.
- We do not train a policy — that's the policy layer.
- We do not solve full Newtonian dynamics (gravity, momentum, elastic restitution) — we provide kinematic structure (joint angles, part rigidity), not dynamical physics. MPM-based methods (PhysDreamer) handle that orthogonally.
- We do not generalize across scenes — per-scene optimization is required. Generalization is a feed-forward problem (4DGT, DIFF4SPLAT) we don't compete with.

## What we explicitly do claim

- A per-scene perception module that takes a single static image as input and produces a 4D scene representation with articulated kinematic structure.
- Demonstrated downstream usability: 4DGS imports into Genesis/PyBullet such that an articulated-part IK solver returns sane joint angles. (W4 experiment.)
- Quantitative improvements on the articulated subsets of D-NeRF, DyCheck, and HyperNeRF over the strongest current baselines (ViDAR for diffusion-supervised + scene-level; RigGS for articulation + monocular; DIFF4SPLAT for single-image + feed-forward).
- A new metric — **inter-part angular consistency** — that directly measures piecewise-rigid joint behavior, complementary to PSNR/SSIM/LPIPS (which conflate articulated and elastic reconstructions).

## Paper outline (intro section)

§1.1 — Visual world models need perception modules that operate from realistic deployment-time inputs (single image, no captured video, no multi-view rig).

§1.2 — Existing perception modules split into two non-overlapping clusters: diffusion-supervised 4D (no articulation) and articulated 4D (requires real video).

§1.3 — The downstream cost of missing articulation: IK, contact, policy training all fail on elastic-bend reconstructions.

§1.4 — Our contribution: articulation-aware video-diffusion-supervised 4DGS from a single image. SAM-2 part labels + per-edge ARAP + ARAP-energy gating + frequency curriculum.

§1.5 — Evaluation: D-NeRF articulated subset + DyCheck + HyperNeRF; inter-part angular consistency + simulator-import demo.

§1.6 — Limitations: per-scene optimization; depends on SAM 2 part quality; provides kinematic not dynamical structure.

---

## Decision log

### 2026-05-12 — Deployment regime reframed to "single short observation"

**Issue.** SV4D 2.0 architecturally requires a 21-frame input video, not a single static image. The original framing in §1.2 above ("single static image, no captured video") is not achievable with SV4D 2.0 alone.

**Routes considered.**
- (A) Reframe "single observation = one short clip (≤21 frames at 576²)". Keeps the spirit of the deployment argument — embodied agents observing for ~1 second satisfies this — without architectural overhaul. **Cost: rhetorical concession.**
- (B) Insert an image-to-video model (SVD, CogVideoX-I2V, Wan-2.2-I2V) in front of SV4D. **Cost: ~3 days engineering + cascaded generative drift.** The cascaded drift partially overlaps with what Components 2 and 3 of our hook are designed to absorb, so this route also serves as additional empirical justification for the gating + curriculum components.

**Decision.** Ship Route A for the paper. Tag Route B as future work / appendix experiment. Embodied robotics audiences (DreamerV3 / Cosmos / V-JEPA 2 / GAIA-2 lit) routinely treat short observation clips as valid perception input; reviewers in that community will not object to the relaxed framing.

**Implications for the paper.**
- §1.1 / §1.2 framing language should read "single short observation" rather than "single static image."
- The deployment-time argument is unchanged — an agent in the wild can produce ≤1 s of camera footage trivially.
- The contribution table in §3 ("Why articulation specifically is load-bearing") is unchanged.
- The pipeline diagram in §6 should clarify that the input is `1–21 frames of a single camera trajectory`, not literally one frame.

### 2026-05-12 — Three-component contribution made explicit

The three components named in §3 above (SAM-2-grounded piecewise-rigid ARAP / ARAP-energy gating / frequency-domain curriculum) are not independently novel; the diffusion-supervised cluster has each piece in isolation. The novelty is the **intersection** — specifically, Component 1 (articulation) operating in a regime where it would otherwise collapse, with Components 2 and 3 providing the drift-resistance that makes Component 1 viable under generative supervision. The progress report (`docs/reports/2026-05-12_progress.md` §1) gives the precise component → pathology mapping; intro section §1.4 of the paper should foreground this intersection framing rather than treating the three components as additive contributions.
