# Slide plan — group meeting 2026-05-13

> **Format:** 30 min total = 20 min presentation + 10 min Q&A.
> **Goal:** land the W1 closeout + the 3-component framing + the W2/W3 plan.
> **Headline figure to project:** `outputs/_cross_scene_failure/HEADLINE.png`
> **Companion docs:** `docs/reports/2026-05-12_progress.md`, `docs/runbooks/demo_runbook.md`

---

## Time budget (20 min main presentation)

| Section | Slides | Time | Notes |
|---|---|---|---|
| A. Hook + contribution | 3 | 3.5 min | Open with the visual hook before any framing claim |
| B. Related work | 2 | 3 min | Two clusters, four direct competitors |
| C. The 3 components | 3 | 4 min | DIRECT (1) vs PROTECTIVE (2, 3) framing |
| D. W1 evidence | 4 | 4 min | The radial profile is the money chart |
| E. Path forward | 3 | 4 min | Tier 1 / 2 / 3 dataset stack |
| F. Closing | 2 | 1.5 min | Weaknesses + asks |
| **Total** | **17 slides** | **20 min** | Avg ~70 s per slide |

Keep dense slides (HEADLINE, radial profile, ablation matrix, related-work table) at 90–120 s; keep transition slides at 30–45 s.

---

## Section A — Hook + Contribution (3 slides, 3.5 min)

### Slide 1 — Title (15 s)

- **MotionPrior-4DGS: W1 closeout + W2/W3 plan**
- Owner · group meeting · 2026-05-13
- Subtitle: *"Articulation-aware 4D Gaussian Splatting under generative-video supervision"*

### Slide 2 — The visual hook (1.5 min)

- **Visual:** `outputs/_cross_scene_failure/HEADLINE.png` at full-slide size.
- **Three sentences to say over the slide:**
  1. "Four D-NeRF scenes, identical SC-GS config. The error heatmaps and time-interp std maps tell us SC-GS has a structured residual that is articulation-specific."
  2. "Notice the radial profile at the bottom — jumpingjacks ramps 5–6× from core to edge; bouncingballs is flat."
  3. "This is the W1 evidence. The rest of this talk is about what it does and doesn't mean — and what we built to test it properly."
- **Why open here:** earns 90 s of attention before any framing claim has to defend itself.

### Slide 3 — Contribution in one slide (1.5 min)

- **Title:** *"A 3-component joint system. The novelty is the intersection."*
- **Center visual:** 3-circle Venn intersection labeled:
  - Diffusion-supervised 4D (ViDAR, 4D-Fly, DIFF4SPLAT)
  - Articulation-aware 4D (RigGS, VideoArtGS, Part2GS)
  - Single-observation input regime
- **One-liner under the visual:** "RigGS does articulated+ARAP+monocular. ViDAR does diffusion-supervised+4D. Nobody does both — because RigGS's skeleton extraction breaks under generative drift, and the diffusion-supervised cluster has no drift-rejection mechanism."
- **Closing line:** "Our contribution is delivering articulation in a regime where it would otherwise collapse."

---

## Section B — Related Work (2 slides, 3 min)

### Slide 4 — Landscape: four clusters, where we sit (1 min)

- **Title:** *"The May-2026 landscape — four clusters, one unoccupied corner."*
- **Visual:** 2×2 grid of cluster boxes (layout shown below as ASCII reference):

```
┌────────────────────────────────┬────────────────────────────────┐
│  DIFFUSION-SUPERVISED 4D       │  ARTICULATED 4D                │
│  ViDAR · 4D-Fly · DIFF4SPLAT   │  RigGS · VideoArtGS ·          │
│  CAT4D · 4Diffusion            │  Part2GS · GaussianArt         │
│                                │                                │
│  + Single-image-friendly       │  + Kinematic structure         │
│  − No articulation             │  − Requires real video         │
└────────────────────────────────┴────────────────────────────────┘
┌────────────────────────────────┬────────────────────────────────┐
│  FEED-FORWARD 4D               │  MPM-BASED PHYSICS             │
│  4DGT · DIFF4SPLAT             │  PhysDreamer · DreamPhysics ·  │
│                                │  Physics3D                     │
│  + Fast inference              │  + Newtonian simulation        │
│  − No per-scene control        │  − Object-centric only         │
│  − No articulation             │  − Material spec required      │
└────────────────────────────────┴────────────────────────────────┘
                    ↓
            ★ Our position ★
   intersection of (top-left + top-right) plus
   drift-handling that neither cluster has
```

- **Verbal one-liner:** "Top-left does our front-end. Top-right does our regularizer. Neither does both — because their assumptions about supervision quality are incompatible. We sit at the intersection by adding the drift-resistance that makes it work."

### Slide 5 — Direct competitors table (2 min)

- **Title:** *"What each direct competitor contributed — and the limitation we address."*
- **The table is the slide. Spend 2 min walking through it:**

| Paper | Cluster | What they contributed | Limitation we address |
|---|---|---|---|
| **RigGS** (CVPR 2025) | Articulated 4D | ARAP + monocular video + skeleton-aware 4DGS on general objects. **Component 1 alone.** | Skeleton extraction needs consistent 2D keypoints across frames — **fails on generative video** because diffusion-supervised motion is drifty. |
| **VideoArtGS** (Sep 2025) | Articulated 4D | Articulated monocular 4D via motion clustering of trajectories | Motion clustering breaks under noisy / inconsistent trajectories — **their own stated limitation**. We sidestep by reading parts from the static frame (SAM-2). |
| **ViDAR** (2025) | Diffusion-supervised 4D | DreamBooth + diffusion supervision; dynamic-region appearance mask | Uniform-elastic deformation — **no kinematic structure**. Appearance mask doesn't detect physically implausible motion. |
| **4D-Fly** (CVPR 2025) | Diffusion-supervised 4D | VGM front-end + 4DGS reconstruction; the closest pipeline shape to ours | **Explicitly names "highly complex articulated motions" as a failure mode** (their §6). Uniform ARAP; no drift-rejection. |

- **Honorable mentions (one sentence each, fast):**
  - DIFF4SPLAT — single-image + feed-forward, no articulation (competitor on input regime, not articulation).
  - Part2GS / GaussianArt — articulated 4DGS for object-centric multi-view; doesn't scale to scene-level.
  - MoSca — motion scaffold; treats deformation as a soft prior, no explicit rigidity.

- **Closing verbal (15 s, worth saying aloud):**
  > "Against each competitor, our delta is concrete and **falsifiable**. RigGS publishes 40.82 PSNR on jumpingjacks; we either beat them under generative supervision or we don't. 4D-Fly names articulated motion as their failure mode by name; we either fix it on the same scenes or we don't. We're not claiming a vague improvement on a vague axis."

---

## Section C — The 3 Components (3 slides, 4 min)

### Slide 6 — Component 1: SAM-2 piecewise-rigid ARAP (1.5 min)

- **Title:** *"Component 1 (DIRECT): SAM-2-grounded piecewise-rigid ARAP."*
- **Left panel:** the loss
  - `L_arap = Σ_(i,j) λ_ij · ||(x_i^t − x_j^t) − R_i (x_i − x_j)||²`
  - `λ_ij = λ_intra` if SAM-2 part(i) == part(j), else `λ_inter`
  - `λ_intra ≫ λ_inter` (e.g. 10:1)
- **Right panel:** `runs_aux/jumpingjacks_label_overlay.png` (the colorful part overlay)
- **One-liner:** "Part structure is fixed on the **static** input frame — robust to whatever motion noise enters downstream."

### Slide 7 — Components 2 & 3 on one slide (1.5 min)

- **Title:** *"Components 2 and 3 (PROTECTIVE): drift-rejection mechanisms."*
- **Left half — Component 2: ARAP-energy gating**
  - Formula: `L_photo(t) = exp(−α(t) · E_t) · L_recon(t)` where `E_t` is ARAP energy of lifted sparse trajectories
  - Tiny α(t) ramp visual: low α early (trust all frames) → high α late (aggressively reject drifty frames)
  - One line: "Down-weight frames where lifted trajectories aren't near-rigid."
- **Right half — Component 3: Frequency curriculum**
  - Visual: band-mask schedule over iterations on the temporal PE — early iters mask high-freq channels, bands unlock progressively
  - One line: "Force macro-articulation to be learned before micro-jitter."

### Slide 8 — DIRECT vs PROTECTIVE (1 min)

- **Title:** *"Only Component 1 adds articulation. Components 2 and 3 protect it."*
- **Activation table:**

| Component | Role | Active under clean supervision? | Active under generative supervision? |
|---|---|---|---|
| 1 (art-ARAP) | DIRECT | Yes | Yes |
| 2 (gating) | PROTECTIVE | Dormant (uniform weight) | Yes |
| 3 (curriculum) | PROTECTIVE | Marginal | Yes |

- **Critical closing line:** "Our W1 evidence — on clean multi-view D-NeRF — only exercises Component 1. The full joint system isn't tested yet. That's why we need Tier 2."

---

## Section D — W1 Evidence (4 slides, 4 min)

### Slide 9 — W1 setup + failure mode visualized (1 min)

- **Title:** *"Same SC-GS config, four D-NeRF scenes, 30K iters."*
- **Setup line at top:** SC-GS-default, node_num=512, hyper_dim=8, ~18 min wall on lab A4500×3.
- **PSNR strip:** jumpingjacks 40.85 / hellwarrior 42.97 / bouncingballs 41.64 / standup 47.22.
- **2×2 grid below:** temporal-std heatmap thumbnails from each scene's `inspection/qualitative/temporal_strobe.png` (middle panel).
- **Failure signature labels:**
  - jumpingjacks → limb-end fan trails
  - hellwarrior → spiky radial fringe (rotating weapon)
  - bouncingballs → arc streaks (ball trajectories)
  - standup → vertical ladder slabs (upward motion)
- **One-liner:** "Different morphologies, same underlying issue — the deformation field can't keep peripheral regions stable."

### Slide 10 — Periphery/core ratio (1 min)

- **Title:** *"Periphery/core error ratio tracks articulation complexity."*
- **Visual:** `outputs/_cross_scene_failure/core_vs_periphery_cross_scene.png` (full slide).
- **Annotation callouts:**
  - jumpingjacks: ×2.28 ← W1 headline
  - bouncingballs: ×1.03 ← non-articulated control
- **Key takeaway:** "The control sits at unity. The signal is *articulation-specific*, not high-motion-in-general."

### Slide 11 — The radial profile (the money chart) (1 min)

- **Title:** *"Mean error grows monotonically with distance from object centroid."*
- **Visual:** `outputs/_cross_scene_failure/radial_profile_cross_scene.png` (full slide).
- **Annotations:**
  - jumpingjacks: 5–6× ramp from center to edge
  - bouncingballs: flat through 80% of normalized radius
  - hellwarrior, standup: monotonic but milder
- **Closing line:** "This is the cleanest single argument that uniform-rigidity ARAP under-fits articulated kinematic chains."
- **Presenter tip:** let the curve speak for 10 s before adding words.

### Slide 12 — Honest reading of W1 (1 min)

- **Title:** *"What the W1 evidence does and doesn't support."*
- **Two-column layout:**
  - **Supports (defensibly):**
    - Structured spatial residual that scales with articulation complexity
    - Articulation-specific (bouncingballs control sits at unity)
    - Lower bound on failure under harder supervision regimes
  - **Does NOT yet support:**
    - That uniform-ARAP *causes* it (could be capacity / iters / hyper_dim — W2 settles this)
    - That Component 1 *fixes* it (W2 ablations settle this)
    - That Components 2 or 3 matter at all (dormant under clean supervision — needs Tier 2)
- **Credibility line:** "We're not claiming 40.85 PSNR is a failure. We're claiming the residual structure predicts a worse failure under the deployment regime — which is the experiment that justifies the hook."

---

## Section E — Path Forward (3 slides, 4 min)

### Slide 13 — Three dataset tiers (1.5 min)

- **Title:** *"Dataset stack, not dataset choice."*
- **Table:**

| Tier | Dataset | Supervision | Active components | Role |
|---|---|---|---|---|
| 1 | D-NeRF multi-view | 100 train cameras, dense | 1 only | Controlled ablation baseline |
| 2 | D-NeRF + SV4D | 5–9 azimuths × 21 frames, generative | 1 + 2 + 3 jointly | **Load-bearing experiment** |
| 3 | DyCheck | Real monocular handheld | 1 + 2 + 3 jointly | Real-world existence proof |

- **Answer to "D-NeRF is too easy":** "D-NeRF's value is the clean evaluation oracle, not easy supervision. Under SV4D supervision (Tier 2), the same scenes become genuinely ill-posed while we keep a fair eval oracle. DyCheck cannot match that — it has both ill-posed supervision and noisy GT."

### Slide 14 — Tier 1 ablation matrix (1.5 min)

- **Title:** *"Tier 1: discriminating ablations to turn W1 observations into a causal claim."*
- **Table:**

| Row | Config | Rules out |
|---|---|---|
| A | Component 1 with **shuffled SAM-2 labels** | "Any extra regularizer helps" |
| B | uniform ARAP, **node_num 512 → 1024** | "Just needs more capacity" |
| C | Component 1 (real SAM-2 labels) | (the claim) |
| D | C + 40K iters (matching SC-GS paper) | "Just needs more training" |

- **Sequencing:**
  - Day 1: ablation B on jumpingjacks (eliminates the capacity objection)
  - Day 2: ablation C on jumpingjacks (the W1 gate scene — pass/fail decision)
  - Days 3–4: A, B, C × 4 D-NeRF scenes, 3-way parallel on the lab box
  - Day 5: radial-bin Δ-PSNR aggregation; W2 report

- **Stop-gate:** if C does not beat B on the radial profile by end of day 2, pivot per `HANDOFF §6.4`.

### Slide 15 — Tier 2 pipeline ready (1 min)

- **Title:** *"SV4D 2.0 → SC-GS pipeline: built, CPU-tested, ready for first GPU run."*
- **4-stage diagram (horizontal flow):**

```
D-NeRF scene  →  SV4D 2.0 inference  →  format converter  →  SC-GS training
(1 frame         (subprocess wrapper,    (5 views × 21       (same config as
 trajectory)      ~10–15 min on H100)     frames; auto         W1 baseline)
                                          transforms_train)
```

- **Bullets:**
  - 117/117 pytest tests pass (CPU smoke runs full pipeline minus real SV4D)
  - First real GPU run pending — 48 GB card (A6000 / L40S) at ~$0.70 for full 4-scene sweep
  - Eval against original D-NeRF test cameras — same oracle as W1 baseline; numbers directly comparable

---

## Section F — Closing (2 slides, 1.5 min)

### Slide 16 — Honest weaknesses (45 s)

- **Title:** *"Four weaknesses; only one has a fundamental ceiling."*
- **Table:**

| # | Weakness | Kind | Fixable? |
|---|---|---|---|
| 1 | "Single-image" → "single short clip" reframe (SV4D 2.0 needs video input) | Architectural | Yes — Option A reframe, zero engineering cost |
| 2 | 3-component story ambitious for 4 weeks | Project management | Yes — triage; Component 3 is the safest scope-cut |
| 3 | Inter-part angular consistency metric un-validated | Engineering | Yes — 1–2 days on RigGS / MoSca outputs |
| 4 | Gating bounded by trajectory-lifting quality | Partially architectural | **Engineering wins help; consistently-wrong SV4D outputs are the fundamental ceiling.** Rare in practice. |

### Slide 17 — Asks for the group (45 s)

- **Title:** *"What I need decisions on today."*
- **Top 3 to pose live (rest stay on the slide as a list):**
  1. Sign-off on Tier 1 ablation matrix (especially the shuffled-labels row)
  2. Single-image vs single-short-clip framing — **Option A**, or push for SVD I2V front-end?
  3. SV4D variant: **4-view (sv4d2)** or **8-view (sv4d2_8views)**? 8-view fills the 120° azimuth gap.

- **Also on the slide (don't read aloud unless raised):**
  - DyCheck preprocessing — start now in parallel with Tier 1 training?
  - Component 3 scope-cut criterion if it underperforms in Tier 2?
  - Per-Gaussian (3D) part label lift — W2 day 1 or day 5?

- **Settled decisions on the slide for reference:**
  - SV4D 2.0 license — Community License free for our use, no enterprise needed
  - GPU rental — A6000/L40S 48 GB on RunPod, ~$5–10 total for the 4-scene sweep

---

## Backup slides — have ready for Q&A (don't present unless asked)

| B# | Title | Likely trigger question |
|---|---|---|
| B1 | RigGS published 40.82 vs our SC-GS-default 40.85 on jumpingjacks | "Doesn't RigGS already beat you on this scene?" |
| B2 | Per-scene contact sheets + joint zooms | "Can I see the actual failure cases?" |
| B3 | SV4D GPU pick + cost breakdown | "How much compute does this need?" |
| B4 | 4-view azimuth gap visualized | "Why does view count matter?" |
| B5 | DyCheck data format / Nerfies-adapter sketch | "When does DyCheck get going?" |
| B6 | Component 2 trajectory-lifting pipeline detail | "How does gating actually work?" |
| B7 | Inter-part angular consistency formula + metric definition | "What's the new metric?" |
| B8 | Tier 2 ablation matrix preview (rows 1, 5, 8, 9 on SV4D-supervised) | "What does the joint-system experiment look like?" |

---

## Talking-point tips

### General

- **The radial profile (slide 11) and the honest reading (slide 12) are the load-bearing pieces.** If you nail those two, the rest of the talk lands.
- **Resist the urge to over-explain Components 2 and 3.** They're stubs that haven't been validated yet. Be clear that Tier 2 is where they get measured.
- **Let visuals breathe.** On slide 2 (HEADLINE) and slide 11 (radial profile), pause for ~10 s after the visual appears before adding words. The figure does the argument.

### For the related-work section

- **Don't read the table line by line.** Group the 4 papers as paired contrasts: "RigGS does our Component 1 — but needs clean video. ViDAR does our front-end — but no articulation." That makes the intersection claim natural rather than enumerative.
- **The 4D-Fly mention is the strongest single sentence.** They named the failure mode by name. Slow down for that sentence; let it land.
- **Order on slide 5 matters.** Lead with RigGS (the most threatening competitor — closest in technique). Pre-empts "why aren't you just RigGS" in your own presentation.

### Likely Q&A and prepared responses

- **"How is this different from RigGS?"** → *"RigGS does Component 1 alone, on captured video. We deliver Component 1 in a regime where their skeleton extraction breaks — because we use SAM-2 parts on the static frame (Component 1's only structural requirement) and we filter out the drift that would otherwise corrupt it (Components 2 and 3). The 3-way intersection is the contribution; Component 1 alone has been published."*

- **"Why is 40.85 PSNR on jumpingjacks a failure?"** → *"It's not a failure number — it's a structured residual. The radial profile shows the error concentrating at limb extremities with a 5–6× ramp, while the non-articulated control sits flat. The residual structure predicts a worse failure under the deployment regime, and the W2 ablations turn that prediction into a causal claim."*

- **"D-NeRF is too simple — why not switch datasets?"** → *"D-NeRF's value is the clean evaluation oracle. Under SV4D supervision (Tier 2), the same scenes become genuinely ill-posed while keeping a fair eval oracle. That's a property DyCheck can't match. We add Tier 3 (DyCheck) for the real-world existence proof, but Tier 2 is where the load-bearing measurements happen."*

- **"What if Component 3 doesn't work?"** → *"We have a scope-cut plan. If Tier 2 ablations show Component 3 adds less than ~0.3 PSNR on top of Components 1+2, we demote it to an appendix experiment and ship the 2-component story. Components 1+2 alone (articulation + drift-rejection) is still a defensible joint contribution against the existing clusters."*

- **"How long until first SV4D-supervised result?"** → *"Pipeline is implemented and CPU-tested today (117/117 tests pass). First GPU run is ~$5–10 on RunPod for the 4-scene sweep, ~1 hour of GPU wall time including setup. SC-GS training on the converted scenes is ~15 min per scene on the lab box. Realistic timeline: first Tier 2 number within 2 days of starting the GPU run."*

---

## Practical prep

- **Pre-load** `outputs/_cross_scene_failure/HEADLINE.png` on a second monitor / tab for live deep-dive if asked.
- **Have** `docs/reports/2026-05-12_progress.md` open in a browser tab — answers most Q&A questions directly.
- **Have** `docs/runbooks/demo_runbook.md` ready — if someone asks "can you run X right now," section 8's reset commands rebuild any artifact in under 3 min.
- **Rehearse slides 11 (radial profile) and 12 (honest reading) once each.** These are the load-bearing slides.
- **Time check at slide 9** (entering Section D). If behind, compress slide 12 (honest reading) — it can shrink to 30 s if needed without losing the credibility framing.
- **Time check at slide 15** (entering Section F). If behind, drop slide 16 (weaknesses) to 30 s — list the four weaknesses on screen but only verbally elaborate on #4 (the fundamental one).

---

## Source artifacts referenced

| Slide | Asset | Path |
|---|---|---|
| 2 | Composite headline figure | `outputs/_cross_scene_failure/HEADLINE.png` |
| 6 | SAM-2 part overlay | `runs_aux/jumpingjacks_label_overlay.png` |
| 9 | Per-scene temporal-std heatmaps | `outputs/<scene>_scgs_default_node/inspection/qualitative/temporal_strobe.png` |
| 10 | Periphery/core bar chart | `outputs/_cross_scene_failure/core_vs_periphery_cross_scene.png` |
| 11 | Radial profile | `outputs/_cross_scene_failure/radial_profile_cross_scene.png` |
| 12, 14, 16, 17 | Progress report tables (copy-paste source) | `docs/reports/2026-05-12_progress.md` |
| B2 | Per-scene contact sheets + joint zooms | `outputs/<scene>_scgs_default_node/inspection/qualitative/{worst_frames_contact_sheet,joint_zoom_*}.png` |
| B8 | Tier 2 ablation matrix | `docs/reports/2026-05-12_progress.md` §5 + `docs/design/experiments.md` §3 |

All numbers in the deck come from `outputs/_cross_scene_failure/spatial_error_summary.json` and per-scene `inspection/REPORT.md` files. Reproducible via `docs/runbooks/demo_runbook.md` section 8.
