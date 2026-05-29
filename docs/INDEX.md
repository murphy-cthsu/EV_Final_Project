# `docs/` index

> Naming rule:
> - **Snapshots** (dated artifacts that captured state at a point in time) live
>   under `reports/` or `planning/` with a `YYYY-MM-DD_*.md` prefix so they
>   sort chronologically.
> - **Living references** (design docs, runbooks, API refs that we keep
>   updating) use stable lowercase descriptive names — no date prefix, because
>   a date in the filename would lie as soon as we edit them.

## Layout

```
docs/
├── design/                            # living references — what we're building & why
│   ├── motion_design.md               # current motion-design direction (latest, supersedes CVCG/C3)
│   ├── scgs_hook_design.md            # 3 patch sites + contract for SC-GS
│   ├── vwm_framing.md                 # paper intro source-of-truth
│   ├── sv4d2_api.md                   # SV4D 2.0 inference API reference
│   ├── experiments.md                 # ablation matrix + baselines (spec)
│   └── pipeline.png                   # block diagram
├── runbooks/                          # how to run / how to reproduce
│   ├── demo_runbook.md                # copy-paste reproduction commands (<2 min each)
│   └── sv4d_runbook.md                # CPU smoke / lab A4500 / RunPod
├── reports/                           # dated snapshots — sorted chronologically
│   ├── 2026-05-12_progress.md         # W1 closeout progress report
│   ├── 2026-05-13_pipeline_state.md   # pipeline-state snapshot for group meeting
│   ├── 2026-05-29_checkpoint.md       # mid-experiment honest inventory
│   └── 2026-05-29_final_report.md     # **final project report**
├── planning/                          # ownership + slides
│   ├── ownership.md                   # team split (no date — living)
│   └── 2026-05-13_slide_plan.md       # group-meeting slide plan (dated)
└── superpowers/                       # legacy bootstrap plans (do not edit)
    └── plans/2026-05-11-codebase-bootstrap.md
```

## Time-ordered reading list (most recent first)

| Date | File | What it is |
|---|---|---|
| 2026-05-29 | `reports/2026-05-29_final_report.md` | Final project report (W1 + W2 + W3) |
| 2026-05-29 | `reports/2026-05-29_checkpoint.md` | Mid-experiment inventory; superseded by the final report |
| 2026-05-29 | `design/motion_design.md` | Current motion-design rationale |
| 2026-05-28 | `design/scgs_hook_design.md` | SC-GS patch site contract (active) |
| 2026-05-13 | `planning/2026-05-13_slide_plan.md` | Slide plan for the 5/13 advisor meeting |
| 2026-05-13 | `reports/2026-05-13_pipeline_state.md` | Pipeline-state snapshot |
| 2026-05-13 | `runbooks/demo_runbook.md` | Reproduction commands |
| 2026-05-13 | `runbooks/sv4d_runbook.md` | SV4D operational runbook |
| 2026-05-13 | `design/vwm_framing.md` | Paper-intro framing |
| 2026-05-13 | `design/experiments.md` | Experiment design spec |
| 2026-05-12 | `reports/2026-05-12_progress.md` | W1 closeout progress report |
| 2026-05-12 | `design/sv4d2_api.md` | SV4D 2.0 API reference |
| 2026-05-12 | `planning/ownership.md` | Team split |

## Related artifact indexes

- [`outputs/INDEX.md`](../outputs/INDEX.md) — what every run directory is and which ones are still active
- [`runs_aux/INDEX.md`](../runs_aux/INDEX.md) — same, for intermediate artifacts
