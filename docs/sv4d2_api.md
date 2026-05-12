# SV4D 2.0 inference API — reference

> Source: `third_party/generative-models/scripts/sampling/simple_video_sample_4d2.py` (rev: clone 2026-05-12)
>
> Read once. Lookup table for our adapter and converter.

## Entry point

```bash
cd third_party/generative-models
python scripts/sampling/simple_video_sample_4d2.py \
    --input_path <video.gif|video.mp4|dir_of_pngs> \
    --model_path checkpoints/sv4d2.safetensors \
    --output_folder outputs/ \
    --n_frames 21 \
    --img_size 576 \
    --seed 23 \
    --encoding_t 8 \
    --decoding_t 4 \
    --elevations_deg 0.0 \
    --azimuths_deg None     # use defaults
```

CLI is via [`fire.Fire(sample)`]. The function does **not return**; it writes mp4 files to `<output_folder>/sv4d2/`.

## Output layout

```
<output_folder>/sv4d2/
  {input}.mp4              # the preprocessed input video (view 0)
  000000_v001.mp4          # novel view at azimuth 0°
  000000_v002.mp4          # novel view at azimuth 60°
  000000_v003.mp4          # novel view at azimuth 120°
  000000_v004.mp4          # novel view at azimuth 180°
```

If `output_folder/sv4d2/` already has N files matching `*.mp4`, the next run uses `base_count = N // n_views`. So re-runs auto-increment instead of overwriting.

Each output video has `n_frames` (default 21) frames at `img_size × img_size` (default 576).

## Camera convention

Per `simple_video_sample_4d2.py:149-162`:

| variant | n_views (incl. input) | Default azimuths (degrees) | Elevation |
|---|---:|---|---|
| **sv4d2** (the 4-view model) | 5 | `[0, 60, 120, 180, 240]`  — input view is azimuth **240°**, novel views at 0/60/120/180° | configurable; default 0° for all |
| sv4d2_8views | 9 | `[0, 30, 75, 120, 165, 210, 255, 300, 330]` — input view at 330°, 8 novel views | configurable; default 0° for all |

Important: SV4D **does not output any extrinsic-matrix metadata**. We must construct camera transforms ourselves from the azimuths above. The model is object-centric: cameras orbit a unit-ish sphere with the object at origin. The internal coordinate frame is consistent with standard novel-view-synthesis pipelines, but the orbit radius is not exposed — we choose it when converting to D-NeRF format.

## Input constraints

| | Value |
|---|---|
| Format | gif, mp4, or dir of jpg/jpeg/png |
| Frames | exactly `n_frames` (default 21); excess truncated, fewer fails |
| Resolution | resized to `img_size × img_size` (default 576) |
| Background | white preferred; use `--remove_bg=True` for rembg if input has plain bg; or pre-segment with SAM 2 |
| `image_frame_ratio` | 0.917 default — controls object centering/cropping. Lower → object fills more of the frame |

D-NeRF train images are 800×800 RGBA with transparent backgrounds. To use as input video: composite alpha onto white at the alpha threshold, resize 800→576, save 21 frames as a gif.

## Compute envelope

| Setting | VRAM target | Notes |
|---|---|---|
| `encoding_t=8, decoding_t=4` | ~40 GB | Default; H100/A100 80GB |
| `encoding_t=1, decoding_t=1` | unknown lower bound | README says "low VRAM"; experiment to find threshold |
| `--img_size 512` | smaller still | README's other low-VRAM suggestion |

**Tentative**: A4500 (20 GB) with `encoding_t=1, decoding_t=1, img_size=512` *might* fit. Empirical test needed. If not, RunPod H100.

## Internal call graph (for reference if we need to refactor)

```
simple_video_sample_4d2.sample()
├── sv4d_helpers.preprocess_video()       # crop/resize/rembg
├── sv4d_helpers.read_video()             # → (T, 3, H, W) on device
├── sv4d_helpers.load_model()             # → SGM model object
└── for each t0 in range(0, n_frames, T-1):
    ├── sv4d_helpers.run_img2vid()        # the diffusion sampling
    └── img_matrix[t][v] = samples[...]
└── for v in view_indices:
    └── sv4d_helpers.save_video()         # mp4 per view
```

## Wrapping decisions for our pipeline

1. **Subprocess wrapper, not import wrapper.** Their script wires too much top-level state (`load_model`, `Fire`, global config dicts). Cleaner to launch as subprocess from `SV4D2Adapter.generate()`, then parse the output mp4s.
2. **Save inputs to a temp dir.** We composite D-NeRF RGBA→white-RGB and write `frame_NNN.png` to `<scene>/_sv4d_input/`, then point `--input_path` at that dir.
3. **Output convention.** SV4D writes mp4s with autoincrementing `base_count`. We give it a fresh output dir per scene+seed so `base_count == 0`.
4. **Reading outputs.** Use `imageio` to demux the 4 output mp4s + the saved input mp4 into per-frame PNGs. SC-GS expects PNGs anyway.
5. **Constructing `transforms_train.json`.** For each of the 5 views at azimuths `[240, 0, 60, 120, 180]`, build a 4×4 transform_matrix in D-NeRF/Blender convention (camera at `(r·cos(elev)·sin(az), r·sin(elev), r·cos(elev)·cos(az))`, looking at origin, up = +Y). Radius can match D-NeRF's typical 4.0 to keep PSNR comparable to original-supervision baseline.
6. **Time field.** D-NeRF uses `time ∈ [0,1]`. With 21 frames: `time_i = i / 20`.
7. **Eval oracle alignment.** Held-out test cameras come from the original D-NeRF `transforms_test.json` (unchanged) — those 20 test cameras at their original timestamps. We evaluate the SV4D-supervised SC-GS run against the same D-NeRF GT we used for the baseline. The numbers are directly comparable.
