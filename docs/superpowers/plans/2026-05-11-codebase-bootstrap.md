# MotionPrior-4DGS — Codebase Bootstrap Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `EV_Final_Project` repository today with: vendored upstream deps (SC-GS, AnySplat), a CPU-testable Python package `motionprior` implementing the four novel loss/curriculum components (gating, frequency curriculum, articulation-aware ARAP, rest-state anchor), a RunPod bootstrap script, and an ownership-mapped README so two collaborators can pick up parallel work tomorrow on H100.

**Architecture:** Two-layer split. Upstream code (SC-GS, AnySplat, Wan-2.2) is cloned into a gitignored `third_party/` directory by a pinned bootstrap script — we do not vendor it into the repo. Our novel contributions live in `motionprior/`, a small Python package that depends only on `torch` + `numpy` and is import-compatible with SC-GS's environment. Every component in `motionprior/` is pure-tensor math and CPU-unit-tested before any GPU integration. Integration with SC-GS's training loop is deferred to a follow-up plan because that requires reading SC-GS source code under GPU.

**Tech Stack:** Python 3.10, PyTorch 2.1 (CPU for local dev / CUDA for H100), NumPy, PyYAML, pytest. Upstream: SC-GS (pinned `3a9d2ad4`), AnySplat (pinned `5f5e208a`), Wan2.2-I2V-A14B (HuggingFace), SAM 2 (Meta).

**Scope:** Today only. SC-GS↔motionprior hook integration, RAFT optical flow precomputation pipeline, full training script, and any GPU smoke tests are explicitly out of scope and belong to follow-up plans.

**Hardware reality:** This box is CPU-only WSL2. All tests must run on CPU torch. GPU code (rasterizer, SAM 2 inference, video diffusion) is structured as injectable / mockable so its callers stay testable.

---

## File Structure

Created today:

```
EV_Final_Project/
├── .gitignore
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml                   # motionprior package metadata
├── requirements-dev.txt             # cpu torch + pytest for local dev
├── requirements-gpu.txt             # cuda torch + flash-attn etc., applied on RunPod
├── docs/
│   ├── superpowers/plans/2026-05-11-codebase-bootstrap.md   (this file, already exists)
│   └── ownership.md                 # who owns what, per A/B split
├── third_party/                     # gitignored
│   └── README.md                    # explains how bootstrap populates this
├── scripts/
│   ├── bootstrap_third_party.sh     # clone SC-GS + AnySplat at pinned SHAs
│   └── setup_runpod.sh              # H100 env bootstrap
├── motionprior/
│   ├── __init__.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── gating.py
│   │   ├── arap_articulated.py
│   │   └── rest_state.py
│   ├── curriculum/
│   │   ├── __init__.py
│   │   └── frequency.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── arap_prior.py            # offline E_ARAP_prior precomputation
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── parts.py                 # SAM 2 part-mask → Gaussian labels
│   └── configs/
│       └── default.yaml
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_gating.py
    ├── test_arap_articulated.py
    ├── test_rest_state.py
    ├── test_frequency.py
    ├── test_arap_prior.py
    └── test_parts.py
```

Module responsibilities:

- `motionprior/losses/gating.py` — adaptive-α gating weights `w_t = exp(−α(t) · E_t)` with EMA normalization. Pure function on tensors.
- `motionprior/losses/arap_articulated.py` — per-edge λ weighting (λ_intra / λ_inter) given a K-NN edge list and per-Gaussian part labels. Applied as a multiplier on SC-GS's existing ARAP edge energies.
- `motionprior/losses/rest_state.py` — fixed-weight L2 anchor to canonical positions. One line, no schedule (per refined plan).
- `motionprior/curriculum/frequency.py` — sinusoidal-PE band mask: given iteration `i` and a schedule, returns a boolean mask over Fourier frequency channels.
- `motionprior/geometry/arap_prior.py` — given (lifted 3D control-point trajectories, K-NN graph), compute E_ARAP_prior(t) per frame. Offline, runs once per scene.
- `motionprior/segmentation/parts.py` — wraps SAM 2 to produce per-pixel part labels, then projects through a depth map to label Gaussians by part. SAM 2 calls are abstracted behind a `PartSegmenter` protocol so tests can inject masks.

---

## Task 1: Repo bootstrap — .gitignore, LICENSE, initial README, branch hygiene

**Files:**
- Create: `/home/phymersh/EV_Final_Project/.gitignore`
- Create: `/home/phymersh/EV_Final_Project/LICENSE`
- Create: `/home/phymersh/EV_Final_Project/README.md`

- [ ] **Step 1: Create .gitignore**

Write `/home/phymersh/EV_Final_Project/.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual envs
.venv/
venv/
env/
.conda/

# Vendored upstream code (populated by scripts/bootstrap_third_party.sh)
third_party/SC-GS/
third_party/AnySplat/
third_party/Wan2.2/

# Data, weights, outputs
data/
checkpoints/
outputs/
runs/
wandb/
*.ckpt
*.pth
*.safetensors

# Editors
.vscode/
.idea/
*.swp
.DS_Store

# Local config
.env
*.local.yaml
```

- [ ] **Step 2: Create LICENSE (MIT)**

Write `/home/phymersh/EV_Final_Project/LICENSE`:

```
MIT License

Copyright (c) 2026 EV Final Project contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create initial README**

Write `/home/phymersh/EV_Final_Project/README.md`:

```markdown
# MotionPrior-4DGS

Physics-constrained deformable 4D Gaussian Splatting from a single static image, supervised by a video generative model. Selectively trusts the video prior — gates physically implausible frames before they enter the photometric gradient, applies a frequency-domain curriculum during deformation MLP training, and uses an articulation-aware ARAP regularizer for piecewise-rigid structure.

> Status: under active development. Target submission: ICCV 2026 / 3DV 2026.

## Pipeline

```
Static Image
   → Wan-2.2 I2V (image-to-video diffusion)
   → AnySplat (feed-forward 3DGS + camera pose)
   → SC-GS deformation MLP, trained with:
       · physically-gated photometric loss
       · frequency-domain curriculum
       · articulation-aware ARAP
       · rest-state L2 anchor
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
```

- [ ] **Step 4: Initial commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add .gitignore LICENSE README.md docs/superpowers/plans/2026-05-11-codebase-bootstrap.md
git commit -m "chore: repo skeleton + plan"
```

---

## Task 2: Python package skeleton + dependency manifests

**Files:**
- Create: `/home/phymersh/EV_Final_Project/pyproject.toml`
- Create: `/home/phymersh/EV_Final_Project/requirements-dev.txt`
- Create: `/home/phymersh/EV_Final_Project/requirements-gpu.txt`
- Create: `/home/phymersh/EV_Final_Project/motionprior/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/losses/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/curriculum/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/geometry/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/segmentation/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/tests/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/tests/conftest.py`

- [ ] **Step 1: Write pyproject.toml**

Write `/home/phymersh/EV_Final_Project/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "motionprior"
version = "0.1.0"
description = "Physics-gated deformation field training for video-prior-supervised 4DGS."
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
authors = [
    {name = "Murphy Hsu"},
]
dependencies = [
    "numpy>=1.24",
    "torch>=2.1",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov>=4"]

[tool.setuptools.packages.find]
where = ["."]
include = ["motionprior*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 2: Write requirements-dev.txt (CPU)**

Write `/home/phymersh/EV_Final_Project/requirements-dev.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.2+cpu
numpy>=1.24,<2.0
pyyaml>=6.0
pytest>=7
pytest-cov>=4
```

- [ ] **Step 3: Write requirements-gpu.txt (H100, applied by setup_runpod.sh)**

Write `/home/phymersh/EV_Final_Project/requirements-gpu.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.1.2+cu121
torchvision==0.16.2+cu121
numpy>=1.24,<2.0
pyyaml>=6.0
pytest>=7
opencv-python-headless>=4.9
imageio>=2.34
imageio-ffmpeg>=0.4.9
einops>=0.7
transformers>=4.40
accelerate>=0.28
huggingface_hub>=0.22
```

- [ ] **Step 4: Create empty package __init__.py files**

Write `/home/phymersh/EV_Final_Project/motionprior/__init__.py`:

```python
"""MotionPrior-4DGS: physics-gated deformation field training utilities."""

__version__ = "0.1.0"
```

Write `/home/phymersh/EV_Final_Project/motionprior/losses/__init__.py`:

```python
from motionprior.losses.gating import compute_gating_weights, AdaptiveAlpha
from motionprior.losses.arap_articulated import articulated_edge_weights
from motionprior.losses.rest_state import rest_state_l2

__all__ = [
    "compute_gating_weights",
    "AdaptiveAlpha",
    "articulated_edge_weights",
    "rest_state_l2",
]
```

Write `/home/phymersh/EV_Final_Project/motionprior/curriculum/__init__.py`:

```python
from motionprior.curriculum.frequency import FrequencyCurriculum, frequency_band_mask

__all__ = ["FrequencyCurriculum", "frequency_band_mask"]
```

Write `/home/phymersh/EV_Final_Project/motionprior/geometry/__init__.py`:

```python
from motionprior.geometry.arap_prior import compute_arap_prior_energy

__all__ = ["compute_arap_prior_energy"]
```

Write `/home/phymersh/EV_Final_Project/motionprior/segmentation/__init__.py`:

```python
from motionprior.segmentation.parts import (
    PartSegmenter,
    assign_part_labels,
)

__all__ = ["PartSegmenter", "assign_part_labels"]
```

- [ ] **Step 5: Create tests/__init__.py and conftest.py**

Write `/home/phymersh/EV_Final_Project/tests/__init__.py` as empty file (one blank line).

Write `/home/phymersh/EV_Final_Project/tests/conftest.py`:

```python
import torch
import pytest


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


@pytest.fixture
def device():
    return torch.device("cpu")
```

- [ ] **Step 6: Verify package importability (modules don't exist yet — just the namespaces)**

The `__init__.py` files reference modules we haven't written yet. That's intentional — the next tasks will create them and the imports will resolve. We do NOT run pytest yet.

- [ ] **Step 7: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add pyproject.toml requirements-dev.txt requirements-gpu.txt motionprior tests
git commit -m "chore: python package skeleton + dependency manifests"
```

---

## Task 3: Frequency curriculum module (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_frequency.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/curriculum/frequency.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_frequency.py`:

```python
import torch
import pytest

from motionprior.curriculum.frequency import (
    FrequencyCurriculum,
    frequency_band_mask,
)


def test_frequency_band_mask_zeroes_high_bands():
    # 4 frequency bands, sin+cos pairs => 8 channels
    mask = frequency_band_mask(num_bands=4, k_max=2)
    assert mask.shape == (8,)
    # bands 0,1 active (channels 0..3), bands 2,3 zero (channels 4..7)
    assert mask[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert mask[4:].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_frequency_band_mask_full_when_kmax_equals_bands():
    mask = frequency_band_mask(num_bands=3, k_max=3)
    assert mask.tolist() == [1.0] * 6


def test_frequency_band_mask_empty_when_kmax_zero():
    mask = frequency_band_mask(num_bands=3, k_max=0)
    assert mask.tolist() == [0.0] * 6


def test_frequency_band_mask_invalid_kmax_raises():
    with pytest.raises(ValueError):
        frequency_band_mask(num_bands=3, k_max=5)
    with pytest.raises(ValueError):
        frequency_band_mask(num_bands=3, k_max=-1)


def test_curriculum_schedule_unlocks_bands_at_milestones():
    sched = FrequencyCurriculum(
        num_bands=6,
        milestones=[0, 5000, 10000],
        k_at_milestone=[2, 4, 6],
    )
    assert sched.k_max(0) == 2
    assert sched.k_max(4999) == 2
    assert sched.k_max(5000) == 4
    assert sched.k_max(9999) == 4
    assert sched.k_max(10000) == 6
    assert sched.k_max(20000) == 6


def test_curriculum_mask_at_iteration_matches_band_mask():
    sched = FrequencyCurriculum(
        num_bands=4,
        milestones=[0, 5000],
        k_at_milestone=[1, 4],
    )
    expected = frequency_band_mask(num_bands=4, k_max=1)
    torch.testing.assert_close(sched.mask(iteration=100), expected)


def test_curriculum_apply_zeros_correct_channels():
    sched = FrequencyCurriculum(num_bands=2, milestones=[0], k_at_milestone=[1])
    # batch x channels (2 bands * 2 (sin+cos) = 4)
    encoded = torch.ones(3, 4)
    out = sched.apply(encoded, iteration=0)
    # k_max=1 -> only band 0 (channels 0..1) active
    assert out.shape == (3, 4)
    assert torch.all(out[:, :2] == 1.0)
    assert torch.all(out[:, 2:] == 0.0)


def test_curriculum_rejects_misaligned_schedule():
    with pytest.raises(ValueError):
        FrequencyCurriculum(num_bands=4, milestones=[0, 5000], k_at_milestone=[2])  # length mismatch
    with pytest.raises(ValueError):
        FrequencyCurriculum(num_bands=4, milestones=[5000, 0], k_at_milestone=[2, 4])  # not monotonic
    with pytest.raises(ValueError):
        FrequencyCurriculum(num_bands=4, milestones=[0, 5000], k_at_milestone=[4, 2])  # k decreasing
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pip install -e . -q 2>/dev/null; pytest tests/test_frequency.py -q
```
Expected: collection error or ImportError on `motionprior.curriculum.frequency`.

- [ ] **Step 3: Implement frequency.py**

Write `/home/phymersh/EV_Final_Project/motionprior/curriculum/frequency.py`:

```python
"""Frequency-domain curriculum for the deformation MLP's temporal positional encoding.

Standard sinusoidal PE produces 2 channels per frequency band (sin, cos). We
mask out high-frequency bands during early training so the MLP can only model
slow, macro-trajectory motion. High-frequency hallucinations from the video
prior become unrepresentable until later iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


def frequency_band_mask(num_bands: int, k_max: int) -> torch.Tensor:
    """Boolean mask over a sinusoidal PE's 2*num_bands channels.

    Channels are laid out as [sin_0, cos_0, sin_1, cos_1, ...]. The first
    k_max bands are active (1.0), the rest are zeroed (0.0).
    """
    if k_max < 0 or k_max > num_bands:
        raise ValueError(
            f"k_max must be in [0, {num_bands}], got {k_max}"
        )
    mask = torch.zeros(2 * num_bands)
    if k_max > 0:
        mask[: 2 * k_max] = 1.0
    return mask


@dataclass
class FrequencyCurriculum:
    """Step-function schedule that unlocks frequency bands over training.

    Args:
        num_bands: total number of sinusoidal PE bands.
        milestones: iteration thresholds (sorted ascending, first must be 0).
        k_at_milestone: number of bands active at each milestone (monotonic
            non-decreasing, last entry <= num_bands).
    """

    num_bands: int
    milestones: Sequence[int]
    k_at_milestone: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.milestones) != len(self.k_at_milestone):
            raise ValueError(
                "milestones and k_at_milestone must have the same length"
            )
        if len(self.milestones) == 0 or self.milestones[0] != 0:
            raise ValueError("milestones must start at 0")
        if any(b < a for a, b in zip(self.milestones, self.milestones[1:])):
            raise ValueError("milestones must be non-decreasing")
        if any(b < a for a, b in zip(self.k_at_milestone, self.k_at_milestone[1:])):
            raise ValueError("k_at_milestone must be non-decreasing")
        if self.k_at_milestone[-1] > self.num_bands:
            raise ValueError(
                f"final k_at_milestone {self.k_at_milestone[-1]} exceeds num_bands {self.num_bands}"
            )

    def k_max(self, iteration: int) -> int:
        k = self.k_at_milestone[0]
        for m, km in zip(self.milestones, self.k_at_milestone):
            if iteration >= m:
                k = km
        return k

    def mask(self, iteration: int) -> torch.Tensor:
        return frequency_band_mask(self.num_bands, self.k_max(iteration))

    def apply(self, encoded: torch.Tensor, iteration: int) -> torch.Tensor:
        """Multiply the last dim of `encoded` by the band mask at `iteration`."""
        m = self.mask(iteration).to(encoded.device).to(encoded.dtype)
        return encoded * m
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_frequency.py -q
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/curriculum/frequency.py tests/test_frequency.py
git commit -m "feat(curriculum): step-schedule frequency band mask for temporal PE"
```

---

## Task 4: Physically-gated photometric weights (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_gating.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/losses/gating.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_gating.py`:

```python
import torch
import pytest

from motionprior.losses.gating import compute_gating_weights, AdaptiveAlpha


def test_gating_weights_decrease_with_energy():
    energies = torch.tensor([0.0, 0.5, 1.0, 2.0])
    w = compute_gating_weights(energies, alpha=1.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] < w[0]
    assert w[2] < w[1]
    assert w[3] < w[2]
    assert torch.all(w >= 0.0) and torch.all(w <= 1.0)


def test_gating_weights_zero_alpha_returns_ones():
    energies = torch.tensor([0.1, 5.0, 100.0])
    w = compute_gating_weights(energies, alpha=0.0)
    torch.testing.assert_close(w, torch.ones_like(w))


def test_gating_weights_rejects_negative_alpha():
    with pytest.raises(ValueError):
        compute_gating_weights(torch.tensor([0.0]), alpha=-1.0)


def test_gating_weights_rejects_negative_energy():
    with pytest.raises(ValueError):
        compute_gating_weights(torch.tensor([-0.1]), alpha=1.0)


def test_adaptive_alpha_initial_value_returns_alpha0():
    a = AdaptiveAlpha(alpha0=2.0, momentum=0.99)
    # First update with energy=1 should return alpha0 / 1 = 2.0
    assert a(torch.tensor(1.0)).item() == pytest.approx(2.0)


def test_adaptive_alpha_normalizes_by_ema():
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.5)
    # Step 1: ema = 0.5 * 0 + 0.5 * 4 = 2.0; alpha = 1 / 2 = 0.5
    out1 = a(torch.tensor(4.0))
    assert out1.item() == pytest.approx(0.5)
    # Step 2: ema = 0.5 * 2 + 0.5 * 2 = 2.0; alpha = 1 / 2 = 0.5
    out2 = a(torch.tensor(2.0))
    assert out2.item() == pytest.approx(0.5)


def test_adaptive_alpha_clamps_zero_ema():
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.5, eps=1e-6)
    out = a(torch.tensor(0.0))
    # Must not divide by zero
    assert torch.isfinite(out).all()


def test_gating_end_to_end_matches_formula():
    energies = torch.tensor([0.1, 0.2, 0.4])
    a = AdaptiveAlpha(alpha0=1.0, momentum=0.0)  # no smoothing
    # mean energy = (0.1+0.2+0.4)/3 = 0.2333...; alpha = 1/0.2333 = 4.2857...
    alpha = a(energies.mean())
    w = compute_gating_weights(energies, alpha=alpha.item())
    expected = torch.exp(-alpha * energies)
    torch.testing.assert_close(w, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_gating.py -q
```
Expected: collection error / ImportError on `motionprior.losses.gating`.

- [ ] **Step 3: Implement gating.py**

Write `/home/phymersh/EV_Final_Project/motionprior/losses/gating.py`:

```python
"""Physically-gated supervision weights.

For each frame t with precomputed ARAP-prior energy E_t, the per-frame
photometric loss is multiplied by

    w_t = exp(-alpha(t) * E_t)

where alpha(t) is adaptive — normalized by an EMA of the scene's global
ARAP energy so the gating is scene-invariant.

E_t is computed offline from the video prior's intrinsic geometry (optical
flow lifted to sparse 3D control points via the static-3DGS depth map). It
does NOT depend on the deformation MLP's output, so the gating signal is
independent of training state.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gating_weights(energies: Tensor, alpha: float) -> Tensor:
    """Per-frame gating weights.

    Args:
        energies: nonnegative tensor of shape (T,) — per-frame ARAP-prior energy.
        alpha: nonnegative scalar — gating strength (typically from AdaptiveAlpha).

    Returns:
        Tensor of shape (T,), values in (0, 1].
    """
    if alpha < 0:
        raise ValueError(f"alpha must be nonnegative, got {alpha}")
    if torch.any(energies < 0):
        raise ValueError("energies must be nonnegative")
    return torch.exp(-alpha * energies)


class AdaptiveAlpha:
    """alpha(t) = alpha0 / EMA_t(E_prior).

    EMA on a running scalar (the mean ARAP-prior energy). Call once per
    training step with the current step's mean energy; it returns the
    current alpha to use for gating.
    """

    def __init__(self, alpha0: float, momentum: float = 0.99, eps: float = 1e-6) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if alpha0 < 0:
            raise ValueError(f"alpha0 must be nonnegative, got {alpha0}")
        self.alpha0 = float(alpha0)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._ema: float | None = None

    def __call__(self, energy_mean: Tensor) -> Tensor:
        e = float(energy_mean.detach().cpu().item())
        if self._ema is None:
            self._ema = e
        else:
            self._ema = self.momentum * self._ema + (1.0 - self.momentum) * e
        return torch.tensor(self.alpha0 / max(self._ema, self.eps))

    @property
    def ema(self) -> float | None:
        return self._ema
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_gating.py -q
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/losses/gating.py tests/test_gating.py
git commit -m "feat(losses): adaptive-alpha physical gating weights"
```

---

## Task 5: Articulation-aware ARAP edge weights (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_arap_articulated.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/losses/arap_articulated.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_arap_articulated.py`:

```python
import torch
import pytest

from motionprior.losses.arap_articulated import articulated_edge_weights


def test_same_part_edges_get_intra_weight():
    # 4 control points; parts: [0,0,1,1]; edges: (0,1) and (2,3) intra; (1,2) inter
    parts = torch.tensor([0, 0, 1, 1])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    w = articulated_edge_weights(edges, parts, lambda_intra=1.0, lambda_inter=0.05)
    assert w.shape == (3,)
    assert w[0].item() == pytest.approx(1.0)
    assert w[1].item() == pytest.approx(0.05)
    assert w[2].item() == pytest.approx(1.0)


def test_single_part_collapses_to_uniform_intra():
    parts = torch.zeros(5, dtype=torch.long)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    w = articulated_edge_weights(edges, parts, lambda_intra=0.7, lambda_inter=0.05)
    torch.testing.assert_close(w, torch.full((4,), 0.7))


def test_static_part_label_zeroed():
    # part label STATIC_PART = -1 means "do not deform" — both endpoints static => weight 0
    parts = torch.tensor([-1, -1, 0, 0])
    edges = torch.tensor([[0, 1], [2, 3], [1, 2]])
    w = articulated_edge_weights(
        edges, parts, lambda_intra=1.0, lambda_inter=0.05, static_label=-1
    )
    # (0,1): both static => 0
    # (2,3): both part 0 => intra
    # (1,2): static<->dynamic => inter
    assert w[0].item() == pytest.approx(0.0)
    assert w[1].item() == pytest.approx(1.0)
    assert w[2].item() == pytest.approx(0.05)


def test_rejects_negative_lambda():
    parts = torch.tensor([0, 0])
    edges = torch.tensor([[0, 1]])
    with pytest.raises(ValueError):
        articulated_edge_weights(edges, parts, lambda_intra=-1.0, lambda_inter=0.05)
    with pytest.raises(ValueError):
        articulated_edge_weights(edges, parts, lambda_intra=1.0, lambda_inter=-0.05)


def test_rejects_out_of_range_edge_index():
    parts = torch.tensor([0, 0, 1])
    edges = torch.tensor([[0, 5]])  # 5 is out of range
    with pytest.raises(IndexError):
        articulated_edge_weights(edges, parts, lambda_intra=1.0, lambda_inter=0.05)


def test_output_is_differentiable_in_lambdas():
    # We pass scalar floats, so we test that we can wrap them in tensors and grad flows.
    parts = torch.tensor([0, 0, 1])
    edges = torch.tensor([[0, 1], [1, 2]])
    li = torch.tensor(1.0, requires_grad=True)
    le = torch.tensor(0.05, requires_grad=True)
    w = articulated_edge_weights(edges, parts, lambda_intra=li, lambda_inter=le)
    w.sum().backward()
    assert li.grad is not None and li.grad.item() == pytest.approx(1.0)
    assert le.grad is not None and le.grad.item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_arap_articulated.py -q
```
Expected: ImportError on `motionprior.losses.arap_articulated`.

- [ ] **Step 3: Implement arap_articulated.py**

Write `/home/phymersh/EV_Final_Project/motionprior/losses/arap_articulated.py`:

```python
"""Articulation-aware per-edge weights for SC-GS's ARAP regularizer.

SC-GS applies a uniform λ to every K-NN edge between control points. That
penalty over-smooths joints — at a pendulum hinge or an elbow, neighboring
points belong to different rigid parts and should be allowed to rotate
independently. We replace the uniform λ with a per-edge weight:

    λ_intra  for edges where both endpoints share a part label
    λ_inter  for edges that cross a part boundary  (much smaller, slack)
    0        for edges where both endpoints are labelled STATIC (no deformation)

Part labels come from `motionprior.segmentation.parts`. A label of
`static_label` (default -1) means the Gaussian is in the frozen background.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

LambdaLike = Union[float, Tensor]


def _as_tensor(x: LambdaLike, ref: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=ref.device, dtype=torch.float32)
    return torch.as_tensor(float(x), dtype=torch.float32, device=ref.device)


def articulated_edge_weights(
    edges: Tensor,
    parts: Tensor,
    lambda_intra: LambdaLike,
    lambda_inter: LambdaLike,
    static_label: int = -1,
) -> Tensor:
    """Compute per-edge ARAP scaling weights.

    Args:
        edges: LongTensor of shape (E, 2) — index pairs into the parts tensor.
        parts: LongTensor of shape (N,) — per-control-point part label.
            Use `static_label` (default -1) for background / non-deformable.
        lambda_intra: weight for edges where both endpoints share a non-static part.
        lambda_inter: weight for edges crossing a part boundary, or static<->dynamic.
        static_label: sentinel value for static (frozen) control points.

    Returns:
        FloatTensor of shape (E,) with values in {0, λ_inter, λ_intra}.
    """
    li = _as_tensor(lambda_intra, parts.float())
    le = _as_tensor(lambda_inter, parts.float())
    if li.item() < 0 or le.item() < 0:
        raise ValueError(
            f"lambda values must be nonnegative; got intra={li.item()}, inter={le.item()}"
        )
    n = parts.shape[0]
    if torch.any(edges < 0) or torch.any(edges >= n):
        raise IndexError(
            f"edge indices out of range for parts tensor of length {n}"
        )

    a = parts[edges[:, 0]]
    b = parts[edges[:, 1]]
    both_static = (a == static_label) & (b == static_label)
    same_part = (a == b) & (~both_static)
    # Anything else is inter (including static<->dynamic; one rigid + one frozen
    # endpoint is correctly slack — the frozen point's rest position serves as
    # the anchor, and inter weight permits small relative motion).

    weights = torch.zeros(edges.shape[0], dtype=torch.float32, device=parts.device)
    weights = torch.where(same_part, weights + li, weights)
    weights = torch.where(~(same_part | both_static), weights + le, weights)
    return weights
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_arap_articulated.py -q
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/losses/arap_articulated.py tests/test_arap_articulated.py
git commit -m "feat(losses): articulation-aware ARAP edge weights (intra/inter/static)"
```

---

## Task 6: Rest-state L2 anchor loss (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_rest_state.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/losses/rest_state.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_rest_state.py`:

```python
import torch
import pytest

from motionprior.losses.rest_state import rest_state_l2


def test_zero_deformation_zero_loss():
    deformed = torch.zeros(10, 3)
    rest = torch.zeros(10, 3)
    loss = rest_state_l2(deformed, rest)
    assert loss.item() == pytest.approx(0.0)


def test_unit_displacement_returns_correct_norm():
    rest = torch.zeros(3, 3)
    deformed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss = rest_state_l2(deformed, rest)
    # Mean of ||(1,0,0)||^2 etc = 1.0
    assert loss.item() == pytest.approx(1.0)


def test_mismatched_shapes_raise():
    rest = torch.zeros(3, 3)
    deformed = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        rest_state_l2(deformed, rest)


def test_supports_per_point_weight():
    rest = torch.zeros(2, 3)
    deformed = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    weights = torch.tensor([1.0, 0.0])  # zero out second point
    loss = rest_state_l2(deformed, rest, weights=weights)
    # Only first point counts: ||(1,0,0)||^2 = 1; weighted mean = 1 / (1+0) = 1
    assert loss.item() == pytest.approx(1.0)


def test_is_differentiable():
    rest = torch.zeros(4, 3)
    deformed = torch.randn(4, 3, requires_grad=True)
    loss = rest_state_l2(deformed, rest)
    loss.backward()
    assert deformed.grad is not None
    assert deformed.grad.shape == (4, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_rest_state.py -q
```
Expected: ImportError on `motionprior.losses.rest_state`.

- [ ] **Step 3: Implement rest_state.py**

Write `/home/phymersh/EV_Final_Project/motionprior/losses/rest_state.py`:

```python
"""Rest-state L2 anchor.

A single fixed-weight L2 term pulling each Gaussian's deformed position
toward its canonical rest position (the static input image's 3DGS). No
schedule, no decay, no floor — the elaborate version was descoped per
2026-05-11 advisor feedback ("energy may be too ambitious").
"""

from __future__ import annotations

import torch
from torch import Tensor


def rest_state_l2(
    deformed_positions: Tensor,
    rest_positions: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Weighted mean of squared displacement from the canonical rest state.

    Args:
        deformed_positions: (N, D) positions predicted by the deformation MLP.
        rest_positions: (N, D) canonical positions from the static 3DGS.
        weights: optional (N,) per-point weights. Useful for downweighting
            static-labelled Gaussians.

    Returns:
        Scalar tensor. Multiply by η (small constant, e.g. 1e-3) externally.
    """
    if deformed_positions.shape != rest_positions.shape:
        raise ValueError(
            f"shape mismatch: deformed {tuple(deformed_positions.shape)} "
            f"vs rest {tuple(rest_positions.shape)}"
        )
    sq = (deformed_positions - rest_positions).pow(2).sum(dim=-1)  # (N,)
    if weights is None:
        return sq.mean()
    if weights.shape != sq.shape:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} must match {tuple(sq.shape)}"
        )
    denom = weights.sum().clamp(min=1e-12)
    return (sq * weights).sum() / denom
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_rest_state.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/losses/rest_state.py tests/test_rest_state.py
git commit -m "feat(losses): rest-state L2 anchor (fixed weight, no schedule)"
```

---

## Task 7: Offline ARAP-prior energy precomputation (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_arap_prior.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/geometry/arap_prior.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_arap_prior.py`:

```python
import torch
import pytest

from motionprior.geometry.arap_prior import compute_arap_prior_energy


def test_static_trajectory_returns_zero_energy():
    # 5 control points, 3 frames, no motion
    P = torch.zeros(3, 5, 3)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    E = compute_arap_prior_energy(P, edges)
    assert E.shape == (3,)
    torch.testing.assert_close(E, torch.zeros(3))


def test_rigid_translation_returns_zero_energy():
    # Two frames, second is first shifted by (1, 0, 0) — ARAP should ignore this
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p1 = p0 + torch.tensor([1.0, 0.0, 0.0])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1], [0, 2]])
    E = compute_arap_prior_energy(P, edges)
    torch.testing.assert_close(E, torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_stretching_produces_nonzero_energy():
    # Frame 1 stretches one edge by 2x — non-rigid; should produce energy
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p1 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1]])
    E = compute_arap_prior_energy(P, edges)
    assert E[0].item() == pytest.approx(0.0)
    assert E[1].item() > 0.0


def test_energy_grows_with_distortion():
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    p_mild = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    p_strong = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1]])
    E_mild = compute_arap_prior_energy(torch.stack([p0, p_mild]), edges)
    E_strong = compute_arap_prior_energy(torch.stack([p0, p_strong]), edges)
    assert E_strong[1].item() > E_mild[1].item()


def test_energy_is_per_frame_mean_over_edges():
    p0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    p1 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    P = torch.stack([p0, p1])
    edges = torch.tensor([[0, 1], [1, 2]])
    E = compute_arap_prior_energy(P, edges)
    # Both edges stretch from 1 to 2 — should give equal per-edge energy
    # Per-frame energy is the mean over edges; t=0 -> 0, t=1 -> positive
    assert E[0].item() == pytest.approx(0.0)
    assert E[1].item() > 0.0


def test_invalid_input_shape_raises():
    P = torch.zeros(3, 5)  # missing the xyz dim
    edges = torch.tensor([[0, 1]])
    with pytest.raises(ValueError):
        compute_arap_prior_energy(P, edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_arap_prior.py -q
```
Expected: ImportError on `motionprior.geometry.arap_prior`.

- [ ] **Step 3: Implement arap_prior.py**

Write `/home/phymersh/EV_Final_Project/motionprior/geometry/arap_prior.py`:

```python
"""Offline ARAP-prior energy precomputation.

Given sparse control-point trajectories (typically ~512 points obtained by
lifting RAFT optical flow into 3D via the static-3DGS depth map), compute
a per-frame energy score that captures how badly the video prior violates
local rigidity. The energy is independent of any deformation MLP — it is
a fixed property of the video prior, computed once per scene.

For each edge (i, j) and frame t, the violation is the deviation between
the rotated rest-edge vector and the observed edge vector at frame t. We
use the closed-form best-fit rotation per edge neighborhood (single-edge
case: the best rotation aligns rest direction with observed direction, so
the residual reduces to a length difference). This is a tractable proxy
for full ARAP energy that is exact in the rigid limit and grows with
distortion.

Formally, for an edge with rest length L0 and frame-t length Lt:
    e_t = (Lt - L0)^2

The per-frame energy is the mean of e_t over all edges.

This proxy is invariant under rigid motion (rotation + translation): both
preserve edge lengths. It captures stretching, shearing, and joint-induced
length changes — exactly the failure modes of video-prior supervision.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_arap_prior_energy(
    positions: Tensor,
    edges: Tensor,
    rest_frame: int = 0,
) -> Tensor:
    """Per-frame ARAP-prior energy.

    Args:
        positions: (T, N, 3) — control-point positions over T frames.
        edges: (E, 2) — index pairs into the N control points.
        rest_frame: which frame defines the rest configuration. Default 0.

    Returns:
        (T,) tensor; values are nonnegative, with positions[rest_frame] giving 0.
    """
    if positions.dim() != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape (T, N, 3); got {tuple(positions.shape)}"
        )
    if edges.dim() != 2 or edges.shape[-1] != 2:
        raise ValueError(
            f"edges must have shape (E, 2); got {tuple(edges.shape)}"
        )

    rest = positions[rest_frame]                      # (N, 3)
    rest_vec = rest[edges[:, 0]] - rest[edges[:, 1]]  # (E, 3)
    rest_len = rest_vec.norm(dim=-1)                  # (E,)

    obs_vec = positions[:, edges[:, 0]] - positions[:, edges[:, 1]]  # (T, E, 3)
    obs_len = obs_vec.norm(dim=-1)                                    # (T, E)

    per_edge = (obs_len - rest_len.unsqueeze(0)).pow(2)               # (T, E)
    return per_edge.mean(dim=-1)                                      # (T,)
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_arap_prior.py -q
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/geometry/arap_prior.py tests/test_arap_prior.py
git commit -m "feat(geometry): offline ARAP-prior energy from sparse trajectories"
```

---

## Task 8: SAM 2 part-segmentation wrapper with injectable backend (TDD)

**Files:**
- Create: `/home/phymersh/EV_Final_Project/tests/test_parts.py`
- Create: `/home/phymersh/EV_Final_Project/motionprior/segmentation/parts.py`

The real SAM 2 model is GPU-only and not available on this box. We define a `PartSegmenter` protocol; tests use a fake implementation that returns known masks. The label-assignment logic (project 2D part-id image through a depth map → per-Gaussian part label) is the testable part.

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_parts.py`:

```python
import numpy as np
import torch
import pytest

from motionprior.segmentation.parts import (
    PartSegmenter,
    assign_part_labels,
)


class FakeSegmenter(PartSegmenter):
    """A toy segmenter that returns a precomputed label map."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels

    def segment(self, image: np.ndarray) -> np.ndarray:
        return self.labels


def test_assign_labels_via_2d_pixel_index():
    # 2x2 label map: top-left part 0, rest part 1
    label_map = torch.tensor([[0, 1], [1, 1]])
    # 3 Gaussians whose projected pixel coords are:
    pixel_coords = torch.tensor([[0, 0], [1, 1], [1, 0]])  # (gauss, [y, x])
    labels = assign_part_labels(label_map, pixel_coords)
    assert labels.tolist() == [0, 1, 1]


def test_assign_labels_marks_offscreen_as_static():
    label_map = torch.tensor([[0, 0], [0, 0]])
    pixel_coords = torch.tensor([[0, 0], [5, 5], [-1, 0]])  # 2 offscreen
    labels = assign_part_labels(label_map, pixel_coords, static_label=-1)
    assert labels[0].item() == 0
    assert labels[1].item() == -1
    assert labels[2].item() == -1


def test_fake_segmenter_protocol_satisfies():
    labels = np.array([[0, 0], [1, 1]], dtype=np.int64)
    seg = FakeSegmenter(labels)
    out = seg.segment(np.zeros((2, 2, 3), dtype=np.uint8))
    np.testing.assert_array_equal(out, labels)


def test_assign_labels_rejects_mismatched_label_map_dim():
    label_map = torch.zeros(2, 2, 2)  # 3-D — wrong
    pixel_coords = torch.tensor([[0, 0]])
    with pytest.raises(ValueError):
        assign_part_labels(label_map, pixel_coords)


def test_assign_labels_rejects_mismatched_pixel_coords_shape():
    label_map = torch.zeros(2, 2)
    pixel_coords = torch.tensor([0, 0])  # (2,) instead of (N, 2)
    with pytest.raises(ValueError):
        assign_part_labels(label_map, pixel_coords)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_parts.py -q
```
Expected: ImportError on `motionprior.segmentation.parts`.

- [ ] **Step 3: Implement parts.py**

Write `/home/phymersh/EV_Final_Project/motionprior/segmentation/parts.py`:

```python
"""Part segmentation wrapper.

`PartSegmenter` is a Protocol — any object exposing `.segment(image) -> ndarray`
counts. The real SAM 2 backend lives in this module too (lazy-imported so this
file is importable on a CPU-only box), but tests use injected fakes.

`assign_part_labels` is the pure-tensor function: given a 2-D per-pixel part-id
label map and a list of pixel coordinates (one per Gaussian, obtained by
projecting the canonical Gaussian centers through the static-3DGS camera), it
returns a per-Gaussian part label. Out-of-frame Gaussians are labelled static.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class PartSegmenter(Protocol):
    """Anything callable with `.segment(image: ndarray) -> ndarray` qualifies.

    The returned array is a 2-D int label map of the same height/width as the
    input image, with int part ids in [0, K-1] for K parts.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        ...


def assign_part_labels(
    label_map: Tensor,
    pixel_coords: Tensor,
    static_label: int = -1,
) -> Tensor:
    """Look up each Gaussian's part label from a 2-D part-id image.

    Args:
        label_map: (H, W) int tensor. label_map[y, x] is the part id at pixel (y, x).
        pixel_coords: (N, 2) int tensor. pixel_coords[i] = (y, x) of Gaussian i.
        static_label: label assigned to Gaussians whose pixel coords fall outside
            the label map.

    Returns:
        (N,) int tensor of part labels.
    """
    if label_map.dim() != 2:
        raise ValueError(
            f"label_map must be 2-D (H, W); got shape {tuple(label_map.shape)}"
        )
    if pixel_coords.dim() != 2 or pixel_coords.shape[-1] != 2:
        raise ValueError(
            f"pixel_coords must have shape (N, 2); got {tuple(pixel_coords.shape)}"
        )

    h, w = label_map.shape
    ys = pixel_coords[:, 0]
    xs = pixel_coords[:, 1]
    in_frame = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)

    out = torch.full((pixel_coords.shape[0],), static_label, dtype=label_map.dtype)
    if in_frame.any():
        clamped_y = ys.clamp(min=0, max=h - 1)
        clamped_x = xs.clamp(min=0, max=w - 1)
        looked_up = label_map[clamped_y, clamped_x]
        out = torch.where(in_frame, looked_up, out)
    return out


class SAM2Segmenter:
    """Thin wrapper around SAM 2 hierarchical mask generation.

    Lazy-imports `sam2` so this file can be imported on machines without
    SAM 2 installed. Construction triggers the import.

    Usage (on GPU box only):

        seg = SAM2Segmenter(checkpoint_path="checkpoints/sam2_hiera_large.pt")
        part_id_map = seg.segment(image_hwc_uint8)
    """

    def __init__(self, checkpoint_path: str, model_cfg: str = "sam2_hiera_l.yaml") -> None:
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SAM 2 is not installed in this environment. "
                "Install on the RunPod box via setup_runpod.sh."
            ) from exc

        self._model = build_sam2(model_cfg, checkpoint_path)
        self._generator = SAM2AutomaticMaskGenerator(self._model)

    def segment(self, image: np.ndarray) -> np.ndarray:
        masks = self._generator.generate(image)
        # masks is a list of dicts with 'segmentation' (bool HxW) and 'area'.
        # Sort by area descending so smaller, more specific parts overwrite larger ones.
        masks_sorted = sorted(masks, key=lambda m: -m["area"])
        h, w = image.shape[:2]
        label_map = np.full((h, w), -1, dtype=np.int64)
        for i, m in enumerate(masks_sorted):
            label_map[m["segmentation"]] = i
        return label_map
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_parts.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/segmentation/parts.py tests/test_parts.py
git commit -m "feat(segmentation): part-label assignment + injectable SAM 2 wrapper"
```

---

## Task 9: Default config (D-NeRF jumpingjacks) and config loader

**Files:**
- Create: `/home/phymersh/EV_Final_Project/motionprior/configs/default.yaml`
- Create: `/home/phymersh/EV_Final_Project/motionprior/configs/dnerf_jumpingjacks.yaml`
- Create: `/home/phymersh/EV_Final_Project/motionprior/configs/__init__.py`
- Create: `/home/phymersh/EV_Final_Project/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Write `/home/phymersh/EV_Final_Project/tests/test_config.py`:

```python
from pathlib import Path

from motionprior.configs import load_config, REPO_CONFIG_DIR


def test_default_config_loads():
    cfg = load_config("default")
    assert cfg["gating"]["alpha0"] > 0
    assert isinstance(cfg["frequency_curriculum"]["milestones"], list)
    assert isinstance(cfg["articulation"]["lambda_intra"], float)


def test_named_config_resolves():
    cfg = load_config("dnerf_jumpingjacks")
    assert cfg["scene"]["name"] == "jumpingjacks"


def test_explicit_path_loads():
    cfg_path = Path(REPO_CONFIG_DIR) / "default.yaml"
    cfg = load_config(str(cfg_path))
    assert cfg["gating"]["alpha0"] > 0
```

- [ ] **Step 2: Write configs/default.yaml**

Write `/home/phymersh/EV_Final_Project/motionprior/configs/default.yaml`:

```yaml
# Default training-time config for motionprior components.
# Scene-specific configs (e.g. dnerf_jumpingjacks.yaml) override fields here.

scene:
  name: unset
  dataset_root: data/

gating:
  alpha0: 1.0            # base gating strength
  ema_momentum: 0.99     # EMA over scene-mean ARAP-prior energy
  eps: 1.0e-6

frequency_curriculum:
  num_bands: 6
  milestones: [0, 5000, 10000]
  k_at_milestone: [2, 4, 6]

articulation:
  lambda_intra: 1.0      # in-part ARAP weight
  lambda_inter: 0.05     # cross-part (joint) ARAP weight
  static_label: -1

rest_state:
  eta: 0.001             # fixed L2 anchor weight

# Loss term weights summed into the deformation MLP's total loss.
loss_weights:
  photometric: 1.0
  arap: 1.0              # multiplied into per-edge λs above
  rest_state: 1.0        # multiplied into rest-state L2

training:
  iterations: 30000
  batch_frames: 4
  log_every: 100
```

- [ ] **Step 3: Write configs/dnerf_jumpingjacks.yaml**

Write `/home/phymersh/EV_Final_Project/motionprior/configs/dnerf_jumpingjacks.yaml`:

```yaml
# D-NeRF synthetic scene with strong articulation. Use as the headline
# articulation ablation row.

scene:
  name: jumpingjacks
  dataset_root: data/dnerf/jumpingjacks

# Slightly stronger joint slack for highly articulated motion.
articulation:
  lambda_intra: 1.0
  lambda_inter: 0.10
  static_label: -1

# D-NeRF clips are short; pull milestones inward.
frequency_curriculum:
  num_bands: 6
  milestones: [0, 3000, 8000]
  k_at_milestone: [2, 4, 6]
```

- [ ] **Step 4: Write configs/__init__.py with loader**

Write `/home/phymersh/EV_Final_Project/motionprior/configs/__init__.py`:

```python
"""YAML config loader for motionprior."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_CONFIG_DIR = str(Path(__file__).parent)


def _resolve_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".yaml" and p.exists():
        return p
    candidate = Path(REPO_CONFIG_DIR) / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not resolve config '{name_or_path}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name_or_path: str) -> dict[str, Any]:
    """Load a YAML config, deep-merging on top of default.yaml.

    `name_or_path` accepts either a bare name (e.g. "dnerf_jumpingjacks") which
    resolves to <REPO_CONFIG_DIR>/<name>.yaml, or an absolute/relative path to
    a YAML file.
    """
    path = _resolve_path(name_or_path)
    default_path = Path(REPO_CONFIG_DIR) / "default.yaml"

    with open(default_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if path.resolve() != default_path.resolve():
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    return cfg


__all__ = ["load_config", "REPO_CONFIG_DIR"]
```

- [ ] **Step 5: Run tests — expect pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest tests/test_config.py -q
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add motionprior/configs tests/test_config.py
git commit -m "feat(configs): yaml loader + default + dnerf_jumpingjacks"
```

---

## Task 10: Bootstrap script for vendored third-party code

**Files:**
- Create: `/home/phymersh/EV_Final_Project/scripts/bootstrap_third_party.sh`
- Create: `/home/phymersh/EV_Final_Project/third_party/README.md`

- [ ] **Step 1: Write scripts/bootstrap_third_party.sh**

Write `/home/phymersh/EV_Final_Project/scripts/bootstrap_third_party.sh`:

```bash
#!/usr/bin/env bash
# Clone upstream repositories at pinned SHAs into third_party/.
# third_party/ is gitignored — this script is the only authoritative source
# for which upstream commits we depend on. Update SHAs here when bumping.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${REPO_ROOT}/third_party"
mkdir -p "${TP}"

# ---- SC-GS (deformable backbone, ARAP already implemented) ----
SCGS_REPO="https://github.com/yihua7/SC-GS.git"
SCGS_SHA="3a9d2ad4e4fc058b0763d446ae9e6b1be120b872"
SCGS_DIR="${TP}/SC-GS"

if [ ! -d "${SCGS_DIR}/.git" ]; then
    echo "Cloning SC-GS @ ${SCGS_SHA}..."
    git clone "${SCGS_REPO}" "${SCGS_DIR}"
    git -C "${SCGS_DIR}" checkout "${SCGS_SHA}"
    git -C "${SCGS_DIR}" submodule update --init --recursive
else
    echo "SC-GS already present; checking pin..."
    git -C "${SCGS_DIR}" fetch --quiet
    git -C "${SCGS_DIR}" checkout "${SCGS_SHA}"
    git -C "${SCGS_DIR}" submodule update --init --recursive
fi

# ---- AnySplat (feed-forward 3DGS + pose) ----
ANYSPLAT_REPO="https://github.com/OpenRobotLab/AnySplat.git"
ANYSPLAT_SHA="5f5e208a7dd57d52e43ea0d553a95eab526e8775"
ANYSPLAT_DIR="${TP}/AnySplat"

if [ ! -d "${ANYSPLAT_DIR}/.git" ]; then
    echo "Cloning AnySplat @ ${ANYSPLAT_SHA}..."
    git clone "${ANYSPLAT_REPO}" "${ANYSPLAT_DIR}"
    git -C "${ANYSPLAT_DIR}" checkout "${ANYSPLAT_SHA}"
    git -C "${ANYSPLAT_DIR}" submodule update --init --recursive
else
    echo "AnySplat already present; checking pin..."
    git -C "${ANYSPLAT_DIR}" fetch --quiet
    git -C "${ANYSPLAT_DIR}" checkout "${ANYSPLAT_SHA}"
    git -C "${ANYSPLAT_DIR}" submodule update --init --recursive
fi

echo ""
echo "third_party/ ready. Next steps:"
echo "  - SC-GS:    cd third_party/SC-GS    && follow its README to build CUDA extensions"
echo "  - AnySplat: cd third_party/AnySplat && pip install -r requirements.txt"
echo "  - Wan-2.2 weights: download from HuggingFace (Wan-AI/Wan2.2-I2V-A14B)"
```

- [ ] **Step 2: Make script executable**

Run:
```bash
chmod +x /home/phymersh/EV_Final_Project/scripts/bootstrap_third_party.sh
```

- [ ] **Step 3: Write third_party/README.md**

Write `/home/phymersh/EV_Final_Project/third_party/README.md`:

```markdown
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
```

- [ ] **Step 4: Smoke-test the bootstrap script (clones into third_party/ which is gitignored)**

Run:
```bash
bash /home/phymersh/EV_Final_Project/scripts/bootstrap_third_party.sh 2>&1 | tail -20
ls /home/phymersh/EV_Final_Project/third_party/
```
Expected output: directories `SC-GS/` and `AnySplat/` exist; their `git -C ... rev-parse HEAD` matches the pinned SHAs.

(If network is slow or unavailable, skip this step and run it on the RunPod box. The script itself is the deliverable.)

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add scripts/bootstrap_third_party.sh third_party/README.md
git commit -m "build: bootstrap script for pinned third-party upstreams"
```

---

## Task 11: RunPod environment setup script

**Files:**
- Create: `/home/phymersh/EV_Final_Project/scripts/setup_runpod.sh`

- [ ] **Step 1: Write scripts/setup_runpod.sh**

Write `/home/phymersh/EV_Final_Project/scripts/setup_runpod.sh`:

```bash
#!/usr/bin/env bash
# One-shot RunPod (H100) environment setup.
# Run from the repo root on a fresh pod. Idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---- System packages ----
apt-get update -qq
apt-get install -y -qq \
    git wget curl unzip build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    libegl1 libglu1-mesa \
    > /dev/null

# ---- Miniconda (if not present) ----
if [ ! -d /opt/miniconda3 ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="/opt/miniconda3/bin:${PATH}"
conda init bash > /dev/null
. /opt/miniconda3/etc/profile.d/conda.sh

# ---- Three envs: motionprior dev, sc-gs, anysplat ----

# (1) motionprior-dev: lightweight, for running our pytest + losses on the pod
if ! conda env list | grep -q '^motionprior '; then
    conda create -y -n motionprior python=3.10
fi
conda activate motionprior
pip install -q -r requirements-gpu.txt
pip install -q -e .
pytest -q
conda deactivate

# (2) scgs: SC-GS training env. Defer to SC-GS's own environment.yml.
# We install motionprior into this env too so the training loop can import it.
if [ -f third_party/SC-GS/environment.yml ]; then
    if ! conda env list | grep -q '^scgs '; then
        conda env create -n scgs -f third_party/SC-GS/environment.yml
    fi
    conda activate scgs
    pip install -e .
    conda deactivate
else
    echo "WARN: third_party/SC-GS not bootstrapped yet. Run bootstrap_third_party.sh first."
fi

# (3) anysplat: AnySplat inference env. Used to produce the static 3DGS + pose.
if [ -d third_party/AnySplat ]; then
    if ! conda env list | grep -q '^anysplat '; then
        conda create -y -n anysplat python=3.10
    fi
    conda activate anysplat
    pip install -q torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    if [ -f third_party/AnySplat/requirements.txt ]; then
        pip install -q -r third_party/AnySplat/requirements.txt
    fi
    conda deactivate
else
    echo "WARN: third_party/AnySplat not bootstrapped yet. Run bootstrap_third_party.sh first."
fi

echo ""
echo "RunPod setup complete. Available envs:"
echo "  conda activate motionprior   # local tests + our package only"
echo "  conda activate scgs          # SC-GS training (with motionprior installed)"
echo "  conda activate anysplat      # AnySplat inference"
echo ""
echo "Wan-2.2 weights:"
echo "  huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir checkpoints/wan22"
```

- [ ] **Step 2: Make script executable**

Run:
```bash
chmod +x /home/phymersh/EV_Final_Project/scripts/setup_runpod.sh
```

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add scripts/setup_runpod.sh
git commit -m "build: RunPod H100 env bootstrap (motionprior + scgs + anysplat envs)"
```

---

## Task 12: Ownership map + final README polish + push

**Files:**
- Create: `/home/phymersh/EV_Final_Project/docs/ownership.md`
- Modify: `/home/phymersh/EV_Final_Project/README.md` (status section update)

- [ ] **Step 1: Write docs/ownership.md**

Write `/home/phymersh/EV_Final_Project/docs/ownership.md`:

```markdown
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

- `master`: stable, every commit passes `pytest -q`
- Feature branches optional; for fast iteration in this 1-month window, commit straight to `master` with frequent small commits
- Push after every commit so the other person sees state immediately
```

- [ ] **Step 2: Update README status section**

Edit `/home/phymersh/EV_Final_Project/README.md`. Replace the line starting with `> Status:` with:

```
> Status: bootstrap complete (2026-05-11). Components implemented and CPU-tested: gating, frequency curriculum, articulation-aware ARAP, rest-state L2, ARAP-prior precomputation, part-label assignment. Next: SC-GS hook + Wan-2.2/AnySplat front-end on H100.
```

- [ ] **Step 3: Final full test pass**

Run:
```bash
cd /home/phymersh/EV_Final_Project && pytest -q
```
Expected: ≥ 33 passed. (8 freq + 8 gating + 6 arap_articulated + 5 rest_state + 6 arap_prior + 5 parts + 3 config = 41)

- [ ] **Step 4: Commit and push**

Run:
```bash
cd /home/phymersh/EV_Final_Project
git add docs/ownership.md README.md
git commit -m "docs: ownership map + bootstrap-complete status"
git push -u origin master
```

Expected: push succeeds; teammate can `git clone https://github.com/murphy-cthsu/EV_Final_Project.git` to start.

---

## Out of scope today (next plan)

- SC-GS source-code reading + hook integration of all four motionprior losses into SC-GS's training step.
- RAFT optical-flow runner + 2D-to-3D lifting via static-3DGS depth → real per-scene `E_ARAP_prior(t)` tensors.
- AnySplat front-end runner script.
- Wan-2.2 I2V inference wrapper.
- End-to-end `scripts/run_training.py`.
- Real SAM 2 part-segmentation on D-NeRF scenes (requires GPU; for now we have the wrapper and the labeling logic).
- Baselines (SC-GS default, GaussVideoDreamer-style masking, CAT4D-style).
- Evaluation pipeline (PSNR/SSIM/LPIPS, inter-part angular consistency).

A follow-up plan `2026-05-12-scgs-integration.md` will cover the SC-GS hook once we're on a GPU box.
