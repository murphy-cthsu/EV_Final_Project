"""Video Generative Model (VGM) front-end abstraction.

The Option C pipeline begins with a static input image and uses a video
generative model to produce supervision video. Two backends:

* ``SV4D2Adapter`` -- primary; Stability AI's SV4D 2.0 (multi-view-aware,
  commercial license). Better single-image-to-4D fit.
* ``Wan22I2VAdapter`` -- fallback; Alibaba's Wan-2.2 I2V (monocular, Apache 2.0).
  Used if SV4D 2.0 licensing is blocked.

Both backends lazy-import heavyweight GPU deps so this module imports cleanly
on the CPU dev box. The ``VGM`` Protocol lets training code accept *any*
implementation (real backends, ``FakeVGM`` for tests, future backends).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class VGM(Protocol):
    """Protocol any video generative model wrapper must satisfy.

    A VGM takes a single static image and produces a sequence of video frames
    that the downstream pipeline treats as pseudo-multiview supervision.
    """

    def generate(
        self,
        image: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """Generate ``num_frames`` frames from the input image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.
            num_frames: number of frames to generate.

        Returns:
            ``(num_frames, H, W, 3)`` uint8 array. Implementations must NOT
            change ``H, W`` -- if their model has a fixed output resolution,
            they must resize back to match the input.
        """
        ...


class FakeVGM:
    """Test double. Returns the input image as frame 0 and zero-filled frames
    afterwards (so output is deterministic and lightweight).
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def generate(self, image: np.ndarray, num_frames: int) -> np.ndarray:
        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Image shape {image.shape} doesn't match FakeVGM "
                f"({self.height}, {self.width}, 3)"
            )
        out = np.zeros((num_frames, self.height, self.width, 3), dtype=np.uint8)
        out[0] = image
        return out


def normalize_video_frames(frames: np.ndarray) -> torch.Tensor:
    """uint8 (T, H, W, 3) -> float32 (T, 3, H, W) in [0, 1].

    The standard model-facing layout. Use after ``vgm.generate(...)`` and
    before feeding the supervision into AnySplat / SC-GS.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (T, H, W, 3); got {frames.shape}"
        )
    t = torch.from_numpy(frames).float() / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


def save_video_frames(
    frames: np.ndarray,
    out_dir: Path | str,
    prefix: str = "frame",
) -> list[Path]:
    """Save each frame as a PNG. Returns the list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError:
        try:
            import imageio as iio  # type: ignore  # noqa: F401
        except ImportError:
            # Fall back to PIL for tests on a minimal CPU env that doesn't
            # have imageio. Real pipeline use will have imageio in the GPU env.
            from PIL import Image
            paths = []
            for i, f in enumerate(frames):
                p = out_dir / f"{prefix}_{i:04d}.png"
                Image.fromarray(f).save(p)
                paths.append(p)
            return paths
    paths = []
    for i, f in enumerate(frames):
        p = out_dir / f"{prefix}_{i:04d}.png"
        iio.imwrite(str(p), f)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------- #
# Real backends -- lazy-imported so this module imports on CPU                  #
# ---------------------------------------------------------------------------- #


class SV4D2Adapter:
    """SV4D 2.0 wrapper. GPU-required.

    Usage (on the pod):

        from motionprior.integration.vgm import SV4D2Adapter
        vgm = SV4D2Adapter(checkpoint="checkpoints/sv4d2.0", num_views=21)
        frames = vgm.generate(image, num_frames=16)
        # frames: (16, H, W, 3) uint8, where 16 here means 16 views (SV4D 2.0
        # produces multi-view samples for one timestep; for multi-timestep
        # output, call generate() per timestep or use the model's native
        # 4D mode -- see SV4D 2.0 docs).

    NOTE: SV4D 2.0's exact API will pin the implementation on the pod. The
    constructor signature here is a placeholder; the real adapter will refine
    it after we read the SV4D 2.0 release docs on the GPU box.
    """

    def __init__(
        self,
        checkpoint: str,
        num_views: int = 21,
        num_frames: int = 21,
        device: str = "cuda",
    ) -> None:
        try:
            import diffusers  # type: ignore # noqa: F401
            import transformers  # type: ignore # noqa: F401
            # SV4D 2.0-specific imports go here. As of 2026-05-12, exact
            # pipeline class is `stabilityai/sv4d2.0`; the loader is TBD.
            raise ImportError(
                "SV4D 2.0 adapter not yet pinned -- finalize on the GPU pod "
                "where stabilityai/sv4d2.0 weights are accessible. "
                "(Lazy-checking diffusers + transformers were importable; "
                "real adapter coming W1 of the experiment timeline.)"
            )
        except ImportError as exc:
            raise ImportError(
                f"SV4D 2.0 requires GPU + diffusers + transformers + "
                f"stabilityai/sv4d2.0 weights. Install on the RunPod box: "
                f"`conda activate sv4d2 && pip install diffusers transformers`. "
                f"Underlying error: {exc}"
            ) from exc

    def generate(self, image: np.ndarray, num_frames: int) -> np.ndarray:
        raise NotImplementedError("Fill in on the GPU pod once SV4D 2.0 API is loaded.")


class Wan22I2VAdapter:
    """Wan-2.2 I2V-A14B wrapper. GPU-required.

    Usage (on the pod):

        vgm = Wan22I2VAdapter(checkpoint="Wan-AI/Wan2.2-I2V-A14B")
        frames = vgm.generate(image, num_frames=80)

    Wan-2.2 is monocular (single output view per call) -- to use it as a
    multi-view supervisor, run with a camera trajectory specification and let
    AnySplat downstream extract multi-view samples from the dense temporal
    output. Less robust than SV4D 2.0's explicit multi-view-aware path.
    """

    def __init__(
        self,
        checkpoint: str = "Wan-AI/Wan2.2-I2V-A14B",
        device: str = "cuda",
        resolution: int = 480,
    ) -> None:
        try:
            import diffusers  # type: ignore # noqa: F401
            import transformers  # type: ignore # noqa: F401
            raise ImportError(
                "Wan-2.2 I2V adapter not yet pinned -- finalize on the GPU pod "
                "where the model weights are downloadable from HuggingFace. "
                "(Lazy-checking diffusers + transformers were importable; "
                "real adapter coming W1 of the experiment timeline.)"
            )
        except ImportError as exc:
            raise ImportError(
                f"Wan-2.2 I2V requires GPU + diffusers + transformers + Wan-AI "
                f"weights. Install on the RunPod box: `conda activate wan22 && "
                f"pip install diffusers transformers accelerate`. "
                f"Underlying error: {exc}"
            ) from exc

    def generate(self, image: np.ndarray, num_frames: int) -> np.ndarray:
        raise NotImplementedError("Fill in on the GPU pod once Wan-2.2 API is loaded.")
