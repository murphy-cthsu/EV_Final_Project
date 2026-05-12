"""Video Generative Model (VGM) front-end abstraction.

The Option C pipeline begins with a static input image (or a short
single-view video) and uses a video generative model to produce a
multi-view video matrix that the downstream pipeline treats as
pseudo-multi-view supervision. Two backends:

* ``SV4D2Adapter`` -- primary; Stability AI's SV4D 2.0. Multi-view-aware:
  takes one input video and emits 4 (or 8) novel-view videos of the same
  length. Wraps ``third_party/generative-models/scripts/sampling/
  simple_video_sample_4d2.py`` as a subprocess.
* ``Wan22I2VAdapter`` -- fallback; Alibaba's Wan-2.2 I2V (monocular, Apache 2.0).
  Used if SV4D 2.0 licensing is blocked.

Both backends lazy-import heavyweight GPU deps and call the upstream code
via subprocess so this module imports cleanly on a CPU dev box.

See docs/sv4d2_api.md for the API reference these wrappers follow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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


REPO_ROOT = Path(__file__).resolve().parents[2]
SV4D_REPO = REPO_ROOT / "third_party" / "generative-models"
SV4D_SCRIPT = SV4D_REPO / "scripts" / "sampling" / "simple_video_sample_4d2.py"

# Hardcoded camera convention from simple_video_sample_4d2.py:149-162
SV4D2_AZIMUTHS_DEG = {
    "sv4d2": [240.0, 0.0, 60.0, 120.0, 180.0],
    "sv4d2_8views": [330.0, 0.0, 30.0, 75.0, 120.0, 165.0, 210.0, 255.0, 300.0],
}
SV4D2_N_FRAMES = {"sv4d2": 21, "sv4d2_8views": 21}


@dataclass
class SV4DResult:
    """The output of one SV4D 2.0 run, parsed into a uniform per-view-per-frame layout.

    ``frames`` is a numpy array of shape ``(V, T, H, W, 3)`` uint8 with the
    convention that view 0 is the input video (saved by SV4D after preprocessing)
    and views 1..V-1 are the novel views generated by the model.

    ``azimuths_deg`` and ``elevations_deg`` are length-V; they list each view's
    camera angle so the downstream converter can build transform matrices.
    """
    frames: np.ndarray                       # (V, T, H, W, 3) uint8
    azimuths_deg: list[float]                 # length V
    elevations_deg: list[float]               # length V
    variant: str                              # "sv4d2" or "sv4d2_8views"
    seed: int
    img_size: int
    n_frames: int
    output_dir: Path                          # where the SV4D mp4s live
    metadata: dict = field(default_factory=dict)


class SV4D2Adapter:
    """SV4D 2.0 wrapper. GPU-required at ``generate`` time.

    Implementation strategy: launches the upstream
    ``simple_video_sample_4d2.py`` as a subprocess (the script wires global
    state via Fire + module-level config dicts; calling in-process risks
    polluting the host process). After the subprocess finishes, we parse the
    output mp4s back into a uniform ``(V, T, H, W, 3)`` tensor.

    Usage::

        adapter = SV4D2Adapter(
            checkpoint=Path("checkpoints/sv4d2.safetensors"),
            variant="sv4d2",
            img_size=576,
            n_frames=21,
        )
        result = adapter.run(
            input_video_dir=Path("data/sv4d_inputs/jumpingjacks_v0"),
            output_dir=Path("outputs/sv4d_supervised/jumpingjacks"),
        )
        # result.frames is (5, 21, 576, 576, 3) uint8.
    """

    def __init__(
        self,
        checkpoint: Path | str = "checkpoints/sv4d2.safetensors",
        variant: str = "sv4d2",
        img_size: int = 576,
        n_frames: int = 21,
        elevation_deg: float = 0.0,
        seed: int = 23,
        encoding_t: int = 8,
        decoding_t: int = 4,
        device: str = "cuda",
        python_exe: str | None = None,
    ) -> None:
        if variant not in SV4D2_AZIMUTHS_DEG:
            raise ValueError(
                f"variant must be one of {list(SV4D2_AZIMUTHS_DEG)}, got {variant!r}"
            )
        self.checkpoint = Path(checkpoint)
        self.variant = variant
        self.img_size = img_size
        self.n_frames = n_frames
        self.elevation_deg = elevation_deg
        self.seed = seed
        self.encoding_t = encoding_t
        self.decoding_t = decoding_t
        self.device = device
        self.python_exe = python_exe or sys.executable

        self.azimuths_deg = list(SV4D2_AZIMUTHS_DEG[variant])
        self.n_views = len(self.azimuths_deg)

        if not SV4D_SCRIPT.is_file():
            raise FileNotFoundError(
                f"SV4D 2.0 script missing at {SV4D_SCRIPT}. "
                f"Run: cd third_party && git clone "
                f"https://github.com/Stability-AI/generative-models.git"
            )

    def run(
        self,
        input_video_dir: Path,
        output_dir: Path,
        extra_args: list[str] | None = None,
    ) -> SV4DResult:
        """Run SV4D 2.0 inference and parse the outputs.

        Args:
            input_video_dir: directory containing the input video as ordered PNGs
                (``frame_0000.png`` ... ``frame_NNNN.png``). Must have at least
                ``self.n_frames`` files. SV4D will read them in sorted order.
            output_dir: where to write the SV4D mp4 outputs. Created if absent.
            extra_args: optional extra CLI args passed through to the upstream script.
        """
        input_video_dir = Path(input_video_dir).resolve()
        output_dir = Path(output_dir).resolve()
        if not input_video_dir.is_dir():
            raise FileNotFoundError(f"{input_video_dir} not found")

        n_frames_found = len(list(input_video_dir.glob("*.png")))
        if n_frames_found < self.n_frames:
            raise ValueError(
                f"input_video_dir has {n_frames_found} PNGs, need >= "
                f"{self.n_frames}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_exe,
            "-u",
            "scripts/sampling/simple_video_sample_4d2.py",
            f"--input_path={input_video_dir}",
            f"--model_path={self.checkpoint.resolve()}",
            f"--output_folder={output_dir}",
            f"--n_frames={self.n_frames}",
            f"--img_size={self.img_size}",
            f"--seed={self.seed}",
            f"--encoding_t={self.encoding_t}",
            f"--decoding_t={self.decoding_t}",
            f"--elevations_deg={self.elevation_deg}",
            f"--device={self.device}",
        ]
        if extra_args:
            cmd.extend(extra_args)

        log_path = output_dir / "sv4d_run.log"
        env = os.environ.copy()
        # The script does sys.path.append from __file__'s directory; running
        # from SV4D_REPO ensures this resolves correctly.
        with log_path.open("w") as f:
            f.write(f"# cwd: {SV4D_REPO}\n")
            f.write(f"# cmd: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(SV4D_REPO),
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"SV4D 2.0 subprocess failed (exit {proc.returncode}). "
                f"See {log_path}"
            )

        return self._parse_outputs(output_dir)

    # ------------------------------------------------------------------ #
    # parsing helpers
    # ------------------------------------------------------------------ #
    def _parse_outputs(self, output_dir: Path) -> SV4DResult:
        """Demux the {base_count}_vNNN.mp4 + preprocessed-input mp4 to (V, T, H, W, 3)."""
        sv4d_sub = output_dir / self.variant
        if not sv4d_sub.is_dir():
            raise FileNotFoundError(
                f"Expected SV4D output under {sv4d_sub}; found "
                f"{list(output_dir.iterdir())}"
            )

        # Find the most-recent base_count by sorting v001 files
        v001_candidates = sorted(sv4d_sub.glob("*_v001.mp4"))
        if not v001_candidates:
            raise FileNotFoundError(f"No *_v001.mp4 found under {sv4d_sub}")
        base = v001_candidates[-1].stem.split("_")[0]

        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio  # type: ignore

        per_view: list[np.ndarray] = []
        # view 0: the preprocessed input video — SV4D saves it inside output_dir
        # with the same basename as the input directory.
        input_mp4 = self._find_preprocessed_input(sv4d_sub, base)
        if input_mp4 is None:
            # Some versions save the preprocessed input one level up; check there too
            input_mp4 = self._find_preprocessed_input(output_dir, base)
        if input_mp4 is None:
            raise FileNotFoundError(
                f"Could not locate preprocessed input mp4 for base={base} "
                f"in {sv4d_sub} or {output_dir}"
            )
        per_view.append(np.asarray(iio.imread(input_mp4)))

        # views 1..V-1: novel views
        for v in range(1, self.n_views):
            mp4 = sv4d_sub / f"{base}_v{v:03d}.mp4"
            if not mp4.is_file():
                raise FileNotFoundError(f"Missing novel-view output {mp4}")
            per_view.append(np.asarray(iio.imread(mp4)))

        # Ensure consistent T,H,W across views; truncate any extras to n_frames.
        per_view = [v[: self.n_frames] for v in per_view]
        T = min(v.shape[0] for v in per_view)
        per_view = [v[:T] for v in per_view]
        frames = np.stack(per_view, axis=0)  # (V, T, H, W, 3) uint8

        result = SV4DResult(
            frames=frames,
            azimuths_deg=list(self.azimuths_deg),
            elevations_deg=[self.elevation_deg] * self.n_views,
            variant=self.variant,
            seed=self.seed,
            img_size=self.img_size,
            n_frames=T,
            output_dir=output_dir,
            metadata={"base": base},
        )
        # Sidecar JSON for reproducibility
        (output_dir / "sv4d_result.json").write_text(json.dumps({
            "variant": self.variant,
            "seed": self.seed,
            "img_size": self.img_size,
            "n_frames": T,
            "n_views": self.n_views,
            "azimuths_deg": self.azimuths_deg,
            "elevations_deg": [self.elevation_deg] * self.n_views,
            "checkpoint": str(self.checkpoint),
            "base": base,
            "frames_shape": list(frames.shape),
        }, indent=2))
        return result

    @staticmethod
    def _find_preprocessed_input(folder: Path, base: str) -> Path | None:
        """SV4D saves the preprocessed input as `<something>.mp4` (no _vNNN suffix)."""
        for p in folder.glob("*.mp4"):
            if "_v" in p.stem:
                continue
            return p
        return None

    # ------------------------------------------------------------------ #
    # protocol-compatible single-image-style call (convenience)
    # ------------------------------------------------------------------ #
    def generate(self, image: np.ndarray, num_frames: int) -> np.ndarray:
        """VGM-protocol entry point. Replicates ``image`` ``num_frames`` times,
        feeds it to SV4D as a static input video, returns only view 1 to match
        the (T, H, W, 3) Protocol signature.

        For the multi-view output that SC-GS actually consumes, call ``run()``
        and use the full ``SV4DResult.frames``.
        """
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"image must be (H, W, 3) uint8; got {image.shape} {image.dtype}"
            )
        if num_frames != self.n_frames:
            raise ValueError(
                f"This adapter is configured for n_frames={self.n_frames}; "
                f"got {num_frames}"
            )
        import tempfile
        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio  # type: ignore
        with tempfile.TemporaryDirectory() as td_in, tempfile.TemporaryDirectory() as td_out:
            td_in_p = Path(td_in)
            for i in range(num_frames):
                iio.imwrite(td_in_p / f"frame_{i:04d}.png", image)
            result = self.run(td_in_p, Path(td_out))
            return result.frames[1].copy()  # the first novel view


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
