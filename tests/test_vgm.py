"""Tests for the VGM wrapper module.

Real VGM backends (SV4D 2.0, Wan-2.2-I2V) need GPU + large model weights to
run, so we test the abstraction with a FakeVGM that returns scripted frames.
The wrapper code under test is CPU-only.
"""

import numpy as np
import pytest
import torch

from motionprior.integration.vgm import (
    VGM,
    FakeVGM,
    save_video_frames,
    normalize_video_frames,
)


def test_fake_vgm_returns_requested_count():
    vgm = FakeVGM(height=64, width=64)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = vgm.generate(img, num_frames=16)
    assert frames.shape == (16, 64, 64, 3)
    assert frames.dtype == np.uint8


def test_fake_vgm_propagates_input_pixel():
    vgm = FakeVGM(height=4, width=4)
    img = np.full((4, 4, 3), 137, dtype=np.uint8)
    frames = vgm.generate(img, num_frames=3)
    # The first frame should be exactly the input
    np.testing.assert_array_equal(frames[0], img)


def test_normalize_video_frames_uint8_to_float():
    frames = np.full((3, 8, 8, 3), 255, dtype=np.uint8)
    out = normalize_video_frames(frames)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 3, 8, 8)  # NCHW
    assert out.dtype == torch.float32
    assert out.min().item() == pytest.approx(1.0)


def test_normalize_video_frames_rejects_wrong_shape():
    frames = np.zeros((8, 8, 3), dtype=np.uint8)  # missing time dim
    with pytest.raises(ValueError):
        normalize_video_frames(frames)


def test_save_video_frames_writes_files(tmp_path):
    frames = np.random.randint(0, 256, (5, 16, 16, 3), dtype=np.uint8)
    out_dir = tmp_path / "vgm_out"
    paths = save_video_frames(frames, out_dir, prefix="frame")
    assert len(paths) == 5
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_vgm_protocol_accepts_any_generate_method():
    """Any object with `.generate(image, num_frames)` satisfies the Protocol."""

    class CustomVGM:
        def generate(self, image, num_frames):
            return np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)

    vgm: VGM = CustomVGM()  # structural typing
    out = vgm.generate(np.zeros((4, 4, 3), dtype=np.uint8), num_frames=2)
    assert out.shape == (2, 4, 4, 3)


def test_sv4d2_adapter_construction_is_cpu_safe():
    """Constructor must not require a GPU. It only records parameters and
    confirms the upstream script is on disk; subprocess launch is deferred
    until .run() is called."""
    from motionprior.integration.vgm import SV4D2Adapter, SV4D2_AZIMUTHS_DEG
    a = SV4D2Adapter(checkpoint="checkpoints/sv4d2.safetensors")
    assert a.variant == "sv4d2"
    assert a.n_views == 5
    assert a.azimuths_deg == SV4D2_AZIMUTHS_DEG["sv4d2"]
    assert a.n_frames == 21


def test_sv4d2_adapter_8views_variant():
    from motionprior.integration.vgm import SV4D2Adapter
    a = SV4D2Adapter(
        checkpoint="checkpoints/sv4d2_8views.safetensors",
        variant="sv4d2_8views",
    )
    assert a.n_views == 9
    assert a.azimuths_deg[0] == 330.0   # input view at azimuth 330° for 8-view variant


def test_sv4d2_adapter_rejects_unknown_variant():
    from motionprior.integration.vgm import SV4D2Adapter
    with pytest.raises(ValueError):
        SV4D2Adapter(checkpoint="fake", variant="sv4d3_made_up")


def test_sv4d2_adapter_rejects_short_input_dir(tmp_path):
    """`.run` validates that the input dir has enough PNGs before launching."""
    from motionprior.integration.vgm import SV4D2Adapter
    # only 5 PNGs but adapter wants 21
    for i in range(5):
        (tmp_path / f"frame_{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # fake but path exists
    a = SV4D2Adapter(checkpoint="checkpoints/sv4d2.safetensors", n_frames=21)
    with pytest.raises(ValueError, match="need"):
        a.run(input_video_dir=tmp_path, output_dir=tmp_path / "out")


def test_wan22_adapter_import_fails_gracefully_on_cpu():
    from motionprior.integration.vgm import Wan22I2VAdapter
    with pytest.raises(ImportError) as ei:
        Wan22I2VAdapter(checkpoint="fake/path")
    assert "GPU" in str(ei.value) or "diffusers" in str(ei.value).lower() or \
           "transformers" in str(ei.value).lower() or \
           "wan" in str(ei.value).lower()
