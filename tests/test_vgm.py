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


def test_sv4d2_adapter_import_fails_gracefully_on_cpu():
    """The SV4D 2.0 backend lazy-imports heavy GPU deps; on CPU box it should
    raise an informative ImportError on construction, not at import time."""
    from motionprior.integration.vgm import SV4D2Adapter
    with pytest.raises(ImportError) as ei:
        SV4D2Adapter(checkpoint="fake/path")
    assert "GPU" in str(ei.value) or "diffusers" in str(ei.value).lower() or \
           "transformers" in str(ei.value).lower() or \
           "stabilityai" in str(ei.value).lower()


def test_wan22_adapter_import_fails_gracefully_on_cpu():
    from motionprior.integration.vgm import Wan22I2VAdapter
    with pytest.raises(ImportError) as ei:
        Wan22I2VAdapter(checkpoint="fake/path")
    assert "GPU" in str(ei.value) or "diffusers" in str(ei.value).lower() or \
           "transformers" in str(ei.value).lower() or \
           "wan" in str(ei.value).lower()
