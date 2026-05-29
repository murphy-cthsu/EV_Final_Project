"""CPU-only tests for the SV4D→SC-GS pipeline plumbing.

These tests do NOT invoke real SV4D 2.0 inference — that requires a GPU. They
test:
  - convert_sv4d_to_dnerf() produces a SC-GS-readable scene from any
    SV4DResult-shaped input (verified via SC-GS's own dataset_readers).
  - prepare_input_video() correctly composites D-NeRF RGBA → RGB and writes
    21 frames at the requested resolution.
  - The driver's fake_sv4d_result() path runs end-to-end on CPU.

The real SV4D inference path is exercised separately on the GPU pod via the
smoke runbook in docs/design/sv4d2_api.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from motionprior.integration.vgm import SV4D2_AZIMUTHS_DEG, SV4DResult
from scripts.sv4d_to_dnerf import (
    convert_sv4d_to_dnerf,
    look_at_blender,
    orbit_camera,
)


@pytest.fixture
def fake_dnerf_scene(tmp_path: Path) -> Path:
    """Minimal D-NeRF-shaped scene with 21 RGBA train images + transforms_train/test/val."""
    scene = tmp_path / "fake_dnerf"
    (scene / "train").mkdir(parents=True)
    (scene / "test").mkdir()
    (scene / "val").mkdir()

    rng = np.random.default_rng(0)
    frames_train = []
    for i in range(21):
        # RGBA: an opaque colored disk on a transparent bg
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        cy, cx = 32, 32
        y, x = np.indices((64, 64))
        mask = (y - cy) ** 2 + (x - cx) ** 2 < 20 ** 2
        arr[..., 0][mask] = (i * 12) % 255
        arr[..., 1][mask] = (i * 30) % 255
        arr[..., 2][mask] = 200
        arr[..., 3][mask] = 255
        Image.fromarray(arr).save(scene / "train" / f"r_{i:03d}.png")
        frames_train.append({
            "file_path": f"./train/r_{i:03d}",
            "rotation": 0.0,
            "time": i / 20.0,
            "transform_matrix": np.eye(4).tolist(),
        })
    (scene / "transforms_train.json").write_text(json.dumps({
        "camera_angle_x": 0.6911112070083618,
        "frames": frames_train,
    }))

    # Test split: 2 images with valid transforms
    frames_test = []
    for i in range(2):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[..., :3] = 100
        arr[..., 3] = 255
        Image.fromarray(arr).save(scene / "test" / f"r_{i:03d}.png")
        frames_test.append({
            "file_path": f"./test/r_{i:03d}",
            "rotation": 0.0,
            "time": 0.5,
            "transform_matrix": np.eye(4).tolist(),
        })
    (scene / "transforms_test.json").write_text(json.dumps({
        "camera_angle_x": 0.6911112070083618,
        "frames": frames_test,
    }))
    (scene / "transforms_val.json").write_text(json.dumps({
        "camera_angle_x": 0.6911112070083618,
        "frames": [],
    }))
    return scene


def make_fake_sv4d_result(tmp_path: Path, V: int = 5, T: int = 21,
                          H: int = 64, W: int = 64) -> SV4DResult:
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 255, (V, T, H, W, 3), dtype=np.uint8)
    out_dir = tmp_path / "sv4d_out"
    out_dir.mkdir(exist_ok=True)
    return SV4DResult(
        frames=frames,
        azimuths_deg=list(SV4D2_AZIMUTHS_DEG["sv4d2"][:V]),
        elevations_deg=[0.0] * V,
        variant="sv4d2",
        seed=42,
        img_size=H,
        n_frames=T,
        output_dir=out_dir,
    )


def test_orbit_camera_radius_preserved():
    c2w = orbit_camera(0.0, 0.0, radius=4.0)
    np.testing.assert_allclose(np.linalg.norm(c2w[:3, 3]), 4.0, atol=1e-6)


def test_orbit_camera_azimuth_60_position():
    """Azimuth 60° at radius 4 should put camera at (sin60·r, 0, cos60·r)."""
    c2w = orbit_camera(60.0, 0.0, radius=4.0)
    expected = np.array([4.0 * np.sin(np.pi / 3), 0.0, 4.0 * np.cos(np.pi / 3)])
    np.testing.assert_allclose(c2w[:3, 3], expected, atol=1e-6)


def test_orbit_camera_looks_at_origin():
    """The camera's local -Z direction should point at the origin."""
    c2w = orbit_camera(90.0, 30.0, radius=5.0)
    pos = c2w[:3, 3]
    cam_z = c2w[:3, 2]
    # forward = -cam_z (Blender convention); should match -normalize(pos)
    forward = -cam_z
    expected_forward = -pos / np.linalg.norm(pos)
    np.testing.assert_allclose(forward, expected_forward, atol=1e-6)


def test_convert_writes_flat_indexed_filenames(tmp_path, fake_dnerf_scene):
    """SC-GS sorts by int() of the last `_`-separated chunk of filename;
    we must emit names whose last chunk parses as int."""
    result = make_fake_sv4d_result(tmp_path)
    out_dir = tmp_path / "converted"
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir, overwrite=True)

    transforms = json.loads((out_dir / "transforms_train.json").read_text())
    for f in transforms["frames"]:
        stem = Path(f["file_path"]).stem  # e.g. r_00000
        last = stem.split("_")[-1]
        int(last)  # must not raise


def test_convert_writes_expected_number_of_train_images(tmp_path, fake_dnerf_scene):
    V, T = 5, 21
    result = make_fake_sv4d_result(tmp_path, V=V, T=T)
    out_dir = tmp_path / "converted"
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir, overwrite=True)
    pngs = list((out_dir / "train").glob("*.png"))
    assert len(pngs) == V * T


def test_convert_copies_test_split(tmp_path, fake_dnerf_scene):
    result = make_fake_sv4d_result(tmp_path)
    out_dir = tmp_path / "converted"
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir, overwrite=True)
    assert (out_dir / "test").is_dir()
    assert (out_dir / "transforms_test.json").is_file()
    assert (out_dir / "transforms_val.json").is_file()
    # Test PNGs come through unchanged
    test_pngs = list((out_dir / "test").glob("*.png"))
    assert len(test_pngs) == 2


def test_convert_emits_metadata(tmp_path, fake_dnerf_scene):
    result = make_fake_sv4d_result(tmp_path, V=5, T=21)
    out_dir = tmp_path / "converted"
    meta = convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir,
                                  orbit_radius=4.0, overwrite=True)
    assert meta["n_views"] == 5
    assert meta["n_frames_per_view"] == 21
    assert meta["n_train_images_written"] == 105
    assert (out_dir / "sv4d_metadata.json").is_file()


def test_convert_per_view_transforms_differ_by_azimuth(tmp_path, fake_dnerf_scene):
    """Different views should yield different transform_matrix entries."""
    result = make_fake_sv4d_result(tmp_path)
    out_dir = tmp_path / "converted"
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir, overwrite=True)
    transforms = json.loads((out_dir / "transforms_train.json").read_text())
    by_view: dict[int, list] = {}
    for f in transforms["frames"]:
        by_view.setdefault(f["view_idx"], []).append(f["transform_matrix"])
    # Every view should have identical transform across frames (orbit is time-independent)
    # but different views must have different transforms.
    matrices = []
    for v in sorted(by_view):
        first = by_view[v][0]
        for m in by_view[v]:
            assert m == first, f"view {v} transforms vary across time"
        matrices.append(first)
    # Each view's matrix must be distinct
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            assert matrices[i] != matrices[j], (
                f"views {i} and {j} have the same transform_matrix"
            )


def test_convert_time_field_normalized_0_to_1(tmp_path, fake_dnerf_scene):
    result = make_fake_sv4d_result(tmp_path, T=21)
    out_dir = tmp_path / "converted"
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, out_dir, overwrite=True)
    transforms = json.loads((out_dir / "transforms_train.json").read_text())
    times = [f["time"] for f in transforms["frames"]]
    assert min(times) == 0.0
    assert max(times) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------- #
# pipeline driver: PREPARE stage smoke test
# ---------------------------------------------------------------------- #
def test_prepare_stage_writes_n_frames(tmp_path, fake_dnerf_scene):
    from scripts.run_sv4d_supervised_pipeline import prepare_input_video
    out = tmp_path / "input"
    meta = prepare_input_video(fake_dnerf_scene, out, n_frames=21, img_size=128)
    pngs = sorted(out.glob("frame_*.png"))
    assert len(pngs) == 21
    arr = np.asarray(Image.open(pngs[0]))
    assert arr.shape == (128, 128, 3)
    # White background composite: pixels outside the colored disk should be ~white
    corner = arr[0, 0]
    assert (corner > 240).all(), f"expected white background, got {corner}"
    assert "sampled_timestamps" in meta
    assert len(meta["sampled_timestamps"]) == 21


# ---------------------------------------------------------------------- #
# End-to-end smoke test: prepare → fake_sv4d → convert → SC-GS-reader-parses
# ---------------------------------------------------------------------- #
def test_end_to_end_with_fake_sv4d_is_scgs_readable(tmp_path, fake_dnerf_scene):
    """Full pipeline end-to-end on CPU using fake SV4D. Confirms that the final
    output is parseable by SC-GS's own dataset_readers without errors."""
    from scripts.run_sv4d_supervised_pipeline import (
        prepare_input_video, fake_sv4d_result,
    )

    sv4d_input = tmp_path / "sv4d_input"
    sv4d_output = tmp_path / "sv4d_output"
    converted = tmp_path / "converted"

    prepare_input_video(fake_dnerf_scene, sv4d_input, n_frames=21, img_size=128)
    result = fake_sv4d_result(sv4d_input, sv4d_output, variant="sv4d2",
                               img_size=128, n_frames=21)
    convert_sv4d_to_dnerf(result, fake_dnerf_scene, converted, overwrite=True)

    # Try to read with SC-GS's own reader
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "SC-GS"))
    from scene.dataset_readers import readNerfSyntheticInfo  # noqa: E402
    info = readNerfSyntheticInfo(
        str(converted),
        white_background=True,
        eval=True,
        extension=".png",
    )
    assert len(info.train_cameras) == 5 * 21
    assert len(info.test_cameras) == 2
    # fid spans [0, 1]
    fids = [c.fid for c in info.train_cameras]
    assert min(fids) == pytest.approx(0.0, abs=1e-6)
    assert max(fids) == pytest.approx(1.0, abs=1e-6)
