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
