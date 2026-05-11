"""Frequency-domain curriculum for the deformation MLP's temporal positional encoding.

Standard sinusoidal PE produces 2 channels per frequency band (sin, cos). We
mask out high-frequency bands during early training so the MLP can only model
slow, macro-trajectory motion. High-frequency hallucinations from the video
prior become unrepresentable until later iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
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
