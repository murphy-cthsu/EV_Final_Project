"""Part segmentation wrapper.

`PartSegmenter` is a Protocol -- any object exposing `.segment(image) -> ndarray`
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
