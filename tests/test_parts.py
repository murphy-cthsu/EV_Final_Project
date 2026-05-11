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
    label_map = torch.zeros(2, 2, 2)  # 3-D -- wrong
    pixel_coords = torch.tensor([[0, 0]])
    with pytest.raises(ValueError):
        assign_part_labels(label_map, pixel_coords)


def test_assign_labels_rejects_mismatched_pixel_coords_shape():
    label_map = torch.zeros(2, 2)
    pixel_coords = torch.tensor([0, 0])  # (2,) instead of (N, 2)
    with pytest.raises(ValueError):
        assign_part_labels(label_map, pixel_coords)
