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
    # part label STATIC_PART = -1 means "do not deform" -- both endpoints static => weight 0
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
