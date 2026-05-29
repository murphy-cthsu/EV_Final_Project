import pytest
import torch

from motionprior.losses.cross_view_consistency import (
    AdaptiveBeta,
    build_sibling_map,
    compute_cross_view_gate,
)


def test_gate_decreases_with_residual():
    r_low = torch.tensor([0.01, 0.01, 0.01])
    r_high = torch.tensor([0.5, 0.5, 0.5])
    w_low = compute_cross_view_gate(r_low, beta=1.0)
    w_high = compute_cross_view_gate(r_high, beta=1.0)
    assert w_low > w_high
    assert 0.0 < w_low.item() <= 1.0
    assert 0.0 < w_high.item() <= 1.0


def test_gate_zero_beta_returns_one():
    r = torch.tensor([0.5, 1.0])
    w = compute_cross_view_gate(r, beta=0.0)
    assert torch.allclose(w, torch.tensor(1.0))


def test_gate_empty_residuals_returns_one():
    r = torch.tensor([])
    w = compute_cross_view_gate(r, beta=1.0)
    assert torch.allclose(w, torch.tensor(1.0))


def test_gate_rejects_negative_beta():
    with pytest.raises(ValueError):
        compute_cross_view_gate(torch.tensor([0.1]), beta=-1.0)


def test_gate_rejects_negative_residual():
    with pytest.raises(ValueError):
        compute_cross_view_gate(torch.tensor([-0.01]), beta=1.0)


def test_adaptive_beta_initial_value_returns_beta0():
    b = AdaptiveBeta(beta0=2.0, momentum=0.99)
    out = b(torch.tensor(1.0))
    # First update warm-starts ema=1.0, so beta = 2.0 / 1.0 = 2.0
    assert pytest.approx(out.item(), abs=1e-6) == 2.0


def test_adaptive_beta_normalizes_by_ema():
    b = AdaptiveBeta(beta0=1.0, momentum=0.5)
    # Step 1: ema=4.0; beta = 1/4 = 0.25
    out1 = b(torch.tensor(4.0))
    assert pytest.approx(out1.item(), abs=1e-6) == 0.25
    # Step 2: ema = 0.5*4 + 0.5*2 = 3.0; beta = 1/3
    out2 = b(torch.tensor(2.0))
    assert pytest.approx(out2.item(), abs=1e-6) == 1.0 / 3.0


def test_adaptive_beta_rejects_negative_beta0():
    with pytest.raises(ValueError):
        AdaptiveBeta(beta0=-1.0)


def test_sibling_map_5view_21frame():
    """Our canonical layout: 5 views x 21 frames, flat-indexed.
    Cam i has view = i // 21, frame = i % 21.
    """
    views = [i // 21 for i in range(105)]
    frames = [i % 21 for i in range(105)]
    m = build_sibling_map(views, frames)
    # Cam 0 (view 0, frame 0) should have 4 siblings at frame 0, views 1..4
    assert sorted(m[0]) == [21, 42, 63, 84]
    # Cam 50 (view 2, frame 8) -> siblings at frame 8, views 0,1,3,4
    assert sorted(m[50]) == [8, 29, 71, 92]


def test_sibling_map_train_test_split_84_train():
    """After holding out view 2: train has views {0,1,3,4} x 21 frames = 84 cams.
    Each cam should have 3 siblings (one fewer because view 2 is gone)."""
    views_kept = [0, 1, 3, 4]
    views = [v for v in views_kept for _ in range(21)]
    frames = [t for _ in views_kept for t in range(21)]
    m = build_sibling_map(views, frames)
    for i in range(len(views)):
        sibs = m[i]
        # Each cam has 3 siblings (one per other kept view) at the same frame.
        assert len(sibs) == 3
        # All siblings share the frame and have different view.
        for j in sibs:
            assert frames[j] == frames[i]
            assert views[j] != views[i]


def test_sibling_map_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        build_sibling_map([0, 1], [0])
