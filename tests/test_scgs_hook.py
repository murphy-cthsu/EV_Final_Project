"""Tests for the SC-GS integration hook.

We don't have SC-GS installed in the CPU test env, so we test the hook's
public interface with mocked inputs that mimic what SC-GS would pass in.
"""

import torch
import pytest

from motionprior.integration import MotionPriorHook


def _minimal_config() -> dict:
    return {
        "gating": {"alpha0": 1.0, "ema_momentum": 0.99, "eps": 1e-6, "enabled": True},
        "frequency_curriculum": {
            "num_bands": 4,
            "milestones": [0, 1000, 2000],
            "k_at_milestone": [1, 2, 4],
            "enabled": True,
        },
        "articulation": {"lambda_intra": 1.0, "lambda_inter": 0.05, "static_label": -1, "enabled": True},
        "rest_state": {"eta": 0.01, "enabled": True},
        "warm_up": 0,  # for tests, no warm-up
    }


def test_hook_constructs_without_arguments_beyond_required():
    hook = MotionPriorHook(
        config=_minimal_config(),
        arap_prior_energies=torch.tensor([0.0, 0.1, 0.2, 0.3]),
        part_labels=torch.tensor([0, 0, 1, 1]),
        edges=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        rest_positions=torch.zeros(4, 3),
    )
    assert hook is not None


def test_gate_temporal_encoding_applies_freq_mask_when_layout_matches():
    hook = MotionPriorHook(
        config=_minimal_config(),
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    # 4 bands x 2 (sin, cos) = 8 channels
    time_emb = torch.ones(3, 8)
    # At iteration 0, k_max=1 -> only bands 0..0 (channels 0..1) active
    out = hook.gate_temporal_encoding(time_emb, iteration=0)
    assert out.shape == time_emb.shape
    assert (out[:, :2] == 1.0).all()
    assert (out[:, 2:] == 0.0).all()


def test_gate_temporal_encoding_no_op_when_layout_mismatched():
    hook = MotionPriorHook(
        config=_minimal_config(),
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    # 7 channels -> not 2*num_bands, not 1+2*num_bands; fallback to no-op
    time_emb = torch.ones(3, 7)
    out = hook.gate_temporal_encoding(time_emb, iteration=0)
    torch.testing.assert_close(out, time_emb)


def test_gate_temporal_encoding_handles_identity_prefixed_layout():
    """SC-GS / NeRF PE layout: [identity, sin_0, cos_0, sin_1, cos_1, ...]."""
    hook = MotionPriorHook(
        config=_minimal_config(),
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    # 1 (identity) + 4 bands x 2 = 9 channels
    time_emb = torch.ones(3, 9)
    # At iteration 0, k_max=1 -> identity passes, only first sin/cos band active
    out = hook.gate_temporal_encoding(time_emb, iteration=0)
    assert out.shape == time_emb.shape
    # identity unchanged
    assert (out[:, 0] == 1.0).all()
    # band 0 (sin_0, cos_0) active
    assert (out[:, 1:3] == 1.0).all()
    # bands 1, 2, 3 zeroed
    assert (out[:, 3:] == 0.0).all()


def test_gate_temporal_encoding_disabled_returns_input():
    cfg = _minimal_config()
    cfg["frequency_curriculum"]["enabled"] = False
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    time_emb = torch.ones(3, 8)
    out = hook.gate_temporal_encoding(time_emb, iteration=0)
    torch.testing.assert_close(out, time_emb)


def test_photometric_gating_returns_one_during_warmup():
    cfg = _minimal_config()
    cfg["warm_up"] = 500
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0, 0.5, 1.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    w = hook.photometric_gating(fid=0.5, iteration=100)
    assert w == pytest.approx(1.0)


def test_photometric_gating_after_warmup_uses_energy():
    cfg = _minimal_config()
    cfg["warm_up"] = 0
    energies = torch.tensor([0.0, 0.5, 2.0])
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=energies,
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    # fid=0.0 -> energy 0.0 -> gating=1.0
    w0 = hook.photometric_gating(fid=0.0, iteration=10)
    assert w0 == pytest.approx(1.0)
    # fid=1.0 -> energy 2.0 -> gating < 1.0
    w_end = hook.photometric_gating(fid=1.0, iteration=10)
    assert 0.0 < w_end < 1.0


def test_photometric_gating_disabled_always_one():
    cfg = _minimal_config()
    cfg["gating"]["enabled"] = False
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0, 5.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    w = hook.photometric_gating(fid=1.0, iteration=10)
    assert w == pytest.approx(1.0)


def test_extra_losses_returns_rest_state_when_enabled():
    cfg = _minimal_config()
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0, 0]),
        edges=torch.tensor([[0, 1]]),
        rest_positions=torch.zeros(2, 3),
    )
    # d_xyz = displacement from canonical (which is zero). ||(1,0,0)||^2 mean = 1
    d_xyz = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    loss = hook.extra_losses(d_xyz, iteration=10)
    # eta * 1 = 0.01
    assert loss.item() == pytest.approx(0.01)


def test_extra_losses_zero_when_disabled():
    cfg = _minimal_config()
    cfg["rest_state"]["enabled"] = False
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0, 0]),
        edges=torch.tensor([[0, 1]]),
        rest_positions=torch.zeros(2, 3),
    )
    d_xyz = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    loss = hook.extra_losses(d_xyz, iteration=10)
    assert loss.item() == pytest.approx(0.0)


def test_articulated_edge_weights_returned_for_inspection():
    """The hook exposes its articulated weights so SC-GS's ARAPDeformer can use them."""
    cfg = _minimal_config()
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0, 0, 1, 1]),
        edges=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        rest_positions=torch.zeros(4, 3),
    )
    w = hook.articulated_edge_weights()
    assert w.shape == (3,)
    assert w[0].item() == pytest.approx(1.0)    # intra
    assert w[1].item() == pytest.approx(0.05)   # inter
    assert w[2].item() == pytest.approx(1.0)    # intra


def test_articulated_edge_weights_uniform_when_disabled():
    cfg = _minimal_config()
    cfg["articulation"]["enabled"] = False
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0]),
        part_labels=torch.tensor([0, 0, 1, 1]),
        edges=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        rest_positions=torch.zeros(4, 3),
    )
    w = hook.articulated_edge_weights()
    # Uniform weight equal to lambda_intra
    assert w.shape == (3,)
    torch.testing.assert_close(w, torch.full((3,), 1.0))


def test_photometric_gating_fid_out_of_range_clamps():
    cfg = _minimal_config()
    hook = MotionPriorHook(
        config=cfg,
        arap_prior_energies=torch.tensor([0.0, 0.5, 1.0]),
        part_labels=torch.tensor([0]),
        edges=torch.tensor([[0, 0]]),
        rest_positions=torch.zeros(1, 3),
    )
    # fid out of [0, 1] should clamp to nearest frame, not crash
    w_neg = hook.photometric_gating(fid=-0.5, iteration=10)
    w_big = hook.photometric_gating(fid=2.0, iteration=10)
    assert 0.0 < w_neg <= 1.0
    assert 0.0 < w_big <= 1.0
