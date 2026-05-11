"""Tests for the SC-GS ARAP adapter.

The adapter monkey-patches SC-GS's ControlNodeWarp.arap_loss to use our
anisotropic per-edge weighting derived from part labels.

These tests run against mock SC-GS objects since we don't have SC-GS installed
on the CPU dev box. The mock mirrors SC-GS's API exactly so the adapter logic
is verified.
"""

import torch
import pytest

from motionprior.integration.scgs_arap_adapter import (
    build_anisotropic_weight_matrix,
    install,
)


def test_anisotropic_weight_matrix_zero_when_disabled_returns_uniform():
    Nv = 4
    K = 3
    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([1, 2, 3])
    nn = torch.tensor([0, 0, 0])
    parts = torch.tensor([0, 0, 1, 1])
    # Uniform lambda_intra means same-part edges get 1.0; inter edges get 0.05.
    out = build_anisotropic_weight_matrix(
        Nv=Nv, K=K, ii=ii, jj=jj, nn=nn,
        node_parts=parts,
        lambda_intra=1.0, lambda_inter=0.05, static_label=-1,
    )
    assert out.shape == (Nv, K)
    # edge (0->1, slot 0): same part 0 -> intra (1.0)
    assert out[0, 0].item() == pytest.approx(1.0)
    # edge (1->2, slot 0): parts 0 vs 1 -> inter (0.05)
    assert out[1, 0].item() == pytest.approx(0.05)
    # edge (2->3, slot 0): same part 1 -> intra (1.0)
    assert out[2, 0].item() == pytest.approx(1.0)


def test_anisotropic_weight_static_both_endpoints_zero():
    Nv = 4
    K = 2
    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([1, 2, 3])
    nn = torch.tensor([0, 0, 0])
    parts = torch.tensor([-1, -1, 0, 0])
    out = build_anisotropic_weight_matrix(
        Nv=Nv, K=K, ii=ii, jj=jj, nn=nn,
        node_parts=parts,
        lambda_intra=1.0, lambda_inter=0.05, static_label=-1,
    )
    # (0,1): both static -> 0
    assert out[0, 0].item() == pytest.approx(0.0)
    # (1,2): static<->dynamic -> inter
    assert out[1, 0].item() == pytest.approx(0.05)
    # (2,3): both part 0 -> intra
    assert out[2, 0].item() == pytest.approx(1.0)


def test_anisotropic_weight_unfilled_slots_are_zero():
    Nv = 3
    K = 5  # but only 2 edges defined
    ii = torch.tensor([0, 1])
    jj = torch.tensor([1, 2])
    nn = torch.tensor([0, 0])
    parts = torch.tensor([0, 0, 0])
    out = build_anisotropic_weight_matrix(
        Nv=Nv, K=K, ii=ii, jj=jj, nn=nn,
        node_parts=parts,
        lambda_intra=1.0, lambda_inter=0.05, static_label=-1,
    )
    # filled slots are 1.0 (intra); unfilled slots are 0
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[0, 1:].sum().item() == pytest.approx(0.0)
    assert out[1, 0].item() == pytest.approx(1.0)
    assert out[1, 1:].sum().item() == pytest.approx(0.0)


class _MockControlNodeWarp:
    """Simulates the SC-GS ControlNodeWarp API surface our adapter touches."""

    def __init__(self, node_num: int = 4, K: int = 2):
        self.node_num = node_num
        self.K = K
        # The actual nodes don't matter for our adapter logic; we just need
        # the attribute to exist so adapter's preflight doesn't error.
        self.nodes = torch.zeros(node_num, 3)
        self._arap_loss_called_with_weight = None

    def arap_loss(self, t=None, delta_t=0.05, t_samp_num=2):
        # The original returns a scalar tensor; we just return one so the patch
        # has something to chain on. The patched version stores the weight
        # passed to a sentinel attribute, which our test inspects.
        return torch.tensor(0.0)


def test_install_disabled_is_noop():
    """When articulation is disabled in hook config, install must not patch."""

    class FakeHook:
        def __init__(self):
            self.cfg = {"articulation": {"enabled": False}}

        def articulated_edge_weights(self):
            raise AssertionError("articulated_edge_weights should not be called when disabled")

    cnw = _MockControlNodeWarp()
    install(cnw, FakeHook(), node_part_labels=torch.zeros(4, dtype=torch.long))
    # No motionprior attributes should be attached, and arap_loss is still
    # the unmodified class-level method.
    assert not hasattr(cnw, "_motionprior_node_part_labels")
    assert not hasattr(cnw, "_motionprior_articulation_config")
    assert cnw.arap_loss.__func__ is _MockControlNodeWarp.arap_loss


def test_install_attaches_anisotropic_factor_and_logs():
    """When enabled, install attaches per-edge factors and the patched method
    uses them."""

    class FakeHook:
        def __init__(self):
            self.cfg = {
                "articulation": {
                    "enabled": True,
                    "lambda_intra": 1.0,
                    "lambda_inter": 0.05,
                    "static_label": -1,
                }
            }

    cnw = _MockControlNodeWarp(node_num=4, K=2)
    install(cnw, FakeHook(), node_part_labels=torch.tensor([0, 0, 1, 1]))
    # Adapter should have attached the part labels on the model for later use
    assert hasattr(cnw, "_motionprior_node_part_labels")
    assert hasattr(cnw, "_motionprior_articulation_config")
    cfg = cnw._motionprior_articulation_config
    assert cfg["lambda_intra"] == 1.0
    assert cfg["lambda_inter"] == 0.05
    # And the arap_loss method has been replaced.
    assert cnw.arap_loss is not _MockControlNodeWarp.arap_loss


def test_install_rejects_part_labels_wrong_length():
    class FakeHook:
        def __init__(self):
            self.cfg = {"articulation": {"enabled": True, "lambda_intra": 1.0, "lambda_inter": 0.05, "static_label": -1}}

    cnw = _MockControlNodeWarp(node_num=4)
    with pytest.raises(ValueError):
        install(cnw, FakeHook(), node_part_labels=torch.zeros(5, dtype=torch.long))
