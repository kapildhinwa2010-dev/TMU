from __future__ import annotations

import torch

from tmu.core import TimeManagementUnit, load_tmu_state_dict
from tmu.fusion import TemporalFusionLayer, TemporalFusionStack, encode_fused_time_embed


def test_fusion_layer_is_identity_at_init():
    layer = TemporalFusionLayer(hidden_dim=16, expansion=2)

    x = torch.randn(3, 16)
    y = layer(x)

    assert torch.allclose(x, y, atol=1e-6)


def test_fusion_stack_with_zero_layers_is_identity():
    stack = TemporalFusionStack(hidden_dim=16, num_layers=0)

    assert stack.is_identity
    assert len(list(stack.parameters())) == 0

    x = torch.randn(2, 16)
    assert torch.allclose(stack(x), x)


def test_fusion_stack_changes_output_after_training():
    stack = TemporalFusionStack(hidden_dim=16, num_layers=1)

    for p in stack.parameters():
        if p.dim() > 0:
            p.data.normal_(0.0, 0.1)

    x = torch.randn(2, 16)
    assert not torch.allclose(stack(x), x)


def test_fusion_stack_keeps_shape_and_is_differentiable():
    stack = TemporalFusionStack(hidden_dim=16, num_layers=2)

    x = torch.randn(4, 16, requires_grad=True)
    y = stack(x)

    assert y.shape == x.shape

    y.sum().backward()

    grads = [p.grad for p in stack.parameters() if p.requires_grad]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_tmu_with_fusion_is_identity_at_init():
    tmu = TimeManagementUnit(hidden_dim=16, fusion_layers=1)

    x = torch.randn(2, 16)
    te = tmu.encode_time(time_stamp=1_700_000_000.0, batch_size=2)

    assert torch.allclose(tmu.apply_to(x, te), x, atol=1e-6)


def test_encode_time_accepts_extra_embed():
    tmu = TimeManagementUnit(hidden_dim=16, fusion_layers=1)

    base = tmu.encode_time(time_stamp=1_700_000_000.0, batch_size=2)
    extra = torch.randn(2, 16)

    combined = tmu.encode_time(
        time_stamp=1_700_000_000.0,
        batch_size=2,
        extra_embed=extra,
    )

    assert combined.shape == base.shape
    assert not torch.allclose(combined, base)


def test_encode_fused_time_embed_requires_adapter_for_internal_features():
    tmu = TimeManagementUnit(hidden_dim=16, fusion_layers=1)

    try:
        encode_fused_time_embed(
            tmu=tmu,
            time_stamp=1_700_000_000.0,
            internal_features=torch.randn(2, 24),
            adapter=None,
            batch_size=2,
        )
    except ValueError:
        return

    raise AssertionError("adapter ke bina internal features raise karega.")


def test_load_tmu_state_dict_is_backward_compatible():
    old_tmu = TimeManagementUnit(hidden_dim=16, fusion_layers=0)
    old_state = old_tmu.state_dict()

    new_tmu = TimeManagementUnit(hidden_dim=16, fusion_layers=1)
    missing = load_tmu_state_dict(new_tmu, old_state, verbose=False)

    assert any("fusion" in k for k in missing)
