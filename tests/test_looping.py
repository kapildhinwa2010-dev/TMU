import torch
import torch.nn as nn

from tmu.looping import (
    AdvancedLoopHeads,
    InputAnchor,
    LayerRefinerCollection,
    LoopConfig,
    make_loop_plan,
)


class DummyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, hidden_states=None, attention_mask=None, use_cache=False, *args, **kwargs):
        if hidden_states is None and args:
            hidden_states = args[0]
        return (self.lin(hidden_states),)


def test_input_anchor_zero_gate():
    anchor = InputAnchor(hidden_dim=16, gate_init=0.0)
    x = torch.randn(2, 5, 16)
    a = anchor(x)
    assert a.shape == (2, 16)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-6)


def test_layer_refiner_noop_when_loops_zero():
    d = 16
    block = DummyBlock(d)
    refiner = LayerRefinerCollection(
        num_layers=1,
        hidden_dim=d,
        config=LoopConfig(enabled=True, max_loops=2),
    )

    h = torch.randn(1, 4, d)
    mask = torch.zeros(1, 1, 4, 4)
    anchor = torch.randn(1, d)

    out = refiner.forward_block(
        layer_idx=0,
        block=block,
        hidden=h,
        attention_mask=mask,
        anchor=anchor,
        loops=0,
    )

    assert torch.allclose(out, h, atol=1e-6)


def test_advanced_loop_heads_returns_six_outputs():
    heads = AdvancedLoopHeads(hidden_dim=32, num_layers=4, max_tokens=16)
    time_embed = torch.randn(2, 32)
    budget = torch.tensor([0.3, 0.8])

    out = heads(time_embed, budget)

    assert len(out) == 6
    rhythm_pred, pace, layer_frac, token_frac, loop_frac, pace_logit = out

    assert rhythm_pred.shape == (2, 12)
    assert pace.shape == (2,)
    assert layer_frac.shape == (2,)
    assert token_frac.shape == (2,)
    assert loop_frac.shape == (2,)
    assert pace_logit.shape == (2,)


def test_make_loop_plan_easy_prompt():
    plan = make_loop_plan(
        num_layers=12,
        loop_frac=0.0,
        difficulty=0.1,
        budget=0.2,
        traffic_load=0.0,
        thermal_load=0.0,
        config=LoopConfig(max_loops=3),
    )

    assert plan["loop_scalar"] == 0
    assert all(l == 0 for l in plan["layer_loops"])
    assert plan["final_refine_passes"] == 0


def test_make_loop_plan_hard_prompt():
    plan = make_loop_plan(
        num_layers=12,
        loop_frac=0.9,
        difficulty=0.95,
        budget=0.9,
        traffic_load=0.0,
        thermal_load=0.0,
        config=LoopConfig(max_loops=3),
    )

    assert plan["loop_scalar"] > 0
    assert max(plan["layer_loops"]) > 0