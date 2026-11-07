import torch
import torch.nn as nn
import torch.nn.functional as F

from tmu import (
    AdvancedPaceController,
    TMUPlugin,
    TimeManagementUnit,
    cyclic_time_features,
)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 32, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_cyclic_time_features_shape():
    feats = cyclic_time_features(1_700_000_000.0)
    assert feats.shape == (1, 12)


def test_tmu_zero_init_no_change():
    tmu = TimeManagementUnit(hidden_dim=16)
    x = torch.randn(2, 5, 16)
    te = tmu.encode_time(time_stamp=1_700_000_000.0, batch_size=2)
    y = tmu.apply_to(x, te)
    assert torch.allclose(x, y, atol=1e-6)


def test_plugin_hooks_preserve_shape():
    model = TinyModel(vocab_size=50, d_model=32, num_layers=2)
    plugin = TMUPlugin(
        base_model=model,
        hidden_dim=32,
        mode="hooks",
        target_modules=["TinyBlock"],
    )

    ids = torch.randint(0, 50, size=(2, 4))
    out = plugin(ids, time_stamp=1_700_000_000.0)

    assert out.shape == (2, 4, 50)


def test_advanced_pace_controller_bounds():
    ctrl = AdvancedPaceController(num_layers=12)
    plan = ctrl.from_outputs(
        pace=0.2,
        layer_frac=0.2,
        token_frac=0.2,
        budget=0.25,
        traffic={"queue_delay_norm": 0.0, "active_norm": 0.0, "deadline_pressure": 0.0, "p95_norm": 0.0, "error_rate": 0.0},
        system={"temp_norm": 0.0, "power_norm": 0.0, "throttle_risk": 0.0, "gpu_util_norm": 0.0, "mem_pressure": 0.0},
        difficulty=0.5,
    )

    assert 1 <= plan["active_layers"] <= 12
    assert plan["max_new_tokens"] >= 1