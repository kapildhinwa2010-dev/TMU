import torch
import torch.nn as nn
import torch.nn.functional as F

from tmu import (
    AdvancedInternalAdapter,
    AdvancedTmuCapabilityHeads,
    TMUPlugin,
)
from tmu.clock import advanced_snapshot_to_tensor
from tmu.generate import predict_advanced_time_embed
from tmu.utils import ADV_INTERNAL_DIM


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
        self.config = type("Config", (), {"n_embd": d_model})()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_predict_advanced_time_embed_shapes():
    d_model = 32
    model = TinyModel(vocab_size=50, d_model=d_model, num_layers=2)

    plugin = TMUPlugin(
        base_model=model,
        hidden_dim=d_model,
        mode="hooks",
        target_modules=["TinyBlock"],
    )

    heads = AdvancedTmuCapabilityHeads(hidden_dim=d_model, num_layers=2, max_tokens=16)
    adapter = AdvancedInternalAdapter(hidden_dim=d_model)

    internal = torch.zeros(1, ADV_INTERNAL_DIM)
    internal_features = advanced_snapshot_to_tensor(
        type("Snap", (), {"feature_list": internal[0].tolist()})(),
        batch_size=1,
        device=torch.device("cpu"),
    )

    pace, layer_frac, token_frac, time_embed, rhythm_pred = predict_advanced_time_embed(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        timestamp=1767229200.0,
        budget=0.5,
        batch_size=1,
        device=torch.device("cpu"),
        internal_features=internal_features,
    )

    assert pace.shape == (1,)
    assert layer_frac.shape == (1,)
    assert token_frac.shape == (1,)
    assert time_embed.shape == (1, d_model)
    assert rhythm_pred.shape == (1, 12)
