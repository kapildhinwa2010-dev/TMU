from __future__ import annotations

import torch
import torch.nn as nn

from .utils import ADV_INTERNAL_DIM


class AdvancedInternalAdapter(nn.Module):
    def __init__(self, hidden_dim: int, internal_dim: int = ADV_INTERNAL_DIM):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.internal_dim = int(internal_dim)

        self.norm = nn.LayerNorm(self.internal_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.internal_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, internal_features: torch.Tensor) -> torch.Tensor:
        x = self.norm(internal_features)
        return self.proj(x)