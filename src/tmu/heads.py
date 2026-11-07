from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .core import TimeManagementUnit


class TmuCapabilityHeads(nn.Module):
    def __init__(self, hidden_dim: int, budget_dim: int = 1):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.budget_dim = int(budget_dim)

        self.budget_proj = nn.Linear(self.budget_dim, self.hidden_dim)
        nn.init.zeros_(self.budget_proj.weight)
        nn.init.zeros_(self.budget_proj.bias)

        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, TimeManagementUnit.FEATURE_DIM),
        )

        pace_hidden = max(8, self.hidden_dim // 2)

        self.pace_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
        if not torch.is_tensor(budget):
            budget = torch.tensor(float(budget), device=time_embed.device)

        budget = budget.to(device=time_embed.device, dtype=time_embed.dtype)

        if budget.dim() == 0:
            budget = budget.unsqueeze(0)

        if budget.dim() == 1:
            budget = budget.unsqueeze(-1)

        if budget.shape[0] != time_embed.shape[0]:
            if budget.shape[0] == 1:
                budget = budget.expand(time_embed.shape[0], -1)
            elif time_embed.shape[0] == 1:
                time_embed = time_embed.expand(budget.shape[0], -1)
            else:
                raise ValueError(
                    f"Budget batch {budget.shape[0]} != time_embed batch {time_embed.shape[0]}"
                )

        conditioned = time_embed + self.budget_proj(budget)

        rhythm_pred = self.rhythm_head(conditioned)
        pace_logit = self.pace_head(conditioned).squeeze(-1)
        pace = torch.sigmoid(pace_logit)

        return rhythm_pred, pace, pace_logit


class AdvancedTmuCapabilityHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 12,
        max_tokens: int = 64,
        budget_dim: int = 1,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_tokens = int(max_tokens)
        self.budget_dim = int(budget_dim)

        self.budget_proj = nn.Linear(self.budget_dim, self.hidden_dim)
        nn.init.zeros_(self.budget_proj.weight)
        nn.init.zeros_(self.budget_proj.bias)

        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, TimeManagementUnit.FEATURE_DIM),
        )

        pace_hidden = max(8, self.hidden_dim // 2)

        self.pace_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        self.layer_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        self.token_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

        nn.init.zeros_(self.layer_head[-1].weight)
        nn.init.constant_(self.layer_head[-1].bias, 0.0)

        nn.init.zeros_(self.token_head[-1].weight)
        nn.init.constant_(self.token_head[-1].bias, 0.0)

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
        if not torch.is_tensor(budget):
            budget = torch.tensor(float(budget), device=time_embed.device)

        budget = budget.to(device=time_embed.device, dtype=time_embed.dtype)

        if budget.dim() == 0:
            budget = budget.unsqueeze(0)

        if budget.dim() == 1:
            budget = budget.unsqueeze(-1)

        if budget.shape[0] != time_embed.shape[0]:
            if budget.shape[0] == 1:
                budget = budget.expand(time_embed.shape[0], -1)
            elif time_embed.shape[0] == 1:
                time_embed = time_embed.expand(budget.shape[0], -1)
            else:
                raise ValueError(
                    f"Budget batch {budget.shape[0]} != time_embed batch {time_embed.shape[0]}"
                )

        conditioned = time_embed + self.budget_proj(budget)

        rhythm_pred = self.rhythm_head(conditioned)

        pace_logit = self.pace_head(conditioned).squeeze(-1)
        layer_logit = self.layer_head(conditioned).squeeze(-1)
        token_logit = self.token_head(conditioned).squeeze(-1)

        pace = torch.sigmoid(pace_logit)
        layer_frac = torch.sigmoid(layer_logit)
        token_frac = torch.sigmoid(token_logit)

        return rhythm_pred, pace, layer_frac, token_frac, pace_logit