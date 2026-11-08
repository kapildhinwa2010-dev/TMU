from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionLayer(nn.Module):
    """Additional residual layer for the TMU time embedding.

    Ye layer TMU ke time embedding ke upar ek gated residual MLP lagata hai:

        y = x + gate(h) * W_out(gelu(W_in(LayerNorm(x))))

    Design notes:

    - ``W_out`` zero-init hota hai, isliye layer init pe exact identity hai.
      Isse existing checkpoints backwards-compatible rehte hain aur naya layer
      add karne se pretrained behaviour turant bigadta nahi.
    - ``gate`` sigmoid hai, toh layer apna contribution khud control kar sakta hai.
    - Pre-LayerNorm use hota hai, isliye stack deep hone pe bhi stable rehta hai.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 2,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.expansion = max(1, int(expansion))
        self.dropout_p = float(dropout)

        inner_dim = self.hidden_dim * self.expansion

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.fc_in = nn.Linear(self.hidden_dim, inner_dim)
        self.fc_out = nn.Linear(inner_dim, self.hidden_dim)
        self.gate = nn.Linear(inner_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        if zero_init:
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return x

        h = self.norm(x)
        h = F.gelu(self.fc_in(h))

        delta = self.fc_out(h)
        gate = torch.sigmoid(self.gate(h))

        return x + self.dropout(delta * gate)


class TemporalFusionStack(nn.Module):
    """Stack of :class:`TemporalFusionLayer` blocks.

    ``num_layers=0`` pe stack pure identity hai (koi parameter nahi, koi output
    change nahi). Isliye callers isko blindly call kar sakte hain.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        expansion: int = 2,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(0, int(num_layers))

        self.layers = nn.ModuleList(
            [
                TemporalFusionLayer(
                    hidden_dim=self.hidden_dim,
                    expansion=expansion,
                    dropout=dropout,
                    zero_init=zero_init,
                )
                for _ in range(self.num_layers)
            ]
        )

    @property
    def is_identity(self) -> bool:
        return self.num_layers == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def encode_fused_time_embed(
    tmu: nn.Module,
    time_stamp=None,
    internal_features: Optional[torch.Tensor] = None,
    adapter: Optional[nn.Module] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """External (+ internal) time embedding ko ek hi jagah se encode karta hai.

    Advanced path mein external aur internal embeddings pehle alag-alag bante the
    aur baad me seedha add ho jaate the. Ab internal embedding ko
    ``extra_embed`` ke roop me TMU encoder ko diya jaata hai, jisse additional
    fusion layer ko **combined** vector dikhta hai.

    Args:
        tmu: :class:`tmu.core.TimeManagementUnit` (ya compatible module).
        time_stamp: World timestamp (see :data:`tmu.time_features.TimeLike`).
        internal_features: Optional internal clock feature tensor.
        adapter: Optional adapter jo internal features ko hidden dim me le jata hai.
        batch_size: Model batch size.
        device: Target device.

    Returns:
        Combined time embedding of shape ``(batch_size, hidden_dim)``.
    """

    if device is None:
        weight = getattr(getattr(tmu, "input_proj", None), "weight", None)
        device = weight.device if torch.is_tensor(weight) else torch.device("cpu")

    extra_embed: Optional[torch.Tensor] = None

    if internal_features is not None:
        if adapter is None:
            raise ValueError("internal_features diya hai lekin adapter missing hai.")

        internal_features = internal_features.to(device=device, dtype=torch.float32)
        extra_embed = adapter(internal_features)

        if extra_embed.dim() == 1:
            extra_embed = extra_embed.unsqueeze(0)

    return tmu.encode_time(
        time_stamp=time_stamp,
        batch_size=batch_size,
        device=device,
        extra_embed=extra_embed,
    )
