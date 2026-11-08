from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import TemporalFusionStack
from .time_features import TimeLike, cyclic_time_features


class TimeManagementUnit(nn.Module):
    FEATURE_DIM = 12

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        ema_momentum: float = 0.1,
        dropout: float = 0.0,
        use_ema: bool = True,
        apply_mode: str = "scale",
        fusion_layers: int = 1,
        fusion_expansion: int = 2,
        fusion_dropout: float = 0.0,
    ):
        super().__init__()

        if apply_mode not in {"scale", "add"}:
            raise ValueError("apply_mode must be 'scale' or 'add'")

        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else None
        self.ema_momentum = float(ema_momentum)
        self.dropout = float(dropout)
        self.use_ema = bool(use_ema)
        self.apply_mode = apply_mode
        self.fusion_layers = max(0, int(fusion_layers))
        self.fusion_expansion = max(1, int(fusion_expansion))
        self.fusion_dropout = float(fusion_dropout)

        self.input_proj: Optional[nn.Linear] = None
        self.temporal_mlp: Optional[nn.Module] = None
        self.scale_proj: Optional[nn.Linear] = None
        self.fusion: Optional[TemporalFusionStack] = None

        self.register_buffer("ema_time", torch.zeros(1, self.FEATURE_DIM), persistent=True)

        if self.hidden_dim is not None:
            self.build(self.hidden_dim)

    def build(self, hidden_dim: int) -> None:
        hidden_dim = int(hidden_dim)

        if self.input_proj is not None:
            if self.hidden_dim != hidden_dim:
                raise ValueError(
                    f"TMU already built for hidden_dim={self.hidden_dim}, requested={hidden_dim}"
                )
            return

        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.FEATURE_DIM, hidden_dim)

        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.scale_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

        self.fusion = TemporalFusionStack(
            hidden_dim=hidden_dim,
            num_layers=self.fusion_layers,
            expansion=self.fusion_expansion,
            dropout=self.fusion_dropout,
        )

    def _match_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor

        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)

        if batch_size == 1:
            return tensor[:1]

        raise ValueError(f"Time batch size {tensor.shape[0]} != model batch size {batch_size}")

    def encode_time(
        self,
        time_stamp: TimeLike = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        extra_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_proj is None or self.temporal_mlp is None:
            raise RuntimeError("TMU build nahi hua. Pehle build(hidden_dim) call karein.")

        if device is None:
            device = self.input_proj.weight.device

        if self.input_proj.weight.device != device:
            self.to(device)

        tf = cyclic_time_features(time_stamp).to(device=device, dtype=torch.float32)
        tf = self._match_batch(tf, int(batch_size))

        if self.use_ema:
            m = float(self.ema_momentum)

            with torch.no_grad():
                current_mean = tf.mean(dim=0, keepdim=True)

                if self.ema_time.device != current_mean.device:
                    self.ema_time.data = self.ema_time.data.to(current_mean.device)

                self.ema_time.data.mul_(1.0 - m).add_(m * current_mean)

            tf = m * tf + (1.0 - m) * self.ema_time

        embed = self.input_proj(tf)
        embed = self.temporal_mlp(embed)

        if extra_embed is not None:
            extra = extra_embed.to(device=embed.device, dtype=embed.dtype)

            if extra.dim() == 1:
                extra = extra.unsqueeze(0)

            if extra.shape[0] == 1 and embed.shape[0] > 1:
                extra = extra.expand(embed.shape[0], -1)

            embed = embed + extra

        if self.fusion is not None and not self.fusion.is_identity:
            embed = self.fusion(embed)

        return embed

    def _reshape_time_for_x(self, time_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if time_embed.dim() == 1:
            time_embed = time_embed.unsqueeze(0)

        B = x.shape[0]

        if time_embed.shape[0] == 1 and B > 1:
            time_embed = time_embed.expand(B, -1)
        elif time_embed.shape[0] != B:
            if B == 1:
                time_embed = time_embed[:1]
            else:
                raise ValueError(f"Time embed batch {time_embed.shape[0]} != input batch {B}")

        if x.dim() == 2:
            return time_embed

        return time_embed.reshape(B, *([1] * (x.dim() - 2)), time_embed.shape[-1])

    def apply_to(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return x

        if x.dim() < 2:
            return x

        if self.hidden_dim is None or x.shape[-1] != self.hidden_dim:
            return x

        if self.scale_proj is None:
            raise RuntimeError("TMU scale_proj build nahi hua.")

        t = self._reshape_time_for_x(time_embed.to(device=x.device, dtype=torch.float32), x)
        scale = self.scale_proj(t)

        if self.apply_mode == "scale":
            mod = 1.0 + torch.tanh(scale)

            if self.dropout > 0.0:
                mod = F.dropout(mod, p=self.dropout, training=self.training)

            return x * mod.to(x.dtype)

        mod = scale

        if self.dropout > 0.0:
            mod = F.dropout(mod, p=self.dropout, training=self.training)

        return x + mod.to(x.dtype)

    def forward(self, x: torch.Tensor, time_stamp: TimeLike = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("TimeManagementUnit.forward ko torch.Tensor chahiye.")

        te = self.encode_time(time_stamp, batch_size=x.shape[0], device=x.device)
        return self.apply_to(x, te)

def load_tmu_state_dict(
    tmu: "TimeManagementUnit",
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> List[str]:
    """Backward-compatible TMU state dict load.

    Fusion layers naye parameters add karte hain, isliye purane checkpoints
    (jinmein fusion weights nahi hote) load karte waqt missing keys aate hain.
    Unhe silently ignore kiya jaata hai kyunki naye layers zero-init hain aur
    identity behave karte hain.

    Returns:
        Missing keys jo checkpoint me nahi mile.
    """

    incompatible = tmu.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if verbose and missing:
        print(f"[tmu] checkpoint me missing keys (naye fusion params, identity init): {missing}")

    if verbose and unexpected:
        print(f"[tmu] checkpoint me unexpected keys (ignore kiye gaye): {unexpected}")

    return missing
