from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .core import TimeManagementUnit


class PerLayerTimer:
    def __init__(
        self,
        num_layers: int,
        enabled: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_layers = int(num_layers)
        self.device = device
        self.enabled = bool(enabled) and str(device).startswith("cuda") and torch.cuda.is_available()

        self.start_events: List[Any] = [None] * self.num_layers
        self.end_events: List[Any] = [None] * self.num_layers
        self.used: List[bool] = [False] * self.num_layers

        if self.enabled:
            for i in range(self.num_layers):
                self.start_events[i] = torch.cuda.Event(enable_timing=True)
                self.end_events[i] = torch.cuda.Event(enable_timing=True)

    def reset(self) -> None:
        self.used = [False] * self.num_layers

    def begin(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        if 0 <= layer_idx < self.num_layers:
            self.start_events[layer_idx].record()
            self.used[layer_idx] = True

    def end(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        if 0 <= layer_idx < self.num_layers and self.used[layer_idx]:
            self.end_events[layer_idx].record()

    def collect(self) -> List[float]:
        if not self.enabled:
            return []

        torch.cuda.synchronize()
        times: List[float] = []

        for i in range(self.num_layers):
            if self.used[i]:
                ms = float(self.start_events[i].elapsed_time(self.end_events[i]))
            else:
                ms = 0.0
            times.append(ms)

        self.reset()
        return times


def _make_causal_padding_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    B, T = input_ids.shape

    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()

    if attention_mask is None:
        pad = torch.ones(B, T, dtype=torch.bool, device=device)
    else:
        pad = attention_mask.to(device).bool()

    mask = causal.view(1, 1, T, T) & pad.view(B, 1, 1, T)

    out = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    out.masked_fill_(~mask, torch.finfo(dtype).min)

    return out


def custom_gpt2_forward_with_tmu(
    model: nn.Module,
    tmu: TimeManagementUnit,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    time_embed: torch.Tensor,
    active_layers: Optional[int] = None,
    modulate_blocks: bool = True,
) -> Dict[str, torch.Tensor]:
    transformer = model.transformer
    B, T = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)

    if hasattr(transformer, "drop"):
        hidden = transformer.drop(hidden)

    hidden = tmu.apply_to(hidden, time_embed)

    attn_mask = _make_causal_padding_mask(input_ids, attention_mask, hidden.dtype, device)

    num_layers = len(transformer.h)

    if active_layers is None:
        active_layers = num_layers

    active_layers = max(1, min(int(active_layers), num_layers))

    for i, block in enumerate(transformer.h):
        if i >= active_layers:
            continue

        outputs = block(hidden, attention_mask=attn_mask, use_cache=False)
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if modulate_blocks:
            hidden = tmu.apply_to(hidden, time_embed)

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    return {"logits": logits}


def custom_gpt2_forward_advanced(
    model: nn.Module,
    tmu: TimeManagementUnit,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    time_embed: torch.Tensor,
    active_layers: Optional[int] = None,
    timer: Optional[PerLayerTimer] = None,
    modulate_blocks: bool = True,
) -> Tuple[torch.Tensor, Optional[List[float]]]:
    transformer = model.transformer
    B, T = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)

    if hasattr(transformer, "drop"):
        hidden = transformer.drop(hidden)

    hidden = tmu.apply_to(hidden, time_embed)

    attn_mask = _make_causal_padding_mask(input_ids, attention_mask, hidden.dtype, device)

    num_layers = len(transformer.h)

    if active_layers is None:
        active_layers = num_layers

    active_layers = max(1, min(int(active_layers), num_layers))

    if timer is not None:
        timer.reset()

    for i, block in enumerate(transformer.h):
        if i >= active_layers:
            continue

        if timer is not None:
            timer.begin(i)

        outputs = block(hidden, attention_mask=attn_mask, use_cache=False)
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if timer is not None:
            timer.end(i)

        if modulate_blocks:
            hidden = tmu.apply_to(hidden, time_embed)

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    layer_times = timer.collect() if timer is not None else None

    return logits, layer_times


@contextmanager
def prefix_layer_skip_context(model: nn.Module, active_layers: int):
    original_h = model.transformer.h
    total = len(original_h)
    active_layers = max(1, min(int(active_layers), total))

    model.transformer.h = nn.ModuleList(list(original_h)[:active_layers])

    try:
        yield active_layers
    finally:
        model.transformer.h = original_h
