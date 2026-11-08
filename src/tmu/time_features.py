from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Sequence, Union

import torch

TimeLike = Union[None, int, float, torch.Tensor, datetime, Sequence[Union[int, float]]]


def _to_time_tensor(time_stamp: TimeLike) -> torch.Tensor:
    if time_stamp is None:
        return torch.tensor([time.time()], dtype=torch.float64)

    if isinstance(time_stamp, datetime):
        return torch.tensor([time_stamp.timestamp()], dtype=torch.float64)

    if isinstance(time_stamp, (int, float)):
        return torch.tensor([float(time_stamp)], dtype=torch.float64)

    if isinstance(time_stamp, torch.Tensor):
        t = time_stamp.detach().cpu().double()
    else:
        t = torch.tensor(list(time_stamp), dtype=torch.float64)

    if t.dim() == 0:
        t = t.unsqueeze(0)

    if t.dim() > 1:
        t = t.reshape(-1)

    return t


def cyclic_time_features(time_stamp: TimeLike) -> torch.Tensor:
    t = _to_time_tensor(time_stamp).unsqueeze(-1)

    periods = (
        60.0,
        3600.0,
        86400.0,
        86400.0 * 7.0,
        86400.0 * 30.0,
        86400.0 * 365.0,
    )

    two_pi = 2.0 * math.pi
    feats = []

    for p in periods:
        frac = torch.remainder(t, p) / p
        feats.append(torch.sin(two_pi * frac))
        feats.append(torch.cos(two_pi * frac))

    return torch.cat(feats, dim=-1).float()
