from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

ADV_INTERNAL_DIM = 24


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def log_norm(x: float, scale: float) -> float:
    x = max(0.0, float(x))
    scale = max(1e-9, float(scale))
    return clamp01(math.log1p(x) / math.log1p(scale))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    b = float(b)
    if abs(b) < 1e-12:
        return float(default)
    return float(a) / b


def float_or_none(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if not s or s.upper() == "N/A":
            return None
        v = float(s)
        return v if v >= 0 else None
    except Exception:
        return None


def cuda_device_index(device: Union[str, torch.device]) -> int:
    if isinstance(device, torch.device):
        if device.index is not None:
            return int(device.index)
        return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0

    s = str(device)
    if s.startswith("cuda"):
        parts = s.split(":")
        if len(parts) > 1:
            return int(parts[1])
        return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0

    return 0


def get_first_field(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            v = d[k]
            if v is not None and v != "":
                return v

    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kk = k.lower()
        if kk in lower:
            v = lower[kk]
            if v is not None and v != "":
                return v

    return None


def prepare_budget_tensor(
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(budget):
        budget = torch.tensor(float(budget), device=device, dtype=torch.float32)

    budget = budget.to(device=device, dtype=torch.float32)

    if budget.dim() == 0:
        budget = budget.unsqueeze(0)

    if budget.dim() > 1:
        budget = budget.reshape(-1)

    if budget.shape[0] == 1 and batch_size > 1:
        budget = budget.expand(batch_size)
    elif budget.shape[0] != batch_size:
        if batch_size == 1:
            budget = budget[:1]
        else:
            raise ValueError(f"Budget batch {budget.shape[0]} != batch_size {batch_size}")

    return budget