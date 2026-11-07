# gpt2_tmu_full.py
#
# GPT-2 + TMU (Time Management Unit)
#
# Is code me 3 cheezein ek saath hain:
# 1. Real timestamped data loading (JSONL / CSV)
# 2. Pace ko actual generation / layer-skipping loop se join karna
# 3. Time coherence evaluation
#
# Chalaiye:
#   python gpt2_tmu_full.py
#
# Real data ke saath:
#   python gpt2_tmu_full.py --data my_data.jsonl
#
# CSV format example:
#   text,timestamp,budget
#   "Morning meeting started",1700000000,0.4
#
# JSONL format example:
#   {"text": "Morning meeting started", "timestamp": "2026-01-01T09:30:00", "budget": 0.4}

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1) TIME UTILITIES
# ============================================================

TimeLike = Union[None, int, float, torch.Tensor, datetime, Sequence[Union[int, float]]]


def _to_time_tensor(time_stamp: TimeLike) -> torch.Tensor:
    """
    Different types ke time input ko 1D float64 tensor me convert karta hai.

    Output shape: (B,)
    """
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
    """
    Time se 12 cyclic features banata hai:
    second, minute/hour, day, week, month, year ke sin/cos.

    Output shape: (B, 12)
    """
    t = _to_time_tensor(time_stamp).unsqueeze(-1)  # (B, 1)

    periods = (
        60.0,                 # second cycle
        3600.0,               # minute / hour cycle
        86400.0,              # day cycle
        86400.0 * 7.0,        # week cycle
        86400.0 * 30.0,       # approximate month cycle
        86400.0 * 365.0,      # approximate year cycle
    )

    two_pi = 2.0 * math.pi
    feats = []

    for p in periods:
        frac = torch.remainder(t, p) / p
        feats.append(torch.sin(two_pi * frac))
        feats.append(torch.cos(two_pi * frac))

    return torch.cat(feats, dim=-1).float()  # (B, 12)


# ============================================================
# 2) TMU CORE: TIME MANAGEMENT UNIT
# ============================================================

class TimeManagementUnit(nn.Module):
    """
    TMU core module.

    Ye time features ko hidden_dim me convert karta hai
    aur activations par time-based modulation lagata hai.

    Important:
    - Shuru me scale_proj zero-initialized hota hai,
      isliye original model ka output turant change nahi hota.
    - Training ke baad TMU time coherence seekhta hai.
    """

    FEATURE_DIM = 12

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        ema_momentum: float = 0.1,
        dropout: float = 0.0,
        use_ema: bool = True,
        apply_mode: str = "scale",
    ):
        super().__init__()

        if apply_mode not in {"scale", "add"}:
            raise ValueError("apply_mode must be 'scale' or 'add'")

        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else None
        self.ema_momentum = float(ema_momentum)
        self.dropout = float(dropout)
        self.use_ema = bool(use_ema)
        self.apply_mode = apply_mode

        self.input_proj: Optional[nn.Linear] = None
        self.temporal_mlp: Optional[nn.Module] = None
        self.scale_proj: Optional[nn.Linear] = None

        # Time coherence ke liye EMA buffer
        self.register_buffer("ema_time", torch.zeros(1, self.FEATURE_DIM), persistent=True)

        if self.hidden_dim is not None:
            self.build(self.hidden_dim)

    def build(self, hidden_dim: int) -> None:
        """
        TMU layers banata hai.
        hidden_dim model ki embedding/activation dimension honi chahiye.
        """
        hidden_dim = int(hidden_dim)

        if self.input_proj is not None:
            if self.hidden_dim != hidden_dim:
                raise ValueError(
                    f"TMU already built for hidden_dim={self.hidden_dim}, requested={hidden_dim}"
                )
            return

        self.hidden_dim = hidden_dim

        # 12 time features -> hidden_dim
        self.input_proj = nn.Linear(self.FEATURE_DIM, hidden_dim)

        # Small temporal MLP
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time modulation scale
        self.scale_proj = nn.Linear(hidden_dim, hidden_dim)

        # Zero init: shuru me TMU ka effect almost zero
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

    def _match_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Time tensor ka batch size model input ke batch size se match karta hai.
        """
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
    ) -> torch.Tensor:
        """
        Time stamp se time embedding banata hai.

        Output shape: (B, hidden_dim)
        """
        if self.input_proj is None or self.temporal_mlp is None:
            raise RuntimeError("TMU build nahi hua. Pehle build(hidden_dim) call karein ya TMUPlugin use karein.")

        if device is None:
            device = self.input_proj.weight.device

        if self.input_proj.weight.device != device:
            self.to(device)

        tf = cyclic_time_features(time_stamp).to(device=device, dtype=torch.float32)
        tf = self._match_batch(tf, int(batch_size))

        # EMA smoothing: time coherence banaye rakhne ke liye
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

        return embed

    def _reshape_time_for_x(self, time_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        time_embed ko x ke shape ke hisab se broadcast-ready banata hai.

        x:
          (B, D)        -> time_embed (B, D)
          (B, L, D)     -> time_embed (B, 1, D)
          (B, L1, L2, D)-> time_embed (B, 1, 1, D)
        """
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
        """
        TMU modulation apply karta hai.

        Agar x ki last dimension hidden_dim nahi hai, to x unchanged rahega.
        Isse logits/output heads galat tarike se modify nahi honge.
        """
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
            # residual multiplicative modulation
            mod = 1.0 + torch.tanh(scale)

            if self.dropout > 0.0:
                mod = F.dropout(mod, p=self.dropout, training=self.training)

            return x * mod.to(x.dtype)

        # apply_mode == "add"
        mod = scale

        if self.dropout > 0.0:
            mod = F.dropout(mod, p=self.dropout, training=self.training)

        return x + mod.to(x.dtype)

    def forward(self, x: torch.Tensor, time_stamp: TimeLike = None) -> torch.Tensor:
        """
        Direct TMU use ke liye convenience forward.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("TimeManagementUnit.forward ko torch.Tensor chahiye.")

        te = self.encode_time(time_stamp, batch_size=x.shape[0], device=x.device)
        return self.apply_to(x, te)


def infer_hidden_dim(model: nn.Module) -> int:
    """
    Kisi bhi PyTorch model se hidden_dim auto-detect karne ki koshish karta hai.
    """
    if hasattr(model, "config"):
        config = model.config

        for attr in ("hidden_size", "n_embd", "d_model", "embedding_size", "width"):
            if hasattr(config, attr):
                value = getattr(config, attr)
                if isinstance(value, int) and value > 0:
                    return int(value)

    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return int(module.embedding_dim)

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)

    raise ValueError("Model se hidden_dim auto-detect nahi ho paya. TMUPlugin me hidden_dim manually pass karein.")


class TMUPlugin(nn.Module):
    """
    Kisi bhi PyTorch model ko TMU ke saath wrap karta hai.

    Modes:
      - "hooks": model ke andar layers/blocks par hook lagata hai
      - "input": input tensor par TMU apply karta hai
      - "off": TMU maujood hai lekin effect nahi dalta
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: Optional[int] = None,
        mode: str = "hooks",
        target_modules: Optional[Sequence[str]] = None,
        ema_momentum: float = 0.1,
        dropout: float = 0.0,
        use_ema: bool = True,
        apply_mode: str = "scale",
    ):
        super().__init__()

        if mode not in {"hooks", "input", "off"}:
            raise ValueError("mode must be 'hooks', 'input', or 'off'")

        self.base_model = base_model

        if hidden_dim is None:
            hidden_dim = infer_hidden_dim(base_model)

        self.tmu = TimeManagementUnit(
            hidden_dim=int(hidden_dim),
            ema_momentum=ema_momentum,
            dropout=dropout,
            use_ema=use_ema,
            apply_mode=apply_mode,
        )

        self.mode = mode
        self.target_modules = list(target_modules) if target_modules is not None else []
        self._current_time_embed: Optional[torch.Tensor] = None
        self.last_time_embed: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

        if self.mode == "hooks":
            self._register_hooks()

    def _is_target(self, name: str, module: nn.Module) -> bool:
        """
        Ye decide karta hai ki kis module par TMU hook lagega.
        """
        cls = module.__class__.__name__.lower()
        name = name.lower()

        if not self.target_modules:
            keywords = (
                "block",
                "layer",
                "encoder",
                "decoder",
                "attention",
                "mlp",
                "feedforward",
                "fluid",
            )
            return any(k in cls or k in name for k in keywords)

        return any(p.lower() in name or p.lower() in cls for p in self.target_modules)

    def _register_hooks(self) -> None:
        """
        Base model ke target modules par forward hooks lagata hai.
        """
        self.remove_hooks()
        matched = False

        for name, module in self.base_model.named_modules():
            if name == "":
                continue

            if self._is_target(name, module):
                handle = module.register_forward_hook(self._hook_fn)
                self._hooks.append(handle)
                matched = True

        # Agar koi internal target nahi mila, to root output par hook lagao.
        if not matched:
            handle = self.base_model.register_forward_hook(self._hook_fn)
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """
        Lage hue sabhi TMU hooks hata deta hai.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _hook_fn(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        """
        Forward hook function.
        """
        return self._apply_to_output(output)

    def _apply_to_output(self, output: Any) -> Any:
        """
        Different output formats ko handle karta hai:
          - Tensor
          - tuple
          - list
          - dict
          - HuggingFace-style object with last_hidden_state
        """
        if self._current_time_embed is None:
            return output

        if isinstance(output, torch.Tensor):
            return self.tmu.apply_to(output, self._current_time_embed)

        # HuggingFace-style outputs
        if hasattr(output, "last_hidden_state") and isinstance(getattr(output, "last_hidden_state"), torch.Tensor):
            try:
                output.last_hidden_state = self.tmu.apply_to(
                    output.last_hidden_state,
                    self._current_time_embed,
                )
                return output
            except Exception:
                pass

        if isinstance(output, tuple):
            items = list(output)
            for i, item in enumerate(items):
                if isinstance(item, torch.Tensor):
                    items[i] = self.tmu.apply_to(item, self._current_time_embed)
                    break
            return tuple(items)

        if isinstance(output, list):
            items = list(output)
            for i, item in enumerate(items):
                if isinstance(item, torch.Tensor):
                    items[i] = self.tmu.apply_to(item, self._current_time_embed)
                    break
            return items

        if isinstance(output, dict):
            out = dict(output)
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    out[k] = self.tmu.apply_to(v, self._current_time_embed)
                    break
            return out

        return output

    def _find_sample_tensor(self, x: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        batch_size aur device janane ke liye pehla tensor dhundhta hai.
        """
        if isinstance(x, torch.Tensor):
            return x

        for a in args:
            if isinstance(a, torch.Tensor):
                return a

        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v

        return None

    def forward(
        self,
        x: Any = None,
        *args: Any,
        time_stamp: TimeLike = None,
        time_embed: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Wrapped model forward.

        Usage:
          out = plugin(input_ids, time_stamp=None)
          out = plugin(embeddings, time_stamp=123456.0)
        """
        sample = self._find_sample_tensor(x, args, kwargs)

        if sample is None:
            raise ValueError("TMUPlugin ko batch_size/device janane ke liye kam se kam ek torch.Tensor chahiye.")

        batch_size = int(sample.shape[0])
        device = sample.device

        if time_embed is None:
            time_embed = self.tmu.encode_time(
                time_stamp=time_stamp,
                batch_size=batch_size,
                device=device,
            )
        else:
            time_embed = time_embed.to(device=device, dtype=torch.float32)

        self._current_time_embed = time_embed
        self.last_time_embed = time_embed

        try:
            if self.mode == "input" and isinstance(x, torch.Tensor):
                x = self.tmu.apply_to(x, time_embed)

            if x is None:
                if args:
                    output = self.base_model(*args, **kwargs)
                else:
                    output = self.base_model(**kwargs)
            else:
                output = self.base_model(x, *args, **kwargs)

            return output

        finally:
            self._current_time_embed = None


def freeze_base_and_enable_tmu(plugin: TMUPlugin) -> List[nn.Parameter]:
    """
    Base model freeze karo, sirf TMU train karo.
    Ye quick training ke liye useful hai.
    """
    for p in plugin.base_model.parameters():
        p.requires_grad_(False)

    params: List[nn.Parameter] = []
    for p in plugin.tmu.parameters():
        p.requires_grad_(True)
        params.append(p)

    return params


# ============================================================
# 3) CAPABILITY HEADS: RHYTHM + PACE
# ============================================================

TIME_FEATURE_DIM = getattr(TimeManagementUnit, "FEATURE_DIM", 12)


class TmuCapabilityHeads(nn.Module):
    """
    TMU ke saath training heads:
    1. rhythm_head  -> future/cyclic time pattern predict karna
    2. pace_head    -> reasoning speed / compute budget predict karna
    """

    def __init__(self, hidden_dim: int, budget_dim: int = 1):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.budget_dim = int(budget_dim)

        # Budget embedding: initially zero, taki shuru me model behavior na badle
        self.budget_proj = nn.Linear(self.budget_dim, self.hidden_dim)
        nn.init.zeros_(self.budget_proj.weight)
        nn.init.zeros_(self.budget_proj.bias)

        # World rhythm prediction head
        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, TIME_FEATURE_DIM),
        )

        # Thinking speed / pace prediction head
        pace_hidden = max(8, self.hidden_dim // 2)
        self.pace_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        # Pace initially neutral
        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
        """
        time_embed: (B, hidden_dim)
        budget:     (B,) ya scalar, normalized 0..1

        returns:
            rhythm_pred: (B, TIME_FEATURE_DIM)
            pace:        (B,) in [0, 1]
            pace_logit:  (B,)
        """
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


# ============================================================
# 4) REAL TIMESTAMPED DATA LOADING
# ============================================================

def parse_timestamp(value: Any) -> float:
    """
    Timestamp ko Unix float seconds me convert karta hai.

    Supported:
    - 1700000000
    - "1700000000"
    - "2026-01-01T09:30:00"
    - "2026-01-01 09:30:00"
    - "2026-01-01"
    - datetime object
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime):
        return value.timestamp()

    s = str(value).strip()
    if not s:
        raise ValueError("Empty timestamp")

    # Try direct float Unix timestamp
    try:
        return float(s)
    except ValueError:
        pass

    # Try ISO format
    s_iso = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s_iso).timestamp()
    except ValueError:
        pass

    # Try common formats
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue

    raise ValueError(f"Timestamp parse nahi ho paya: {value}")


def _get_first(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    """
    Dict me se pehla available field nikalta hai.
    Case-insensitive fallback bhi karta hai.
    """
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]

    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kk = k.lower()
        if kk in lower and lower[kk] not in (None, ""):
            return lower[kk]

    return None


def load_timestamped_records(path: str) -> Tuple[List[str], List[float], List[float]]:
    """
    Real timestamped data load karta hai.

    Supported files:
    - .jsonl
    - .json
    - .csv

    Expected fields:
    - text / content / prompt / message / sentence
    - timestamp / time / datetime / date / created_at
    - optional budget / time_budget / budget_norm
    """
    ext = os.path.splitext(path)[1].lower()
    raw: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("records") or data.get("data") or []

        if not isinstance(data, list):
            raise ValueError("JSON file me list of records honi chahiye, ya {'records': [...]} format chahiye.")

        raw = data

    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # BOM remove
                row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
                raw.append(row)

    else:
        raise ValueError("Sirf .jsonl, .json, .csv supported hain.")

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []

    for rec in raw:
        if not isinstance(rec, dict):
            continue

        text = _get_first(rec, ["text", "content", "prompt", "message", "sentence"])
        ts = _get_first(rec, ["timestamp", "time", "datetime", "date", "created_at"])

        if text is None or ts is None:
            continue

        b = _get_first(rec, ["budget", "time_budget", "budget_norm"])
        if b is None:
            b = 0.5

        try:
            b = float(b)
        except Exception:
            b = 0.5

        b = min(1.0, max(0.05, b))

        texts.append(str(text))
        timestamps.append(parse_timestamp(ts))
        budgets.append(b)

    if not texts:
        raise ValueError("Koi valid record nahi mila. Fields check karein: text, timestamp.")

    return texts, timestamps, budgets


def build_demo_records(
    num_samples: int = 192,
    base_time: float = 1_700_000_000.0,
    seed: int = 42,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Demo ke liye synthetic timestamped records banata hai.
    Real data ke liye --data flag use karein.
    """
    base_texts = [
        "The morning sun rises and the city starts working.",
        "At noon the office lunch break happens quickly.",
        "In the evening traffic increases on every road.",
        "At night the model should answer shorter messages.",
        "During the day people ask many practical questions.",
        "Early morning thoughts are usually calm and clear.",
        "Late night reasoning can become slower and deeper.",
        "The weekly routine repeats in a predictable rhythm.",
        "Seasons change slowly and affect human plans.",
        "A busy schedule needs faster decisions.",
        "A free schedule allows longer thinking.",
        "Time pressure changes the quality of answers.",
        "The clock controls meetings and deadlines.",
        "Deadlines force compact reasoning.",
        "Open time encourages exploration.",
        "Rhythm helps prediction and coherence.",
    ]

    g = torch.Generator()
    g.manual_seed(int(seed))

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []

    t = float(base_time)

    for i in range(int(num_samples)):
        text = base_texts[i % len(base_texts)] + f" Record {i}."

        if i > 0:
            delta = torch.rand(1, generator=g).item() * 3600.0 + 10.0
            t += delta

        budget = torch.rand(1, generator=g).item() * 0.9 + 0.1

        texts.append(text)
        timestamps.append(t)
        budgets.append(budget)

    return texts, timestamps, budgets


# ============================================================
# 5) GPT-2 TIMESTAMPED DATASET
# ============================================================

class GPT2TimestampedDataset(Dataset):
    """
    GPT-2 ke liye time-stamped text dataset.

    Har sample me:
      - input_ids
      - attention_mask
      - labels
      - t: timestamp
      - budget: allowed normalized time 0..1
      - y_rhythm: future cyclic time features
      - y_pace: desired reasoning speed, capped by budget
    """

    def __init__(
        self,
        texts: Sequence[str],
        timestamps: Sequence[float],
        budgets: Sequence[float],
        tokenizer: Any,
        seq_len: int = 64,
    ):
        super().__init__()

        if len(texts) != len(timestamps) or len(texts) != len(budgets):
            raise ValueError("texts, timestamps, budgets ki length same honi chahiye.")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.samples: List[Dict[str, Any]] = []

        for text, t, b in zip(texts, timestamps, budgets):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            t_tensor = torch.tensor(float(t), dtype=torch.float32)
            budget_tensor = torch.tensor(float(b), dtype=torch.float32)

            # Rhythm target: 10 minutes future ke cyclic features
            y_rhythm = cyclic_time_features(
                torch.tensor([float(t) + 600.0], dtype=torch.float64)
            ).squeeze(0)

            # Pace target: text difficulty + time-of-day need, budget se capped
            valid_len = int(attention_mask.sum().item())
            difficulty = min(1.0, max(0.0, valid_len / float(max(1, self.seq_len))))

            day_phase = (float(t) % 86400.0) / 86400.0
            time_need = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * day_phase))

            desired_pace = 0.5 * difficulty + 0.5 * time_need
            y_pace_value = min(float(desired_pace), float(b))
            y_pace = torch.tensor(y_pace_value, dtype=torch.float32)

            self.samples.append(
                {
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "t": t_tensor,
                    "budget": budget_tensor,
                    "y_rhythm": y_rhythm,
                    "y_pace": y_pace,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 6) TRAINING: SIRF TMU + HEADS TRAIN HONGE
# ============================================================

def train_gpt2_tmu(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 2,
    lr: float = 1e-3,
    w_lm: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    """
    Sirf TMU + capability heads train karte hain.
    GPT-2 base model freeze rehta hai, isliye quick toy training possible hai.
    """
    plugin.to(device)
    heads.to(device)

    # Training me cache band rakho
    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = list(tmu_params) + list(heads.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    history = []

    for epoch in range(int(epochs)):
        plugin.train()
        # Base model eval mode me rakho taaki dropout noise kam ho,
        # lekin TMU train mode me rahe.
        plugin.base_model.eval()
        plugin.tmu.train()
        heads.train()

        total_loss = 0.0
        total_lm = 0.0
        total_rhythm = 0.0
        total_pace = 0.0
        total_over = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)

            y_rhythm = batch["y_rhythm"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)

            B = int(input_ids.shape[0])

            # Ek baar time_embed banao, phir plugin aur heads dono me use karo
            time_embed = plugin.tmu.encode_time(
                time_stamp=t,
                batch_size=B,
                device=device,
            )

            outputs = plugin(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                time_embed=time_embed,
                use_cache=False,
                return_dict=True,
            )

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss_lm = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
                loss_lm = outputs["loss"]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                loss_lm = outputs[0]
            else:
                raise RuntimeError("GPT-2 output se language modeling loss nahi mil paya.")

            rhythm_pred, pace_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace_bce = F.binary_cross_entropy(pace_safe, y_pace)

            # Budget exceed penalty: pace > budget hone par saja
            over_budget = F.relu(pace_pred - budget).mean()
            loss_pace = loss_pace_bce + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * loss_pace
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            total_loss += float(loss.item())
            total_lm += float(loss_lm.item())
            total_rhythm += float(loss_rhythm.item())
            total_pace += float(loss_pace.item())
            total_over += float(over_budget.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_lm = total_lm / max(1, num_batches)
        avg_rhythm = total_rhythm / max(1, num_batches)
        avg_pace = total_pace / max(1, num_batches)
        avg_over = total_over / max(1, num_batches)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "loss_lm": avg_lm,
                "loss_rhythm": avg_rhythm,
                "loss_pace": avg_pace,
                "over_budget": avg_over,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"lm={avg_lm:.4f} | "
                f"rhythm={avg_rhythm:.4f} | "
                f"pace={avg_pace:.4f} | "
                f"over={avg_over:.4f}"
            )

    return history


# ============================================================
# 7) PACE CONTROLLER: PACE -> LAYERS / TOKENS / GENERATION PLAN
# ============================================================

@dataclass
class PaceConfig:
    min_layers: int = 2
    max_layers: int = 12
    min_tokens: int = 8
    max_tokens: int = 64
    min_temperature: float = 0.6
    max_temperature: float = 1.0
    refine_min: int = 1
    refine_max: int = 3


class PaceController:
    """
    Pace (0..1) ko actual execution plan me convert karta hai:
    - active_layers: kitne GPT-2 blocks chalenge
    - max_new_tokens: kitne tokens generate honge
    - temperature: sampling strictness
    - refinement_passes: optional multi-pass reasoning
    """

    def __init__(self, num_layers: int, config: Optional[PaceConfig] = None):
        self.num_layers = int(num_layers)
        self.config = config or PaceConfig()

        self.config.max_layers = min(int(self.config.max_layers), self.num_layers)
        self.config.min_layers = max(1, min(int(self.config.min_layers), self.config.max_layers))

    def from_pace(self, pace: float, budget: Optional[float] = None) -> Dict[str, Union[int, float]]:
        pace = float(max(0.0, min(1.0, pace)))

        if budget is None:
            budget_eff = 1.0
        else:
            budget_eff = float(max(0.0, min(1.0, budget)))

        cfg = self.config

        active_layers = int(round(cfg.min_layers + pace * (cfg.max_layers - cfg.min_layers)))
        active_layers = max(1, min(self.num_layers, active_layers))

        token_budget = int(round(cfg.min_tokens + budget_eff * (cfg.max_tokens - cfg.min_tokens)))
        pace_tokens = int(round(cfg.min_tokens + pace * (cfg.max_tokens - cfg.min_tokens)))

        # Budget hard cap hai: pace zyada ho lekin budget kam ho to kam tokens
        max_new_tokens = max(1, min(token_budget, pace_tokens))

        temperature = float(cfg.min_temperature + pace * (cfg.max_temperature - cfg.min_temperature))
        refinement_passes = int(round(cfg.refine_min + pace * (cfg.refine_max - cfg.refine_min)))

        return {
            "pace": pace,
            "active_layers": active_layers,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "refinement_passes": refinement_passes,
        }


# ============================================================
# 8) CUSTOM GPT-2 FORWARD WITH TMU + LAYER SKIPPING
# ============================================================

def _make_causal_padding_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    GPT-2 blocks ke liye 4D causal + padding mask banata hai.

    Shape: (B, 1, T, T)
    """
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
    """
    Custom GPT-2 forward:
    - TMU manually apply hota hai
    - Layer skipping possible hai
    - KV cache use nahi hota, isliye slow but controllable

    active_layers:
        - None = sab layers
        - int = sirf pehli N layers chalayenge
    """
    transformer = model.transformer
    B, T = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)

    if hasattr(transformer, "drop"):
        hidden = transformer.drop(hidden)

    # Input par TMU modulation
    hidden = tmu.apply_to(hidden, time_embed)

    attn_mask = _make_causal_padding_mask(input_ids, attention_mask, hidden.dtype, device)

    num_layers = len(transformer.h)

    if active_layers is None:
        active_layers = num_layers

    active_layers = max(1, min(int(active_layers), num_layers))

    for i, block in enumerate(transformer.h):
        # Layer skipping: active_layers ke baad ke blocks skip
        if i >= active_layers:
            continue

        outputs = block(hidden, attention_mask=attn_mask, use_cache=False)
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Har active block ke baad TMU modulation
        if modulate_blocks:
            hidden = tmu.apply_to(hidden, time_embed)

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    return {"logits": logits}


def predict_pace(
    tmu: TimeManagementUnit,
    heads: TmuCapabilityHeads,
    timestamp: TimeLike,
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TMU + heads se pace predict karta hai.

    Returns:
        pace: (B,)
        time_embed: (B, hidden_dim)
    """
    time_embed = tmu.encode_time(
        time_stamp=timestamp,
        batch_size=batch_size,
        device=device,
    )

    if torch.is_tensor(budget):
        budget_t = budget.to(device=device, dtype=torch.float32)
    else:
        budget_t = torch.full((batch_size,), float(budget), device=device, dtype=torch.float32)

    if budget_t.dim() == 0:
        budget_t = budget_t.unsqueeze(0)

    if budget_t.shape[0] == 1 and batch_size > 1:
        budget_t = budget_t.expand(batch_size)
    elif budget_t.shape[0] != batch_size:
        if batch_size == 1:
            budget_t = budget_t[:1]
        else:
            raise ValueError(f"Budget batch {budget_t.shape[0]} != batch_size {batch_size}")

    _, pace, _ = heads(time_embed, budget_t)

    return pace, time_embed


# ============================================================
# 9) GENERATION: LAYER-SKIP + PLUGIN CACHE DO MODES
# ============================================================

@torch.no_grad()
def custom_generate_with_layer_skip(
    model: nn.Module,
    tmu: TimeManagementUnit,
    heads: TmuCapabilityHeads,
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    pace_config: Optional[PaceConfig] = None,
    max_new_override: Optional[int] = None,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_k: int = 50,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    True layer-skipping generation.

    Ye slow hai kyunki KV cache use nahi hota,
    lekin pace ke hisab se active layers control kar sakta hai.
    """
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    B = 1

    pace, time_embed = predict_pace(tmu, heads, timestamp, budget, B, device)
    pace_val = float(pace[0].item())

    num_layers = len(model.transformer.h)
    ctrl = PaceController(num_layers, pace_config)
    plan = ctrl.from_pace(pace_val, budget)

    max_new = int(plan["max_new_tokens"])
    if max_new_override is not None:
        max_new = min(max_new, int(max_new_override))

    temp = float(plan["temperature"]) if temperature is None else float(temperature)
    temp = max(1e-3, temp)

    if seed is not None:
        torch.manual_seed(int(seed))

    generated = input_ids.clone()

    for _ in range(max_new):
        attn = torch.ones_like(generated)

        out = custom_gpt2_forward_with_tmu(
            model=model,
            tmu=tmu,
            input_ids=generated,
            attention_mask=attn,
            time_embed=time_embed,
            active_layers=plan["active_layers"],
            modulate_blocks=True,
        )

        logits = out["logits"][:, -1, :]

        if not do_sample:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temp

            if top_k is None or top_k <= 0:
                k = logits.size(-1)
            else:
                k = min(int(top_k), logits.size(-1))

            topk_logits, topk_indices = torch.topk(logits, k)
            probs = F.softmax(topk_logits, dim=-1)
            selected = torch.multinomial(probs, 1)
            next_token = topk_indices.gather(-1, selected)

        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

    prompt_len = input_ids.shape[1]
    text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)

    plan_out = dict(plan)
    plan_out["pace"] = pace_val
    plan_out["generated_tokens"] = int(generated.shape[1] - prompt_len)

    return text, plan_out


@torch.no_grad()
def generate_with_plugin_cache(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    pace_config: Optional[PaceConfig] = None,
    max_new_override: Optional[int] = None,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_k: int = 50,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fast HuggingFace generate path.

    Isme true layer skipping nahi hota,
    lekin pace se max_new_tokens control hota hai.
    Ye production-like fast path hai.
    """
    device = next(plugin.base_model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    pace, time_embed = predict_pace(plugin.tmu, heads, timestamp, budget, 1, device)
    pace_val = float(pace[0].item())

    num_layers = len(plugin.base_model.transformer.h)
    plan = PaceController(num_layers, pace_config).from_pace(pace_val, budget)

    max_new = int(plan["max_new_tokens"])
    if max_new_override is not None:
        max_new = min(max_new, int(max_new_override))

    temp = float(plan["temperature"]) if temperature is None else float(temperature)
    temp = max(1e-3, temp)

    old_current = plugin._current_time_embed
    old_cache = None

    if hasattr(plugin.base_model, "config"):
        old_cache = plugin.base_model.config.use_cache
        plugin.base_model.config.use_cache = True

    # Generation ke dauraan har internal forward me yahi time_embed use hoga
    plugin._current_time_embed = time_embed

    try:
        if seed is not None:
            torch.manual_seed(int(seed))

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_k"] = top_k

        output_ids = plugin.base_model.generate(**gen_kwargs)

    finally:
        plugin._current_time_embed = old_current
        if hasattr(plugin.base_model, "config") and old_cache is not None:
            plugin.base_model.config.use_cache = old_cache

    prompt_len = input_ids.shape[1]
    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    plan_out = dict(plan)
    plan_out["pace"] = pace_val
    plan_out["generated_tokens"] = int(output_ids.shape[1] - prompt_len)

    return text, plan_out


# ============================================================
# 10) EVALUATION: TIME COHERENCE
# ============================================================

def _flatten_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Padding positions hatakar valid logits flatten karta hai.

    logits: (B, T, V)
    attention_mask: (B, T)
    returns: (N, V)
    """
    mask = attention_mask.bool()
    return logits[mask]


def _compare_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Do time points ke logits compare karta hai.

    Metrics:
    - mad: mean absolute difference
    - cos: cosine similarity
    - agree: argmax next-token agreement
    """
    a = _flatten_valid_logits(logits_a, attention_mask)
    b = _flatten_valid_logits(logits_b, attention_mask)

    if a.numel() == 0 or b.numel() == 0:
        return {"mad": 0.0, "cos": 1.0, "agree": 1.0}

    mad = (a - b).abs().mean().item()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()

    return {"mad": float(mad), "cos": float(cos), "agree": float(agree)}


def _generation_overlap(text_a: str, text_b: str, tokenizer: Any) -> Dict[str, float]:
    """
    Do generated texts ka token overlap nikalta hai.

    Metrics:
    - jaccard: set overlap
    - sequence_ratio: order-aware similarity
    """
    ids_a = tokenizer.encode(text_a, add_special_tokens=False)
    ids_b = tokenizer.encode(text_b, add_special_tokens=False)

    if not ids_a and not ids_b:
        return {"jaccard": 1.0, "sequence_ratio": 1.0}

    set_a = set(ids_a)
    set_b = set(ids_b)
    union = set_a | set_b

    jaccard = len(set_a & set_b) / len(union) if union else 0.0
    seq_ratio = SequenceMatcher(None, ids_a, ids_b).ratio()

    return {"jaccard": float(jaccard), "sequence_ratio": float(seq_ratio)}


def _get_logits_for_time(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    x: torch.Tensor,
    am: torch.Tensor,
    t: torch.Tensor,
    budget: float,
    use_layer_skip: bool,
    pace_config: Optional[PaceConfig],
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Optional[float]]:
    """
    Ek time point par logits nikalta hai.

    use_layer_skip=True:
        custom forward + active layers
    use_layer_skip=False:
        normal plugin hooks forward
    """
    device = x.device
    B = int(x.shape[0])

    if use_layer_skip:
        pace, time_embed = predict_pace(plugin.tmu, heads, t, budget, B, device)
        pace_val = float(pace[0].item())

        num_layers = len(plugin.base_model.transformer.h)
        plan = PaceController(num_layers, pace_config).from_pace(pace_val, budget)

        out = custom_gpt2_forward_with_tmu(
            model=plugin.base_model,
            tmu=plugin.tmu,
            input_ids=x,
            attention_mask=am,
            time_embed=time_embed,
            active_layers=plan["active_layers"],
            modulate_blocks=True,
        )

        return out["logits"], plan, pace_val

    else:
        out = plugin(
            input_ids=x,
            attention_mask=am,
            time_stamp=t,
            use_cache=False,
            return_dict=True,
        )
        return out.logits, None, None


def evaluate_time_coherence(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    tokenizer: Any,
    dataset: GPT2TimestampedDataset,
    device: str,
    num_samples: int = 2,
    periods: Sequence[float] = (86400.0, 604800.0, 31536000.0),
    cross_period: float = 43200.0,
    max_gen_tokens: int = 8,
    use_layer_skip: bool = False,
    pace_config: Optional[PaceConfig] = None,
    budget: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Time coherence eval karta hai.

    Idea:
    - Same text ko t aur t+period par run karo.
    - Agar model ne time rhythm seekha hai, to:
        * t aur t+24h similar hona chahiye
        * t aur t+7d similar hona chahiye
        * t aur t+1year similar hona chahiye
    - Lekin t aur t+12h me kuch difference hona chahiye,
      kyunki din ka half change hua.

    Metrics:
    - mad: logit mean absolute difference, kam = stable
    - cos: logit cosine similarity, zyada = stable
    - agree: next-token argmax agreement, zyada = stable
    - gen_jaccard / gen_sequence_ratio: generated text overlap
    """
    plugin.eval()
    heads.eval()

    all_periods: List[float] = []
    for p in list(periods) + [cross_period]:
        if p not in all_periods:
            all_periods.append(float(p))

    period_metrics: Dict[float, Dict[str, List[float]]] = {
        p: {"mad": [], "cos": [], "agree": [], "gen_jaccard": [], "gen_sequence_ratio": []}
        for p in all_periods
    }

    n = min(int(num_samples), len(dataset))

    for i in range(n):
        sample = dataset[i]

        x = sample["input_ids"].unsqueeze(0).to(device)
        am = sample["attention_mask"].unsqueeze(0).to(device)
        t = sample["t"].to(device)

        valid_ids = x[0][am[0].bool()]
        prompt = tokenizer.decode(valid_ids, skip_special_tokens=True)

        logits_t, plan_t, pace_t = _get_logits_for_time(
            plugin=plugin,
            heads=heads,
            x=x,
            am=am,
            t=t,
            budget=budget,
            use_layer_skip=use_layer_skip,
            pace_config=pace_config,
        )

        if use_layer_skip:
            gen_t, _ = custom_generate_with_layer_skip(
                model=plugin.base_model,
                tmu=plugin.tmu,
                heads=heads,
                tokenizer=tokenizer,
                prompt=prompt,
                timestamp=t,
                budget=budget,
                pace_config=pace_config,
                max_new_override=max_gen_tokens,
                do_sample=False,
                seed=123,
            )
        else:
            gen_t, _ = generate_with_plugin_cache(
                plugin=plugin,
                heads=heads,
                tokenizer=tokenizer,
                prompt=prompt,
                timestamp=t,
                budget=budget,
                pace_config=pace_config,
                max_new_override=max_gen_tokens,
                do_sample=False,
                seed=123,
            )

        for p in all_periods:
            t_p = t + float(p)

            logits_tp, plan_tp, pace_tp = _get_logits_for_time(
                plugin=plugin,
                heads=heads,
                x=x,
                am=am,
                t=t_p,
                budget=budget,
                use_layer_skip=use_layer_skip,
                pace_config=pace_config,
            )

            comp = _compare_logits(logits_t, logits_tp, am)

            if use_layer_skip:
                gen_tp, _ = custom_generate_with_layer_skip(
                    model=plugin.base_model,
                    tmu=plugin.tmu,
                    heads=heads,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    timestamp=t_p,
                    budget=budget,
                    pace_config=pace_config,
                    max_new_override=max_gen_tokens,
                    do_sample=False,
                    seed=123,
                )
            else:
                gen_tp, _ = generate_with_plugin_cache(
                    plugin=plugin,
                    heads=heads,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    timestamp=t_p,
                    budget=budget,
                    pace_config=pace_config,
                    max_new_override=max_gen_tokens,
                    do_sample=False,
                    seed=123,
                )

            ov = _generation_overlap(gen_t, gen_tp, tokenizer)

            period_metrics[p]["mad"].append(comp["mad"])
            period_metrics[p]["cos"].append(comp["cos"])
            period_metrics[p]["agree"].append(comp["agree"])
            period_metrics[p]["gen_jaccard"].append(ov["jaccard"])
            period_metrics[p]["gen_sequence_ratio"].append(ov["sequence_ratio"])

        if verbose:
            print(f"[Eval sample {i + 1}/{n}] prompt preview: {prompt[:80]}...")

    summary: Dict[str, Any] = {}

    for p in all_periods:
        vals = period_metrics[p]
        if not vals["cos"]:
            continue

        key = f"period_{int(p)}s"
        summary[key] = {
            "mad": float(sum(vals["mad"]) / len(vals["mad"])),
            "cos": float(sum(vals["cos"]) / len(vals["cos"])),
            "agree": float(sum(vals["agree"]) / len(vals["agree"])),
            "gen_jaccard": float(sum(vals["gen_jaccard"]) / len(vals["gen_jaccard"])),
            "gen_sequence_ratio": float(sum(vals["gen_sequence_ratio"]) / len(vals["gen_sequence_ratio"])),
        }

    # Useful derived scores
    if "period_86400s" in summary:
        summary["daily_coherence_cos"] = summary["period_86400s"]["cos"]

    if "period_43200s" in summary:
        summary["half_day_sensitivity_1_minus_cos"] = 1.0 - summary["period_43200s"]["cos"]

    if "period_604800s" in summary:
        summary["weekly_coherence_cos"] = summary["period_604800s"]["cos"]

    if "period_31536000s" in summary:
        summary["yearly_coherence_cos"] = summary["period_31536000s"]["cos"]

    return summary


# ============================================================
# 11) SAVE / LOAD
# ============================================================

def save_system(plugin: TMUPlugin, heads: TmuCapabilityHeads, path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
        },
        path,
    )


def load_system(plugin: TMUPlugin, heads: TmuCapabilityHeads, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    return plugin, heads


# ============================================================
# 12) MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-2 + TMU full pipeline")

    parser.add_argument("--data", type=str, default=None, help="Real timestamped JSONL/JSON/CSV file")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")

    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=192, help="Demo dataset size agar --data nahi hai")
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max-gen-tokens", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--layer-skip-eval", action="store_true", help="Eval me custom layer-skip path use karo")

    parser.add_argument("--save", type=str, default="gpt2_tmu_full.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Device:", device)

    # ---------------- Model + Tokenizer ----------------
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False

    hidden_dim = int(model.config.n_embd)
    num_layers = len(model.transformer.h)

    print("GPT-2 hidden_dim:", hidden_dim)
    print("GPT-2 layers:", num_layers)

    # ---------------- TMU Plugin ----------------
    plugin = TMUPlugin(
        base_model=model,
        hidden_dim=hidden_dim,
        mode="hooks",
        target_modules=["GPT2Block", "block"],
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
    )

    heads = TmuCapabilityHeads(hidden_dim=hidden_dim)

    # ---------------- Data ----------------
    if args.data:
        print(f"Loading real timestamped data: {args.data}")
        texts, timestamps, budgets = load_timestamped_records(args.data)
    else:
        print("No --data provided. Using demo timestamped records.")
        texts, timestamps, budgets = build_demo_records(
            num_samples=args.samples,
            base_time=1_700_000_000.0,
            seed=42,
        )

    print("Records:", len(texts))

    dataset = GPT2TimestampedDataset(
        texts=texts,
        timestamps=timestamps,
        budgets=budgets,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ---------------- Train or Load ----------------
    if args.load:
        print(f"Loading TMU capabilities from: {args.load}")
        load_system(plugin, heads, args.load, device=device)
    else:
        print("\n=== Training TMU capabilities on GPT-2 ===")
        train_gpt2_tmu(
            plugin=plugin,
            heads=heads,
            train_loader=loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            w_lm=1.0,
            w_rhythm=1.0,
            w_pace=0.7,
            budget_penalty=2.0,
            verbose=True,
        )

        if args.save:
            save_system(plugin, heads, args.save)
            print(f"\nSaved: {args.save}")

    # ---------------- Pace Config ----------------
    pace_config = PaceConfig(
        min_layers=2,
        max_layers=num_layers,
        min_tokens=4,
        max_tokens=max(8, args.max_gen_tokens),
        min_temperature=0.6,
        max_temperature=1.0,
        refine_min=1,
        refine_max=3,
    )

    # ---------------- Generation Demo ----------------
    print("\n=== Generation demo: low budget vs high budget ===")

    prompt = "The morning routine is"
    demo_time = time.time()

    for b in [0.2, 0.8]:
        print(f"\nBudget = {b}")

        text_skip, plan_skip = custom_generate_with_layer_skip(
            model=model,
            tmu=plugin.tmu,
            heads=heads,
            tokenizer=tokenizer,
            prompt=prompt,
            timestamp=demo_time,
            budget=b,
            pace_config=pace_config,
            max_new_override=args.max_gen_tokens,
            do_sample=False,
            seed=1,
        )

        print("[Layer-skip custom generate]")
        print("Output:", text_skip)
        print("Plan:", plan_skip)

        text_fast, plan_fast = generate_with_plugin_cache(
            plugin=plugin,
            heads=heads,
            tokenizer=tokenizer,
            prompt=prompt,
            timestamp=demo_time,
            budget=b,
            pace_config=pace_config,
            max_new_override=args.max_gen_tokens,
            do_sample=False,
            seed=1,
        )

        print("[Fast plugin cache generate]")
        print("Output:", text_fast)
        print("Plan:", plan_fast)

    # ---------------- Time Coherence Eval ----------------
    print("\n=== Time coherence evaluation ===")

    metrics = evaluate_time_coherence(
        plugin=plugin,
        heads=heads,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_samples=args.eval_samples,
        periods=(86400.0, 604800.0, 31536000.0),
        cross_period=43200.0,
        max_gen_tokens=min(8, args.max_gen_tokens),
        use_layer_skip=args.layer_skip_eval,
        pace_config=pace_config,
        budget=0.5,
        verbose=True,
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()