# gpt2_tmu.py
import os
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------
# 1) TMU Core: Time Management Unit
# --------------------------------------------------

TimeLike = Union[None, int, float, torch.Tensor, datetime, Sequence[Union[int, float]]]


def _to_time_tensor(time_stamp: TimeLike) -> torch.Tensor:
    """
    विभिन्न time inputs को 1D float64 tensor में convert करता है।
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
    Time से 12 cyclic features बनाता है:
    second, minute, hour, day, week, month/year approx के sin/cos।

    Output shape: (B, 12)
    """
    t = _to_time_tensor(time_stamp).unsqueeze(-1)  # (B, 1)

    periods = (
        60.0,                 # second cycle
        3600.0,               # minute cycle
        86400.0,              # hour/day cycle
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


class TimeManagementUnit(nn.Module):
    """
    TMU core module।

    यह time features को hidden_dim में convert करता है
    और activations पर time-based modulation लगाता है।
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

        # Time coherence के लिए EMA buffer
        self.register_buffer("ema_time", torch.zeros(1, self.FEATURE_DIM), persistent=True)

        if self.hidden_dim is not None:
            self.build(self.hidden_dim)

    def build(self, hidden_dim: int) -> None:
        """
        TMU layers बनाता है।
        hidden_dim model की embedding/activation dimension होनी चाहिए।
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

        # Important:
        # शुरू में TMU का effect zero रखो ताकि original model behavior change न हो।
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

    def _match_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Time tensor का batch size model input के batch size से match करता है।
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
        Time stamp से time embedding बनाता है।

        Output shape: (B, hidden_dim)
        """
        if self.input_proj is None or self.temporal_mlp is None:
            raise RuntimeError("TMU build नहीं हुआ। पहले build(hidden_dim) call करें या TMUPlugin use करें।")

        if device is None:
            device = self.input_proj.weight.device

        if self.input_proj.weight.device != device:
            self.to(device)

        tf = cyclic_time_features(time_stamp).to(device=device, dtype=torch.float32)
        tf = self._match_batch(tf, int(batch_size))

        # EMA smoothing: time coherence बनाए रखने के लिए
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
        time_embed को x के shape के हिसाब से broadcast-ready बनाता है।

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
        TMU modulation apply करता है।

        अगर x की last dimension hidden_dim नहीं है, तो x unchanged रहेगा।
        इससे logits/output heads गलत तरीके से modify नहीं होंगे।
        """
        if not isinstance(x, torch.Tensor):
            return x

        if x.dim() < 2:
            return x

        if self.hidden_dim is None or x.shape[-1] != self.hidden_dim:
            return x

        if self.scale_proj is None:
            raise RuntimeError("TMU scale_proj build नहीं हुआ।")

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
        Direct TMU use के लिए convenience forward।
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("TimeManagementUnit.forward को torch.Tensor चाहिए।")

        te = self.encode_time(time_stamp, batch_size=x.shape[0], device=x.device)
        return self.apply_to(x, te)


def infer_hidden_dim(model: nn.Module) -> int:
    """
    किसी भी PyTorch model से hidden_dim auto-detect करने की कोशिश करता है।
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

    raise ValueError("Model से hidden_dim auto-detect नहीं हो पाया। TMUPlugin में hidden_dim manually pass करें।")


class TMUPlugin(nn.Module):
    """
    किसी भी PyTorch model को TMU के साथ wrap करता है।

    Modes:
      - "hooks": model के अंदर layers/blocks पर hook लगाता है
      - "input": input tensor पर TMU apply करता है
      - "off": TMU मौजूद है लेकिन effect नहीं डालता
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
        यह decide करता है कि किस module पर TMU hook लगेगा।
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
        Base model के target modules पर forward hooks लगाता है।
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

        # अगर कोई internal target नहीं मिला, तो root output पर hook लगाओ।
        if not matched:
            handle = self.base_model.register_forward_hook(self._hook_fn)
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """
        लगे हुए सभी TMU hooks हटाता है।
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _hook_fn(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        """
        Forward hook function।
        """
        return self._apply_to_output(output)

    def _apply_to_output(self, output: Any) -> Any:
        """
        विभिन्न output formats को handle करता है:
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
        batch_size और device जानने के लिए पहला tensor ढूंढता है।
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
        Wrapped model forward।

        Usage:
          out = plugin(input_ids, time_stamp=None)
          out = plugin(embeddings, time_stamp=123456.0)
        """
        sample = self._find_sample_tensor(x, args, kwargs)

        if sample is None:
            raise ValueError("TMUPlugin को batch_size/device जानने के लिए कम से कम एक torch.Tensor चाहिए।")

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
    Base model freeze करो, सिर्फ TMU train करो।
    यह 10-minute style quick training के लिए useful है।
    """
    for p in plugin.base_model.parameters():
        p.requires_grad_(False)

    params: List[nn.Parameter] = []
    for p in plugin.tmu.parameters():
        p.requires_grad_(True)
        params.append(p)

    return params


# --------------------------------------------------
# 2) Capability Heads: rhythm + pace
# --------------------------------------------------

TIME_FEATURE_DIM = getattr(TimeManagementUnit, "FEATURE_DIM", 12)


class TmuCapabilityHeads(nn.Module):
    """
    TMU के साथ training heads:
    1. rhythm_head  -> future/cyclic time pattern predict करना
    2. pace_head    -> reasoning speed / compute budget predict करना
    """

    def __init__(self, hidden_dim: int, budget_dim: int = 1):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.budget_dim = int(budget_dim)

        # Budget embedding: initially zero, ताकि शुरू में model behavior न बदले
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

        # Pace initially neutral ~0.5
        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
        """
        time_embed: (B, hidden_dim)
        budget:     (B,) या scalar, normalized 0..1
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


# --------------------------------------------------
# 3) GPT-2 Time Dataset
# --------------------------------------------------

class GPT2TimestampedDataset(Dataset):
    """
    GPT-2 के लिए time-stamped text dataset।

    हर sample में:
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
            raise ValueError("texts, timestamps, budgets की length same होनी चाहिए।")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.samples: List[Dict[str, torch.Tensor]] = []

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

            # Rhythm target: 10 minutes future के cyclic features
            y_rhythm = cyclic_time_features(
                torch.tensor([float(t) + 600.0], dtype=torch.float64)
            ).squeeze(0)

            # Pace target: text difficulty + time-of-day need, budget से capped
            valid_len = int(attention_mask.sum().item())
            difficulty = min(1.0, max(0.0, valid_len / float(max(1, self.seq_len))))

            day_phase = (float(t) % 86400.0) / 86400.0
            time_need = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * day_phase))

            desired_pace = 0.5 * difficulty + 0.5 * time_need
            y_pace_value = min(float(desired_pace), float(b))
            y_pace = torch.tensor(y_pace_value, dtype=torch.float32)

            self.samples.append(
                {
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


def build_demo_gpt2_dataset(
    tokenizer: Any,
    seq_len: int = 64,
    num_samples: int = 192,
    base_time: float = 1_700_000_000.0,
    seed: int = 42,
) -> GPT2TimestampedDataset:
    """
    Demo के लिए छोटा time-stamped text dataset बनाता है।
    अपने real data को यहां replace कर सकते हो।
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

    for i in range(int(num_samples)):
        text = base_texts[i % len(base_texts)] + f" Record {i}."

        if i == 0:
            t = float(base_time)
        else:
            delta = torch.rand(1, generator=g).item() * 3600.0 + 10.0
            t = timestamps[-1] + delta

        budget = torch.rand(1, generator=g).item() * 0.9 + 0.1

        texts.append(text)
        timestamps.append(t)
        budgets.append(budget)

    return GPT2TimestampedDataset(
        texts=texts,
        timestamps=timestamps,
        budgets=budgets,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )


# --------------------------------------------------
# 4) GPT-2 TMU Training
# --------------------------------------------------

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
    सिर्फ TMU + capability heads train करते हैं।
    GPT-2 base model freeze रहता है, इसलिए quick toy training possible है।
    """
    plugin.to(device)
    heads.to(device)

    # Training में cache बंद रखो
    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = list(tmu_params) + list(heads.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    history = []

    for epoch in range(int(epochs)):
        plugin.train()
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

            # एक बार time_embed बनाओ, फिर plugin और heads दोनों में use करो
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
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                loss_lm = outputs[0]
            else:
                raise RuntimeError("GPT-2 output से language modeling loss नहीं मिल पाया।")

            rhythm_pred, pace_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace_bce = F.binary_cross_entropy(pace_safe, y_pace)

            # Budget exceed penalty: pace > budget होने पर सजा
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


# --------------------------------------------------
# 5) Save / Load
# --------------------------------------------------

def save_gpt2_tmu(plugin: TMUPlugin, heads: TmuCapabilityHeads, path: str):
    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
        },
        path,
    )


def load_gpt2_tmu(plugin: TMUPlugin, heads: TmuCapabilityHeads, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    return plugin, heads


# --------------------------------------------------
# 6) Inference: pace/reasoning steps recommendation
# --------------------------------------------------

@torch.no_grad()
def recommend_reasoning_steps(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    sample_tensor: torch.Tensor,
    time_stamp: Union[torch.Tensor, float],
    budget: Union[torch.Tensor, float],
    min_steps: int = 1,
    max_steps: int = 8,
):
    """
    TMU + heads से recommend करता है कि कितने reasoning steps / compute use करने चाहिए।

    Actual generation में आप इसे ऐसे use कर सकते हो:
        max_new_tokens = recommended_steps * tokens_per_step
    या dynamic layer skipping में:
        active_layers = recommended_steps
    """
    plugin.eval()
    heads.eval()

    device = sample_tensor.device
    batch_size = int(sample_tensor.shape[0])

    time_embed = plugin.tmu.encode_time(
        time_stamp=time_stamp,
        batch_size=batch_size,
        device=device,
    )

    _, pace, _ = heads(time_embed, budget)

    min_steps = float(min_steps)
    max_steps = float(max_steps)

    steps_float = min_steps + pace * (max_steps - min_steps)
    steps = torch.round(steps_float).clamp(min=min_steps, max=max_steps).long()

    return pace.cpu(), steps.cpu()


@torch.no_grad()
def generate_with_tmu(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    tokenizer: Any,
    prompt: str,
    time_stamp: TimeLike = None,
    budget: float = 0.5,
    max_tokens_per_step: int = 8,
    min_steps: int = 1,
    max_steps: int = 8,
    temperature: float = 0.8,
    top_k: int = 50,
):
    """
    TMU के साथ GPT-2 generation।

    pace के हिसाब से max_new_tokens तय करता है:
      कम budget -> कम tokens
      ज्यादा budget -> ज्यादा tokens
    """
    plugin.eval()
    heads.eval()

    device = next(plugin.base_model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    B = int(input_ids.shape[0])
    input_length = int(input_ids.shape[1])

    time_embed = plugin.tmu.encode_time(
        time_stamp=time_stamp,
        batch_size=B,
        device=device,
    )

    budget_t = torch.full((B,), float(budget), device=device, dtype=torch.float32)

    _, pace, _ = heads(time_embed, budget_t)

    steps_float = float(min_steps) + pace * (float(max_steps) - float(min_steps))
    steps = torch.round(steps_float).clamp(min=float(min_steps), max=float(max_steps)).long()

    recommended_steps = int(steps[0].item())
    max_new_tokens = max(1, recommended_steps * int(max_tokens_per_step))

    old_current = plugin._current_time_embed
    old_use_cache = True
    if hasattr(plugin.base_model, "config"):
        old_use_cache = getattr(plugin.base_model.config, "use_cache", True)
        plugin.base_model.config.use_cache = True

    # Generation के दौरान हर internal forward में यही time_embed use होगा
    plugin._current_time_embed = time_embed

    try:
        output_ids = plugin.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        plugin._current_time_embed = old_current
        if hasattr(plugin.base_model, "config"):
            plugin.base_model.config.use_cache = old_use_cache

    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    return generated_text, float(pace[0].item()), recommended_steps, max_new_tokens


# --------------------------------------------------
# 7) Main Demo
# --------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # कम VRAM के लिए ये values घटा सकते हो:
    # SEQ_LEN = 32, BATCH_SIZE = 2, EPOCHS = 1, NUM_SAMPLES = 96
    MODEL_NAME = "gpt2"
    SEQ_LEN = 64
    BATCH_SIZE = 4
    EPOCHS = 2
    NUM_SAMPLES = 192
    LR = 1e-3

    print("Loading GPT-2 tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.use_cache = False

    hidden_dim = int(model.config.n_embd)
    print("GPT-2 hidden_dim:", hidden_dim)

    # GPT-2 के अंदर सिर्फ Transformer blocks पर TMU hook लगेगा
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

    print("Building demo time-stamped dataset...")
    dataset = build_demo_gpt2_dataset(
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        num_samples=NUM_SAMPLES,
        base_time=1_700_000_000.0,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    print("\n=== Training TMU capabilities on GPT-2 ===")
    history = train_gpt2_tmu(
        plugin=plugin,
        heads=heads,
        train_loader=loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        w_lm=1.0,
        w_rhythm=1.0,
        w_pace=0.7,
        budget_penalty=2.0,
        verbose=True,
    )

    save_path = "gpt2_tmu_capabilities.pt"
    save_gpt2_tmu(plugin, heads, save_path)
    print(f"\nSaved: {save_path}")

    print("\n=== Time coherence check ===")
    sample = dataset[0]
    x = sample["input_ids"].unsqueeze(0).to(device)
    am = sample["attention_mask"].unsqueeze(0).to(device)
    t1 = sample["t"].to(device)
    t2 = t1 + torch.tensor(86400.0, device=device)  # 24 hours बाद

    with torch.no_grad():
        plugin.eval()
        out1 = plugin(input_ids=x, attention_mask=am, time_stamp=t1, return_dict=True)
        out2 = plugin(input_ids=x, attention_mask=am, time_stamp=t2, return_dict=True)

        logits1 = out1.logits
        logits2 = out2.logits
        diff = (logits1 - logits2).abs().mean().item()

    print("Same text, 24-hour difference में average logit change:", diff)

    print("\n=== Generation with time budget ===")
    prompt = "The morning routine is"
    generated, pace, steps, max_new = generate_with_tmu(
        plugin=plugin,
        heads=heads,
        tokenizer=tokenizer,
        prompt=prompt,
        time_stamp=float(t1.item()),
        budget=0.35,
        max_tokens_per_step=8,
        min_steps=1,
        max_steps=8,
    )

    print("Prompt:", prompt)
    print("Generated:", generated)
    print("Pace:", pace)
    print("Recommended steps:", steps)
    print("Max new tokens:", max_new)