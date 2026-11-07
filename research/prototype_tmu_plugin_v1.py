import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TimeLike = Union[None, int, float, torch.Tensor, datetime, Sequence[Union[int, float]]]


def _to_time_tensor(time_stamp: TimeLike) -> torch.Tensor:
    """
    विभिन्न types के time input को 1D float64 tensor में convert करता है।
    Shape: (B,)
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

        for attr in ("hidden_size", "d_model", "embedding_size", "n_embd", "width"):
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

    def forward(self, x: Any = None, *args: Any, time_stamp: TimeLike = None, **kwargs: Any) -> Any:
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

        self._current_time_embed = self.tmu.encode_time(
            time_stamp=time_stamp,
            batch_size=batch_size,
            device=device,
        )

        try:
            if self.mode == "input" and isinstance(x, torch.Tensor):
                x = self.tmu.apply_to(x, self._current_time_embed)

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


# ---------------- Example / Test ----------------
if __name__ == "__main__":
    torch.manual_seed(0)

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
        def __init__(self, vocab_size: int = 100, d_model: int = 64, num_layers: int = 2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([TinyBlock(d_model) for _ in range(num_layers)])
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(input_ids)
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    d_model = 64
    base = TinyModel(vocab_size=100, d_model=d_model, num_layers=2)

    # TMU plugin: model के अंदर के blocks पर automatically hook लगेगा
    plugin = TMUPlugin(
        base_model=base,
        hidden_dim=d_model,
        mode="hooks",
        target_modules=["block"],  # TinyBlock / FluidLayer जैसे names के लिए
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
    )

    ids = torch.randint(0, 100, size=(2, 5))

    # पहला run: system time use करेगा
    out_a = plugin(ids, time_stamp=None)

    # Explicit time stamps से test
    out_1 = plugin(ids, time_stamp=1000.0)
    out_2 = plugin(ids, time_stamp=1001.0)

    print("Output shape:", out_a.shape)
    print(
        "Initial zero-initialized TMU की वजह से same input पर output almost same रहेगा:",
        torch.allclose(out_1, out_2, atol=1e-6),
    )

    # Demo के लिए थोड़ा सा manual time-effect on करो
    with torch.no_grad():
        plugin.tmu.scale_proj.weight.fill_(0.01)
        plugin.tmu.scale_proj.bias.fill_(0.0)

    out_3 = plugin(ids, time_stamp=1000.0)
    out_4 = plugin(ids, time_stamp=100000.0)

    print(
        "Time change करने पर अब output differ करता है:",
        not torch.allclose(out_3, out_4, atol=1e-6),
    )