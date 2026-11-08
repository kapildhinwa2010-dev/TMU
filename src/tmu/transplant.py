from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import TimeManagementUnit
from .plugin import TMUPlugin
from .time_features import TimeLike


@dataclass
class TmuLayerConfig:
    """Portable config of a trained TMU layer."""

    hidden_dim: int
    feature_dim: int = TimeManagementUnit.FEATURE_DIM
    apply_mode: str = "scale"
    ema_momentum: float = 0.1
    dropout: float = 0.0
    use_ema: bool = True
    fusion_layers: int = 1
    fusion_expansion: int = 2
    fusion_dropout: float = 0.0
    format_version: int = 1

    @classmethod
    def from_tmu(cls, tmu: TimeManagementUnit) -> "TmuLayerConfig":
        return cls(
            hidden_dim=int(tmu.hidden_dim or 0),
            feature_dim=int(tmu.FEATURE_DIM),
            apply_mode=str(tmu.apply_mode),
            ema_momentum=float(tmu.ema_momentum),
            dropout=float(tmu.dropout),
            use_ema=bool(tmu.use_ema),
            fusion_layers=int(getattr(tmu, "fusion_layers", 0)),
            fusion_expansion=int(getattr(tmu, "fusion_expansion", 1)),
            fusion_dropout=float(getattr(tmu, "fusion_dropout", 0.0)),
        )

    def to_tmu(self) -> TimeManagementUnit:
        tmu = TimeManagementUnit(
            hidden_dim=self.hidden_dim,
            ema_momentum=self.ema_momentum,
            dropout=self.dropout,
            use_ema=self.use_ema,
            apply_mode=self.apply_mode,
            fusion_layers=self.fusion_layers,
            fusion_expansion=self.fusion_expansion,
            fusion_dropout=self.fusion_dropout,
        )
        return tmu


def export_tmu_layer(
    tmu: TimeManagementUnit,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Trained TMU layer ko ek portable file me export karta hai.

    File format (torch.save):
        {"config": {...}, "state_dict": {...}, "metadata": {...}}

    Isse layer ko kisi bhi plain GPT-2 par "transplant" kiya ja sakta hai
    bina original training setup ke.
    """

    if tmu.hidden_dim is None:
        raise ValueError("TMU build nahi hua; export se pehle hidden_dim set karo.")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    payload = {
        "config": asdict(TmuLayerConfig.from_tmu(tmu)),
        "state_dict": tmu.state_dict(),
        "metadata": dict(metadata or {}),
    }

    torch.save(payload, path)
    return path


def load_tmu_layer(
    path: str,
    map_location: Any = "cpu",
    device: Optional[torch.device] = None,
) -> TimeManagementUnit:
    """Exported TMU layer file ko wapas ``TimeManagementUnit`` me load karta hai."""

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path} ek valid TMU layer export nahi hai.")

    cfg = TmuLayerConfig(**payload["config"])
    tmu = cfg.to_tmu()
    tmu.load_state_dict(payload["state_dict"], strict=True)

    if device is not None:
        tmu.to(device)

    tmu.eval()
    return tmu


def load_tmu_layer_metadata(path: str) -> Dict[str, Any]:
    """Export file ka metadata sirf padhta hai (weights load kiye bina)."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("metadata", {}))
    meta["config"] = payload.get("config", {})
    return meta


def mount_tmu_on_gpt2(
    model: nn.Module,
    tmu: TimeManagementUnit,
    mode: str = "hooks",
    target_modules: Optional[Sequence[str]] = None,
) -> TMUPlugin:
    """Trained TMU layer ko plain GPT-2 (ya kisi bhi PyTorch model) par mount karta hai.

    Ye hi "TMU layer ko GPT-2 ke upar lana" hai: base model untouched rehta hai,
    sirf TMU plugin hooks ke through attach hota hai.
    """

    if target_modules is None:
        target_modules = ["GPT2Block", "block"]

    plugin = TMUPlugin(
        base_model=model,
        hidden_dim=int(tmu.hidden_dim or 0) or None,
        mode=mode,
        target_modules=list(target_modules),
        ema_momentum=float(tmu.ema_momentum),
        dropout=float(tmu.dropout),
        use_ema=bool(tmu.use_ema),
        apply_mode=str(tmu.apply_mode),
        fusion_layers=int(getattr(tmu, "fusion_layers", 0)),
        fusion_expansion=int(getattr(tmu, "fusion_expansion", 1)),
        fusion_dropout=float(getattr(tmu, "fusion_dropout", 0.0)),
    )

    # Mounted layer ka apna trained weight load karo (fusion layers included).
    plugin.tmu.load_state_dict(tmu.state_dict(), strict=True)

    return plugin


class GPT2WithTMU(nn.Module):
    """Plain GPT-2 + transplanted TMU layer ka ready-to-use wrapper.

    Wrapper base model ko bilkul freeze nahi karta (users decide karte hain),
    lekin default ``freeze_base=True`` PEFT-style behaviour deta hai: sirf TMU
    layer trainable rehta hai.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tmu: TimeManagementUnit,
        mode: str = "hooks",
        target_modules: Optional[Sequence[str]] = None,
        freeze_base: bool = True,
    ):
        super().__init__()

        self.plugin = mount_tmu_on_gpt2(
            model=base_model,
            tmu=tmu,
            mode=mode,
            target_modules=target_modules,
        )

        if freeze_base:
            for p in self.plugin.base_model.parameters():
                p.requires_grad_(False)

        for p in self.plugin.tmu.parameters():
            p.requires_grad_(True)

    @property
    def base_model(self) -> nn.Module:
        return self.plugin.base_model

    @property
    def tmu(self) -> TimeManagementUnit:
        return self.plugin.tmu

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        time_stamp: TimeLike = None,
        **kwargs: Any,
    ):
        return self.plugin(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_stamp=time_stamp,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        time_stamp: TimeLike = None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Time-aware greedy/sampling decoding (TMU embed har step par refresh hota hai)."""

        self.eval()

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        out = self.plugin(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_stamp=time_stamp,
            use_cache=True,
            return_dict=True,
        )

        logits = out.logits[:, -1, :]
        past = getattr(out, "past_key_values", None)

        cur_mask = attention_mask
        generated: List[torch.Tensor] = []

        for _ in range(int(max_new_tokens)):
            if do_sample:
                scaled = logits / max(1e-6, float(temperature))

                if top_k and top_k > 0:
                    k = min(int(top_k), scaled.shape[-1])
                    threshold = torch.topk(scaled, k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            generated.append(next_id)

            cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)

            out = self.plugin(
                input_ids=next_id,
                attention_mask=cur_mask,
                time_stamp=time_stamp,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )

            logits = out.logits[:, -1, :]
            past = getattr(out, "past_key_values", None)

        new_tokens = torch.cat(generated, dim=1) if generated else torch.zeros(
            (input_ids.shape[0], 0), dtype=input_ids.dtype, device=device
        )

        return torch.cat([input_ids, new_tokens], dim=1)


def _lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    time_stamp: Any = None,
    use_tmu: bool = False,
) -> torch.Tensor:
    if use_tmu:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_stamp=time_stamp,
            use_cache=False,
            return_dict=True,
        )
    else:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = out.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_transplant(
    base_model: nn.Module,
    plugin: TMUPlugin,
    samples: Sequence[Dict[str, Any]],
    device: str = "cpu",
    time_shift_seconds: float = 43200.0,
    max_samples: int = 64,
) -> Dict[str, Any]:
    """Baseline GPT-2 vs transplanted TMU layer ka comparison metrics.

    Metrics:
        ``baseline_loss`` / ``baseline_perplexity``
        ``transplant_loss`` / ``transplant_perplexity``
        ``logit_shift_mad``  - 12h time shift par logits ka mean abs change
                               (0 hoga toh model time use hi nahi kar raha)
        ``time_sensitive``   - ``logit_shift_mad > 0`` ya nahi
    """

    base_model.eval()
    plugin.eval()

    n = min(int(max_samples), len(samples))
    if n <= 0:
        raise ValueError("Evaluate karne ke liye kam se kam ek sample chahiye.")

    base_losses: List[float] = []
    tmu_losses: List[float] = []
    shift_mads: List[float] = []

    for i in range(n):
        s = samples[i]

        input_ids = s["input_ids"].unsqueeze(0).to(device)
        attention_mask = s.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        attention_mask = attention_mask.unsqueeze(0).to(device)
        labels = s["labels"].unsqueeze(0).to(device)

        t = s.get("t")
        t = t.to(device) if torch.is_tensor(t) else t

        base_losses.append(
            float(
                _lm_loss(
                    base_model,
                    input_ids,
                    attention_mask,
                    labels,
                    time_stamp=None,
                    use_tmu=False,
                ).item()
            )
        )

        tmu_losses.append(
            float(
                _lm_loss(
                    plugin,
                    input_ids,
                    attention_mask,
                    labels,
                    time_stamp=t,
                    use_tmu=True,
                ).item()
            )
        )

        out_t = plugin(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_stamp=t,
            use_cache=False,
            return_dict=True,
        )

        shifted = t + float(time_shift_seconds) if torch.is_tensor(t) else t

        out_s = plugin(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_stamp=shifted,
            use_cache=False,
            return_dict=True,
        )

        mask = attention_mask.bool()
        a = out_t.logits[mask]
        b = out_s.logits[mask]

        if a.numel() > 0:
            shift_mads.append(float((a - b).abs().mean().item()))

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    base_loss = _avg(base_losses)
    tmu_loss = _avg(tmu_losses)
    shift_mad = _avg(shift_mads)

    tmu_params = sum(p.numel() for p in plugin.tmu.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in plugin.tmu.parameters())

    return {
        "num_samples": n,
        "baseline_loss": base_loss,
        "baseline_perplexity": float(torch.exp(torch.tensor(base_loss)).item()),
        "transplant_loss": tmu_loss,
        "transplant_perplexity": float(torch.exp(torch.tensor(tmu_loss)).item()),
        "loss_delta": tmu_loss - base_loss,
        "time_shift_seconds": float(time_shift_seconds),
        "logit_shift_mad": shift_mad,
        "time_sensitive": bool(shift_mad > 0.0),
        "tmu_trainable_params": int(tmu_params),
        "tmu_total_params": int(total_params),
    }


def write_transplant_report(
    metrics: Dict[str, Any],
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Metrics ko JSON me likhta hai (results/ folder ke liye)."""

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    payload = dict(metrics)
    if extra:
        payload.update(extra)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def format_transplant_report(metrics: Dict[str, Any]) -> str:
    """Human-readable summary (README/results blocks ke liye)."""

    lines = [
        "| Metric | Baseline GPT-2 | GPT-2 + TMU layer |",
        "|---|---|---|",
        f"| Eval loss | {metrics.get('baseline_loss', 0.0):.4f} | {metrics.get('transplant_loss', 0.0):.4f} |",
        f"| Eval perplexity | {metrics.get('baseline_perplexity', 0.0):.2f} | {metrics.get('transplant_perplexity', 0.0):.2f} |",
        "",
        f"- Samples: {metrics.get('num_samples', 0)}",
        f"- Logit change on {metrics.get('time_shift_seconds', 0.0):.0f}s time shift: {metrics.get('logit_shift_mad', 0.0):.6f}"
        f" ({'time-sensitive' if metrics.get('time_sensitive') else 'NOT time-sensitive'})",
        f"- TMU trainable params: {metrics.get('tmu_trainable_params', 0):,}",
    ]

    return "\n".join(lines)
