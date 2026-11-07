from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .core import TimeManagementUnit
from .time_features import TimeLike


def infer_hidden_dim(model: nn.Module) -> int:
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

    raise ValueError("Model se hidden_dim auto-detect nahi ho paya.")


class TMUPlugin(nn.Module):
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
        self.remove_hooks()
        matched = False

        for name, module in self.base_model.named_modules():
            if name == "":
                continue

            if self._is_target(name, module):
                handle = module.register_forward_hook(self._hook_fn)
                self._hooks.append(handle)
                matched = True

        if not matched:
            handle = self.base_model.register_forward_hook(self._hook_fn)
            self._hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _hook_fn(self, module: nn.Module, inputs: Any, output: Any) -> Any:
        return self._apply_to_output(output)

    def _apply_to_output(self, output: Any) -> Any:
        if self._current_time_embed is None:
            return output

        if isinstance(output, torch.Tensor):
            return self.tmu.apply_to(output, self._current_time_embed)

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

    def _find_sample_tensor(
        self,
        x: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
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
        sample = self._find_sample_tensor(x, args, kwargs)

        if sample is None:
            raise ValueError("TMUPlugin ko batch_size/device ke liye kam se kam ek tensor chahiye.")

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
    for p in plugin.base_model.parameters():
        p.requires_grad_(False)

    params: List[nn.Parameter] = []
    for p in plugin.tmu.parameters():
        p.requires_grad_(True)
        params.append(p)

    return params