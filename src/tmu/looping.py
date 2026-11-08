# src/tmu/looping.py
#
# DLR-IA: Dynamic Layer Refinement with Input Anchoring
#
# This module is add-only. It does not require changing existing TMU files.
#
# Features:
# - Input anchor from original prompt
# - Per-layer loop execution
# - Final refinement passes
# - Loop fraction head
# - Loop-aware dynamic generation
# - Loop-aware training
# - Save/load loop refiner

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .adapter import AdvancedInternalAdapter
from .clock import (
    AdvancedClockSnapshot,
    AdvancedInternalClock,
    advanced_snapshot_to_tensor,
)
from .core import TimeManagementUnit, load_tmu_state_dict
from .data import estimate_text_difficulty
from .fusion import encode_fused_time_embed
from .heads import AdvancedTmuCapabilityHeads
from .modeling import PerLayerTimer, _make_causal_padding_mask
from .pacing import AdvancedPaceConfig, AdvancedPaceController
from .plugin import TMUPlugin, freeze_base_and_enable_tmu
from .system import SystemMonitor
from .time_features import TimeLike
from .traffic import TrafficMonitor
from .utils import clamp01, prepare_budget_tensor

# ============================================================
# Loop Config
# ============================================================

@dataclass
class LoopConfig:
    enabled: bool = True

    # Per layer maximum extra loops
    max_loops: int = 3
    min_loops: int = 0

    # Last layer extra refinement passes
    max_final_refine_passes: int = 2

    # Gate initialization.
    # 0 means initially loop/anchor has no effect, so base model stays safe.
    anchor_gate_init: float = 0.0
    loop_gate_init: float = 0.0

    # Hard prompt ko extra loops dene ka boost
    difficulty_boost: float = 1.5

    # Traffic/thermal load badhne par loops ghatao
    load_penalty: float = 0.75

    # Late layers ko thoda zyada refinement bias
    late_layer_bias: float = 0.5


# ============================================================
# Input Anchor
# ============================================================

class InputAnchor(nn.Module):
    """
    Original prompt/input ko summarize karke ek anchor vector banata hai.

    Purpose:
        Loop me model original task/input na bhule.

    Input:
        prompt_hidden: (B, T, D) or (B, D)
        prompt_mask:   (B, T), optional

    Output:
        anchor: (B, D)
    """

    def __init__(self, hidden_dim: int, gate_init: float = 0.0):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Per-channel gate.
        # Zero init se shuru me anchor effect zero rahega.
        self.gate = nn.Parameter(torch.full((self.hidden_dim,), float(gate_init)))

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prompt_hidden.dim() == 2:
            pooled = prompt_hidden
        else:
            if prompt_mask is None:
                pooled = prompt_hidden.mean(dim=1)
            else:
                m = prompt_mask.to(dtype=prompt_hidden.dtype).unsqueeze(-1)  # (B, T, 1)
                denom = m.sum(dim=1).clamp_min(1.0)
                pooled = (prompt_hidden * m).sum(dim=1) / denom

        anchored = self.proj(self.norm(pooled))

        # tanh gate keeps injection bounded
        return torch.tanh(self.gate) * anchored


# ============================================================
# Block Loop Params
# ============================================================

class BlockLoopParams(nn.Module):
    """
    Ek transformer block ko loop chalane ke liye extra params.

    Important:
        Ye block ko own nahi karta. Block forward me pass hota hai.
        Isse base model parameters duplicate nahi hote.
    """

    def __init__(self, hidden_dim: int, config: Optional[LoopConfig] = None):
        super().__init__()

        self.config = config or LoopConfig()
        self.hidden_dim = int(hidden_dim)

        self.pre_norm = nn.LayerNorm(self.hidden_dim)
        self.post_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_norm = nn.LayerNorm(self.hidden_dim)

        # Loop contribution gate.
        # Zero init: shuru me loop no-op.
        self.loop_gate = nn.Parameter(
            torch.full((self.hidden_dim,), float(self.config.loop_gate_init))
        )

    def forward(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor: torch.Tensor,
        loops: int,
    ) -> torch.Tensor:
        """
        hidden: (B, T, D)
        attention_mask: 4D mask compatible with block
        anchor: (B, D)
        loops: int
        """
        loops = int(max(0, min(int(loops), int(self.config.max_loops))))

        if (not self.config.enabled) or loops <= 0:
            return hidden

        h = hidden

        # anchor (B, D) -> (B, 1, D) for broadcasting over tokens
        if anchor.dim() == 2:
            a = anchor.unsqueeze(1)
        else:
            a = anchor

        for _ in range(loops):
            # Input anchoring: original prompt signal wapas inject karo
            h_in = self.pre_norm(h + a)

            # Run base block
            try:
                out = block(
                    hidden_states=h_in,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except TypeError:
                out = block(
                    h_in,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            new_h = out[0] if isinstance(out, (tuple, list)) else out

            # Residual delta, normalized for stability
            delta = self.delta_norm(new_h - h_in)

            # Gated residual update
            h = h + torch.tanh(self.loop_gate) * delta

        return self.post_norm(h)


# ============================================================
# Layer Refiner Collection
# ============================================================

class LayerRefinerCollection(nn.Module):
    """
    Sab layers ke liye loop params + shared input anchor.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        config: Optional[LoopConfig] = None,
    ):
        super().__init__()

        self.config = config or LoopConfig()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

        self.anchor = InputAnchor(
            hidden_dim=self.hidden_dim,
            gate_init=self.config.anchor_gate_init,
        )

        self.block_params = nn.ModuleList(
            [
                BlockLoopParams(self.hidden_dim, self.config)
                for _ in range(self.num_layers)
            ]
        )

    def prepare_anchor(
        self,
        prompt_hidden: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.anchor(prompt_hidden, prompt_mask)

    def forward_block(
        self,
        layer_idx: int,
        block: nn.Module,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor: torch.Tensor,
        loops: int,
    ) -> torch.Tensor:
        layer_idx = max(0, min(int(layer_idx), self.num_layers - 1))
        return self.block_params[layer_idx](
            block=block,
            hidden=hidden,
            attention_mask=attention_mask,
            anchor=anchor,
            loops=loops,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Sirf loop/anchor ke naye parameters.
        Base GPT-2 block parameters include nahi hote.
        """
        params: List[nn.Parameter] = []
        params.extend(list(self.anchor.parameters()))
        for module in self.block_params:
            params.extend(list(module.parameters()))
        return params


# ============================================================
# Loop Plan
# ============================================================

def make_loop_plan(
    num_layers: int,
    loop_frac: float,
    difficulty: float,
    budget: float,
    traffic_load: float = 0.0,
    thermal_load: float = 0.0,
    config: Optional[LoopConfig] = None,
) -> Dict[str, Any]:
    """
    Predicted loop fraction + prompt difficulty + system load se
    per-layer loop schedule banata hai.
    """
    cfg = config or LoopConfig()

    loop_frac = clamp01(loop_frac)
    difficulty = clamp01(difficulty)
    budget = clamp01(budget)
    traffic_load = clamp01(traffic_load)
    thermal_load = clamp01(thermal_load)

    load = clamp01(0.60 * traffic_load + 0.40 * thermal_load)

    # Base loops from predicted loop fraction
    base = float(cfg.min_loops) + loop_frac * float(cfg.max_loops - cfg.min_loops)

    # Hard prompt ko extra loops
    boost = float(cfg.difficulty_boost) * max(0.0, difficulty - 0.35) * budget

    raw = base + boost

    # Load badhne par loops ghatao
    factor = max(0.20, 1.0 - float(cfg.load_penalty) * load)

    scalar = int(round(raw * factor))
    scalar = max(int(cfg.min_loops), min(int(cfg.max_loops), scalar))

    layer_loops: List[int] = []

    for i in range(int(num_layers)):
        if scalar <= 0:
            layer_loops.append(0)
            continue

        pos = float(i + 1) / float(max(1, int(num_layers)))

        # Late layers ko thoda zyada refinement mil sakta hai
        bias = 1.0 + float(cfg.late_layer_bias) * (pos - 0.5)

        l = int(round(float(scalar) * float(bias) * float(factor)))
        l = max(int(cfg.min_loops), min(int(cfg.max_loops), l))

        layer_loops.append(l)

    final_refine_passes = int(
        round(
            clamp01(loop_frac * difficulty)
            * float(cfg.max_final_refine_passes)
            * float(factor)
        )
    )

    if scalar <= 0:
        final_refine_passes = 0

    return {
        "loop_frac": loop_frac,
        "difficulty": difficulty,
        "budget": budget,
        "traffic_load": traffic_load,
        "thermal_load": thermal_load,
        "load": load,
        "loop_scalar": scalar,
        "layer_loops": layer_loops,
        "final_refine_passes": final_refine_passes,
    }


def apply_final_refine(
    hidden: torch.Tensor,
    refiner: LayerRefinerCollection,
    blocks: Sequence[nn.Module],
    attention_mask: torch.Tensor,
    anchor: torch.Tensor,
    passes: int,
    last_layer_idx: int,
) -> torch.Tensor:
    """
    Last active layer ko extra passes se output refine karta hai.
    """
    passes = int(max(0, passes))

    if passes <= 0 or refiner is None or anchor is None:
        return hidden

    if len(blocks) == 0:
        return hidden

    idx = max(0, min(int(last_layer_idx), len(blocks) - 1, refiner.num_layers - 1))

    h = hidden

    for _ in range(passes):
        h = refiner.forward_block(
            layer_idx=idx,
            block=blocks[idx],
            hidden=h,
            attention_mask=attention_mask,
            anchor=anchor,
            loops=1,
        )

    return h


# ============================================================
# Advanced Loop Heads
# ============================================================

class AdvancedLoopHeads(AdvancedTmuCapabilityHeads):
    """
    Existing advanced heads + loop fraction head.

    Returns:
        rhythm_pred, pace, layer_frac, token_frac, loop_frac, pace_logit
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 12,
        max_tokens: int = 64,
        budget_dim: int = 1,
        loop_bias_init: float = 0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_tokens=max_tokens,
            budget_dim=budget_dim,
        )

        pace_hidden = max(8, int(hidden_dim) // 2)

        self.loop_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        nn.init.zeros_(self.loop_head[-1].weight)
        nn.init.constant_(self.loop_head[-1].bias, float(loop_bias_init))

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
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
        layer_logit = self.layer_head(conditioned).squeeze(-1)
        token_logit = self.token_head(conditioned).squeeze(-1)
        loop_logit = self.loop_head(conditioned).squeeze(-1)

        pace = torch.sigmoid(pace_logit)
        layer_frac = torch.sigmoid(layer_logit)
        token_frac = torch.sigmoid(token_logit)
        loop_frac = torch.sigmoid(loop_logit)

        return rhythm_pred, pace, layer_frac, token_frac, loop_frac, pace_logit


# ============================================================
# Loop Label Dataset Wrapper
# ============================================================

class LoopLabelDatasetWrapper(Dataset):
    """
    Existing AdvancedTimestampedDataset ke samples me y_loops add karta hai.

    Isse purane dataset class ko change nahi karna padega.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        metas: Optional[Sequence[Dict[str, Any]]] = None,
        max_loops: int = 3,
    ):
        super().__init__()

        self.base_dataset = base_dataset
        self.metas = list(metas) if metas is not None else [{} for _ in range(len(base_dataset))]
        self.max_loops = int(max(1, max_loops))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = dict(self.base_dataset[idx])

        meta = {}
        if idx < len(self.metas):
            meta = self.metas[idx] or {}

        desired_loops = meta.get("desired_loops", None)

        try:
            difficulty = float(item.get("difficulty", 0.0))
        except Exception:
            difficulty = 0.0

        try:
            budget = float(item.get("budget", 0.5))
        except Exception:
            budget = 0.5

        if desired_loops is not None:
            try:
                y_loops = clamp01(float(desired_loops) / float(self.max_loops))
            except Exception:
                y_loops = clamp01(difficulty * budget)
        else:
            y_loops = clamp01(difficulty * budget)

        item["y_loops"] = torch.tensor(y_loops, dtype=torch.float32)
        return item


# ============================================================
# Predict loop-aware time embed
# ============================================================

def predict_advanced_loop_embed(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeads,
    adapter: AdvancedInternalAdapter,
    timestamp: TimeLike,
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
    internal_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        pace, layer_frac, token_frac, loop_frac, time_embed, rhythm_pred
    """
    time_embed = encode_fused_time_embed(
        tmu=plugin.tmu,
        time_stamp=timestamp,
        internal_features=internal_features,
        adapter=adapter,
        batch_size=batch_size,
        device=device,
    )

    budget_t = prepare_budget_tensor(budget, batch_size, device)

    rhythm_pred, pace, layer_frac, token_frac, loop_frac, _ = heads(time_embed, budget_t)

    return pace, layer_frac, token_frac, loop_frac, time_embed, rhythm_pred


# ============================================================
# Custom GPT-2 forward with loops + input anchor
# ============================================================

def custom_gpt2_forward_looped(
    model: nn.Module,
    tmu: TimeManagementUnit,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    time_embed: torch.Tensor,
    active_layers: Optional[int] = None,
    timer: Optional[PerLayerTimer] = None,
    modulate_blocks: bool = True,
    layer_refiner: Optional[LayerRefinerCollection] = None,
    prompt_anchor: Optional[torch.Tensor] = None,
    layer_loops: Optional[List[int]] = None,
    final_refine_passes: int = 0,
) -> Tuple[torch.Tensor, Optional[List[float]]]:
    """
    No-cache GPT-2 forward with:
    - TMU modulation
    - dynamic active layers
    - per-layer loops
    - input anchor injection
    - final refinement
    - optional per-layer timing
    """
    transformer = model.transformer
    B, T = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    base_hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)

    hidden = base_hidden

    if hasattr(transformer, "drop"):
        hidden = transformer.drop(hidden)

    hidden = tmu.apply_to(hidden, time_embed)

    attn_mask = _make_causal_padding_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=hidden.dtype,
        device=device,
    )

    # If refiner is enabled but anchor not provided, build anchor from original prompt embeddings.
    if layer_refiner is not None and prompt_anchor is None:
        prompt_anchor = layer_refiner.prepare_anchor(base_hidden, attention_mask)

    num_layers = len(transformer.h)

    if active_layers is None:
        active_layers = num_layers

    active_layers = max(1, min(int(active_layers), num_layers))

    if timer is not None:
        timer.reset()

    blocks = list(transformer.h)

    for i, block in enumerate(blocks):
        if i >= active_layers:
            continue

        loops = 0
        if layer_loops is not None and i < len(layer_loops):
            loops = int(layer_loops[i])

        if timer is not None:
            timer.begin(i)

        if (
            layer_refiner is not None
            and prompt_anchor is not None
            and loops > 0
        ):
            hidden = layer_refiner.forward_block(
                layer_idx=i,
                block=block,
                hidden=hidden,
                attention_mask=attn_mask,
                anchor=prompt_anchor,
                loops=loops,
            )
        else:
            outputs = block(
                hidden,
                attention_mask=attn_mask,
                use_cache=False,
            )
            hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if timer is not None:
            timer.end(i)

        if modulate_blocks:
            hidden = tmu.apply_to(hidden, time_embed)

    # Final refinement: last active layer ko extra passes
    if (
        layer_refiner is not None
        and prompt_anchor is not None
        and int(final_refine_passes) > 0
    ):
        hidden = apply_final_refine(
            hidden=hidden,
            refiner=layer_refiner,
            blocks=blocks,
            attention_mask=attn_mask,
            anchor=prompt_anchor,
            passes=int(final_refine_passes),
            last_layer_idx=active_layers - 1,
        )

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    layer_times = timer.collect() if timer is not None else None

    return logits, layer_times


# ============================================================
# Training with loops
# ============================================================

def train_looped_tmu(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeads,
    adapter: AdvancedInternalAdapter,
    refiner: LayerRefinerCollection,
    train_loader: DataLoader,
    device: str = "cuda",
    clock: Optional[AdvancedInternalClock] = None,
    epochs: int = 2,
    lr: float = 1e-3,
    w_lm: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    w_layer: float = 0.8,
    w_token: float = 0.6,
    w_loop: float = 0.8,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    use_gpu_events: bool = False,
    budget_s: Optional[float] = None,
    verbose: bool = True,
):
    """
    Trains:
    - TMU
    - Advanced loop heads
    - Internal adapter
    - Layer refiner / input anchor

    Base GPT-2 remains frozen.
    """
    plugin.to(device)
    heads.to(device)
    adapter.to(device)
    refiner.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    if clock is None:
        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=budget_s,
            use_gpu_events=use_gpu_events,
            system_monitor=SystemMonitor(device, enable=str(device).startswith("cuda")),
            traffic_monitor=TrafficMonitor(),
        )

    tmu_params = freeze_base_and_enable_tmu(plugin)

    params = (
        list(tmu_params)
        + list(heads.parameters())
        + list(adapter.parameters())
        + refiner.trainable_parameters()
    )

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    history = []

    num_layers = len(plugin.base_model.transformer.h)

    for epoch in range(int(epochs)):
        plugin.base_model.eval()
        plugin.tmu.train()
        heads.train()
        adapter.train()
        refiner.train()

        totals = {
            "loss": 0.0,
            "lm": 0.0,
            "rhythm": 0.0,
            "pace": 0.0,
            "layer": 0.0,
            "token": 0.0,
            "loop": 0.0,
            "over": 0.0,
            "delta_us": 0.0,
            "loop_scalar": 0.0,
            "refine_passes": 0.0,
        }
        n = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)

            y_rhythm = batch["y_rhythm"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)
            y_layers = batch["y_layers"].to(device, non_blocking=True)
            y_tokens = batch["y_tokens"].to(device, non_blocking=True)
            y_loops = batch["y_loops"].to(device, non_blocking=True)
            difficulty = batch["difficulty"].to(device, non_blocking=True)

            B = int(input_ids.shape[0])

            snapshot = clock.tick(
                tokens=int(input_ids.numel()),
                prompt_difficulty=float(difficulty.mean().item()),
                confidence=0.5,
                per_layer_times=None,
                sync=bool(use_gpu_events),
            )

            internal_features = advanced_snapshot_to_tensor(snapshot, B, device)

            time_embed = encode_fused_time_embed(
                tmu=plugin.tmu,
                time_stamp=t,
                internal_features=internal_features,
                adapter=adapter,
                batch_size=B,
                device=device,
            )

            rhythm_pred, pace_pred, layer_pred, token_pred, loop_pred, _ = heads(
                time_embed,
                budget,
            )

            loop_plan = make_loop_plan(
                num_layers=num_layers,
                loop_frac=float(loop_pred.mean().item()),
                difficulty=float(difficulty.mean().item()),
                budget=float(budget.mean().item()),
                traffic_load=float(snapshot.traffic.get("queue_delay_norm", 0.0)),
                thermal_load=float(snapshot.system.get("temp_norm", 0.0)),
                config=refiner.config,
            )

            logits, _ = custom_gpt2_forward_looped(
                model=plugin.base_model,
                tmu=plugin.tmu,
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_embed=time_embed,
                active_layers=None,
                timer=None,
                modulate_blocks=True,
                layer_refiner=refiner,
                prompt_anchor=None,
                layer_loops=loop_plan["layer_loops"],
                final_refine_passes=loop_plan["final_refine_passes"],
            )

            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_labels = shift_labels.view(-1)
            valid = flat_labels != -100

            if bool(valid.any()):
                loss_lm = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1))[valid],
                    flat_labels[valid],
                    ignore_index=-100,
                )
            else:
                loss_lm = torch.zeros((), device=device, dtype=torch.float32)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace = F.binary_cross_entropy(pace_safe, y_pace)

            layer_safe = layer_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_layer = F.binary_cross_entropy(layer_safe, y_layers)

            token_safe = token_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_token = F.binary_cross_entropy(token_safe, y_tokens)

            loop_safe = loop_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_loop = F.binary_cross_entropy(loop_safe, y_loops)

            over_budget = F.relu(pace_pred - budget).mean()
            pace_loss_total = loss_pace + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * pace_loss_total
                + float(w_layer) * loss_layer
                + float(w_token) * loss_token
                + float(w_loop) * loss_loop
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["lm"] += float(loss_lm.item())
            totals["rhythm"] += float(loss_rhythm.item())
            totals["pace"] += float(loss_pace.item())
            totals["layer"] += float(loss_layer.item())
            totals["token"] += float(loss_token.item())
            totals["loop"] += float(loss_loop.item())
            totals["over"] += float(over_budget.item())
            totals["delta_us"] += float(snapshot.delta_us)
            totals["loop_scalar"] += float(loop_plan["loop_scalar"])
            totals["refine_passes"] += float(loop_plan["final_refine_passes"])
            n += 1

        avg = {k: v / max(1, n) for k, v in totals.items()}
        history.append({"epoch": epoch + 1, **avg})

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg['loss']:.4f} | "
                f"lm={avg['lm']:.4f} | "
                f"rhythm={avg['rhythm']:.4f} | "
                f"pace={avg['pace']:.4f} | "
                f"layer={avg['layer']:.4f} | "
                f"token={avg['token']:.4f} | "
                f"loop={avg['loop']:.4f} | "
                f"over={avg['over']:.4f} | "
                f"loop_scalar={avg['loop_scalar']:.2f} | "
                f"refine={avg['refine_passes']:.2f} | "
                f"delta_us={avg['delta_us']:.2f}"
            )

    return history


# ============================================================
# Dynamic generation with loops
# ============================================================

@torch.no_grad()
def generate_advanced_dynamic_loops(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeads,
    adapter: AdvancedInternalAdapter,
    layer_refiner: Optional[LayerRefinerCollection],
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    budget_s: Optional[float] = None,
    max_new_tokens: int = 32,
    pace_config: Optional[AdvancedPaceConfig] = None,
    loop_config: Optional[LoopConfig] = None,
    use_gpu_events: bool = False,
    enable_per_layer_timing: bool = False,
    enable_loops: bool = True,
    system_monitor: Optional[SystemMonitor] = None,
    traffic_monitor: Optional[TrafficMonitor] = None,
    cuda_cycle_timer: Optional[Any] = None,
    do_sample: bool = False,
    top_k: int = 50,
    seed: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Optional[AdvancedClockSnapshot], Dict[str, Any]]:
    """
    True dynamic generation with:
    - internal clock update per token
    - traffic/system aware pacing
    - dynamic active layers
    - per-layer loops
    - input anchoring
    - final refinement
    - confidence early stop

    Note:
        This uses no-cache custom forward, so it is slower but research-friendly.
    """
    model = plugin.base_model
    model.eval()
    heads.eval()
    adapter.eval()

    if layer_refiner is not None:
        layer_refiner.eval()

    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(int(seed))

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = int(input_ids.shape[1])

    difficulty = estimate_text_difficulty(prompt)

    if traffic_monitor is not None and request_id is not None:
        traffic_monitor.request_start(
            request_id=request_id,
            prompt_tokens=prompt_len,
            priority=budget,
            deadline_s=budget_s,
        )

    clock = AdvancedInternalClock(
        device=device,
        max_steps=int(max_new_tokens),
        budget_s=budget_s,
        use_gpu_events=use_gpu_events,
        system_monitor=system_monitor,
        traffic_monitor=traffic_monitor,
    )

    num_layers = len(model.transformer.h)
    hidden_dim = int(model.config.n_embd)

    controller = AdvancedPaceController(num_layers, pace_config)
    cfg = controller.config

    if enable_loops and layer_refiner is None:
        loop_cfg = loop_config or LoopConfig(enabled=True)
        layer_refiner = LayerRefinerCollection(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            config=loop_cfg,
        ).to(device)

    timer = PerLayerTimer(
        num_layers=num_layers,
        enabled=enable_per_layer_timing,
        device=device,
    )

    # Build original prompt anchor once.
    prompt_anchor = None
    if enable_loops and layer_refiner is not None:
        transformer = model.transformer
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        base_hidden = transformer.wte(input_ids) + transformer.wpe(pos)
        prompt_anchor = layer_refiner.prepare_anchor(base_hidden, attention_mask)

    generated = input_ids.clone()
    steps_done = 0
    target_tokens = int(max_new_tokens)

    prev_layer_times: Optional[List[float]] = None
    confidence = 0.5
    last_plan: Dict[str, Any] = {}
    last_loop_plan: Dict[str, Any] = {}
    last_snapshot: Optional[AdvancedClockSnapshot] = None
    cycle_info: Optional[Dict[str, Any]] = None

    if int(max_new_tokens) <= 0:
        snapshot = clock.tick(
            tokens=prompt_len,
            prompt_difficulty=difficulty,
            confidence=confidence,
            per_layer_times=None,
            sync=bool(use_gpu_events),
        )
        return "", {}, snapshot, {}

    while steps_done < int(max_new_tokens):
        tokens_for_tick = prompt_len if steps_done == 0 else 1

        snapshot = clock.tick(
            tokens=tokens_for_tick,
            prompt_difficulty=difficulty,
            confidence=confidence,
            per_layer_times=prev_layer_times,
            sync=bool(use_gpu_events),
        )
        last_snapshot = snapshot

        internal_features = advanced_snapshot_to_tensor(snapshot, 1, device)

        pace, layer_frac, token_frac, loop_frac, time_embed, _ = predict_advanced_loop_embed(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            timestamp=timestamp,
            budget=budget,
            batch_size=1,
            device=device,
            internal_features=internal_features,
        )

        plan = controller.from_outputs(
            pace=float(pace[0].item()),
            layer_frac=float(layer_frac[0].item()),
            token_frac=float(token_frac[0].item()),
            budget=float(budget),
            traffic=snapshot.traffic,
            system=snapshot.system,
            difficulty=difficulty,
        )
        last_plan = plan

        loop_plan = make_loop_plan(
            num_layers=num_layers,
            loop_frac=float(loop_frac[0].item()),
            difficulty=difficulty,
            budget=float(budget),
            traffic_load=float(snapshot.traffic.get("queue_delay_norm", 0.0)),
            thermal_load=float(snapshot.system.get("temp_norm", 0.0)),
            config=layer_refiner.config if layer_refiner is not None else None,
        )
        last_loop_plan = loop_plan

        if steps_done == 0:
            target_tokens = min(int(max_new_tokens), int(plan["max_new_tokens"]))
        else:
            target_tokens = min(int(max_new_tokens), steps_done + int(plan["max_new_tokens"]))

        if steps_done >= target_tokens:
            break

        attn = torch.ones_like(generated)

        def _forward_fn():
            return custom_gpt2_forward_looped(
                model=model,
                tmu=plugin.tmu,
                input_ids=generated,
                attention_mask=attn,
                time_embed=time_embed,
                active_layers=int(plan["active_layers"]),
                timer=timer if enable_per_layer_timing else None,
                modulate_blocks=True,
                layer_refiner=layer_refiner if enable_loops else None,
                prompt_anchor=prompt_anchor if enable_loops else None,
                layer_loops=loop_plan["layer_loops"] if enable_loops else None,
                final_refine_passes=loop_plan["final_refine_passes"] if enable_loops else 0,
            )

        if cuda_cycle_timer is not None:
            (logits, prev_layer_times), cycle_info = cuda_cycle_timer.measure(_forward_fn)
        else:
            logits, prev_layer_times = _forward_fn()

        last_logits = logits[:, -1, :]

        probs = F.softmax(last_logits, dim=-1)
        ent = float(-(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item())
        max_ent = math.log(max(2, probs.size(-1)))
        confidence = clamp01(1.0 - ent / max_ent)

        temp = max(1e-3, float(plan["temperature"]))

        if do_sample:
            scaled = last_logits / temp
            k = min(int(top_k), scaled.size(-1)) if top_k and top_k > 0 else scaled.size(-1)
            topk_logits, topk_indices = torch.topk(scaled, k)
            top_probs = F.softmax(topk_logits, dim=-1)
            selected = torch.multinomial(top_probs, 1)
            next_token = topk_indices.gather(-1, selected)
        else:
            next_token = last_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)
        steps_done += 1

        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

        if snapshot.pressure >= 0.98:
            break

        if confidence >= float(cfg.confidence_stop) and steps_done >= int(cfg.min_steps_before_stop):
            break

    text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)

    plan_out = dict(last_plan)
    plan_out["generated_tokens"] = steps_done
    plan_out["confidence"] = confidence
    plan_out["loop_plan"] = last_loop_plan

    if last_snapshot is not None:
        plan_out["last_delta_ms"] = last_snapshot.delta_ms
        plan_out["last_delta_us"] = last_snapshot.delta_us
        plan_out["last_gpu_delta_us"] = last_snapshot.gpu_delta_us
        plan_out["last_cycles"] = last_snapshot.cycles
        plan_out["sm_clock_hz"] = last_snapshot.sm_clock_hz

    if cycle_info is not None:
        plan_out["cuda_cycles"] = cycle_info.get("cycles")
        plan_out["cuda_globaltimer_ns"] = cycle_info.get("globaltimer_ns")

    metrics = {
        "steps_done": steps_done,
        "confidence": confidence,
        "per_layer_times": prev_layer_times,
        "cuda_cycle_info": cycle_info,
        "loop_plan": last_loop_plan,
    }

    if traffic_monitor is not None and request_id is not None:
        traffic_monitor.request_end(
            request_id=request_id,
            generated_tokens=steps_done,
            success=True,
        )

    return text, plan_out, last_snapshot, metrics


# ============================================================
# Save / Load loop system
# ============================================================

def save_looped_system(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeads,
    adapter: AdvancedInternalAdapter,
    refiner: LayerRefinerCollection,
    path: str,
) -> None:
    import os

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "refiner_state_dict": refiner.state_dict(),
        },
        path,
    )


def load_looped_system(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeads,
    adapter: AdvancedInternalAdapter,
    refiner: LayerRefinerCollection,
    path: str,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)

    load_tmu_state_dict(plugin.tmu, ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    refiner.load_state_dict(ckpt["refiner_state_dict"])

    return plugin, heads, adapter, refiner
