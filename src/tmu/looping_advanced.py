# src/tmu/looping_advanced.py
#
# TMU GPT-2 Advanced Looping v3
#
# Add-only module. Existing repo files do not need modification.
#
# Implements:
# 1. KV-cache compatible partial looping / no-cache true dynamic looping
# 2. Per-token anchor strength schedule
# 3. Loop rejection sampling
# 4. Hard prompt detector using entropy + perplexity
# 5. Layer-wise importance scoring
# 6. Actual latency feedback training loop
#
# Plus advanced traffic/system features:
# - backlog, burstiness, queue growth, cost/token, SLA violation,
#   deadline slack, throughput trend, latency trend,
#   GPU contention, VRAM fragmentation proxy, model ppl/entropy.

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
from .data import AdvancedTimestampedDataset, estimate_text_difficulty
from .fusion import encode_fused_time_embed
from .heads import AdvancedTmuCapabilityHeads
from .looping import (
    AdvancedLoopHeads,
    BlockLoopParams,
    InputAnchor,
    LayerRefinerCollection,
    LoopConfig,
    apply_final_refine,
    make_loop_plan,
)
from .modeling import PerLayerTimer, _make_causal_padding_mask
from .pacing import AdvancedPaceConfig, AdvancedPaceController
from .plugin import TMUPlugin, freeze_base_and_enable_tmu
from .system import SystemMonitor
from .time_features import TimeLike, cyclic_time_features
from .traffic import RequestState, TrafficMonitor
from .utils import clamp01, log_norm, prepare_budget_tensor, safe_div

ADV_TRAFFIC_DIM = 36


# ============================================================
# Advanced Traffic Monitor
# ============================================================

class AdvancedTrafficMonitor(TrafficMonitor):
    """
    Extended traffic/QoS monitor.

    Adds:
    - burstiness
    - backlog
    - queue growth
    - cost per token
    - SLA violation
    - deadline slack
    - latency trend
    - throughput trend
    """

    def __init__(
        self,
        window_s: float = 60.0,
        concurrency_limit: int = 4,
        max_history: int = 10000,
        max_latency_ms: float = 2000.0,
    ):
        super().__init__(
            window_s=window_s,
            concurrency_limit=concurrency_limit,
            max_history=max_history,
        )

        self.max_latency_ms = float(max_latency_ms)

        self.starts: deque = deque(maxlen=max_history)
        self.meta: Dict[str, Dict[str, Any]] = {}

        self.previous_active = 0.0
        self.throughput_ema = 0.0
        self.p95_ema = 0.0
        self.avg_ema = 0.0
        self.error_ema = 0.0

    def request_start(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        priority: float = 0.5,
        deadline_s: Optional[float] = None,
        expected_tokens: int = 32,
    ) -> RequestState:
        state = super().request_start(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            priority=priority,
            deadline_s=deadline_s,
        )

        now = time.perf_counter()

        with self.lock:
            self.starts.append(now)
            self.meta[state.request_id] = {
                "expected_tokens": float(expected_tokens),
                "priority": float(priority),
                "deadline_s": state.deadline_s,
                "start": now,
            }

        return state

    def request_end(
        self,
        request_id: str,
        generated_tokens: int = 0,
        success: bool = True,
    ) -> Optional[Dict[str, Any]]:
        rid = str(request_id)

        with self.lock:
            self.meta.pop(rid, None)

        return super().request_end(
            request_id=request_id,
            generated_tokens=generated_tokens,
            success=success,
        )

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        now = time.perf_counter()

        with self.lock:
            starts = list(self.starts)
            metas = dict(self.meta)
            records = list(self.completed)
            active_count = int(base.get("active_requests", 0))

        # Burstiness from request inter-arrival times.
        if len(starts) >= 2:
            diffs = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
            mean_i = statistics.mean(diffs) if diffs else 1.0
            std_i = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
            burstiness = clamp01(std_i / max(mean_i, 1e-6))
        else:
            burstiness = 0.0

        latencies = [float(r[1]) for r in records] if records else []
        tokens = [int(r[2]) for r in records] if records else []

        avg_lat = statistics.mean(latencies) if latencies else 0.0
        p95 = float(base.get("p95_latency_ms", 0.0))
        throughput = float(base.get("throughput_tok_s", 0.0))
        error_rate = float(base.get("error_rate", 0.0))

        latency_trend = clamp01((p95 - avg_lat) / max(avg_lat, 1.0))

        total_tokens = sum(tokens)
        total_latency = sum(latencies)
        cost_per_token = safe_div(total_latency, max(total_tokens, 1), 0.0)
        cost_per_token_norm = clamp01(cost_per_token / 100.0)

        backlog = float(active_count) + max(
            0.0,
            float(active_count) - float(self.concurrency_limit),
        )
        backlog_norm = clamp01(backlog / max(1.0, float(self.concurrency_limit) * 2.0))

        sla_violations = 0
        deadline_slacks: List[float] = []

        for rid, m in metas.items():
            d = m.get("deadline_s")
            start = float(m.get("start", now))
            if d is None:
                continue

            d = float(d)
            if d <= 1e-6:
                continue

            elapsed = now - start
            slack = max(0.0, d - elapsed)
            deadline_slacks.append(slack)

            if elapsed > d:
                sla_violations += 1

        sla_violation_norm = clamp01(sla_violations / max(active_count, 1))
        min_slack = min(deadline_slacks) if deadline_slacks else max(1.0, self.window_s)
        deadline_slack_norm = clamp01(min_slack / max(1.0, self.window_s))

        queue_growth = max(0.0, float(active_count) - float(self.previous_active))
        self.previous_active = float(active_count)
        queue_growth_norm = clamp01(queue_growth / max(1.0, float(self.concurrency_limit)))

        if self.throughput_ema > 1e-6:
            throughput_trend = clamp01(
                (throughput - self.throughput_ema) / self.throughput_ema
            )
        else:
            throughput_trend = 0.0

        alpha = 0.1
        self.throughput_ema = (1.0 - alpha) * self.throughput_ema + alpha * throughput
        self.p95_ema = (1.0 - alpha) * self.p95_ema + alpha * p95
        self.avg_ema = (1.0 - alpha) * self.avg_ema + alpha * avg_lat
        self.error_ema = (1.0 - alpha) * self.error_ema + alpha * error_rate

        base.update(
            {
                "burstiness_norm": float(burstiness),
                "backlog_norm": float(backlog_norm),
                "latency_trend_norm": float(latency_trend),
                "cost_per_token_norm": float(cost_per_token_norm),
                "sla_violation_norm": float(sla_violation_norm),
                "deadline_slack_norm": float(deadline_slack_norm),
                "queue_growth_norm": float(queue_growth_norm),
                "throughput_trend_norm": float(throughput_trend),
                "avg_latency_ms": float(avg_lat),
                "expected_active_tokens": float(
                    sum(float(m.get("expected_tokens", 0.0)) for m in metas.values())
                ),
            }
        )

        return base


# ============================================================
# Advanced Traffic Feature Builder
# ============================================================

def build_advanced_traffic_feature_list(
    base_snap: AdvancedClockSnapshot,
    prompt_info: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Converts base 24-D clock snapshot + advanced traffic/system/model info
    into 36-D feature list.
    """
    prompt_info = prompt_info or {}

    traf = base_snap.traffic
    sysm = base_snap.system

    gpu_contention = clamp01(
        float(traf.get("active_norm", 0.0)) * float(sysm.get("gpu_util_norm", 0.0))
    )

    mem_fragmentation_proxy = clamp01(
        float(sysm.get("mem_pressure", 0.0)) * float(sysm.get("gpu_util_norm", 0.0))
    )

    extras = [
        clamp01(traf.get("latency_trend_norm", 0.0)),
        clamp01(traf.get("burstiness_norm", 0.0)),
        clamp01(traf.get("backlog_norm", 0.0)),
        clamp01(traf.get("sla_violation_norm", 0.0)),
        clamp01(traf.get("queue_growth_norm", 0.0)),
        clamp01(traf.get("cost_per_token_norm", 0.0)),
        clamp01(traf.get("deadline_slack_norm", 0.0)),
        float(gpu_contention),
        float(mem_fragmentation_proxy),
        clamp01(traf.get("throughput_trend_norm", 0.0)),
        clamp01(prompt_info.get("ppl_norm", 0.5)),
        clamp01(prompt_info.get("entropy_norm", 0.5)),
    ]

    full = list(base_snap.feature_list) + extras

    if len(full) != ADV_TRAFFIC_DIM:
        raise ValueError(
            f"Advanced traffic feature dim mismatch: got {len(full)}, expected {ADV_TRAFFIC_DIM}"
        )

    return full


def advanced_traffic_snapshot_to_tensor(
    snapshot: AdvancedClockSnapshot,
    extra_features: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    vals = list(snapshot.feature_list) + list(extra_features)

    if len(vals) != ADV_TRAFFIC_DIM:
        raise ValueError(
            f"Advanced traffic tensor dim mismatch: got {len(vals)}, expected {ADV_TRAFFIC_DIM}"
        )

    t = torch.tensor(vals, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).expand(int(batch_size), -1).contiguous()
    return t


# ============================================================
# Advanced Traffic Adapter
# ============================================================

class AdvancedTrafficAdapter(nn.Module):
    """
    36-D advanced internal/traffic features -> hidden_dim.
    Zero-initialized final layer for safe startup.
    """

    def __init__(self, hidden_dim: int, internal_dim: int = ADV_TRAFFIC_DIM):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.internal_dim = int(internal_dim)

        self.norm = nn.LayerNorm(self.internal_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.internal_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, internal_features: torch.Tensor) -> torch.Tensor:
        x = self.norm(internal_features)
        return self.proj(x)


# ============================================================
# Prompt Difficulty Detector: entropy + perplexity
# ============================================================

class PromptDifficultyDetector:
    """
    Detects hard prompts using:
    - heuristic text difficulty
    - model perplexity on prompt
    - next-token entropy
    """

    def __init__(
        self,
        enabled: bool = True,
        heuristic_weight: float = 0.35,
        ppl_weight: float = 0.35,
        entropy_weight: float = 0.30,
    ):
        self.enabled = bool(enabled)
        self.heuristic_weight = float(heuristic_weight)
        self.ppl_weight = float(ppl_weight)
        self.entropy_weight = float(entropy_weight)

    @torch.no_grad()
    def detect(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: str = "",
    ) -> Dict[str, float]:
        heuristic = estimate_text_difficulty(text)

        if not self.enabled:
            return {
                "heuristic": heuristic,
                "ppl_norm": heuristic,
                "entropy_norm": heuristic,
                "combined": heuristic,
                "perplexity": 0.0,
                "entropy": 0.0,
            }

        was_training = model.training
        model.eval()

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            if was_training:
                model.train()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        if logits.dim() != 3:
            return {
                "heuristic": heuristic,
                "ppl_norm": heuristic,
                "entropy_norm": heuristic,
                "combined": heuristic,
                "perplexity": 0.0,
                "entropy": 0.0,
            }

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].bool()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)

        token_logprob = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        valid = shift_mask & (shift_labels >= 0)

        if bool(valid.any()):
            mean_logprob = float(token_logprob[valid].mean().item())
            perplexity = math.exp(max(-50.0, min(50.0, -mean_logprob)))

            entropy_per_token = -(probs * log_probs).sum(dim=-1)
            entropy = float(entropy_per_token[valid].mean().item())
        else:
            perplexity = 100.0
            entropy = math.log(max(2, int(logits.size(-1)))) * 0.5

        vocab = max(2, int(logits.size(-1)))
        max_entropy = math.log(vocab)

        ppl_norm = log_norm(perplexity, 1000.0)
        entropy_norm = clamp01(entropy / max_entropy)

        combined = clamp01(
            self.heuristic_weight * heuristic
            + self.ppl_weight * ppl_norm
            + self.entropy_weight * entropy_norm
        )

        return {
            "heuristic": float(heuristic),
            "ppl_norm": float(ppl_norm),
            "entropy_norm": float(entropy_norm),
            "combined": float(combined),
            "perplexity": float(perplexity),
            "entropy": float(entropy),
        }


# ============================================================
# Layer-wise Importance Scorer
# ============================================================

class LayerImportanceScorer:
    """
    Tracks per-layer hidden-state contribution using EMA of delta norms.
    Used to distribute loops toward more important layers.
    """

    def __init__(
        self,
        num_layers: int,
        momentum: float = 0.2,
        floor: float = 0.05,
    ):
        self.num_layers = int(num_layers)
        self.momentum = float(momentum)
        self.floor = float(floor)
        self.raw = torch.ones(self.num_layers, dtype=torch.float32)

    def reset(self) -> None:
        self.raw.fill_(1.0)

    def observe(self, layer_idx: int, delta_norm: float) -> None:
        layer_idx = int(max(0, min(layer_idx, self.num_layers - 1)))
        val = max(0.0, float(delta_norm))
        m = self.momentum
        self.raw[layer_idx] = (1.0 - m) * self.raw[layer_idx] + m * val

    def scores(self) -> List[float]:
        x = self.raw + self.floor
        x = x / x.sum().clamp_min(1e-9)
        return [float(v) for v in x.tolist()]


# ============================================================
# Per-token Anchor Strength Scheduler
# ============================================================

class AnchorScheduler:
    """
    Decides how strongly original input anchor should be injected
    at each generation step.

    Modes:
    - decay: strong at start, weak later
    - rise: weak at start, strong later
    - u: strong at start/end, moderate middle
    """

    def __init__(
        self,
        mode: str = "u",
        min_strength: float = 0.15,
        max_strength: float = 1.0,
        pressure_gain: float = 0.40,
        uncertainty_gain: float = 0.30,
    ):
        self.mode = str(mode).lower()
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.pressure_gain = float(pressure_gain)
        self.uncertainty_gain = float(uncertainty_gain)

    def strength(
        self,
        step: int,
        total_steps: int,
        pressure: float = 0.0,
        confidence: float = 0.5,
        difficulty: float = 0.5,
    ) -> float:
        step = int(max(0, step))
        total_steps = int(max(1, total_steps))
        progress = clamp01(float(step) / float(total_steps))

        mn = self.min_strength
        mx = self.max_strength

        if self.mode == "decay":
            base = mx - (mx - mn) * progress
        elif self.mode == "rise":
            base = mn + (mx - mn) * progress
        else:
            # U-shape: stronger near beginning and end.
            base = mn + (mx - mn) * (0.5 + 0.5 * abs(2.0 * progress - 1.0))

        pressure = clamp01(pressure)
        confidence = clamp01(confidence)
        difficulty = clamp01(difficulty)

        # High pressure / low confidence / hard prompt => stronger anchor.
        strength = base
        strength *= 1.0 + self.pressure_gain * pressure
        strength *= 1.0 + self.uncertainty_gain * (1.0 - confidence)
        strength *= 1.0 + 0.15 * difficulty

        return clamp01(strength)


# ============================================================
# Advanced Block Loop Params
# ============================================================

class AdvancedBlockLoopParams(BlockLoopParams):
    """
    Block loop params with:
    - anchor scale
    - layer importance weighting
    - delta-norm rejection guard
    """

    def __init__(
        self,
        hidden_dim: int,
        config: Optional[LoopConfig] = None,
        max_delta_norm: float = 0.0,
        importance_gain: float = 1.0,
    ):
        super().__init__(hidden_dim=hidden_dim, config=config)
        self.max_delta_norm = float(max_delta_norm)
        self.importance_gain = float(importance_gain)

    def loop_forward(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor: torch.Tensor,
        loops: int,
        anchor_scale: float = 1.0,
        importance_weight: float = 1.0,
        collect_delta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        loops = int(max(0, min(int(loops), int(self.config.max_loops))))

        if (not self.config.enabled) or loops <= 0:
            if collect_delta:
                return hidden, 0.0
            return hidden

        h = hidden
        a = anchor

        if a.dim() == 2:
            a = a.unsqueeze(1)

        scale = float(anchor_scale) * float(importance_weight) * float(self.importance_gain)
        a = a * scale

        total_dn = 0.0
        executed = 0

        for _ in range(loops):
            h_in = self.pre_norm(h + a)

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
            delta = self.delta_norm(new_h - h_in)

            dn = float(delta.norm(dim=-1).mean().item())
            total_dn += dn
            executed += 1

            # Delta-norm guard: if loop update is too violent, reject this loop.
            if self.max_delta_norm > 0.0 and dn > self.max_delta_norm:
                break

            h = h + torch.tanh(self.loop_gate) * delta

        h = self.post_norm(h)
        avg_dn = total_dn / max(1, executed)

        if collect_delta:
            return h, avg_dn
        return h


# ============================================================
# Advanced Layer Refiner
# ============================================================

class AdvancedLayerRefiner(LayerRefinerCollection):
    """
    LayerRefinerCollection with advanced block params.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        config: Optional[LoopConfig] = None,
        max_delta_norm: float = 0.0,
        importance_gain: float = 1.0,
    ):
        super().__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            config=config,
        )

        self.block_params = nn.ModuleList(
            [
                AdvancedBlockLoopParams(
                    hidden_dim=hidden_dim,
                    config=self.config,
                    max_delta_norm=max_delta_norm,
                    importance_gain=importance_gain,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward_block_advanced(
        self,
        layer_idx: int,
        block: nn.Module,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor: torch.Tensor,
        loops: int,
        anchor_scale: float = 1.0,
        importance_weight: float = 1.0,
        collect_delta: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        layer_idx = max(0, min(int(layer_idx), self.num_layers - 1))
        out = self.block_params[layer_idx].loop_forward(
            block=block,
            hidden=hidden,
            attention_mask=attention_mask,
            anchor=anchor,
            loops=loops,
            anchor_scale=anchor_scale,
            importance_weight=importance_weight,
            collect_delta=True,
        )
        return out  # type: ignore[return-value]


# ============================================================
# Advanced Loop Heads with Latency Head
# ============================================================

class AdvancedLoopHeadsV3(AdvancedLoopHeads):
    """
    Adds latency prediction head.

    Returns:
        rhythm_pred, pace, layer_frac, token_frac, loop_frac, latency_pred, pace_logit
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 12,
        max_tokens: int = 64,
        budget_dim: int = 1,
        loop_bias_init: float = 0.0,
        latency_bias_init: float = 0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_tokens=max_tokens,
            budget_dim=budget_dim,
            loop_bias_init=loop_bias_init,
        )

        pace_hidden = max(8, int(hidden_dim) // 2)

        self.latency_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        nn.init.zeros_(self.latency_head[-1].weight)
        nn.init.constant_(self.latency_head[-1].bias, float(latency_bias_init))

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: Union[torch.Tensor, float],
    ):
        rhythm_pred, pace, layer_frac, token_frac, loop_frac, pace_logit = super().forward(
            time_embed,
            budget,
        )

        # Recompute conditioned for latency head.
        if not torch.is_tensor(budget):
            budget_t = torch.tensor(float(budget), device=time_embed.device)
        else:
            budget_t = budget

        budget_t = budget_t.to(device=time_embed.device, dtype=time_embed.dtype)

        if budget_t.dim() == 0:
            budget_t = budget_t.unsqueeze(0)
        if budget_t.dim() == 1:
            budget_t = budget_t.unsqueeze(-1)

        if budget_t.shape[0] != time_embed.shape[0]:
            if budget_t.shape[0] == 1:
                budget_t = budget_t.expand(time_embed.shape[0], -1)
            elif time_embed.shape[0] == 1:
                time_embed = time_embed.expand(budget_t.shape[0], -1)
            else:
                raise ValueError(
                    f"Budget batch {budget_t.shape[0]} != time_embed batch {time_embed.shape[0]}"
                )

        conditioned = time_embed + self.budget_proj(budget_t)
        latency_logit = self.latency_head(conditioned).squeeze(-1)
        latency_pred = torch.sigmoid(latency_logit)

        return rhythm_pred, pace, layer_frac, token_frac, loop_frac, latency_pred, pace_logit


# ============================================================
# Advanced Loop Plan with Layer Importance
# ============================================================

def make_advanced_loop_plan(
    num_layers: int,
    loop_frac: float,
    difficulty: float,
    budget: float,
    traffic_load: float = 0.0,
    thermal_load: float = 0.0,
    layer_scores: Optional[Sequence[float]] = None,
    config: Optional[LoopConfig] = None,
    gamma: float = 1.5,
) -> Dict[str, Any]:
    """
    Builds per-layer loop schedule using:
    - predicted loop fraction
    - prompt difficulty
    - budget
    - traffic/thermal load
    - layer importance scores
    """
    cfg = config or LoopConfig()

    loop_frac = clamp01(loop_frac)
    difficulty = clamp01(difficulty)
    budget = clamp01(budget)
    traffic_load = clamp01(traffic_load)
    thermal_load = clamp01(thermal_load)

    load = clamp01(0.60 * traffic_load + 0.40 * thermal_load)

    base = float(cfg.min_loops) + loop_frac * float(cfg.max_loops - cfg.min_loops)
    boost = float(cfg.difficulty_boost) * max(0.0, difficulty - 0.35) * budget
    raw = base + boost

    factor = max(0.20, 1.0 - float(cfg.load_penalty) * load)
    total = int(round(raw * factor))
    total = max(int(cfg.min_loops), min(int(cfg.max_loops), total))

    num_layers = int(max(1, num_layers))

    if layer_scores is None or len(layer_scores) != num_layers:
        scores = [1.0 / float(num_layers) for _ in range(num_layers)]
    else:
        scores = [max(0.0, float(x)) for x in layer_scores]
        s = sum(scores)
        if s <= 1e-9:
            scores = [1.0 / float(num_layers) for _ in range(num_layers)]
        else:
            scores = [x / s for x in scores]

    late_bias = []
    for i in range(num_layers):
        pos = float(i + 1) / float(num_layers)
        late_bias.append(1.0 + float(cfg.late_layer_bias) * (pos - 0.5))

    weights = []
    for i in range(num_layers):
        w = scores[i] * late_bias[i]
        w = max(0.0, float(w))
        if gamma != 1.0 and w > 0.0:
            w = w ** float(gamma)
        weights.append(w)

    wsum = sum(weights)
    if wsum <= 1e-9:
        weights = [1.0 / float(num_layers) for _ in range(num_layers)]
    else:
        weights = [w / wsum for w in weights]

    layer_loops = [0 for _ in range(num_layers)]

    if total > 0:
        raw_alloc = [float(total) * w for w in weights]
        layer_loops = [int(math.floor(x)) for x in raw_alloc]

        # Clip to max_loops.
        for i in range(num_layers):
            layer_loops[i] = max(
                int(cfg.min_loops),
                min(int(cfg.max_loops), layer_loops[i]),
            )

        remaining = total - sum(layer_loops)

        if remaining > 0:
            fracs = []
            for i in range(num_layers):
                frac = raw_alloc[i] - float(layer_loops[i])
                fracs.append((frac, i))
            fracs.sort(reverse=True)

            for _, i in fracs:
                if remaining <= 0:
                    break
                if layer_loops[i] < int(cfg.max_loops):
                    layer_loops[i] += 1
                    remaining -= 1

    final_refine_passes = int(
        round(
            clamp01(loop_frac * difficulty)
            * float(cfg.max_final_refine_passes)
            * float(factor)
        )
    )

    if total <= 0:
        final_refine_passes = 0

    return {
        "loop_frac": loop_frac,
        "difficulty": difficulty,
        "budget": budget,
        "traffic_load": traffic_load,
        "thermal_load": thermal_load,
        "load": load,
        "loop_scalar": total,
        "layer_loops": layer_loops,
        "loop_weights": weights,
        "final_refine_passes": final_refine_passes,
    }


# ============================================================
# Advanced Custom Forward with Loops + Anchor + Importance
# ============================================================

def custom_gpt2_forward_looped_advanced(
    model: nn.Module,
    tmu: TimeManagementUnit,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    time_embed: torch.Tensor,
    active_layers: Optional[int] = None,
    timer: Optional[PerLayerTimer] = None,
    modulate_blocks: bool = True,
    layer_refiner: Optional[AdvancedLayerRefiner] = None,
    prompt_anchor: Optional[torch.Tensor] = None,
    layer_loops: Optional[List[int]] = None,
    final_refine_passes: int = 0,
    anchor_scale: float = 1.0,
    layer_scores: Optional[Sequence[float]] = None,
    importance_scorer: Optional[LayerImportanceScorer] = None,
    collect_importance: bool = False,
) -> Tuple[torch.Tensor, Optional[List[float]], List[float]]:
    """
    No-cache advanced forward:
    - TMU modulation
    - dynamic active layers
    - per-layer loops
    - input anchor injection with per-token scale
    - layer importance observation
    - final refinement
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

    if layer_refiner is not None and prompt_anchor is None:
        prompt_anchor = layer_refiner.prepare_anchor(base_hidden, attention_mask)

    num_layers = len(transformer.h)

    if active_layers is None:
        active_layers = num_layers

    active_layers = max(1, min(int(active_layers), num_layers))

    if timer is not None:
        timer.reset()

    blocks = list(transformer.h)

    if layer_scores is None:
        norm_scores = None
    else:
        norm_scores = [float(x) for x in layer_scores]

    delta_norms: List[float] = []

    for i, block in enumerate(blocks):
        if i >= active_layers:
            continue

        loops = 0
        if layer_loops is not None and i < len(layer_loops):
            loops = int(layer_loops[i])

        pre_hidden = hidden

        if timer is not None:
            timer.begin(i)

        if layer_refiner is not None and prompt_anchor is not None and loops > 0:
            iw = 1.0
            if norm_scores is not None and i < len(norm_scores):
                iw = max(0.0, float(norm_scores[i]) * float(num_layers))

            hidden, dn = layer_refiner.forward_block_advanced(
                layer_idx=i,
                block=block,
                hidden=hidden,
                attention_mask=attn_mask,
                anchor=prompt_anchor,
                loops=loops,
                anchor_scale=anchor_scale,
                importance_weight=iw,
                collect_delta=True,
            )
        else:
            outputs = block(
                hidden,
                attention_mask=attn_mask,
                use_cache=False,
            )
            hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if importance_scorer is not None and collect_importance:
                dn = float((hidden - pre_hidden).norm(dim=-1).mean().item())
            else:
                dn = 0.0

        if timer is not None:
            timer.end(i)

        if importance_scorer is not None and collect_importance:
            importance_scorer.observe(i, dn)

        delta_norms.append(dn)

        if modulate_blocks:
            hidden = tmu.apply_to(hidden, time_embed)

    if (
        layer_refiner is not None
        and prompt_anchor is not None
        and int(final_refine_passes) > 0
    ):
        idx = max(0, min(int(active_layers - 1), len(blocks) - 1, layer_refiner.num_layers - 1))

        iw = 1.0
        if norm_scores is not None and idx < len(norm_scores):
            iw = max(0.0, float(norm_scores[idx]) * float(num_layers))

        for _ in range(int(final_refine_passes)):
            hidden, dn = layer_refiner.forward_block_advanced(
                layer_idx=idx,
                block=blocks[idx],
                hidden=hidden,
                attention_mask=attn_mask,
                anchor=prompt_anchor,
                loops=1,
                anchor_scale=anchor_scale,
                importance_weight=iw,
                collect_delta=True,
            )

            if importance_scorer is not None and collect_importance:
                importance_scorer.observe(idx, dn)

            delta_norms.append(dn)

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden)

    layer_times = timer.collect() if timer is not None else None

    return logits, layer_times, delta_norms


# ============================================================
# Confidence Helper
# ============================================================

def confidence_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    ent = float(-(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item())
    max_ent = math.log(max(2, int(logits.size(-1))))
    return clamp01(1.0 - ent / max_ent)


# ============================================================
# Predict Advanced Loop Embed v3
# ============================================================

def predict_advanced_loop_embed_v3(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    timestamp: TimeLike,
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
    internal_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        pace, layer_frac, token_frac, loop_frac, latency_pred, time_embed, rhythm_pred
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

    rhythm_pred, pace, layer_frac, token_frac, loop_frac, latency_pred, _ = heads(
        time_embed,
        budget_t,
    )

    return pace, layer_frac, token_frac, loop_frac, latency_pred, time_embed, rhythm_pred


# ============================================================
# Dataset Wrapper: Loop + Latency Labels
# ============================================================

class AdvancedLoopLatencyDatasetWrapper(Dataset):
    """
    Adds y_loops and y_latency to existing AdvancedTimestampedDataset.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        metas: Optional[Sequence[Dict[str, Any]]] = None,
        max_loops: int = 3,
        max_latency_ms: float = 2000.0,
        max_tokens: int = 64,
    ):
        super().__init__()

        self.base_dataset = base_dataset
        self.metas = list(metas) if metas is not None else [{} for _ in range(len(base_dataset))]
        self.max_loops = int(max(1, max_loops))
        self.max_latency_ms = float(max(1.0, max_latency_ms))
        self.max_tokens = int(max(1, max_tokens))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = dict(self.base_dataset[idx])
        meta = {}
        if idx < len(self.metas):
            meta = self.metas[idx] or {}

        try:
            difficulty = float(item.get("difficulty", 0.0))
        except Exception:
            difficulty = 0.0

        try:
            budget = float(item.get("budget", 0.5))
        except Exception:
            budget = 0.5

        desired_loops = meta.get("desired_loops", None)

        try:
            y_loops = clamp01(float(desired_loops) / float(self.max_loops))
        except Exception:
            y_loops = clamp01(difficulty * budget)

        latency_ms = meta.get("latency_ms", None)

        try:
            y_latency = clamp01(float(latency_ms) / float(self.max_latency_ms))
        except Exception:
            y_latency = clamp01(
                0.15
                + 0.45 * budget
                + 0.25 * difficulty
                + 0.15 * y_loops
            )

        item["y_loops"] = torch.tensor(y_loops, dtype=torch.float32)
        item["y_latency"] = torch.tensor(y_latency, dtype=torch.float32)

        return item


# ============================================================
# Advanced Training with Loops + Latency
# ============================================================

def train_advanced_loop_v3(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    refiner: AdvancedLayerRefiner,
    train_loader: DataLoader,
    device: str = "cuda",
    clock: Optional[AdvancedInternalClock] = None,
    system_monitor: Optional[SystemMonitor] = None,
    traffic_monitor: Optional[AdvancedTrafficMonitor] = None,
    epochs: int = 2,
    lr: float = 1e-3,
    w_lm: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    w_layer: float = 0.8,
    w_token: float = 0.6,
    w_loop: float = 0.8,
    w_latency: float = 0.7,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    use_gpu_events: bool = False,
    budget_s: Optional[float] = None,
    enable_loops: bool = True,
    verbose: bool = True,
):
    """
    Trains:
    - TMU
    - AdvancedLoopHeadsV3
    - AdvancedTrafficAdapter
    - AdvancedLayerRefiner

    Base GPT-2 remains frozen.
    """
    plugin.to(device)
    heads.to(device)
    adapter.to(device)
    refiner.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    if clock is None:
        if system_monitor is None:
            system_monitor = SystemMonitor(
                device,
                enable=str(device).startswith("cuda"),
            )

        if traffic_monitor is None:
            traffic_monitor = AdvancedTrafficMonitor()

        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=budget_s,
            use_gpu_events=use_gpu_events,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
        )

    num_layers = len(plugin.base_model.transformer.h)
    importance_scorer = LayerImportanceScorer(num_layers=num_layers)

    tmu_params = freeze_base_and_enable_tmu(plugin)

    params = (
        list(tmu_params)
        + list(heads.parameters())
        + list(adapter.parameters())
        + refiner.trainable_parameters()
    )

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    history = []

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
            "latency": 0.0,
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
            y_latency = batch["y_latency"].to(device, non_blocking=True)
            difficulty = batch["difficulty"].to(device, non_blocking=True)

            B = int(input_ids.shape[0])

            snapshot = clock.tick(
                tokens=int(input_ids.numel()),
                prompt_difficulty=float(difficulty.mean().item()),
                confidence=0.5,
                per_layer_times=None,
                sync=bool(use_gpu_events),
            )

            prompt_info = {
                "ppl_norm": float(difficulty.mean().item()),
                "entropy_norm": 0.5,
            }

            extras = build_advanced_traffic_feature_list(snapshot, prompt_info)
            internal_features = advanced_traffic_snapshot_to_tensor(
                snapshot,
                extras,
                batch_size=B,
                device=device,
            )

            (
                pace_pred,
                layer_pred,
                token_pred,
                loop_pred,
                latency_pred,
                time_embed,
                rhythm_pred,
            ) = predict_advanced_loop_embed_v3(
                plugin=plugin,
                heads=heads,
                adapter=adapter,
                timestamp=t,
                budget=budget,
                batch_size=B,
                device=device,
                internal_features=internal_features,
            )

            layer_scores = importance_scorer.scores() if enable_loops else None

            loop_plan = make_advanced_loop_plan(
                num_layers=num_layers,
                loop_frac=float(loop_pred.mean().item()),
                difficulty=float(difficulty.mean().item()),
                budget=float(budget.mean().item()),
                traffic_load=float(snapshot.traffic.get("queue_delay_norm", 0.0)),
                thermal_load=float(snapshot.system.get("temp_norm", 0.0)),
                layer_scores=layer_scores,
                config=refiner.config,
            )

            logits, _, _ = custom_gpt2_forward_looped_advanced(
                model=plugin.base_model,
                tmu=plugin.tmu,
                input_ids=input_ids,
                attention_mask=attention_mask,
                time_embed=time_embed,
                active_layers=None,
                timer=None,
                modulate_blocks=True,
                layer_refiner=refiner if enable_loops else None,
                prompt_anchor=None,
                layer_loops=loop_plan["layer_loops"] if enable_loops else [0] * num_layers,
                final_refine_passes=loop_plan["final_refine_passes"] if enable_loops else 0,
                anchor_scale=1.0,
                layer_scores=layer_scores,
                importance_scorer=importance_scorer if enable_loops else None,
                collect_importance=bool(enable_loops),
            )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            flat_labels = shift_labels.view(-1)
            valid = flat_labels != -100

            if bool(valid.any()):
                loss_lm = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1))[valid],
                    flat_labels[valid],
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

            loss_latency = F.mse_loss(latency_pred, y_latency)

            over_budget = F.relu(pace_pred - budget).mean()
            pace_loss_total = loss_pace + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * pace_loss_total
                + float(w_layer) * loss_layer
                + float(w_token) * loss_token
                + float(w_loop) * loss_loop
                + float(w_latency) * loss_latency
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
            totals["latency"] += float(loss_latency.item())
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
                f"latency={avg['latency']:.4f} | "
                f"over={avg['over']:.4f} | "
                f"loop_scalar={avg['loop_scalar']:.2f} | "
                f"refine={avg['refine_passes']:.2f} | "
                f"delta_us={avg['delta_us']:.2f}"
            )

    return history


# ============================================================
# Latency Feedback Buffer
# ============================================================

@dataclass
class FeedbackRecord:
    timestamp: float
    budget: float
    internal_features: List[float]
    actual_latency_ms: float
    generated_tokens: int
    confidence: float
    quality: Optional[float] = None


class LatencyFeedbackBuffer:
    """
    Stores runtime feedback for later latency/pace fine-tuning.
    """

    def __init__(
        self,
        max_records: int = 10000,
        max_latency_ms: float = 2000.0,
        max_tokens: int = 64,
    ):
        self.max_records = int(max_records)
        self.max_latency_ms = float(max_latency_ms)
        self.max_tokens = int(max_tokens)
        self.records: deque = deque(maxlen=self.max_records)
        self.lock = threading.Lock()

    def add(
        self,
        timestamp: float,
        budget: float,
        internal_features: Sequence[float],
        actual_latency_ms: float,
        generated_tokens: int,
        confidence: float,
        quality: Optional[float] = None,
    ) -> FeedbackRecord:
        rec = FeedbackRecord(
            timestamp=float(timestamp),
            budget=float(budget),
            internal_features=[float(x) for x in internal_features],
            actual_latency_ms=float(actual_latency_ms),
            generated_tokens=int(generated_tokens),
            confidence=float(confidence),
            quality=float(quality) if quality is not None else None,
        )

        with self.lock:
            self.records.append(rec)

        return rec

    def __len__(self):
        with self.lock:
            return len(self.records)

    def make_loader(self, batch_size: int = 16) -> DataLoader:
        class _FeedbackDataset(Dataset):
            def __init__(self, records: List[FeedbackRecord], max_latency_ms: float, max_tokens: int):
                self.records = records
                self.max_latency_ms = float(max_latency_ms)
                self.max_tokens = int(max_tokens)

            def __len__(self):
                return len(self.records)

            def __getitem__(self, idx):
                r = self.records[idx]

                y_latency = clamp01(r.actual_latency_ms / self.max_latency_ms)
                y_tokens = clamp01(float(r.generated_tokens) / float(self.max_tokens))
                y_pace = y_tokens

                return {
                    "t": torch.tensor(r.timestamp, dtype=torch.float32),
                    "budget": torch.tensor(r.budget, dtype=torch.float32),
                    "internal_features": torch.tensor(r.internal_features, dtype=torch.float32),
                    "y_latency": torch.tensor(y_latency, dtype=torch.float32),
                    "y_tokens": torch.tensor(y_tokens, dtype=torch.float32),
                    "y_pace": torch.tensor(y_pace, dtype=torch.float32),
                    "confidence": torch.tensor(r.confidence, dtype=torch.float32),
                }

        with self.lock:
            records = list(self.records)

        ds = _FeedbackDataset(records, self.max_latency_ms, self.max_tokens)
        return DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)


def fine_tune_latency_feedback(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    buffer: LatencyFeedbackBuffer,
    device: str = "cuda",
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 16,
    w_latency: float = 1.0,
    w_token: float = 0.5,
    w_pace: float = 0.3,
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    """
    Small online/offline feedback fine-tuning.

    Trains:
    - TMU
    - heads
    - adapter

    Does not require text input_ids, only stored time/internal features.
    """
    if len(buffer) == 0:
        if verbose:
            print("Feedback buffer empty. Nothing to fine-tune.")
        return []

    plugin.to(device)
    heads.to(device)
    adapter.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = list(tmu_params) + list(heads.parameters()) + list(adapter.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    loader = buffer.make_loader(batch_size=batch_size)
    history = []

    for epoch in range(int(epochs)):
        plugin.tmu.train()
        heads.train()
        adapter.train()

        total_loss = 0.0
        total_latency = 0.0
        total_token = 0.0
        total_pace = 0.0
        n = 0

        for batch in loader:
            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)
            internal_features = batch["internal_features"].to(device, non_blocking=True)

            y_latency = batch["y_latency"].to(device, non_blocking=True)
            y_tokens = batch["y_tokens"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)

            B = int(t.shape[0])

            (
                pace_pred,
                layer_pred,
                token_pred,
                loop_pred,
                latency_pred,
                time_embed,
                rhythm_pred,
            ) = predict_advanced_loop_embed_v3(
                plugin=plugin,
                heads=heads,
                adapter=adapter,
                timestamp=t,
                budget=budget,
                batch_size=B,
                device=device,
                internal_features=internal_features,
            )

            loss_latency = F.mse_loss(latency_pred, y_latency)

            token_safe = token_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_token = F.binary_cross_entropy(token_safe, y_tokens)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace = F.binary_cross_entropy(pace_safe, y_pace)

            loss = (
                float(w_latency) * loss_latency
                + float(w_token) * loss_token
                + float(w_pace) * loss_pace
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            total_loss += float(loss.item())
            total_latency += float(loss_latency.item())
            total_token += float(loss_token.item())
            total_pace += float(loss_pace.item())
            n += 1

        avg_loss = total_loss / max(1, n)
        avg_latency = total_latency / max(1, n)
        avg_token = total_token / max(1, n)
        avg_pace = total_pace / max(1, n)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "latency": avg_latency,
                "token": avg_token,
                "pace": avg_pace,
            }
        )

        if verbose:
            print(
                f"Feedback Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"latency={avg_latency:.4f} | "
                f"token={avg_token:.4f} | "
                f"pace={avg_pace:.4f}"
            )

    return history


# ============================================================
# Advanced Dynamic Generation v3
# ============================================================

@torch.no_grad()
def generate_advanced_dynamic_loops_v3(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    layer_refiner: Optional[AdvancedLayerRefiner],
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
    enable_rejection: bool = False,
    rejection_margin: float = 0.02,
    prompt_detector: Optional[PromptDifficultyDetector] = None,
    importance_scorer: Optional[LayerImportanceScorer] = None,
    anchor_scheduler: Optional[AnchorScheduler] = None,
    latency_feedback_buffer: Optional[LatencyFeedbackBuffer] = None,
    system_monitor: Optional[SystemMonitor] = None,
    traffic_monitor: Optional[AdvancedTrafficMonitor] = None,
    cuda_cycle_timer: Optional[Any] = None,
    do_sample: bool = False,
    top_k: int = 50,
    seed: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Optional[AdvancedClockSnapshot], Dict[str, Any], Optional[FeedbackRecord]]:
    """
    Advanced dynamic generation with:
    - internal clock
    - advanced traffic/system features
    - model-based prompt difficulty
    - layer importance
    - per-token anchor schedule
    - loop rejection sampling
    - latency feedback recording
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

    # Prompt difficulty detection.
    if prompt_detector is not None and prompt_detector.enabled:
        prompt_info = prompt_detector.detect(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=prompt,
        )
    else:
        h = estimate_text_difficulty(prompt)
        prompt_info = {
            "heuristic": h,
            "ppl_norm": h,
            "entropy_norm": h,
            "combined": h,
            "perplexity": 0.0,
            "entropy": 0.0,
        }

    difficulty = clamp01(prompt_info.get("combined", estimate_text_difficulty(prompt)))

    req_start_perf = time.perf_counter()

    if traffic_monitor is not None and request_id is not None:
        try:
            traffic_monitor.request_start(
                request_id=request_id,
                prompt_tokens=prompt_len,
                priority=budget,
                deadline_s=budget_s,
                expected_tokens=int(max_new_tokens),
            )
        except TypeError:
            traffic_monitor.request_start(
                request_id=request_id,
                prompt_tokens=prompt_len,
                priority=budget,
                deadline_s=budget_s,
            )

    if system_monitor is None:
        system_monitor = SystemMonitor(device, enable=str(device).startswith("cuda"))

    if traffic_monitor is None:
        traffic_monitor = AdvancedTrafficMonitor()

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
        layer_refiner = AdvancedLayerRefiner(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            config=loop_cfg,
            max_delta_norm=0.0,
            importance_gain=1.0,
        ).to(device)

    if importance_scorer is None:
        importance_scorer = LayerImportanceScorer(num_layers=num_layers)
    else:
        importance_scorer.reset()

    if anchor_scheduler is None:
        anchor_scheduler = AnchorScheduler(mode="u")

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

    confidence = 0.5
    last_plan: Dict[str, Any] = {}
    last_loop_plan: Dict[str, Any] = {}
    last_snapshot: Optional[AdvancedClockSnapshot] = None
    last_internal_features_list: Optional[List[float]] = None
    cycle_info: Optional[Dict[str, Any]] = None
    delta_norms: List[float] = []
    anchor_scale = 1.0
    accepted_loop = True

    if int(max_new_tokens) <= 0:
        snapshot = clock.tick(
            tokens=prompt_len,
            prompt_difficulty=difficulty,
            confidence=confidence,
            per_layer_times=None,
            sync=bool(use_gpu_events),
        )
        return "", {}, snapshot, {"prompt_info": prompt_info}, None

    while steps_done < int(max_new_tokens):
        tokens_for_tick = prompt_len if steps_done == 0 else 1

        snapshot = clock.tick(
            tokens=tokens_for_tick,
            prompt_difficulty=difficulty,
            confidence=confidence,
            per_layer_times=delta_norms if enable_per_layer_timing else None,
            sync=bool(use_gpu_events),
        )
        last_snapshot = snapshot

        extras = build_advanced_traffic_feature_list(snapshot, prompt_info)
        internal_features = advanced_traffic_snapshot_to_tensor(
            snapshot,
            extras,
            batch_size=1,
            device=device,
        )
        last_internal_features_list = internal_features[0].detach().cpu().tolist()

        (
            pace,
            layer_frac,
            token_frac,
            loop_frac,
            latency_pred,
            time_embed,
            _,
        ) = predict_advanced_loop_embed_v3(
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

        layer_scores = importance_scorer.scores() if enable_loops else None

        loop_plan = make_advanced_loop_plan(
            num_layers=num_layers,
            loop_frac=float(loop_frac[0].item()),
            difficulty=difficulty,
            budget=float(budget),
            traffic_load=float(snapshot.traffic.get("queue_delay_norm", 0.0)),
            thermal_load=float(snapshot.system.get("temp_norm", 0.0)),
            layer_scores=layer_scores,
            config=layer_refiner.config if layer_refiner is not None else None,
        )
        last_loop_plan = loop_plan

        anchor_scale = anchor_scheduler.strength(
            step=steps_done,
            total_steps=max(1, target_tokens),
            pressure=float(snapshot.pressure),
            confidence=float(confidence),
            difficulty=float(difficulty),
        )

        if steps_done == 0:
            target_tokens = min(int(max_new_tokens), int(plan["max_new_tokens"]))
        else:
            target_tokens = min(int(max_new_tokens), steps_done + int(plan["max_new_tokens"]))

        if steps_done >= target_tokens:
            break

        attn = torch.ones_like(generated)

        def run_forward(
            loops: Optional[List[int]],
            refine: int,
            collect: bool,
        ) -> Tuple[torch.Tensor, Optional[List[float]], List[float]]:
            return custom_gpt2_forward_looped_advanced(
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
                layer_loops=loops,
                final_refine_passes=refine,
                anchor_scale=anchor_scale,
                layer_scores=layer_scores,
                importance_scorer=importance_scorer if enable_loops else None,
                collect_importance=collect,
            )

        if enable_rejection and enable_loops:
            zero_loops = [0] * num_layers

            logits_base, _, _ = run_forward(
                loops=zero_loops,
                refine=0,
                collect=False,
            )

            logits_cand, _, delta_norms_cand = run_forward(
                loops=loop_plan["layer_loops"],
                refine=loop_plan["final_refine_passes"],
                collect=True,
            )

            conf_base = confidence_from_logits(logits_base[:, -1, :])
            conf_cand = confidence_from_logits(logits_cand[:, -1, :])

            # Latency guard: if predicted latency is too high and candidate not clearly better, reject.
            pred_lat = float(latency_pred[0].item())
            latency_guard_ok = pred_lat <= clamp01(float(budget) + 0.25)

            accept = (
                conf_cand >= conf_base - float(rejection_margin)
                and latency_guard_ok
            )

            accepted_loop = bool(accept)

            if accept:
                logits = logits_cand
                confidence = conf_cand
                delta_norms = delta_norms_cand
            else:
                logits = logits_base
                confidence = conf_base
                delta_norms = []
        else:
            logits, _, delta_norms = run_forward(
                loops=loop_plan["layer_loops"] if enable_loops else [0] * num_layers,
                refine=loop_plan["final_refine_passes"] if enable_loops else 0,
                collect=bool(enable_loops),
            )
            confidence = confidence_from_logits(logits[:, -1, :])
            accepted_loop = True

        last_logits = logits[:, -1, :]
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

    actual_latency_ms = max(0.0, (time.perf_counter() - req_start_perf) * 1000.0)

    plan_out = dict(last_plan)
    plan_out["generated_tokens"] = steps_done
    plan_out["confidence"] = confidence
    plan_out["loop_plan"] = last_loop_plan
    plan_out["anchor_scale"] = anchor_scale
    plan_out["accepted_loop"] = accepted_loop
    plan_out["prompt_info"] = prompt_info
    plan_out["actual_latency_ms"] = actual_latency_ms
    plan_out["delta_norms"] = delta_norms

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
        "per_layer_times": delta_norms,
        "cuda_cycle_info": cycle_info,
        "loop_plan": last_loop_plan,
        "prompt_info": prompt_info,
        "actual_latency_ms": actual_latency_ms,
        "accepted_loop": accepted_loop,
    }

    feedback_record = None

    if latency_feedback_buffer is not None and last_internal_features_list is not None:
        if torch.is_tensor(timestamp):
            ts_float = float(timestamp.detach().cpu().reshape(-1)[0].item())
        elif timestamp is None:
            ts_float = time.time()
        else:
            ts_float = float(timestamp)

        feedback_record = latency_feedback_buffer.add(
            timestamp=ts_float,
            budget=float(budget),
            internal_features=last_internal_features_list,
            actual_latency_ms=actual_latency_ms,
            generated_tokens=steps_done,
            confidence=confidence,
            quality=None,
        )

    if traffic_monitor is not None and request_id is not None:
        try:
            traffic_monitor.request_end(
                request_id=request_id,
                generated_tokens=steps_done,
                success=True,
            )
        except Exception:
            pass

    return text, plan_out, last_snapshot, metrics, feedback_record


# ============================================================
# Save / Load Advanced Loop System
# ============================================================

def save_advanced_loop_v3(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    refiner: AdvancedLayerRefiner,
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


def load_advanced_loop_v3(
    plugin: TMUPlugin,
    heads: AdvancedLoopHeadsV3,
    adapter: AdvancedTrafficAdapter,
    refiner: AdvancedLayerRefiner,
    path: str,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)

    load_tmu_state_dict(plugin.tmu, ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    refiner.load_state_dict(ckpt["refiner_state_dict"])

    return plugin, heads, adapter, refiner
