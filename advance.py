# gpt2_tmu_advanced.py
#
# Advanced TMU v2 for GPT-2
#
# Requires previous file:
#   gpt2_tmu_full.py
#
# Optional true CUDA cycle timer:
#   cuda_cycle_timer.py
#
# Features:
# - External human/world time TMU
# - Internal AI/GPU clock: ms/us/cycles
# - Per-layer timing
# - Traffic/QoS monitor
# - GPU thermal/power/memory monitor
# - Actual latency-label training
# - Layer-fraction + token-fraction heads
# - Cache-safe static layer skipping
# - Dynamic no-cache layer skipping
# - Confidence/entropy early stop
# - Optional CUDA clock64/globaltimer

import os
import json
import math
import time
import csv
import threading
import statistics
import subprocess
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

# Previous TMU core
from gpt2_tmu_full import (
    TimeLike,
    TimeManagementUnit,
    TMUPlugin,
    TmuCapabilityHeads,
    PaceConfig,
    PaceController,
    cyclic_time_features,
    freeze_base_and_enable_tmu,
    custom_gpt2_forward_with_tmu,
    _make_causal_padding_mask,
)

# Optional true CUDA cycle timer
try:
    from cuda_cycle_timer import CudaCycleTimer
except Exception:
    CudaCycleTimer = None


ADV_INTERNAL_DIM = 24


# ============================================================
# Helpers
# ============================================================

def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _log_norm(x: float, scale: float) -> float:
    x = max(0.0, float(x))
    scale = max(1e-9, float(scale))
    return _clamp01(math.log1p(x) / math.log1p(scale))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    b = float(b)
    if abs(b) < 1e-12:
        return float(default)
    return float(a) / b


def _float_or_none(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if not s or s.upper() == "N/A":
            return None
        v = float(s)
        return v if v >= 0 else None
    except Exception:
        return None


def _cuda_device_index(device: Union[str, torch.device]) -> int:
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


# ============================================================
# GPU System Monitor: clock, temp, power, memory, utilization
# ============================================================

class SystemMonitor:
    """
    nvidia-smi se GPU health signals leta hai.

    TTL cache use karta hai taaki har tick par subprocess na chale.
    """

    FIELDS = (
        "clocks.sm",
        "clocks.max.sm",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
    )

    def __init__(
        self,
        device: Union[str, torch.device],
        ttl_s: float = 0.5,
        enable: bool = True,
    ):
        self.device = device
        self.ttl_s = float(ttl_s)
        self.enable = bool(enable) and str(device).startswith("cuda") and torch.cuda.is_available()
        self.lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time = 0.0
        self.device_index = _cuda_device_index(device) if self.enable else 0

    def _query(self) -> Dict[str, Optional[float]]:
        try:
            cmd = [
                "nvidia-smi",
                f"--query-gpu={','.join(self.FIELDS)}",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2)
            lines = out.decode().strip().splitlines()
            if self.device_index >= len(lines):
                self.device_index = 0
            vals = [v.strip() for v in lines[self.device_index].split(",")]
            d = {k: _float_or_none(v) for k, v in zip(self.FIELDS, vals)}
            return d
        except Exception:
            return {k: None for k in self.FIELDS}

    def snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()

        with self.lock:
            if self._cache is not None and (now - self._cache_time) < self.ttl_s:
                return dict(self._cache)

            if not self.enable:
                snap = self._default_snapshot()
            else:
                q = self._query()
                snap = self._build_snapshot(q)

            self._cache = snap
            self._cache_time = now
            return dict(snap)

    def _default_snapshot(self) -> Dict[str, Any]:
        sm_hz = 1_400_000_000.0
        max_hz = 2_100_000_000.0
        return {
            "sm_clock_hz": sm_hz,
            "max_sm_clock_hz": max_hz,
            "freq_norm": _clamp01(sm_hz / max_hz),
            "throttle_risk": 0.0,

            "temperature_c": 45.0,
            "temp_norm": _clamp01(45.0 / 100.0),

            "power_draw_w": 80.0,
            "power_limit_w": 300.0,
            "power_norm": _clamp01(80.0 / 300.0),

            "memory_used_mb": 1024.0,
            "memory_total_mb": 16384.0,
            "mem_pressure": _clamp01(1024.0 / 16384.0),

            "gpu_util_pct": 0.0,
            "gpu_util_norm": 0.0,
            "mem_util_pct": 0.0,
            "mem_util_norm": 0.0,
        }

    def _build_snapshot(self, q: Dict[str, Optional[float]]) -> Dict[str, Any]:
        sm_mhz = q.get("clocks.sm")
        max_mhz = q.get("clocks.max.sm")
        temp = q.get("temperature.gpu")
        pdraw = q.get("power.draw")
        plimit = q.get("power.limit")
        mused = q.get("memory.used")
        mtot = q.get("memory.total")
        gutil = q.get("utilization.gpu")
        mutil = q.get("utilization.memory")

        sm_hz = (sm_mhz * 1e6) if sm_mhz else 1_400_000_000.0
        max_hz = (max_mhz * 1e6) if max_mhz else max(sm_hz, 2_100_000_000.0)
        max_hz = max(max_hz, sm_hz, 1.0)

        freq_norm = _clamp01(sm_hz / max_hz)
        throttle_risk = _clamp01(1.0 - freq_norm)

        temp = temp if temp is not None else 45.0
        pdraw = pdraw if pdraw is not None else 80.0
        plimit = plimit if plimit is not None else 300.0
        mused = mused if mused is not None else 1024.0
        mtot = mtot if mtot is not None else 16384.0
        gutil = gutil if gutil is not None else 0.0
        mutil = mutil if mutil is not None else 0.0

        return {
            "sm_clock_hz": float(sm_hz),
            "max_sm_clock_hz": float(max_hz),
            "freq_norm": float(freq_norm),
            "throttle_risk": float(throttle_risk),

            "temperature_c": float(temp),
            "temp_norm": _clamp01(temp / 100.0),

            "power_draw_w": float(pdraw),
            "power_limit_w": float(plimit),
            "power_norm": _clamp01(_safe_div(pdraw, plimit, 0.25)),

            "memory_used_mb": float(mused),
            "memory_total_mb": float(mtot),
            "mem_pressure": _clamp01(_safe_div(mused, mtot, 0.25)),

            "gpu_util_pct": float(gutil),
            "gpu_util_norm": _clamp01(gutil / 100.0),
            "mem_util_pct": float(mutil),
            "mem_util_norm": _clamp01(mutil / 100.0),
        }


# ============================================================
# Traffic / QoS Monitor
# ============================================================

@dataclass
class RequestState:
    request_id: str
    start_perf: float
    prompt_tokens: int = 0
    priority: float = 0.5
    deadline_s: Optional[float] = None


class TrafficMonitor:
    """
    Multi-request traffic signals:
    - active requests
    - queue delay estimate
    - throughput tokens/s
    - p95 latency
    - error rate
    - deadline pressure
    """

    def __init__(
        self,
        window_s: float = 60.0,
        concurrency_limit: int = 4,
        max_history: int = 10000,
    ):
        self.window_s = float(window_s)
        self.concurrency_limit = int(max(1, concurrency_limit))
        self.max_history = int(max_history)

        self.lock = threading.Lock()
        self.active: Dict[str, RequestState] = {}
        self.completed: deque = deque(maxlen=self.max_history)

    def request_start(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        priority: float = 0.5,
        deadline_s: Optional[float] = None,
    ) -> RequestState:
        state = RequestState(
            request_id=str(request_id),
            start_perf=time.perf_counter(),
            prompt_tokens=int(prompt_tokens),
            priority=_clamp01(priority),
            deadline_s=float(deadline_s) if deadline_s is not None else None,
        )
        with self.lock:
            self.active[state.request_id] = state
        return state

    def request_end(
        self,
        request_id: str,
        generated_tokens: int = 0,
        success: bool = True,
    ) -> Optional[Dict[str, Any]]:
        rid = str(request_id)
        now = time.perf_counter()

        with self.lock:
            state = self.active.pop(rid, None)
            if state is None:
                return None

            latency_ms = max(0.0, (now - state.start_perf) * 1000.0)
            record = (now, latency_ms, int(generated_tokens), bool(success))
            self.completed.append(record)

            # Trim old records
            cutoff = now - self.window_s
            while self.completed and self.completed[0][0] < cutoff:
                self.completed.popleft()

        return {
            "request_id": rid,
            "latency_ms": latency_ms,
            "generated_tokens": int(generated_tokens),
            "success": bool(success),
        }

    def snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        cutoff = now - self.window_s

        with self.lock:
            # Trim
            while self.completed and self.completed[0][0] < cutoff:
                self.completed.popleft()

            active_count = len(self.active)
            records = list(self.completed)

        latencies = sorted([r[1] for r in records]) if records else []
        tokens = sum([r[2] for r in records]) if records else 0
        errors = sum([0 if r[3] else 1 for r in records]) if records else 0

        throughput = _safe_div(tokens, self.window_s, 0.0)
        p95 = 0.0
        if latencies:
            idx = min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
            p95 = float(latencies[idx])

        error_rate = _safe_div(errors, len(records), 0.0)

        # Queue delay estimate: agar active > limit, extra requests wait karenge
        queue_delay_s = max(0.0, float(active_count - self.concurrency_limit)) * max(0.02, p95 / 1000.0)

        # Deadline pressure: sabse urgent active request
        deadline_pressure = 0.0
        for st in self.active.values():
            if st.deadline_s is None or st.deadline_s <= 1e-6:
                continue
            elapsed = now - st.start_perf
            d = float(st.deadline_s)
            # Last 30% deadline me pressure badhao
            if elapsed <= 0.7 * d:
                p = 0.0
            else:
                p = _clamp01((elapsed - 0.7 * d) / max(1e-6, 0.3 * d))
            deadline_pressure = max(deadline_pressure, p)

        return {
            "active_requests": int(active_count),
            "active_norm": _clamp01(active_count / float(self.concurrency_limit)),
            "queue_delay_s": float(queue_delay_s),
            "queue_delay_norm": _clamp01(queue_delay_s / 1.0),
            "throughput_tok_s": float(throughput),
            "throughput_norm": _log_norm(throughput, 1000.0),
            "p95_latency_ms": float(p95),
            "p95_norm": _clamp01(p95 / 1000.0),
            "error_rate": _clamp01(error_rate),
            "deadline_pressure": _clamp01(deadline_pressure),
            "completed_in_window": int(len(records)),
        }


# ============================================================
# Per-Layer CUDA Event Timer
# ============================================================

class PerLayerTimer:
    """
    Har GPT-2 block ka CUDA-event based latency measure karta hai.

    Note:
    - collect() synchronize karta hai, isliye slow ho sakta hai.
    - Debug/experiment ke liye use karo, production fast path me off rakho.
    """

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


# ============================================================
# Advanced Internal Clock Snapshot
# ============================================================

@dataclass
class AdvancedClockSnapshot:
    wall_s: float
    elapsed_s: float
    delta_s: float
    delta_ms: float
    delta_us: float

    gpu_delta_ms: Optional[float]
    gpu_delta_us: Optional[float]

    sm_clock_hz: Optional[float]
    max_sm_clock_hz: Optional[float]
    cycles: Optional[float]

    step: int
    step_norm: float
    tokens: int
    tokens_per_s: float

    budget_s: Optional[float]
    remaining_s: Optional[float]
    pressure: float

    prompt_difficulty: float
    confidence: float

    per_layer_times: List[float]
    per_layer_mean_norm: float
    per_layer_p95_norm: float
    per_layer_var_norm: float

    system: Dict[str, Any]
    traffic: Dict[str, Any]

    feature_list: List[float] = field(default_factory=list)


class AdvancedInternalClock:
    """
    AI internal clock:
    - CPU nanosecond timer
    - optional CUDA event total timing
    - GPU SM clock / cycles estimate
    - system thermal/power/memory
    - traffic/QoS
    - per-layer timing summary
    - prompt difficulty / confidence
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        max_steps: int = 64,
        budget_s: Optional[float] = None,
        use_gpu_events: bool = False,
        system_monitor: Optional[SystemMonitor] = None,
        traffic_monitor: Optional[TrafficMonitor] = None,
    ):
        self.device = device
        self.max_steps = int(max_steps)
        self.budget_s = float(budget_s) if budget_s is not None else None
        self.use_gpu_events = bool(use_gpu_events) and str(device).startswith("cuda") and torch.cuda.is_available()

        self.system_monitor = system_monitor
        self.traffic_monitor = traffic_monitor

        self.origin_ns = time.perf_counter_ns()
        self.last_ns = self.origin_ns
        self.wall_origin = time.time()

        self.step = 0
        self.tokens = 0
        self.spent_s = 0.0

        if self.use_gpu_events:
            self.event_prev = torch.cuda.Event(enable_timing=True)
            self.event_curr = torch.cuda.Event(enable_timing=True)
            self.event_prev.record()
        else:
            self.event_prev = None
            self.event_curr = None

    def tick(
        self,
        tokens: int = 0,
        prompt_difficulty: float = 0.0,
        confidence: float = 0.5,
        per_layer_times: Optional[Sequence[float]] = None,
        sync: Optional[bool] = None,
    ) -> AdvancedClockSnapshot:
        now_ns = time.perf_counter_ns()

        delta_s = (now_ns - self.last_ns) / 1e9
        elapsed_s = (now_ns - self.origin_ns) / 1e9
        wall_s = time.time() - self.wall_origin

        self.last_ns = now_ns
        self.tokens += int(tokens)
        self.step += 1

        gpu_delta_ms = None
        should_sync = self.use_gpu_events if sync is None else bool(sync)

        if self.use_gpu_events and should_sync:
            self.event_curr.record()
            torch.cuda.synchronize()
            gpu_delta_ms = float(self.event_prev.elapsed_time(self.event_curr))

            self.event_prev, self.event_curr = self.event_curr, self.event_prev
            self.event_curr.record()

        delta_ms = float(gpu_delta_ms) if gpu_delta_ms is not None else delta_s * 1000.0
        delta_us = delta_ms * 1000.0

        gpu_delta_us = None
        if gpu_delta_ms is not None:
            gpu_delta_us = gpu_delta_ms * 1000.0

        sysm = self.system_monitor.snapshot() if self.system_monitor is not None else {
            "sm_clock_hz": 1_400_000_000.0,
            "max_sm_clock_hz": 2_100_000_000.0,
            "freq_norm": 0.67,
            "throttle_risk": 0.33,
            "temp_norm": 0.45,
            "power_norm": 0.25,
            "mem_pressure": 0.25,
            "gpu_util_norm": 0.0,
        }

        traf = self.traffic_monitor.snapshot() if self.traffic_monitor is not None else {
            "active_requests": 0,
            "active_norm": 0.0,
            "queue_delay_norm": 0.0,
            "throughput_norm": 0.0,
            "p95_norm": 0.0,
            "error_rate": 0.0,
            "deadline_pressure": 0.0,
        }

        sm_hz = sysm.get("sm_clock_hz")
        cycles = delta_s * float(sm_hz) if sm_hz is not None else None

        tokens_per_s = _safe_div(float(tokens), delta_s, 0.0) if delta_s > 1e-9 else 0.0

        if self.budget_s is not None and self.budget_s > 0:
            self.spent_s += delta_s
            remaining_s = max(0.0, self.budget_s - self.spent_s)
            budget_pressure = _clamp01(self.spent_s / self.budget_s)
        else:
            remaining_s = None
            budget_pressure = 0.0

        latency_pressure = _clamp01(delta_ms / 100.0)
        pressure = max(budget_pressure, float(traf.get("deadline_pressure", 0.0)), latency_pressure)

        step_norm = _clamp01(float(self.step) / float(max(1, self.max_steps)))

        # Per-layer stats
        pl_times = list(per_layer_times) if per_layer_times else []
        if pl_times:
            mean_ms = float(statistics.mean(pl_times))
            sorted_times = sorted(pl_times)
            p95_ms = float(sorted_times[min(len(sorted_times) - 1, int(0.95 * (len(sorted_times) - 1)))])
            var = float(statistics.pvariance(pl_times)) if len(pl_times) > 1 else 0.0
            std = math.sqrt(max(0.0, var))
        else:
            mean_ms = 0.0
            p95_ms = 0.0
            std = 0.0

        per_layer_mean_norm = _clamp01(mean_ms / 50.0)
        per_layer_p95_norm = _clamp01(p95_ms / 100.0)
        per_layer_var_norm = _clamp01(std / 50.0)

        feature_list = [
            _log_norm(elapsed_s, 3600.0),             # 0 session age
            _log_norm(delta_s, 1.0),                  # 1 gap since last call
            _clamp01(delta_ms / 100.0),               # 2 latency ms
            _clamp01((gpu_delta_us or delta_us) / 1000.0), # 3 us latency
            _log_norm(cycles or 0.0, 1e9),            # 4 estimated GPU cycles
            _clamp01(sysm.get("freq_norm", 0.5)),     # 5 GPU clock health
            _clamp01(step_norm),                      # 6 generation progress
            _clamp01(pressure),                       # 7 budget/latency/deadline pressure

            _clamp01(traf.get("queue_delay_norm", 0.0)),  # 8 traffic queue
            _clamp01(traf.get("active_norm", 0.0)),       # 9 active requests
            _clamp01(traf.get("throughput_norm", 0.0)),   # 10 throughput
            _clamp01(traf.get("p95_norm", 0.0)),          # 11 p95 latency
            _clamp01(traf.get("error_rate", 0.0)),        # 12 error rate
            _clamp01(traf.get("deadline_pressure", 0.0)), # 13 deadline pressure

            _clamp01(sysm.get("gpu_util_norm", 0.0)),     # 14 GPU utilization
            _clamp01(sysm.get("mem_pressure", 0.0)),      # 15 VRAM pressure
            _clamp01(sysm.get("temp_norm", 0.0)),         # 16 temperature
            _clamp01(sysm.get("power_norm", 0.0)),        # 17 power draw

            _clamp01(sysm.get("throttle_risk", 0.0)),     # 18 throttle risk
            _clamp01(prompt_difficulty),                  # 19 prompt difficulty
            _clamp01(confidence),                         # 20 confidence
            per_layer_mean_norm,                          # 21 per-layer mean
            per_layer_p95_norm,                           # 22 per-layer p95
            per_layer_var_norm,                           # 23 per-layer variance
        ]

        return AdvancedClockSnapshot(
            wall_s=wall_s,
            elapsed_s=elapsed_s,
            delta_s=delta_s,
            delta_ms=delta_ms,
            delta_us=delta_us,
            gpu_delta_ms=gpu_delta_ms,
            gpu_delta_us=gpu_delta_us,
            sm_clock_hz=sm_hz,
            max_sm_clock_hz=sysm.get("max_sm_clock_hz"),
            cycles=cycles,
            step=self.step,
            step_norm=step_norm,
            tokens=self.tokens,
            tokens_per_s=tokens_per_s,
            budget_s=self.budget_s,
            remaining_s=remaining_s,
            pressure=pressure,
            prompt_difficulty=_clamp01(prompt_difficulty),
            confidence=_clamp01(confidence),
            per_layer_times=pl_times,
            per_layer_mean_norm=per_layer_mean_norm,
            per_layer_p95_norm=per_layer_p95_norm,
            per_layer_var_norm=per_layer_var_norm,
            system=sysm,
            traffic=traf,
            feature_list=feature_list,
        )


def advanced_snapshot_to_tensor(
    snapshot: AdvancedClockSnapshot,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    t = torch.tensor(snapshot.feature_list, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).expand(int(batch_size), -1).contiguous()
    return t


# ============================================================
# Advanced Internal Adapter
# ============================================================

class AdvancedInternalAdapter(nn.Module):
    """
    24-D internal clock features -> hidden_dim embedding.
    Final layer zero-initialized, isliye shuru me no-op.
    """

    def __init__(self, hidden_dim: int, internal_dim: int = ADV_INTERNAL_DIM):
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
# Advanced Capability Heads: rhythm + pace + layers + tokens
# ============================================================

class AdvancedTmuCapabilityHeads(nn.Module):
    """
    Heads:
    - rhythm_head: future cyclic time features
    - pace_head: overall thinking speed 0..1
    - layer_head: desired active layer fraction 0..1
    - token_head: desired token fraction 0..1
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 12,
        max_tokens: int = 64,
        budget_dim: int = 1,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_tokens = int(max_tokens)
        self.budget_dim = int(budget_dim)

        self.budget_proj = nn.Linear(self.budget_dim, self.hidden_dim)
        nn.init.zeros_(self.budget_proj.weight)
        nn.init.zeros_(self.budget_proj.bias)

        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, TimeManagementUnit.FEATURE_DIM),
        )

        pace_hidden = max(8, self.hidden_dim // 2)

        self.pace_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        self.layer_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        self.token_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        # Neutral init
        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

        nn.init.zeros_(self.layer_head[-1].weight)
        nn.init.constant_(self.layer_head[-1].bias, 0.0)

        nn.init.zeros_(self.token_head[-1].weight)
        nn.init.constant_(self.token_head[-1].bias, 0.0)

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

        pace = torch.sigmoid(pace_logit)
        layer_frac = torch.sigmoid(layer_logit)
        token_frac = torch.sigmoid(token_logit)

        return rhythm_pred, pace, layer_frac, token_frac, pace_logit


# ============================================================
# Prompt Difficulty Estimator
# ============================================================

def estimate_text_difficulty(text: str) -> float:
    """
    Simple heuristic difficulty score 0..1.
    Real system me isko embedding/logprob based bana sakte ho.
    """
    if not text:
        return 0.0

    t = str(text)
    words = t.split()
    chars = len(t)

    score = 0.0

    # Length
    score += min(0.30, chars / 1000.0)
    score += min(0.15, len(words) / 150.0)

    # Question / instruction complexity
    if "?" in t:
        score += 0.08
    if any(k in t.lower() for k in ("why", "how", "explain", "compare", "analyze", "reason")):
        score += 0.10

    # Numbers/code/math
    if any(c.isdigit() for c in t):
        score += 0.06
    if any(c in t for c in "{}[]()<>=+-*/\\"):
        score += 0.10

    # Vocabulary richness
    letters = [c.lower() for c in t if c.isalpha()]
    if letters:
        uniq_ratio = len(set(letters)) / max(1, len(letters))
        score += 0.08 * _clamp01(uniq_ratio)

    return _clamp01(score)


# ============================================================
# Advanced Data Loading with Latency Labels
# ============================================================

def _get_first_field(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]

    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        kk = k.lower()
        if kk in lower and lower[kk] not in (None, ""):
            return lower[kk]

    return None


def load_advanced_records(path: str) -> Tuple[List[str], List[float], List[float], List[Dict[str, Any]]]:
    """
    Loads JSONL/JSON/CSV with optional latency labels.

    Required:
      text, timestamp

    Optional:
      budget, latency_ms, tokens_used, desired_layers, desired_pace, quality
    """
    ext = os.path.splitext(path)[1].lower()
    raw: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("records") or data.get("data") or []
        if not isinstance(data, list):
            raise ValueError("JSON me list of records ya {'records': [...]} chahiye.")
        raw = data

    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
                raw.append(row)

    else:
        raise ValueError("Sirf .jsonl, .json, .csv supported hain.")

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []
    metas: List[Dict[str, Any]] = []

    # Import timestamp parser from previous file indirectly
    from gpt2_tmu_full import parse_timestamp

    for rec in raw:
        if not isinstance(rec, dict):
            continue

        text = _get_first_field(rec, ["text", "content", "prompt", "message", "sentence"])
        ts = _get_first_field(rec, ["timestamp", "time", "datetime", "date", "created_at"])

        if text is None or ts is None:
            continue

        b = _get_first_field(rec, ["budget", "time_budget", "budget_norm"])
        b = 0.5 if b is None else float(b)
        b = min(1.0, max(0.05, b))

        meta = {
            "latency_ms": _get_first_field(rec, ["latency_ms", "response_time_ms", "latency"]),
            "tokens_used": _get_first_field(rec, ["tokens_used", "generated_tokens", "completion_tokens"]),
            "desired_layers": _get_first_field(rec, ["desired_layers", "active_layers", "layers"]),
            "desired_pace": _get_first_field(rec, ["desired_pace", "pace", "speed"]),
            "quality": _get_first_field(rec, ["quality", "score", "reward"]),
        }

        texts.append(str(text))
        timestamps.append(parse_timestamp(ts))
        budgets.append(b)
        metas.append(meta)

    if not texts:
        raise ValueError("Koi valid record nahi mila.")

    return texts, timestamps, budgets, metas


def build_advanced_demo_records(
    num_samples: int = 192,
    num_layers: int = 12,
    base_time: float = 1_700_000_000.0,
    seed: int = 42,
) -> Tuple[List[str], List[float], List[float], List[Dict[str, Any]]]:
    """
    Demo records with synthetic latency labels.
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
        "Explain why this code fails and how to fix it.",
        "Compare morning and night productivity in detail.",
        "Analyze traffic patterns and suggest a route.",
        "Give a short answer only.",
    ]

    g = torch.Generator()
    g.manual_seed(int(seed))

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []
    metas: List[Dict[str, Any]] = []

    t = float(base_time)

    for i in range(int(num_samples)):
        text = base_texts[i % len(base_texts)] + f" Record {i}."

        if i > 0:
            delta = torch.rand(1, generator=g).item() * 3600.0 + 10.0
            t += delta

        budget = torch.rand(1, generator=g).item() * 0.9 + 0.1
        difficulty = estimate_text_difficulty(text)

        # Synthetic desired layers/tokens/latency
        base_layer_frac = 0.25 + 0.55 * budget + 0.20 * difficulty
        base_layer_frac = _clamp01(base_layer_frac)
        desired_layers = int(round(2 + base_layer_frac * max(0, num_layers - 2)))
        desired_layers = max(1, min(num_layers, desired_layers))

        token_frac = _clamp01(0.15 + 0.70 * budget + 0.15 * difficulty)
        tokens_used = int(round(4 + token_frac * 60))

        latency_ms = 20.0 + tokens_used * 8.0 + torch.rand(1, generator=g).item() * 120.0
        quality = _clamp01(0.55 + 0.25 * budget + 0.20 * (desired_layers / max(1, num_layers)))

        texts.append(text)
        timestamps.append(t)
        budgets.append(budget)
        metas.append({
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "desired_layers": desired_layers,
            "desired_pace": desired_layers / float(num_layers),
            "quality": quality,
        })

    return texts, timestamps, budgets, metas


# ============================================================
# Advanced Dataset
# ============================================================

class AdvancedTimestampedDataset(Dataset):
    """
    GPT-2 dataset with:
    - timestamp
    - budget
    - optional latency labels
    - difficulty
    - rhythm target
    - pace/layer/token targets
    """

    def __init__(
        self,
        texts: Sequence[str],
        timestamps: Sequence[float],
        budgets: Sequence[float],
        metas: Sequence[Dict[str, Any]],
        tokenizer: Any,
        seq_len: int = 64,
        num_layers: int = 12,
        max_label_tokens: int = 64,
    ):
        super().__init__()

        if not (len(texts) == len(timestamps) == len(budgets) == len(metas)):
            raise ValueError("texts, timestamps, budgets, metas ki length same honi chahiye.")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.num_layers = int(num_layers)
        self.max_label_tokens = int(max_label_tokens)
        self.samples: List[Dict[str, Any]] = []

        for text, t, b, meta in zip(texts, timestamps, budgets, metas):
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

            y_rhythm = cyclic_time_features(
                torch.tensor([float(t) + 600.0], dtype=torch.float64)
            ).squeeze(0)

            difficulty = estimate_text_difficulty(text)

            # Target pace/layers/tokens
            desired_pace = meta.get("desired_pace")
            desired_layers = meta.get("desired_layers")
            tokens_used = meta.get("tokens_used")

            if desired_pace is not None:
                target_pace = _clamp01(float(desired_pace))
            elif desired_layers is not None:
                target_pace = _clamp01(float(desired_layers) / float(self.num_layers))
            elif tokens_used is not None:
                target_pace = _clamp01(float(tokens_used) / float(self.max_label_tokens))
            else:
                # Heuristic
                valid_len = int(attention_mask.sum().item())
                length_diff = min(1.0, max(0.0, valid_len / float(max(1, self.seq_len))))
                day_phase = (float(t) % 86400.0) / 86400.0
                time_need = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * day_phase))
                target_pace = _clamp01(0.5 * length_diff + 0.5 * time_need)

            target_pace = min(target_pace, float(b))

            if desired_layers is not None:
                y_layers = _clamp01(float(desired_layers) / float(self.num_layers))
            else:
                y_layers = target_pace

            if tokens_used is not None:
                y_tokens = _clamp01(float(tokens_used) / float(self.max_label_tokens))
            else:
                y_tokens = target_pace

            self.samples.append({
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "t": t_tensor,
                "budget": budget_tensor,
                "y_rhythm": y_rhythm,
                "y_pace": torch.tensor(target_pace, dtype=torch.float32),
                "y_layers": torch.tensor(y_layers, dtype=torch.float32),
                "y_tokens": torch.tensor(y_tokens, dtype=torch.float32),
                "difficulty": torch.tensor(difficulty, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Advanced Training
# ============================================================

def _prepare_budget_tensor(
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


def train_advanced_tmu(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
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
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    use_gpu_events: bool = False,
    budget_s: Optional[float] = None,
    verbose: bool = True,
):
    """
    Trains:
    - TMU
    - Advanced heads
    - Internal adapter

    Base GPT-2 remains frozen.
    """
    plugin.to(device)
    heads.to(device)
    adapter.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    if clock is None:
        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=budget_s,
            use_gpu_events=use_gpu_events,
            system_monitor=SystemMonitor(device, enable=True),
            traffic_monitor=TrafficMonitor(),
        )

    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = (
        list(tmu_params)
        + list(heads.parameters())
        + list(adapter.parameters())
    )

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    history = []

    for epoch in range(int(epochs)):
        plugin.base_model.eval()
        plugin.tmu.train()
        heads.train()
        adapter.train()

        totals = {
            "loss": 0.0,
            "lm": 0.0,
            "rhythm": 0.0,
            "pace": 0.0,
            "layer": 0.0,
            "token": 0.0,
            "over": 0.0,
            "delta_us": 0.0,
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

            external_embed = plugin.tmu.encode_time(
                time_stamp=t,
                batch_size=B,
                device=device,
            )

            internal_embed = adapter(internal_features)
            time_embed = external_embed + internal_embed

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

            rhythm_pred, pace_pred, layer_pred, token_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace = F.binary_cross_entropy(pace_safe, y_pace)

            layer_safe = layer_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_layer = F.binary_cross_entropy(layer_safe, y_layers)

            token_safe = token_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_token = F.binary_cross_entropy(token_safe, y_tokens)

            over_budget = F.relu(pace_pred - budget).mean()
            pace_loss_total = loss_pace + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * pace_loss_total
                + float(w_layer) * loss_layer
                + float(w_token) * loss_token
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
            totals["over"] += float(over_budget.item())
            totals["delta_us"] += float(snapshot.delta_us)
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
                f"over={avg['over']:.4f} | "
                f"delta_us={avg['delta_us']:.2f}"
            )

    return history


# ============================================================
# Advanced Pace Controller
# ============================================================

@dataclass
class AdvancedPaceConfig:
    min_layers: int = 2
    max_layers: int = 12
    min_tokens: int = 4
    max_tokens: int = 64
    min_temperature: float = 0.6
    max_temperature: float = 1.0
    refine_min: int = 1
    refine_max: int = 3

    traffic_penalty_strength: float = 1.0
    thermal_penalty_strength: float = 1.0
    memory_limit: float = 0.88
    confidence_stop: float = 0.93
    min_steps_before_stop: int = 4


class AdvancedPaceController:
    """
    Converts predicted pace/layer/token fractions into execution plan.
    Applies traffic, thermal, memory, difficulty policies.
    """

    def __init__(self, num_layers: int, config: Optional[AdvancedPaceConfig] = None):
        self.num_layers = int(num_layers)
        self.config = config or AdvancedPaceConfig()

        self.config.max_layers = min(int(self.config.max_layers), self.num_layers)
        self.config.min_layers = max(1, min(int(self.config.min_layers), self.config.max_layers))

    def from_outputs(
        self,
        pace: float,
        layer_frac: float,
        token_frac: float,
        budget: float,
        traffic: Dict[str, Any],
        system: Dict[str, Any],
        difficulty: float,
    ) -> Dict[str, Any]:
        cfg = self.config

        pace = _clamp01(pace)
        layer_frac = _clamp01(layer_frac)
        token_frac = _clamp01(token_frac)
        budget = _clamp01(budget)
        difficulty = _clamp01(difficulty)

        active_layers = int(round(cfg.min_layers + layer_frac * (cfg.max_layers - cfg.min_layers)))
        max_new_tokens = int(round(cfg.min_tokens + token_frac * (cfg.max_tokens - cfg.min_tokens)))

        # Hard budget cap
        token_budget = int(round(cfg.min_tokens + budget * (cfg.max_tokens - cfg.min_tokens)))
        max_new_tokens = max(1, min(max_new_tokens, token_budget))

        # Traffic load
        traffic_load = (
            0.25 * float(traffic.get("queue_delay_norm", 0.0))
            + 0.20 * float(traffic.get("active_norm", 0.0))
            + 0.30 * float(traffic.get("deadline_pressure", 0.0))
            + 0.15 * float(traffic.get("p95_norm", 0.0))
            + 0.10 * float(traffic.get("error_rate", 0.0))
        )
        traffic_load = _clamp01(traffic_load)

        # Thermal/power load
        thermal_load = (
            0.35 * float(system.get("temp_norm", 0.0))
            + 0.25 * float(system.get("power_norm", 0.0))
            + 0.25 * float(system.get("throttle_risk", 0.0))
            + 0.15 * float(system.get("gpu_util_norm", 0.0))
        )
        thermal_load = _clamp01(thermal_load)

        load = _clamp01(
            float(cfg.traffic_penalty_strength) * traffic_load
            + float(cfg.thermal_penalty_strength) * thermal_load
        )

        multiplier = max(0.30, 1.0 - 0.65 * load)

        active_layers = int(max(1, min(self.num_layers, round(active_layers * multiplier))))
        max_new_tokens = int(max(1, round(max_new_tokens * multiplier)))

        # Difficulty adjustment: difficult prompts get slightly more layers if budget allows
        diff_adj = int(round(2.0 * (difficulty - 0.5) * budget))
        active_layers = max(1, min(self.num_layers, active_layers + diff_adj))

        # Memory pressure token cap
        mem_pressure = float(system.get("mem_pressure", 0.0))
        if mem_pressure > cfg.memory_limit:
            excess = _clamp01((mem_pressure - cfg.memory_limit) / max(1e-6, 1.0 - cfg.memory_limit))
            max_new_tokens = max(1, int(max_new_tokens * (1.0 - 0.70 * excess)))

        temperature = cfg.min_temperature + pace * (cfg.max_temperature - cfg.min_temperature)
        temperature *= max(0.75, 1.0 - 0.20 * load)
        temperature = max(0.2, min(2.0, temperature))

        refinement_passes = int(round(cfg.refine_min + pace * (cfg.refine_max - cfg.refine_min)))
        refinement_passes = max(1, min(8, refinement_passes))

        return {
            "pace": pace,
            "layer_frac": layer_frac,
            "token_frac": token_frac,
            "active_layers": active_layers,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "refinement_passes": refinement_passes,
            "traffic_load": traffic_load,
            "thermal_load": thermal_load,
            "load_multiplier": multiplier,
            "difficulty": difficulty,
            "budget": budget,
        }


# ============================================================
# Predict advanced pace/time embed
# ============================================================

def predict_advanced_time_embed(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
    timestamp: TimeLike,
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
    internal_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        pace, layer_frac, token_frac, time_embed, rhythm_pred
    """
    external_embed = plugin.tmu.encode_time(
        time_stamp=timestamp,
        batch_size=batch_size,
        device=device,
    )

    internal_embed = adapter(internal_features.to(device=device, dtype=torch.float32))
    time_embed = external_embed + internal_embed

    budget_t = _prepare_budget_tensor(budget, batch_size, device)

    rhythm_pred, pace, layer_frac, token_frac, _ = heads(time_embed, budget_t)

    return pace, layer_frac, token_frac, time_embed, rhythm_pred


# ============================================================
# Cache-safe prefix layer skipping context
# ============================================================

@contextmanager
def prefix_layer_skip_context(model: nn.Module, active_layers: int):
    """
    Temporarily reduces GPT-2 transformer.h to first active_layers blocks.

    This is cache-safe because only trailing layers are skipped.
    Mid-generation layer changes are NOT done in this mode.
    """
    original_h = model.transformer.h
    total = len(original_h)
    active_layers = max(1, min(int(active_layers), total))

    model.transformer.h = nn.ModuleList(list(original_h)[:active_layers])

    try:
        yield active_layers
    finally:
        model.transformer.h = original_h


# ============================================================
# Custom forward with per-layer timing
# ============================================================

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
    """
    No-cache custom GPT-2 forward.
    Supports true dynamic layer skipping and per-layer timing.
    """
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


# ============================================================
# Static cache-safe generation
# ============================================================

@torch.no_grad()
def generate_advanced_static(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    budget_s: Optional[float] = None,
    max_new_tokens: int = 32,
    config: Optional[AdvancedPaceConfig] = None,
    use_gpu_events: bool = False,
    system_monitor: Optional[SystemMonitor] = None,
    traffic_monitor: Optional[TrafficMonitor] = None,
    do_sample: bool = False,
    top_k: int = 50,
    seed: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], AdvancedClockSnapshot]:
    """
    Fast cache-safe generation.

    Layer skipping is fixed for the whole request.
    Internal clock/traffic/system influence initial plan only.
    """
    model = plugin.base_model
    model.eval()
    heads.eval()
    adapter.eval()

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

    snapshot = clock.tick(
        tokens=prompt_len,
        prompt_difficulty=difficulty,
        confidence=0.5,
        per_layer_times=None,
        sync=bool(use_gpu_events),
    )

    internal_features = advanced_snapshot_to_tensor(snapshot, 1, device)

    pace, layer_frac, token_frac, time_embed, _ = predict_advanced_time_embed(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        timestamp=timestamp,
        budget=budget,
        batch_size=1,
        device=device,
        internal_features=internal_features,
    )

    num_layers = len(model.transformer.h)
    controller = AdvancedPaceController(num_layers, config)

    plan = controller.from_outputs(
        pace=float(pace[0].item()),
        layer_frac=float(layer_frac[0].item()),
        token_frac=float(token_frac[0].item()),
        budget=float(budget),
        traffic=snapshot.traffic,
        system=snapshot.system,
        difficulty=difficulty,
    )

    max_new = min(int(max_new_tokens), int(plan["max_new_tokens"]))
    temp = max(1e-3, float(plan["temperature"]))

    old_current = plugin._current_time_embed
    old_cache = None

    if hasattr(model, "config"):
        old_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True

    try:
        with prefix_layer_skip_context(model, int(plan["active_layers"])):
            plugin._current_time_embed = time_embed

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

            output_ids = model.generate(**gen_kwargs)

    finally:
        plugin._current_time_embed = old_current
        if hasattr(model, "config") and old_cache is not None:
            model.config.use_cache = old_cache

    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    generated_tokens = int(output_ids.shape[1] - prompt_len)

    plan_out = dict(plan)
    plan_out["generated_tokens"] = generated_tokens
    plan_out["last_delta_ms"] = snapshot.delta_ms
    plan_out["last_delta_us"] = snapshot.delta_us
    plan_out["last_gpu_delta_us"] = snapshot.gpu_delta_us
    plan_out["last_cycles"] = snapshot.cycles
    plan_out["sm_clock_hz"] = snapshot.sm_clock_hz

    if traffic_monitor is not None and request_id is not None:
        traffic_monitor.request_end(
            request_id=request_id,
            generated_tokens=generated_tokens,
            success=True,
        )

    return text, plan_out, snapshot


# ============================================================
# Dynamic no-cache generation
# ============================================================

@torch.no_grad()
def generate_advanced_dynamic(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    budget_s: Optional[float] = None,
    max_new_tokens: int = 32,
    config: Optional[AdvancedPaceConfig] = None,
    use_gpu_events: bool = False,
    enable_per_layer_timing: bool = False,
    system_monitor: Optional[SystemMonitor] = None,
    traffic_monitor: Optional[TrafficMonitor] = None,
    cuda_cycle_timer: Optional[Any] = None,
    do_sample: bool = False,
    top_k: int = 50,
    seed: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], AdvancedClockSnapshot, Dict[str, Any]]:
    """
    True dynamic generation:
    - har step internal clock update
    - har step pace/layer/token predict
    - active_layers per step change kar sakta hai
    - per-layer timing optional
    - confidence early stop optional

    Slow hai kyunki KV cache use nahi hota.
    """
    model = plugin.base_model
    model.eval()
    heads.eval()
    adapter.eval()

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
    controller = AdvancedPaceController(num_layers, config)
    cfg = controller.config

    timer = PerLayerTimer(
        num_layers=num_layers,
        enabled=enable_per_layer_timing,
        device=device,
    )

    generated = input_ids.clone()
    steps_done = 0
    target_tokens = int(max_new_tokens)

    prev_layer_times: Optional[List[float]] = None
    confidence = 0.5
    last_plan: Dict[str, Any] = {}
    last_snapshot: Optional[AdvancedClockSnapshot] = None
    cycle_info: Optional[Dict[str, Any]] = None

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

        pace, layer_frac, token_frac, time_embed, _ = predict_advanced_time_embed(
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

        if steps_done == 0:
            target_tokens = min(int(max_new_tokens), int(plan["max_new_tokens"]))
        else:
            target_tokens = min(int(max_new_tokens), steps_done + int(plan["max_new_tokens"]))

        if steps_done >= target_tokens:
            break

        attn = torch.ones_like(generated)

        def _forward_fn():
            return custom_gpt2_forward_advanced(
                model=model,
                tmu=plugin.tmu,
                input_ids=generated,
                attention_mask=attn,
                time_embed=time_embed,
                active_layers=int(plan["active_layers"]),
                timer=timer if enable_per_layer_timing else None,
                modulate_blocks=True,
            )

        if cuda_cycle_timer is not None:
            (logits, prev_layer_times), cycle_info = cuda_cycle_timer.measure(_forward_fn)
        else:
            logits, prev_layer_times = _forward_fn()

        last_logits = logits[:, -1, :]

        # Entropy/confidence
        probs = F.softmax(last_logits, dim=-1)
        ent = float(-(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item())
        max_ent = math.log(max(2, probs.size(-1)))
        confidence = _clamp01(1.0 - ent / max_ent)

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
    }

    if traffic_monitor is not None and request_id is not None:
        traffic_monitor.request_end(
            request_id=request_id,
            generated_tokens=steps_done,
            success=True,
        )

    return text, plan_out, last_snapshot, metrics


# ============================================================
# Save / Load
# ============================================================

def save_advanced_system(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
    path: str,
) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
        },
        path,
    )


def load_advanced_system(
    plugin: TMUPlugin,
    heads: AdvancedTmuCapabilityHeads,
    adapter: AdvancedInternalAdapter,
    path: str,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)
    plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    return plugin, heads, adapter


# ============================================================
# Main demo
# ============================================================

def main_advanced():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced GPT-2 TMU v2")

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt2")

    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max-gen-tokens", type=int, default=32)
    parser.add_argument("--budget-s", type=float, default=1.0)

    parser.add_argument("--use-gpu-events", action="store_true")
    parser.add_argument("--per-layer-timing", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="Use no-cache dynamic layer skipping")
    parser.add_argument("--enable-cuda-cycles", action="store_true")

    parser.add_argument("--save", type=str, default="gpt2_tmu_advanced.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False

    hidden_dim = int(model.config.n_embd)
    num_layers = len(model.transformer.h)

    print("hidden_dim:", hidden_dim)
    print("num_layers:", num_layers)

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

    heads = AdvancedTmuCapabilityHeads(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_tokens=max(8, args.max_gen_tokens),
    )

    adapter = AdvancedInternalAdapter(hidden_dim=hidden_dim)

    if args.data:
        texts, timestamps, budgets, metas = load_advanced_records(args.data)
    else:
        texts, timestamps, budgets, metas = build_advanced_demo_records(
            num_samples=args.samples,
            num_layers=num_layers,
        )

    print("Records:", len(texts))

    dataset = AdvancedTimestampedDataset(
        texts=texts,
        timestamps=timestamps,
        budgets=budgets,
        metas=metas,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_layers=num_layers,
        max_label_tokens=max(8, args.max_gen_tokens),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    system_monitor = SystemMonitor(device, enable=str(device).startswith("cuda"))
    traffic_monitor = TrafficMonitor(window_s=60.0, concurrency_limit=4)

    if args.load:
        print("Loading advanced TMU from:", args.load)
        load_advanced_system(plugin, heads, adapter, args.load, device=device)
    else:
        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=args.budget_s,
            use_gpu_events=args.use_gpu_events,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
        )

        print("\n=== Training Advanced TMU ===")
        train_advanced_tmu(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            train_loader=loader,
            device=device,
            clock=clock,
            epochs=args.epochs,
            lr=args.lr,
            use_gpu_events=args.use_gpu_events,
            budget_s=args.budget_s,
            verbose=True,
        )

        save_advanced_system(plugin, heads, adapter, args.save)
        print("Saved:", args.save)

    pace_config = AdvancedPaceConfig(
        min_layers=2,
        max_layers=num_layers,
        min_tokens=4,
        max_tokens=max(8, args.max_gen_tokens),
        min_temperature=0.6,
        max_temperature=1.0,
    )

    cuda_timer = None
    if args.enable_cuda_cycles:
        if CudaCycleTimer is None:
            print("Warning: cuda_cycle_timer.py not available. CUDA cycles disabled.")
        else:
            try:
                cuda_timer = CudaCycleTimer()
                print("CUDA cycle timer enabled.")
            except Exception as e:
                print("CUDA cycle timer failed:", e)

    prompt = "Explain why time-aware reasoning matters for AI systems."
    demo_time = time.time()

    print("\n=== Static cache-safe generation: low budget ===")
    text_low, plan_low, snap_low = generate_advanced_static(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        tokenizer=tokenizer,
        prompt=prompt,
        timestamp=demo_time,
        budget=0.25,
        budget_s=0.35,
        max_new_tokens=args.max_gen_tokens,
        config=pace_config,
        use_gpu_events=args.use_gpu_events,
        system_monitor=system_monitor,
        traffic_monitor=traffic_monitor,
        do_sample=False,
        seed=1,
        request_id="req-low",
    )
    print("Output:", text_low)
    print("Plan:", plan_low)
    print("Delta us:", snap_low.delta_us)

    print("\n=== Static cache-safe generation: high budget ===")
    text_high, plan_high, snap_high = generate_advanced_static(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        tokenizer=tokenizer,
        prompt=prompt,
        timestamp=demo_time,
        budget=0.85,
        budget_s=1.2,
        max_new_tokens=args.max_gen_tokens,
        config=pace_config,
        use_gpu_events=args.use_gpu_events,
        system_monitor=system_monitor,
        traffic_monitor=traffic_monitor,
        do_sample=False,
        seed=1,
        request_id="req-high",
    )
    print("Output:", text_high)
    print("Plan:", plan_high)
    print("Delta us:", snap_high.delta_us)

    if args.dynamic:
        print("\n=== Dynamic no-cache generation ===")
        text_dyn, plan_dyn, snap_dyn, metrics_dyn = generate_advanced_dynamic(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            tokenizer=tokenizer,
            prompt=prompt,
            timestamp=demo_time,
            budget=0.55,
            budget_s=0.8,
            max_new_tokens=min(16, args.max_gen_tokens),
            config=pace_config,
            use_gpu_events=args.use_gpu_events,
            enable_per_layer_timing=args.per_layer_timing,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
            cuda_cycle_timer=cuda_timer,
            do_sample=False,
            seed=1,
            request_id="req-dyn",
        )
        print("Output:", text_dyn)
        print("Plan:", plan_dyn)
        print("Metrics:", metrics_dyn)


if __name__ == "__main__":
    main_advanced()