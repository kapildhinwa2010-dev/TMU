from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .system import SystemMonitor
from .traffic import TrafficMonitor
from .utils import ADV_INTERNAL_DIM, clamp01, log_norm, safe_div


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
        self.use_gpu_events = (
            bool(use_gpu_events)
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )

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

        sysm = (
            self.system_monitor.snapshot()
            if self.system_monitor is not None
            else {
                "sm_clock_hz": 1_400_000_000.0,
                "max_sm_clock_hz": 2_100_000_000.0,
                "freq_norm": 0.67,
                "throttle_risk": 0.33,
                "temp_norm": 0.45,
                "power_norm": 0.25,
                "mem_pressure": 0.25,
                "gpu_util_norm": 0.0,
            }
        )

        traf = (
            self.traffic_monitor.snapshot()
            if self.traffic_monitor is not None
            else {
                "active_requests": 0,
                "active_norm": 0.0,
                "queue_delay_norm": 0.0,
                "throughput_norm": 0.0,
                "p95_norm": 0.0,
                "error_rate": 0.0,
                "deadline_pressure": 0.0,
            }
        )

        sm_hz = sysm.get("sm_clock_hz")
        cycles = delta_s * float(sm_hz) if sm_hz is not None else None

        tokens_per_s = safe_div(float(tokens), delta_s, 0.0) if delta_s > 1e-9 else 0.0

        if self.budget_s is not None and self.budget_s > 0:
            self.spent_s += delta_s
            remaining_s = max(0.0, self.budget_s - self.spent_s)
            budget_pressure = clamp01(self.spent_s / self.budget_s)
        else:
            remaining_s = None
            budget_pressure = 0.0

        latency_pressure = clamp01(delta_ms / 100.0)
        pressure = max(
            budget_pressure,
            float(traf.get("deadline_pressure", 0.0)),
            latency_pressure,
        )

        step_norm = clamp01(float(self.step) / float(max(1, self.max_steps)))

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

        per_layer_mean_norm = clamp01(mean_ms / 50.0)
        per_layer_p95_norm = clamp01(p95_ms / 100.0)
        per_layer_var_norm = clamp01(std / 50.0)

        feature_list = [
            log_norm(elapsed_s, 3600.0),
            log_norm(delta_s, 1.0),
            clamp01(delta_ms / 100.0),
            clamp01((gpu_delta_us or delta_us) / 1000.0),
            log_norm(cycles or 0.0, 1e9),
            clamp01(sysm.get("freq_norm", 0.5)),
            clamp01(step_norm),
            clamp01(pressure),
            clamp01(traf.get("queue_delay_norm", 0.0)),
            clamp01(traf.get("active_norm", 0.0)),
            clamp01(traf.get("throughput_norm", 0.0)),
            clamp01(traf.get("p95_norm", 0.0)),
            clamp01(traf.get("error_rate", 0.0)),
            clamp01(traf.get("deadline_pressure", 0.0)),
            clamp01(sysm.get("gpu_util_norm", 0.0)),
            clamp01(sysm.get("mem_pressure", 0.0)),
            clamp01(sysm.get("temp_norm", 0.0)),
            clamp01(sysm.get("power_norm", 0.0)),
            clamp01(sysm.get("throttle_risk", 0.0)),
            clamp01(prompt_difficulty),
            clamp01(confidence),
            per_layer_mean_norm,
            per_layer_p95_norm,
            per_layer_var_norm,
        ]

        assert len(feature_list) == ADV_INTERNAL_DIM

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
            prompt_difficulty=clamp01(prompt_difficulty),
            confidence=clamp01(confidence),
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
