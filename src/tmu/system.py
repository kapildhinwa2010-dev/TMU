from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Dict, Optional, Union

import torch

from .utils import clamp01, cuda_device_index, float_or_none, safe_div


class SystemMonitor:
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
        self.device_index = cuda_device_index(device) if self.enable else 0

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
            return {k: float_or_none(v) for k, v in zip(self.FIELDS, vals)}
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
            "freq_norm": clamp01(sm_hz / max_hz),
            "throttle_risk": 0.0,
            "temperature_c": 45.0,
            "temp_norm": clamp01(45.0 / 100.0),
            "power_draw_w": 80.0,
            "power_limit_w": 300.0,
            "power_norm": clamp01(80.0 / 300.0),
            "memory_used_mb": 1024.0,
            "memory_total_mb": 16384.0,
            "mem_pressure": clamp01(1024.0 / 16384.0),
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

        freq_norm = clamp01(sm_hz / max_hz)
        throttle_risk = clamp01(1.0 - freq_norm)

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
            "temp_norm": clamp01(temp / 100.0),
            "power_draw_w": float(pdraw),
            "power_limit_w": float(plimit),
            "power_norm": clamp01(safe_div(pdraw, plimit, 0.25)),
            "memory_used_mb": float(mused),
            "memory_total_mb": float(mtot),
            "mem_pressure": clamp01(safe_div(mused, mtot, 0.25)),
            "gpu_util_pct": float(gutil),
            "gpu_util_norm": clamp01(gutil / 100.0),
            "mem_util_pct": float(mutil),
            "mem_util_norm": clamp01(mutil / 100.0),
        }
