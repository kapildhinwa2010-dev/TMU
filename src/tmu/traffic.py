from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils import clamp01, log_norm, safe_div


@dataclass
class RequestState:
    request_id: str
    start_perf: float
    prompt_tokens: int = 0
    priority: float = 0.5
    deadline_s: Optional[float] = None


class TrafficMonitor:
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
            priority=clamp01(priority),
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
            while self.completed and self.completed[0][0] < cutoff:
                self.completed.popleft()

            active_count = len(self.active)
            active_states = tuple(self.active.values())
            records = list(self.completed)

        latencies = sorted([r[1] for r in records]) if records else []
        tokens = sum([r[2] for r in records]) if records else 0
        errors = sum([0 if r[3] else 1 for r in records]) if records else 0

        throughput = safe_div(tokens, self.window_s, 0.0)
        p95 = 0.0

        if latencies:
            idx = min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
            p95 = float(latencies[idx])

        error_rate = safe_div(errors, len(records), 0.0)

        queue_delay_s = max(
            0.0,
            float(active_count - self.concurrency_limit),
        ) * max(0.02, p95 / 1000.0)

        deadline_pressure = 0.0

        for st in active_states:
            if st.deadline_s is None or st.deadline_s <= 1e-6:
                continue

            elapsed = now - st.start_perf
            d = float(st.deadline_s)

            if elapsed <= 0.7 * d:
                p = 0.0
            else:
                p = clamp01((elapsed - 0.7 * d) / max(1e-6, 0.3 * d))

            deadline_pressure = max(deadline_pressure, p)

        return {
            "active_requests": int(active_count),
            "active_norm": clamp01(active_count / float(self.concurrency_limit)),
            "queue_delay_s": float(queue_delay_s),
            "queue_delay_norm": clamp01(queue_delay_s / 1.0),
            "throughput_tok_s": float(throughput),
            "throughput_norm": log_norm(throughput, 1000.0),
            "p95_latency_ms": float(p95),
            "p95_norm": clamp01(p95 / 1000.0),
            "error_rate": clamp01(error_rate),
            "deadline_pressure": clamp01(deadline_pressure),
            "completed_in_window": int(len(records)),
        }
