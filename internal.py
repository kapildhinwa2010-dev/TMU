# ============================================================
# AI INTERNAL CLOCK ADD-ON
# ============================================================

import subprocess
from dataclasses import dataclass
from typing import Optional

INTERNAL_DIM = 8


def _nvidia_smi_values(query: str, device_index: int = 0) -> Optional[float]:
    """
    nvidia-smi se GPU clock query karta hai.

    Example queries:
      - clocks.sm
      - clocks.max.sm
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        lines = out.decode().strip().splitlines()
        if device_index >= len(lines):
            return None

        val = float(lines[device_index].strip())
        return val if val > 0 else None
    except Exception:
        return None


def detect_gpu_clock_hz(device: Union[str, torch.device]) -> Tuple[Optional[float], Optional[float]]:
    """
    GPU SM clock aur max SM clock Hz me detect karta hai.

    Returns:
        (sm_clock_hz, max_sm_clock_hz)

    Agar GPU nahi hai ya query fail hui, fallback values deta hai.
    """
    dev_str = str(device)

    if not dev_str.startswith("cuda") or not torch.cuda.is_available():
        return None, None

    try:
        idx = torch.cuda.current_device()
    except Exception:
        idx = 0

    sm_mhz = _nvidia_smi_values("clocks.sm", idx)
    max_mhz = _nvidia_smi_values("clocks.max.sm", idx)

    if sm_mhz is None:
        # Fallback: common GPU SM clock estimate
        sm_hz = 1_400_000_000.0
    else:
        sm_hz = float(sm_mhz) * 1_000_000.0

    if max_mhz is None:
        max_hz = max(sm_hz, 2_100_000_000.0)
    else:
        max_hz = float(max_mhz) * 1_000_000.0

    return sm_hz, max_hz


@dataclass
class ClockSnapshot:
    """
    Ek internal clock tick ka snapshot.

    Isme human wall-clock nahi, AI internal compute-time signals hain.
    """

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


class InternalClock:
    """
    AI ke liye internal clock.

    Sources:
      1. CPU high-resolution timer: time.perf_counter_ns()
      2. Optional CUDA events for GPU-side ms timing
      3. Optional nvidia-smi / NVML for GPU SM clock speed

    Modes:
      - use_gpu_events=False:
          fast, CPU nanosecond based, no forced sync
      - use_gpu_events=True:
          accurate GPU ms timing, lekin synchronize karta hai, slow ho sakta hai
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        max_steps: int = 64,
        budget_s: Optional[float] = None,
        use_gpu_events: bool = False,
        query_gpu_clock: bool = True,
    ):
        self.device = device
        self.max_steps = int(max_steps)
        self.budget_s = float(budget_s) if budget_s is not None else None
        self.use_gpu_events = bool(use_gpu_events)

        self.use_cuda = (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
        )

        self.use_gpu_events = self.use_gpu_events and self.use_cuda

        # CPU nanosecond origin
        self.origin_ns = time.perf_counter_ns()
        self.last_ns = self.origin_ns

        # Human wall clock for reference only
        self.wall_origin = time.time()

        self.step = 0
        self.tokens = 0
        self.spent_s = 0.0

        # GPU clock info
        if query_gpu_clock:
            self.sm_clock_hz, self.max_sm_clock_hz = detect_gpu_clock_hz(device)
        else:
            self.sm_clock_hz, self.max_sm_clock_hz = None, None

        # CUDA events for GPU-side timing
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
        sync: Optional[bool] = None,
    ) -> ClockSnapshot:
        """
        Ek clock tick.

        Args:
            tokens:
                is step me kitne tokens process hue.
                Throughput estimate ke liye use hota hai.

            sync:
                Agar True, CUDA event timing ke liye synchronize karega.
                Agar None, use_gpu_events par depend karega.

        Returns:
            ClockSnapshot
        """
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
            # Current point par GPU event record karo
            self.event_curr.record()

            # GPU work complete hone ka wait karo
            torch.cuda.synchronize()

            # prev -> curr ka elapsed time milliseconds me
            gpu_delta_ms = float(self.event_prev.elapsed_time(self.event_curr))

            # Next interval ke liye events swap karo
            self.event_prev, self.event_curr = self.event_curr, self.event_prev
            self.event_curr.record()

        # If GPU event not used, CPU delta ko GPU proxy samjho
        delta_ms = float(gpu_delta_ms) if gpu_delta_ms is not None else delta_s * 1000.0
        delta_us = delta_ms * 1000.0

        gpu_delta_us = None
        if gpu_delta_ms is not None:
            gpu_delta_us = gpu_delta_ms * 1000.0

        # Estimated GPU cycles
        if self.sm_clock_hz is not None:
            cycles = delta_s * float(self.sm_clock_hz)
        else:
            cycles = None

        # Throughput
        if delta_s > 1e-9:
            tokens_per_s = float(tokens) / delta_s
        else:
            tokens_per_s = 0.0

        # Budget pressure
        if self.budget_s is not None and self.budget_s > 0:
            self.spent_s += delta_s
            remaining_s = max(0.0, self.budget_s - self.spent_s)
            pressure = min(1.0, max(0.0, self.spent_s / self.budget_s))
        else:
            remaining_s = None
            # No explicit time budget: throughput ko soft pressure proxy banao
            pressure = min(1.0, max(0.0, tokens_per_s / 1000.0))

        step_norm = min(1.0, float(self.step) / float(max(1, self.max_steps)))

        return ClockSnapshot(
            wall_s=wall_s,
            elapsed_s=elapsed_s,
            delta_s=delta_s,
            delta_ms=delta_ms,
            delta_us=delta_us,
            gpu_delta_ms=gpu_delta_ms,
            gpu_delta_us=gpu_delta_us,
            sm_clock_hz=self.sm_clock_hz,
            max_sm_clock_hz=self.max_sm_clock_hz,
            cycles=cycles,
            step=self.step,
            step_norm=step_norm,
            tokens=self.tokens,
            tokens_per_s=tokens_per_s,
            budget_s=self.budget_s,
            remaining_s=remaining_s,
            pressure=pressure,
        )


def _log_norm(x: float, scale: float) -> float:
    """
    Positive scalar ko 0..1 me normalize karta hai using log1p.
    """
    x = max(0.0, float(x))
    scale = max(1e-9, float(scale))
    return min(1.0, max(0.0, math.log1p(x) / math.log1p(scale)))


def snapshot_to_internal_tensor(
    snapshot: ClockSnapshot,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    ClockSnapshot ko TMU-compatible internal feature tensor me convert karta hai.

    Output shape:
        (B, INTERNAL_DIM)

    Features:
        0. elapsed_log        : session age
        1. delta_log          : gap since last call
        2. delta_ms_norm      : latency in ms
        3. gpu_us_norm        : GPU/CPU microsecond latency
        4. cycles_log         : estimated GPU cycles
        5. freq_norm          : current GPU clock / max GPU clock
        6. step_norm          : generation progress
        7. pressure           : budget pressure / throughput proxy
    """
    gpu_us = snapshot.gpu_delta_us if snapshot.gpu_delta_us is not None else snapshot.delta_us
    cycles = snapshot.cycles if snapshot.cycles is not None else 0.0

    if snapshot.sm_clock_hz is None:
        freq_norm = 0.5
    else:
        denom = snapshot.max_sm_clock_hz or snapshot.sm_clock_hz
        freq_norm = min(
            1.0,
            max(0.0, float(snapshot.sm_clock_hz) / max(1.0, float(denom))),
        )

    vals = [
        _log_norm(snapshot.elapsed_s, 3600.0),
        _log_norm(snapshot.delta_s, 1.0),
        min(1.0, max(0.0, snapshot.delta_ms / 100.0)),
        min(1.0, max(0.0, gpu_us / 1000.0)),
        _log_norm(cycles, 1e9),
        freq_norm,
        min(1.0, max(0.0, snapshot.step_norm)),
        min(1.0, max(0.0, snapshot.pressure)),
    ]

    t = torch.tensor(vals, dtype=torch.float32, device=device)
    t = t.unsqueeze(0).expand(int(batch_size), -1).contiguous()

    return t


class InternalClockAdapter(nn.Module):
    """
    Internal clock features ko TMU hidden embedding space me laata hai.

    Important:
        Final linear zero-initialized hai, isliye shuru me model ka behavior
        original GPT-2 jaisa hi rahega. Training ke baad internal clock effect
        seekha jayega.
    """

    def __init__(self, hidden_dim: int, internal_dim: int = INTERNAL_DIM):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.internal_dim = int(internal_dim)

        self.norm = nn.LayerNorm(self.internal_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.internal_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Zero init: initial no-op
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, internal_features: torch.Tensor) -> torch.Tensor:
        """
        internal_features: (B, INTERNAL_DIM)
        returns: (B, hidden_dim)
        """
        x = self.norm(internal_features)
        return self.proj(x)


def _prepare_budget_tensor(
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Budget ko (B,) tensor me convert karta hai.
    """
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
            raise ValueError(
                f"Budget batch {budget.shape[0]} != batch_size {batch_size}"
            )

    return budget


def predict_pace_internal(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    adapter: InternalClockAdapter,
    timestamp: TimeLike,
    budget: Union[torch.Tensor, float],
    batch_size: int,
    device: torch.device,
    internal_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    External TMU time embed + Internal clock embed milakar pace predict karta hai.

    Returns:
        pace: (B,)
        time_embed: (B, hidden_dim)
    """
    external_embed = plugin.tmu.encode_time(
        time_stamp=timestamp,
        batch_size=batch_size,
        device=device,
    )

    if internal_features is None:
        internal_features = torch.zeros(
            batch_size,
            INTERNAL_DIM,
            device=device,
            dtype=torch.float32,
        )
    else:
        internal_features = internal_features.to(device=device, dtype=torch.float32)

    internal_embed = adapter(internal_features)

    # Combined time embedding:
    # world rhythm + AI internal compute rhythm
    time_embed = external_embed + internal_embed

    budget_t = _prepare_budget_tensor(budget, batch_size, device)

    _, pace, _ = heads(time_embed, budget_t)

    return pace, time_embed