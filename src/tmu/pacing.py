from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils import clamp01


@dataclass
class PaceConfig:
    min_layers: int = 2
    max_layers: int = 12
    min_tokens: int = 8
    max_tokens: int = 64
    min_temperature: float = 0.6
    max_temperature: float = 1.0
    refine_min: int = 1
    refine_max: int = 3


class PaceController:
    def __init__(self, num_layers: int, config: Optional[PaceConfig] = None):
        self.num_layers = int(num_layers)
        self.config = config or PaceConfig()

        self.config.max_layers = min(int(self.config.max_layers), self.num_layers)
        self.config.min_layers = max(1, min(int(self.config.min_layers), self.config.max_layers))

    def from_pace(self, pace: float, budget: Optional[float] = None) -> Dict[str, Any]:
        pace = clamp01(pace)

        if budget is None:
            budget_eff = 1.0
        else:
            budget_eff = clamp01(budget)

        cfg = self.config

        active_layers = int(round(cfg.min_layers + pace * (cfg.max_layers - cfg.min_layers)))
        active_layers = max(1, min(self.num_layers, active_layers))

        token_budget = int(round(cfg.min_tokens + budget_eff * (cfg.max_tokens - cfg.min_tokens)))
        pace_tokens = int(round(cfg.min_tokens + pace * (cfg.max_tokens - cfg.min_tokens)))

        max_new_tokens = max(1, min(token_budget, pace_tokens))

        temperature = float(cfg.min_temperature + pace * (cfg.max_temperature - cfg.min_temperature))
        refinement_passes = int(round(cfg.refine_min + pace * (cfg.refine_max - cfg.refine_min)))

        return {
            "pace": pace,
            "active_layers": active_layers,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "refinement_passes": refinement_passes,
        }


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

        pace = clamp01(pace)
        layer_frac = clamp01(layer_frac)
        token_frac = clamp01(token_frac)
        budget = clamp01(budget)
        difficulty = clamp01(difficulty)

        active_layers = int(round(cfg.min_layers + layer_frac * (cfg.max_layers - cfg.min_layers)))
        max_new_tokens = int(round(cfg.min_tokens + token_frac * (cfg.max_tokens - cfg.min_tokens)))

        token_budget = int(round(cfg.min_tokens + budget * (cfg.max_tokens - cfg.min_tokens)))
        max_new_tokens = max(1, min(max_new_tokens, token_budget))

        traffic_load = (
            0.25 * float(traffic.get("queue_delay_norm", 0.0))
            + 0.20 * float(traffic.get("active_norm", 0.0))
            + 0.30 * float(traffic.get("deadline_pressure", 0.0))
            + 0.15 * float(traffic.get("p95_norm", 0.0))
            + 0.10 * float(traffic.get("error_rate", 0.0))
        )
        traffic_load = clamp01(traffic_load)

        thermal_load = (
            0.35 * float(system.get("temp_norm", 0.0))
            + 0.25 * float(system.get("power_norm", 0.0))
            + 0.25 * float(system.get("throttle_risk", 0.0))
            + 0.15 * float(system.get("gpu_util_norm", 0.0))
        )
        thermal_load = clamp01(thermal_load)

        load = clamp01(
            float(cfg.traffic_penalty_strength) * traffic_load
            + float(cfg.thermal_penalty_strength) * thermal_load
        )

        multiplier = max(0.30, 1.0 - 0.65 * load)

        active_layers = int(max(1, min(self.num_layers, round(active_layers * multiplier))))
        max_new_tokens = int(max(1, round(max_new_tokens * multiplier)))

        diff_adj = int(round(2.0 * (difficulty - 0.5) * budget))
        active_layers = max(1, min(self.num_layers, active_layers + diff_adj))

        mem_pressure = float(system.get("mem_pressure", 0.0))

        if mem_pressure > cfg.memory_limit:
            excess = clamp01((mem_pressure - cfg.memory_limit) / max(1e-6, 1.0 - cfg.memory_limit))
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
