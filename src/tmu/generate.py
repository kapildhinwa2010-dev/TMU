from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .adapter import AdvancedInternalAdapter
from .clock import AdvancedClockSnapshot, AdvancedInternalClock, advanced_snapshot_to_tensor
from .data import estimate_text_difficulty
from .fusion import encode_fused_time_embed
from .heads import AdvancedTmuCapabilityHeads
from .modeling import custom_gpt2_forward_advanced, prefix_layer_skip_context
from .pacing import AdvancedPaceConfig, AdvancedPaceController
from .plugin import TMUPlugin
from .system import SystemMonitor
from .time_features import TimeLike
from .traffic import TrafficMonitor
from .utils import clamp01, prepare_budget_tensor


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
    time_embed = encode_fused_time_embed(
        tmu=plugin.tmu,
        time_stamp=timestamp,
        internal_features=internal_features,
        adapter=adapter,
        batch_size=batch_size,
        device=device,
    )

    budget_t = prepare_budget_tensor(budget, batch_size, device)

    rhythm_pred, pace, layer_frac, token_frac, _ = heads(time_embed, budget_t)

    return pace, layer_frac, token_frac, time_embed, rhythm_pred


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
) -> Tuple[str, Dict[str, Any], Optional[AdvancedClockSnapshot], Dict[str, Any]]:
    model = plugin.base_model
    model.eval()
    heads.eval()
    adapter.eval()

    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(int(seed))

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
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

    from .modeling import PerLayerTimer

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

    if int(max_new_tokens) <= 0:
        snapshot = clock.tick(
            tokens=prompt_len,
            prompt_difficulty=difficulty,
            confidence=confidence,
            per_layer_times=None,
            sync=bool(use_gpu_events),
        )
        if traffic_monitor is not None and request_id is not None:
            traffic_monitor.request_end(
                request_id=request_id,
                generated_tokens=0,
                success=True,
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
