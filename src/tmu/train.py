from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter import AdvancedInternalAdapter
from .clock import AdvancedInternalClock, advanced_snapshot_to_tensor
from .heads import AdvancedTmuCapabilityHeads, TmuCapabilityHeads
from .plugin import TMUPlugin, freeze_base_and_enable_tmu
from .system import SystemMonitor
from .traffic import TrafficMonitor
from .utils import prepare_budget_tensor


def train_gpt2_tmu(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 2,
    lr: float = 1e-3,
    w_lm: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    plugin.to(device)
    heads.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = list(tmu_params) + list(heads.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    history = []

    for epoch in range(int(epochs)):
        plugin.base_model.eval()
        plugin.tmu.train()
        heads.train()

        total_loss = 0.0
        total_lm = 0.0
        total_rhythm = 0.0
        total_pace = 0.0
        total_over = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)

            y_rhythm = batch["y_rhythm"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)

            B = int(input_ids.shape[0])

            time_embed = plugin.tmu.encode_time(
                time_stamp=t,
                batch_size=B,
                device=device,
            )

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

            rhythm_pred, pace_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace_bce = F.binary_cross_entropy(pace_safe, y_pace)

            over_budget = F.relu(pace_pred - budget).mean()
            loss_pace = loss_pace_bce + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * loss_pace
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            total_loss += float(loss.item())
            total_lm += float(loss_lm.item())
            total_rhythm += float(loss_rhythm.item())
            total_pace += float(loss_pace.item())
            total_over += float(over_budget.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_lm = total_lm / max(1, num_batches)
        avg_rhythm = total_rhythm / max(1, num_batches)
        avg_pace = total_pace / max(1, num_batches)
        avg_over = total_over / max(1, num_batches)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "loss_lm": avg_lm,
            "loss_rhythm": avg_rhythm,
            "loss_pace": avg_pace,
            "over_budget": avg_over,
        })

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"lm={avg_lm:.4f} | "
                f"rhythm={avg_rhythm:.4f} | "
                f"pace={avg_pace:.4f} | "
                f"over={avg_over:.4f}"
            )

    return history


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