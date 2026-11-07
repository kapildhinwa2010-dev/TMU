from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import AdvancedInternalAdapter
from .clock import AdvancedInternalClock
from .data import (
    AdvancedTimestampedDataset,
    GPT2TimestampedDataset,
    build_advanced_demo_records,
    build_demo_records,
    load_advanced_records,
    load_timestamped_records,
)
from .eval import evaluate_time_coherence
from .generate import generate_advanced_dynamic, generate_advanced_static
from .heads import AdvancedTmuCapabilityHeads, TmuCapabilityHeads
from .pacing import AdvancedPaceConfig
from .plugin import TMUPlugin
from .system import SystemMonitor
from .train import train_advanced_tmu, train_gpt2_tmu
from .traffic import TrafficMonitor


def _pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_basic_data(args, tokenizer):
    if args.data:
        texts, timestamps, budgets = load_timestamped_records(args.data)
    else:
        texts, timestamps, budgets = build_demo_records(
            num_samples=args.samples,
            base_time=1_700_000_000.0,
            seed=42,
        )

    dataset = GPT2TimestampedDataset(
        texts=texts,
        timestamps=timestamps,
        budgets=budgets,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
    )

    return dataset


def _load_advanced_data(args, tokenizer, num_layers: int):
    if args.data:
        texts, timestamps, budgets, metas = load_advanced_records(args.data)
    else:
        texts, timestamps, budgets, metas = build_advanced_demo_records(
            num_samples=args.samples,
            num_layers=num_layers,
            base_time=1_700_000_000.0,
            seed=42,
        )

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

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="TMU-GPT-2 CLI")
    parser.add_argument("--config", type=str, default=None, help="Load defaults from a JSON config file (CLI flags override)")
    parser.add_argument("--version", action="version", version="tmu-gpt2 0.1.1")

    parser.add_argument("--mode", choices=["basic", "advanced"], default="advanced")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt2")

    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max-gen-tokens", type=int, default=32)
    parser.add_argument("--budget-s", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=2)

    parser.add_argument("--use-gpu-events", action="store_true")
    parser.add_argument("--per-layer-timing", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--enable-cuda-cycles", action="store_true")

    parser.add_argument("--save", type=str, default="gpt2_tmu_checkpoint.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # JSON config file support: apply values for args NOT explicitly passed on the CLI
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        _explicit = set()
        for _tok in sys.argv[1:]:
            if _tok.startswith("--"):
                _explicit.add(_tok.lstrip("-").replace("-", "_"))
        for _k, _v in _cfg.items():
            _a = _k.replace("-", "_")
            if hasattr(args, _a) and _a not in _explicit and _a != "config":
                setattr(args, _a, _v)
        print(f"[cli] Loaded config: {args.config}")

    device = _pick_device(args.device)
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

    if args.mode == "basic":
        heads = TmuCapabilityHeads(hidden_dim=hidden_dim)
        dataset = _load_basic_data(args, tokenizer)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        if args.load:
            ckpt = torch.load(args.load, map_location=device)
            plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
            heads.load_state_dict(ckpt["heads_state_dict"])
        else:
            print("\n=== Training basic TMU ===")
            train_gpt2_tmu(
                plugin=plugin,
                heads=heads,
                train_loader=loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                verbose=True,
            )

            torch.save({
                "tmu_state_dict": plugin.tmu.state_dict(),
                "heads_state_dict": heads.state_dict(),
            }, args.save)
            print("Saved:", args.save)

        print("\n=== Time coherence evaluation ===")
        metrics = evaluate_time_coherence(
            plugin=plugin,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            num_samples=args.eval_samples,
            verbose=True,
        )
        print(json.dumps(metrics, indent=2))
        return

    heads = AdvancedTmuCapabilityHeads(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_tokens=max(8, args.max_gen_tokens),
    )

    adapter = AdvancedInternalAdapter(hidden_dim=hidden_dim)

    system_monitor = SystemMonitor(device, enable=str(device).startswith("cuda"))
    traffic_monitor = TrafficMonitor(window_s=60.0, concurrency_limit=4)

    dataset = _load_advanced_data(args, tokenizer, num_layers)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
        heads.load_state_dict(ckpt["heads_state_dict"])
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=args.budget_s,
            use_gpu_events=args.use_gpu_events,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
        )

        print("\n=== Training advanced TMU ===")
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

        torch.save({
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
        }, args.save)
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
        try:
            from .cuda_cycle_timer import CudaCycleTimer
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
        request_id=f"static-low-{int(time.time() * 1000)}",
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
        request_id=f"static-high-{int(time.time() * 1000)}",
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
            request_id=f"dynamic-{int(time.time() * 1000)}",
        )
        print("Output:", text_dyn)
        print("Plan:", plan_dyn)
        print("Metrics:", metrics_dyn)

    print("\n=== Time coherence evaluation ===")
    metrics = evaluate_time_coherence(
        plugin=plugin,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_samples=args.eval_samples,
        verbose=True,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()