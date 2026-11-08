#!/usr/bin/env python3
"""TMU mein additional fusion layer train karke GPT-2 par transplant karta hai.

Pipeline:
1. Plain GPT-2 load (base model freeze).
2. TMU layer banai jati hai jismein additional fusion layers hain.
3. Thoda train (demo data ya real timestamped data par).
4. Trained TMU layer export (portable file).
5. Same layer ko fresh GPT-2 par transplant karke baseline vs transplant metrics.

Example:
    python scripts/train_tmu_layer.py \
        --fusion-layers 1 --epochs 1 --samples 96 \
        --export artifacts/tmu_layer.pt \
        --results results/transplant_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tmu.adapter import AdvancedInternalAdapter  # noqa: E402
from tmu.clock import AdvancedInternalClock  # noqa: E402
from tmu.data import (  # noqa: E402
    AdvancedTimestampedDataset,
    build_advanced_demo_records,
    load_advanced_records,
)
from tmu.eval import evaluate_time_coherence  # noqa: E402
from tmu.heads import AdvancedTmuCapabilityHeads  # noqa: E402
from tmu.plugin import TMUPlugin  # noqa: E402
from tmu.system import SystemMonitor  # noqa: E402
from tmu.traffic import TrafficMonitor  # noqa: E402
from tmu.train import train_advanced_tmu  # noqa: E402
from tmu.transplant import (  # noqa: E402
    evaluate_transplant,
    export_tmu_layer,
    write_transplant_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an additional TMU layer and transplant it onto GPT-2")

    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--data", type=str, default=None, help="JSONL/CSV file with timestamped records")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--fusion-layers", type=int, default=1, help="Additional fusion layers in the TMU layer")
    p.add_argument("--fusion-expansion", type=int, default=2)
    p.add_argument("--fusion-dropout", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-gen-tokens", type=int, default=16)

    p.add_argument("--export", type=str, default="artifacts/tmu_layer.pt")
    p.add_argument("--results", type=str, default="results/transplant_test.json")

    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    print("Device:", device)
    print("Model:", args.model)
    print("Additional fusion layers:", args.fusion_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False

    hidden_dim = int(model.config.n_embd)
    num_layers = len(model.transformer.h)

    plugin = TMUPlugin(
        base_model=model,
        hidden_dim=hidden_dim,
        mode="hooks",
        target_modules=["GPT2Block", "block"],
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
        fusion_layers=args.fusion_layers,
        fusion_expansion=args.fusion_expansion,
        fusion_dropout=args.fusion_dropout,
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

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    clock = AdvancedInternalClock(
        device=device,
        max_steps=64,
        budget_s=1.0,
        use_gpu_events=False,
        system_monitor=SystemMonitor(device, enable=str(device).startswith("cuda")),
        traffic_monitor=TrafficMonitor(window_s=60.0, concurrency_limit=4),
    )

    print("\n=== Training TMU layer (base GPT-2 frozen) ===")

    t0 = time.time()
    history = train_advanced_tmu(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        train_loader=loader,
        device=device,
        clock=clock,
        epochs=args.epochs,
        lr=args.lr,
        use_gpu_events=False,
        budget_s=1.0,
        verbose=True,
    )
    train_seconds = time.time() - t0

    tmu_params = sum(p.numel() for p in plugin.tmu.parameters() if p.requires_grad)
    print(f"Train time: {train_seconds:.1f}s | TMU trainable params: {tmu_params:,}")

    print("\n=== Exporting TMU layer ===")
    export_tmu_layer(
        plugin.tmu,
        args.export,
        metadata={
            "source": "scripts/train_tmu_layer.py",
            "base_model": args.model,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "fusion_layers": args.fusion_layers,
            "epochs": args.epochs,
            "train_seconds": train_seconds,
        },
    )
    print("Exported:", args.export)

    print("\n=== Transplant test: plain GPT-2 + trained TMU layer ===")
    metrics = evaluate_transplant(
        base_model=model,
        plugin=plugin,
        samples=dataset.samples,
        device=device,
        max_samples=min(32, len(dataset.samples)),
    )

    coherence = evaluate_time_coherence(
        plugin=plugin,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        num_samples=2,
        verbose=False,
    )

    payload: Dict[str, Any] = {
        **metrics,
        "coherence": coherence,
        "training": {
            "epochs": args.epochs,
            "samples": len(dataset.samples),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "seconds": train_seconds,
            "history": history,
        },
        "config": {
            "model": args.model,
            "fusion_layers": args.fusion_layers,
            "fusion_expansion": args.fusion_expansion,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "device": device,
        },
    }

    write_transplant_report(payload, args.results)
    print("\nWrote:", args.results)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
