#!/usr/bin/env python3
"""Exported TMU layer ko plain GPT-2 par transplant karke test karta hai.

Ye script training nahi karti - sirf ek exported layer file (ya trained
checkpoint) ko uthati hai, usse fresh GPT-2 par mount karti hai aur:

- baseline GPT-2 vs transplanted GPT-2 ka eval loss / perplexity compare karti hai
- time shift (default 12h) par logits change measure karti hai (time-sensitivity proof)
- time-coherence metrics calculate karti hai
- ek markdown summary README ke liye likhti hai

Example:
    python scripts/test_transplant.py --layer artifacts/tmu_layer.pt \
        --results results/transplant_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tmu.data import (  # noqa: E402
    AdvancedTimestampedDataset,
    build_advanced_demo_records,
    load_advanced_records,
)
from tmu.eval import evaluate_time_coherence  # noqa: E402
from tmu.transplant import (  # noqa: E402
    evaluate_transplant,
    format_transplant_report,
    load_tmu_layer,
    load_tmu_layer_metadata,
    mount_tmu_on_gpt2,
    write_transplant_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transplant a trained TMU layer onto plain GPT-2 and test it")

    p.add_argument("--layer", type=str, required=True, help="Exported TMU layer file (.pt)")
    p.add_argument("--model", type=str, default="gpt2", help="Base model onto which layer is mounted")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--time-shift", type=float, default=43200.0)
    p.add_argument("--results", type=str, default="results/transplant_test.json")
    p.add_argument("--markdown", type=str, default=None, help="Optional markdown summary output path")

    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    print("Base model:", args.model)
    print("TMU layer:", args.layer)

    metadata = load_tmu_layer_metadata(args.layer)
    print("Layer metadata:", json.dumps(metadata, indent=2))

    tmu = load_tmu_layer(args.layer, map_location="cpu", device=torch.device(device))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False

    hidden_dim = int(model.config.n_embd)
    num_layers = len(model.transformer.h)

    if int(tmu.hidden_dim or 0) != hidden_dim:
        raise ValueError(
            f"Layer hidden_dim={tmu.hidden_dim} base model hidden_dim={hidden_dim} mismatch."
        )

    plugin = mount_tmu_on_gpt2(model=model, tmu=tmu, mode="hooks")

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
        max_label_tokens=32,
    )

    metrics = evaluate_transplant(
        base_model=model,
        plugin=plugin,
        samples=dataset.samples,
        device=device,
        time_shift_seconds=args.time_shift,
        max_samples=min(args.samples, len(dataset.samples)),
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
        "layer_metadata": metadata,
        "base_model": args.model,
        "device": device,
    }

    write_transplant_report(payload, args.results)
    print("\nWrote:", args.results)

    summary = format_transplant_report(metrics)
    print("\n" + summary)

    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print("Markdown:", args.markdown)


if __name__ == "__main__":
    main()
