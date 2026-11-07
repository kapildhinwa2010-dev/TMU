# scripts/run_loops_advanced_demo.py
#
# Advanced DLR-IA v3 demo:
# - Dynamic Layer Refinement
# - Input Anchoring
# - Loop Rejection Sampling
# - Prompt Difficulty Detection
# - Layer Importance Scoring
# - Latency Feedback
# - Advanced Traffic/System Features
#
# Run from repo root:
#   python scripts/run_loops_advanced_demo.py
#
# With real data:
#   python scripts/run_loops_advanced_demo.py --data examples/data_loops_advanced.jsonl

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmu.data import (
    AdvancedTimestampedDataset,
    build_advanced_demo_records,
    load_advanced_records,
)
from tmu.looping import LoopConfig
from tmu.looping_advanced import (
    AdvancedLayerRefiner,
    AdvancedLoopHeadsV3,
    AdvancedTrafficAdapter,
    AdvancedTrafficMonitor,
    AnchorScheduler,
    LatencyFeedbackBuffer,
    LayerImportanceScorer,
    PromptDifficultyDetector,
    fine_tune_latency_feedback,
    generate_advanced_dynamic_loops_v3,
    load_advanced_loop_v3,
    make_advanced_loop_plan,
    save_advanced_loop_v3,
    train_advanced_loop_v3,
)
from tmu.adapter import AdvancedInternalAdapter
from tmu.clock import AdvancedInternalClock
from tmu.pacing import AdvancedPaceConfig
from tmu.plugin import TMUPlugin
from tmu.system import SystemMonitor


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_dataset(args, tokenizer, num_layers: int):
    if args.data:
        texts, timestamps, budgets, metas = load_advanced_records(args.data)
    else:
        texts, timestamps, budgets, metas = build_advanced_demo_records(
            num_samples=args.samples,
            num_layers=num_layers,
            base_time=1_700_000_000.0,
            seed=42,
        )

    base_ds = AdvancedTimestampedDataset(
        texts=texts,
        timestamps=timestamps,
        budgets=budgets,
        metas=metas,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_layers=num_layers,
        max_label_tokens=max(8, args.max_gen_tokens),
    )

    from tmu.looping_advanced import AdvancedLoopLatencyDatasetWrapper

    ds = AdvancedLoopLatencyDatasetWrapper(
        base_dataset=base_ds,
        metas=metas,
        max_loops=args.max_loops,
        max_latency_ms=args.max_latency_ms,
        max_tokens=max(8, args.max_gen_tokens),
    )

    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="TMU GPT-2 Advanced Looping v3 demo")

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--data", type=str, default=None)

    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--max-gen-tokens", type=int, default=12)
    parser.add_argument("--budget-s", type=float, default=1.0)
    parser.add_argument("--max-latency-ms", type=float, default=2000.0)

    parser.add_argument("--max-loops", type=int, default=3)
    parser.add_argument("--max-final-refine-passes", type=int, default=2)

    parser.add_argument("--use-gpu-events", action="store_true")
    parser.add_argument("--per-layer-timing", action="store_true")
    parser.add_argument("--disable-loops", action="store_true")

    parser.add_argument("--prompt-detector", action="store_true")
    parser.add_argument("--enable-rejection", action="store_true")
    parser.add_argument("--rejection-margin", type=float, default=0.02)

    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--feedback-epochs", type=int, default=1)
    parser.add_argument("--feedback-batch-size", type=int, default=8)

    parser.add_argument("--save", type=str, default="gpt2_tmu_loops_advanced_v3.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = pick_device(args.device)
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

    heads = AdvancedLoopHeadsV3(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_tokens=max(8, args.max_gen_tokens),
    )

    adapter = AdvancedTrafficAdapter(hidden_dim=hidden_dim)

    loop_config = LoopConfig(
        enabled=not args.disable_loops,
        max_loops=args.max_loops,
        max_final_refine_passes=args.max_final_refine_passes,
    )

    refiner = AdvancedLayerRefiner(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        config=loop_config,
        max_delta_norm=0.0,
        importance_gain=1.0,
    )

    system_monitor = SystemMonitor(device, enable=str(device).startswith("cuda"))
    traffic_monitor = AdvancedTrafficMonitor(
        window_s=60.0,
        concurrency_limit=4,
        max_latency_ms=args.max_latency_ms,
    )

    ds = build_dataset(args, tokenizer, num_layers)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    if args.load:
        print("Loading advanced loop v3 from:", args.load)
        load_advanced_loop_v3(plugin, heads, adapter, refiner, args.load, device=device)
    else:
        clock = AdvancedInternalClock(
            device=device,
            max_steps=64,
            budget_s=args.budget_s,
            use_gpu_events=args.use_gpu_events,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
        )

        print("\n=== Training Advanced Looping v3 ===")
        train_advanced_loop_v3(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            refiner=refiner,
            train_loader=loader,
            device=device,
            clock=clock,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
            epochs=args.epochs,
            lr=args.lr,
            use_gpu_events=args.use_gpu_events,
            budget_s=args.budget_s,
            enable_loops=not args.disable_loops,
            verbose=True,
        )

        save_advanced_loop_v3(plugin, heads, adapter, refiner, args.save)
        print("Saved:", args.save)

    pace_config = AdvancedPaceConfig(
        min_layers=2,
        max_layers=num_layers,
        min_tokens=4,
        max_tokens=max(8, args.max_gen_tokens),
        min_temperature=0.6,
        max_temperature=1.0,
    )

    prompt_detector = PromptDifficultyDetector(enabled=args.prompt_detector)
    anchor_scheduler = AnchorScheduler(mode="u")
    feedback_buffer = LatencyFeedbackBuffer(
        max_records=10000,
        max_latency_ms=args.max_latency_ms,
        max_tokens=max(8, args.max_gen_tokens),
    ) if args.feedback else None

    demo_time = time.time()

    cases = [
        {
            "name": "easy_low_budget",
            "prompt": "What is 2 + 2?",
            "budget": 0.20,
            "budget_s": 0.25,
        },
        {
            "name": "normal_medium_budget",
            "prompt": "Summarize the importance of time-aware AI systems.",
            "budget": 0.50,
            "budget_s": 0.60,
        },
        {
            "name": "hard_high_budget",
            "prompt": "Explain step by step why looped refinement can improve reasoning but may forget the input without anchoring.",
            "budget": 0.90,
            "budget_s": 1.20,
        },
    ]

    for case in cases:
        print("\n" + "=" * 78)
        print("Case:", case["name"])
        print("Prompt:", case["prompt"])
        print("Budget:", case["budget"])
        print("=" * 78)

        text, plan, snapshot, metrics, feedback = generate_advanced_dynamic_loops_v3(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            layer_refiner=refiner,
            tokenizer=tokenizer,
            prompt=case["prompt"],
            timestamp=demo_time,
            budget=case["budget"],
            budget_s=case["budget_s"],
            max_new_tokens=args.max_gen_tokens,
            pace_config=pace_config,
            loop_config=loop_config,
            use_gpu_events=args.use_gpu_events,
            enable_per_layer_timing=args.per_layer_timing,
            enable_loops=not args.disable_loops,
            enable_rejection=args.enable_rejection,
            rejection_margin=args.rejection_margin,
            prompt_detector=prompt_detector,
            importance_scorer=None,
            anchor_scheduler=anchor_scheduler,
            latency_feedback_buffer=feedback_buffer,
            system_monitor=system_monitor,
            traffic_monitor=traffic_monitor,
            do_sample=False,
            seed=1,
            request_id=f"{case['name']}-{int(time.time() * 1000)}",
        )

        print("Output:", text)
        print("Active layers:", plan.get("active_layers"))
        print("Max new tokens plan:", plan.get("max_new_tokens"))
        print("Generated tokens:", plan.get("generated_tokens"))
        print("Confidence:", plan.get("confidence"))
        print("Anchor scale:", plan.get("anchor_scale"))
        print("Accepted loop:", plan.get("accepted_loop"))
        print("Actual latency ms:", plan.get("actual_latency_ms"))

        loop_plan = plan.get("loop_plan", {})
        print("Loop scalar:", loop_plan.get("loop_scalar"))
        print("Layer loops:", loop_plan.get("layer_loops"))
        print("Loop weights:", loop_plan.get("loop_weights"))
        print("Final refine passes:", loop_plan.get("final_refine_passes"))

        prompt_info = plan.get("prompt_info", {})
        print("Prompt difficulty:", prompt_info.get("combined"))
        print("Prompt ppl_norm:", prompt_info.get("ppl_norm"))
        print("Prompt entropy_norm:", prompt_info.get("entropy_norm"))

        if snapshot is not None:
            print("Last delta us:", snapshot.delta_us)
            print("Last GPU delta us:", snapshot.gpu_delta_us)
            print("Estimated cycles:", snapshot.cycles)
            print("Pressure:", snapshot.pressure)

        print("Traffic snapshot:", json.dumps(snapshot.traffic if snapshot else {}, indent=2))
        print("System snapshot:", json.dumps(snapshot.system if snapshot else {}, indent=2))

    if feedback_buffer is not None and len(feedback_buffer) > 0:
        print("\n=== Fine-tuning from latency feedback ===")
        fine_tune_latency_feedback(
            plugin=plugin,
            heads=heads,
            adapter=adapter,
            buffer=feedback_buffer,
            device=device,
            epochs=args.feedback_epochs,
            lr=1e-4,
            batch_size=args.feedback_batch_size,
            verbose=True,
        )

        save_advanced_loop_v3(plugin, heads, adapter, refiner, args.save)
        print("Saved after feedback:", args.save)


if __name__ == "__main__":
    main()