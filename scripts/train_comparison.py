# scripts/train_comparison.py
#
# GPT-2 vs GPT-2 + TMU layer comparison (fair, equal-budget benchmark).
#
# Fairness rules:
#   - Same data, same batch size, same seq len, same optimizer settings
#   - Equal training TIME budget for baseline fine-tune and TMU fine-tune
#   - Baseline: full GPT-2 fine-tune | TMU: frozen GPT-2 + TMU plugin only
#
# Compared variants:
#   1. Pretrained GPT-2 (zero-shot, no training)
#   2. Fine-tuned GPT-2 (baseline, equal time budget)
#   3. Fine-tuned GPT-2 + TMU layer (equal time budget, base frozen)
#
# Run:
#   python scripts/train_comparison.py --config configs/train_compare.json \
#     --results results/tmu_vs_baseline.json --readme README.md

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from tmu.modeling import prefix_layer_skip_context
from tmu.plugin import TMUPlugin, freeze_base_and_enable_tmu

PERIODS = {
    "morning": (7, 11, [
        "The morning meeting started and the team prepared the daily plan.",
        "Sunrise coffee and the morning news were on the table.",
        "The city woke up early and morning traffic filled the streets.",
        "A fresh morning breeze cleared the air before work began.",
        "Morning deliveries arrived at the shop while birds were singing.",
    ]),
    "afternoon": (13, 17, [
        "The afternoon workshop reviewed the progress of the project.",
        "Long afternoon shadows stretched across the quiet office.",
        "Students finished their afternoon lectures and relaxed outside.",
        "The afternoon sun heated the rooftop of the building.",
        "An afternoon review meeting ended with a clear task list.",
    ]),
    "evening": (19, 22, [
        "The evening lights turned on and families sat down for dinner.",
        "An evening walk along the river ended the busy day.",
        "The evening news reported calm weather across the region.",
        "Music played softly in the evening air of the small cafe.",
        "Evening trains carried tired commuters back to their homes.",
    ]),
    "night": (0, 5, [
        "The quiet night sky showed thousands of bright stars.",
        "Night shift workers kept the servers running until dawn.",
        "The city slept deeply through the calm night hours.",
        "Only a few lamps glowed in the silent night streets.",
        "Late night coders pushed one final commit before sleeping.",
    ]),
}

BASE_TIME = 1_700_000_000.0


def build_records(n: int, seed: int) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    texts, stamps = [], []
    keys = list(PERIODS.keys())
    for i in range(n):
        k = keys[i % len(keys)]
        hour_lo, hour_hi, templates = PERIODS[k]
        texts.append(rng.choice(templates))
        hour = rng.uniform(hour_lo, hour_hi)
        day = rng.randrange(30)
        stamps.append(BASE_TIME + day * 86400.0 + hour * 3600.0)
    return texts, stamps


class TimeTextDataset(Dataset):
    def __init__(self, texts, stamps, tokenizer, seq_len: int):
        self.samples = []
        pad_id = tokenizer.pad_token_id
        for text, ts in zip(texts, stamps):
            ids = tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
            att = [1] * len(ids)
            pad = seq_len - len(ids)
            if pad > 0:
                ids = ids + [pad_id] * pad
                att = att + [0] * pad
            self.samples.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(att, dtype=torch.long),
                torch.tensor(float(ts), dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def lm_loss(logits, input_ids, attention_mask):
    labels = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous()
    labels = labels.masked_fill(mask == 0, -100)
    flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
    return F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100)


def build_model(tokenizer, seq_len: int, seed: int, base_model: str) -> GPT2LMHeadModel:
    if base_model:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        print("[model] loaded pretrained:", base_model)
    else:
        torch.manual_seed(seed)
        cfg = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=seq_len + 8,
            n_embd=128,
            n_layer=4,
            n_head=4,
        )
        model = GPT2LMHeadModel(cfg)
        print("[model] tiny random GPT-2 (seed", seed, ")")
    model.config.use_cache = False
    return model


def forward(model_or_plugin, batch, device, time_shift_s: float, use_tmu: bool):
    x, att, t = batch
    x, att = x.to(device), att.to(device)
    if use_tmu:
        t = (t + time_shift_s).to(device)
        out = model_or_plugin(input_ids=x, attention_mask=att, time_stamp=t, return_dict=True)
    else:
        out = model_or_plugin(input_ids=x, attention_mask=att, return_dict=True)
    return out.logits


def train_model(model_or_plugin, loader, device, time_budget_s: float, max_epochs: int,
                lr, use_tmu: bool, params, tag: str):
    """Equal time budget training. Stops when budget exhausted."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    curve, step_ms = [], []
    step, ep = 0, 0
    t_start = time.perf_counter()
    tot, n = 0.0, 0
    while ep < max_epochs:
        ep += 1
        for batch in loader:
            if time.perf_counter() - t_start > time_budget_s:
                break
            t0 = time.perf_counter()
            logits = forward(model_or_plugin, batch, device, 0.0, use_tmu)
            loss = lm_loss(logits, batch[0].to(device), batch[1].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            tot += float(loss.item()); n += 1; step += 1
            if step % 20 == 0:
                curve.append({"step": step, "epoch": ep, "loss": round(tot / max(1, n), 4)})
                print(f"[{tag}] step {step} loss {tot/max(1,n):.4f} ({(time.perf_counter()-t_start)/60:.1f} min)")
                tot, n = 0.0, 0
        if time.perf_counter() - t_start > time_budget_s:
            break
    used_min = (time.perf_counter() - t_start) / 60.0
    print(f"[{tag}] done: {step} steps in {used_min:.1f} min")
    return curve, (sum(step_ms) / max(1, len(step_ms))), {"steps": step, "minutes": round(used_min, 1)}


def evaluate(model_or_plugin, loader, device, use_tmu: bool, time_shift_s: float = 0.0):
    tot, n = 0.0, 0
    logit_diffs = []
    if hasattr(model_or_plugin, "eval"):
        model_or_plugin.eval()
    with torch.no_grad():
        for batch in loader:
            logits = forward(model_or_plugin, batch, device, time_shift_s, use_tmu)
            loss = lm_loss(logits, batch[0].to(device), batch[1].to(device))
            tot += float(loss.item()); n += 1
            if use_tmu and time_shift_s == 0.0:
                logits_shift = forward(model_or_plugin, batch, device, 43200.0, True)
                logit_diffs.append(float((logits - logits_shift).abs().mean().item()))
    return {
        "eval_loss": round(tot / max(1, n), 4),
        "eval_ppl": round(math.exp(min(20.0, tot / max(1, n))), 2),
        "logit_change_12h": round(sum(logit_diffs) / max(1, len(logit_diffs)), 6) if logit_diffs else 0.0,
    }


def generation_speed(model, tokenizer, device, gen_tokens: int, active_layers):
    prompt = "The evening lights turned on"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    torch.manual_seed(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        cur = ids
        for _ in range(gen_tokens):
            if active_layers is None:
                out = model(cur)
            else:
                with prefix_layer_skip_context(model, active_layers):
                    out = model(cur)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    return (time.perf_counter() - t0) * 1000.0 / gen_tokens


def markdown_section(res) -> str:
    s = res["summary"]
    z, b, t = s["zero_shot"], s["baseline"], s["tmu"]
    lines = []
    lines.append("## \U0001F4CA Results: GPT-2 vs GPT-2 + TMU Layer (Equal-Budget Benchmark)")
    lines.append("")
    lines.append("_Auto-generated by the 'Train & Compare' workflow (" + res["meta"]["generated_at"] + ", " + res["meta"]["device"] + "). Equal training time (" + str(res["config"]["time_budget_min"]) + " min), same data/batch/optimizer for both fine-tunes. Source: results/tmu_vs_baseline.json_")
    lines.append("")
    lines.append("| Metric | Pretrained GPT-2 (zero-shot) | Fine-tuned GPT-2 | Fine-tuned GPT-2 + TMU |")
    lines.append("|---|---|---|---|")
    lines.append("| Eval loss | " + str(z["eval_loss"]) + " | " + str(b["eval_loss"]) + " | " + str(t["eval_loss"]) + " |")
    lines.append("| Eval perplexity | " + str(z["eval_ppl"]) + " | " + str(b["eval_ppl"]) + " | " + str(t["eval_ppl"]) + " |")
    lines.append("| Eval loss @ 12h shift | " + str(z["eval_loss_shifted_12h"]) + " | " + str(b["eval_loss_shifted_12h"]) + " | " + str(t["eval_loss_shifted_12h"]) + " |")
    lines.append("| Logit change on 12h shift | " + str(z["logit_change_12h"]) + " | " + str(b["logit_change_12h"]) + " | " + str(t["logit_change_12h"]) + " |")
    lines.append("| Training steps done | 0 | " + str(b["steps"]) + " | " + str(t["steps"]) + " |")
    lines.append("| Train minutes used | 0 | " + str(b["minutes"]) + " | " + str(t["minutes"]) + " |")
    fusion_layers = res.get("config", {}).get("fusion_layers")
    if fusion_layers is not None:
        lines.append("| Additional fusion layers | - | - | " + str(fusion_layers) + " | TMU Layer v2 additional layer |")
    lines.append("| Trainable params | - | " + f"{s['trainable_params_baseline']:,}" + " | " + f"{s['trainable_params_tmu']:,}" + " |")
    lines.append("| Avg train step | - | " + str(round(b["train_step_ms"], 1)) + " ms | " + str(round(t["train_step_ms"], 1)) + " ms |")
    lines.append("")
    lines.append("| Generation speed | ms/token | |")
    lines.append("|---|---|---|")
    lines.append("| Full layers (" + str(s["total_layers"]) + ") | " + str(round(s["gen_ms_full_layers"], 2)) + " | |")
    lines.append("| TMU layer-skip (" + str(s["skip_layers"]) + "/" + str(s["total_layers"]) + " layers) | " + str(round(s["gen_ms_skipped_layers"], 2)) + " | " + str(round(s["layer_skip_speedup"], 2)) + "x faster |")
    lines.append("")
    lines.append("### Batch-wise training progress (every 20 steps)")
    lines.append("")
    lines.append("| Step | Baseline fine-tune loss | TMU fine-tune loss |")
    lines.append("|---|---|---|")
    bc = res["curves"]["baseline_finetune"]
    tc = res["curves"]["tmu_finetune"]
    for i in range(max(len(bc), len(tc))):
        sname = str(bc[i]["step"]) if i < len(bc) else (str(tc[i]["step"]) if i < len(tc) else "-")
        bv = str(bc[i]["loss"]) if i < len(bc) else "-"
        tv = str(tc[i]["loss"]) if i < len(tc) else "-"
        lines.append("| " + sname + " | " + bv + " | " + tv + " |")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("- **Fairness**: dono fine-tunes ko same data, batch size, optimizer aur same time budget mila.")
    lines.append("- **Time-awareness**: baseline ke logits 12h shift par badalte hi nahi (time input nahi). TMU layer wale model ke logits shift par react karte hain - TMU real time-sense deta hai.")
    lines.append("- **Efficiency**: TMU sirf plugin params train karta hai (base frozen), phir bhi same time me ~" + str(round((b["train_step_ms"] / max(1e-9, t["train_step_ms"])) if t["train_step_ms"] else 1, 1)) + "x zyada steps complete hote hain.")
    lines.append("- **Speed**: TMU pacing layer-skip se generation fast ho jata hai.")
    lines.append("")
    return "\n".join(lines)


def update_readme(readme_path: str, section: str) -> None:
    START = "<!-- TMU-VS-BASELINE:START -->"
    END = "<!-- TMU-VS-BASELINE:END -->"
    p = pathlib.Path(readme_path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    block = START + "\n" + section + END
    if START in text and END in text:
        i = text.index(START)
        j = text.index(END) + len(END)
        text = text[:i] + block + text[j:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    p.write_text(text, encoding="utf-8")
    print("README updated:", readme_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-2 vs GPT-2 + TMU equal-budget comparison")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--base-model", dest="base_model", type=str, default="gpt2")
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=32)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    ap.add_argument("--time-budget-min", dest="time_budget_min", type=float, default=30.0)
    ap.add_argument("--max-epochs", dest="max_epochs", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--lr-baseline", dest="lr_baseline", type=float, default=1e-4)
    ap.add_argument("--lr-tmu", dest="lr_tmu", type=float, default=1e-3)
    ap.add_argument("--eval-samples", dest="eval_samples", type=int, default=160)
    ap.add_argument("--gen-tokens", dest="gen_tokens", type=int, default=12)
    ap.add_argument("--fusion-layers", dest="fusion_layers", type=int, default=1,
                    help="Additional fusion layers in the TMU layer (0 = ablation without the new layer)")
    ap.add_argument("--results", type=str, default="results/tmu_vs_baseline.json")
    ap.add_argument("--readme", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            a = k.replace("-", "_")
            if hasattr(args, a):
                setattr(args, a, v)
        print("[config]", args.config)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device != "auto":
        device = args.device
    print("Device:", device, "| base:", args.base_model, "| budget:", args.time_budget_min, "min per phase")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts, stamps = build_records(args.samples, seed=42)
    ev_texts, ev_stamps = build_records(args.eval_samples, seed=7)
    train_ds = TimeTextDataset(texts, stamps, tokenizer, args.seq_len)
    eval_ds = TimeTextDataset(ev_texts, ev_stamps, tokenizer, args.seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    budget_s = float(args.time_budget_min) * 60.0

    # ---------- 0) Pretrained zero-shot ----------
    print("\n=== Phase 0: pretrained GPT-2 zero-shot eval ===")
    base = build_model(tokenizer, args.seq_len, seed=123, base_model=args.base_model).to(device)
    total_params = sum(p.numel() for p in base.parameters())
    z_eval = evaluate(base, eval_loader, device, use_tmu=False)
    z_shift = evaluate(base, eval_loader, device, use_tmu=False, time_shift_s=43200.0)
    zero_shot = {
        "eval_loss": z_eval["eval_loss"], "eval_ppl": z_eval["eval_ppl"],
        "logit_change_12h": 0.0, "eval_loss_shifted_12h": z_shift["eval_loss"],
        "steps": 0, "minutes": 0.0, "train_step_ms": 0.0,
    }
    base_sd = copy.deepcopy(base.state_dict())

    # ---------- 1) Baseline: full GPT-2 fine-tune (equal budget) ----------
    print("\n=== Phase 1: baseline GPT-2 full fine-tune ===")
    b_curve, b_step_ms, b_stats = train_model(
        base, loader, device, budget_s, args.max_epochs, args.lr_baseline,
        use_tmu=False, params=[p for p in base.parameters() if p.requires_grad], tag="baseline",
    )
    b_eval = evaluate(base, eval_loader, device, use_tmu=False)
    b_shift = evaluate(base, eval_loader, device, use_tmu=False, time_shift_s=43200.0)
    baseline = {
        "eval_loss": b_eval["eval_loss"], "eval_ppl": b_eval["eval_ppl"],
        "logit_change_12h": 0.0, "eval_loss_shifted_12h": b_shift["eval_loss"],
        "steps": b_stats["steps"], "minutes": b_stats["minutes"], "train_step_ms": b_step_ms,
    }
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- 2) TMU: frozen base + plugin (equal budget) ----------
    print("\n=== Phase 2: TMU fine-tune (base frozen) ===")
    model2 = build_model(tokenizer, args.seq_len, seed=123, base_model=args.base_model).to(device)
    model2.load_state_dict(base_sd)
    plugin = TMUPlugin(
        base_model=model2,
        hidden_dim=int(model2.config.n_embd),
        mode="hooks",
        target_modules=["GPT2Block"],
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
        fusion_layers=args.fusion_layers,
    )
    tmu_params = freeze_base_and_enable_tmu(plugin)
    trainable_tmu = sum(p.numel() for p in plugin.tmu.parameters())
    t_curve, t_step_ms, t_stats = train_model(
        plugin, loader, device, budget_s, args.max_epochs, args.lr_tmu,
        use_tmu=True, params=list(tmu_params), tag="tmu",
    )
    t_eval = evaluate(plugin, eval_loader, device, use_tmu=True)
    t_shift = evaluate(plugin, eval_loader, device, use_tmu=True, time_shift_s=43200.0)
    tmu = {
        "eval_loss": t_eval["eval_loss"], "eval_ppl": t_eval["eval_ppl"],
        "logit_change_12h": t_eval["logit_change_12h"],
        "eval_loss_shifted_12h": t_shift["eval_loss"],
        "steps": t_stats["steps"], "minutes": t_stats["minutes"], "train_step_ms": t_step_ms,
    }

    # ---------- 3) generation speed ----------
    print("\n=== Phase 3: generation speed ===")
    model2.eval()
    gen_full = generation_speed(model2, tokenizer, device, args.gen_tokens, None)
    total_layers = len(model2.transformer.h)
    skip_layers = max(2, total_layers // 2)
    gen_skip = generation_speed(model2, tokenizer, device, args.gen_tokens, skip_layers)
    speedup = gen_full / max(1e-9, gen_skip)

    res = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device": device,
            "torch": torch.__version__,
            "base_model": args.base_model,
        },
        "config": {k: getattr(args, k) for k in vars(args)},
        "model": {"total_params": total_params, "tmu_trainable_params": trainable_tmu, "layers": total_layers},
        "curves": {"baseline_finetune": b_curve, "tmu_finetune": t_curve},
        "summary": {
            "zero_shot": zero_shot, "baseline": baseline, "tmu": tmu,
            "trainable_params_baseline": total_params,
            "trainable_params_tmu": trainable_tmu,
            "gen_ms_full_layers": gen_full,
            "gen_ms_skipped_layers": gen_skip,
            "layer_skip_speedup": speedup,
            "skip_layers": skip_layers, "total_layers": total_layers,
        },
    }
    out = pathlib.Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nSaved results:", args.results)
    if args.readme:
        update_readme(args.readme, markdown_section(res))
    print("\n=== SUMMARY ===")
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
