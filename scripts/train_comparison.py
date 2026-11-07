# scripts/train_comparison.py
#
# Normal GPT-2 vs GPT-2 + TMU layer comparison.
#
# Ye script:
#   1. Ek chhota GPT-2 (random init) ko time-of-day correlated corpus par pretrain karta hai
#   2. Baseline eval: normal GPT-2 (same weights)
#   3. TMU eval: same pretrained GPT-2 + TMU plugin layer (base frozen, sirf TMU train)
#   4. Compare: eval loss, perplexity, time-sensitivity, params, latency, layer-skip speedup
#
# Run from repo root:
#   python scripts/train_comparison.py --config configs/train_compare.json --results results/tmu_vs_baseline.json --readme README.md

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
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from tmu.modeling import prefix_layer_skip_context
from tmu.plugin import TMUPlugin, freeze_base_and_enable_tmu

# ------------------------------------------------------------------
# Time-of-day correlated synthetic corpus
# ------------------------------------------------------------------

PERIODS = {
    "morning": (7, 11, [
        "The morning meeting started and the team prepared the daily plan.",
        "Sunrise coffee and the morning news were on the table.",
        "The city woke up early and morning traffic filled the streets.",
        "A fresh morning breeze cleared the air before work began.",
    ]),
    "afternoon": (13, 17, [
        "The afternoon workshop reviewed the progress of the project.",
        "Long afternoon shadows stretched across the quiet office.",
        "Students finished their afternoon lectures and relaxed.",
        "The afternoon sun heated the rooftop of the building.",
    ]),
    "evening": (19, 22, [
        "The evening lights turned on and families sat down for dinner.",
        "An evening walk along the river ended the busy day.",
        "The evening news reported calm weather across the region.",
        "Music played softly in the evening air of the small cafe.",
    ]),
    "night": (0, 5, [
        "The quiet night sky showed thousands of bright stars.",
        "Night shift workers kept the servers running until dawn.",
        "The city slept deeply through the calm night hours.",
        "Only a few lamps glowed in the silent night streets.",
    ]),
}

BASE_TIME = 1_700_000_000.0  # unix seconds


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


def build_model(tokenizer, seq_len: int, seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=seq_len + 8,
        n_embd=128,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(cfg)
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


def train_model(model_or_plugin, loader, device, epochs, lr, use_tmu: bool, params, tag: str):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    curve, step_ms = [], []
    step = 0
    for ep in range(epochs):
        model_or_plugin.train() if not use_tmu else None
        tot, n = 0.0, 0
        for batch in loader:
            t0 = time.perf_counter()
            logits = forward(model_or_plugin, batch, device, 0.0, use_tmu)
            loss = lm_loss(logits, batch[0].to(device), batch[1].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            tot += float(loss.item()); n += 1; step += 1
            if step % 5 == 0:
                curve.append({"epoch": ep + 1, "batch": step, "loss": round(tot / max(1, n), 4)})
                tot, n = 0.0, 0
        print(f"[{tag}] epoch {ep+1}/{epochs} done")
    return curve, sum(step_ms) / max(1, len(step_ms))


def evaluate(model_or_plugin, loader, device, use_tmu: bool, time_shift_s: float = 0.0):
    tot, n = 0.0, 0
    logit_diffs = []
    was_training = getattr(model_or_plugin, "training", True)
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
    if hasattr(model_or_plugin, "train") and was_training:
        model_or_plugin.train()
    return {
        "eval_loss": round(tot / max(1, n), 4),
        "eval_ppl": round(math.exp(min(20.0, tot / max(1, n))), 2),
        "logit_change_12h": round(sum(logit_diffs) / max(1, len(logit_diffs)), 6) if logit_diffs else 0.0,
    }


def generation_speed(model, tokenizer, device, gen_tokens: int, active_layers: int):
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
    lines = []
    lines.append("## \U0001F4CA Results: Normal GPT-2 vs GPT-2 + TMU Layer")
    lines.append("")
    lines.append("_Auto-generated by 'Train & Compare' workflow (" + res["meta"]["generated_at"] + ", " + res["meta"]["device"] + "). Source: results/tmu_vs_baseline.json_")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Normal GPT-2 (baseline) | GPT-2 + TMU layer | Note |")
    lines.append("|---|---|---|---|")
    b, t = s["baseline"], s["tmu"]
    lines.append("| Eval loss | " + str(b["eval_loss"]) + " | " + str(t["eval_loss"]) + " | lower = better LM fit |")
    lines.append("| Eval perplexity | " + str(b["eval_ppl"]) + " | " + str(t["eval_ppl"]) + " | |")
    lines.append("| Eval loss @ 12h time shift | " + str(b["eval_loss_shifted_12h"]) + " | " + str(t["eval_loss_shifted_12h"]) + " | TMU expected to react to time shift |")
    lines.append("| Logit change on 12h shift | " + str(b["logit_change_12h"]) + " | " + str(t["logit_change_12h"]) + " | >0 means model actually uses time |")
    lines.append("| Trainable params | " + f"{s['trainable_params_baseline']:,}" + " | " + f"{s['trainable_params_tmu']:,}" + " | TMU trains only the plugin |")
    lines.append("| Avg train step | " + str(round(b["train_step_ms"], 1)) + " ms | " + str(round(t["train_step_ms"], 1)) + " ms | |")
    lines.append("| Generation (full layers) | " + str(round(s["gen_ms_full_layers"], 1)) + " ms/token | | |")
    lines.append("| Generation (TMU layer-skip, " + str(s["skip_layers"]) + "/" + str(s["total_layers"]) + " layers) | " + str(round(s["gen_ms_skipped_layers"], 1)) + " ms/token | | " + str(round(s["layer_skip_speedup"], 2)) + "x faster |")
    lines.append("")
    lines.append("### Batch-wise training progress (every 5th batch)")
    lines.append("")
    lines.append("| Batch | Pretrain loss (shared base) | TMU fine-tune loss |")
    lines.append("|---|---|---|")
    pt = res["curves"]["pretrain"]
    tm = res["curves"]["tmu_finetune"]
    for i in range(max(len(pt), len(tm))):
        a = str(pt[i]["loss"]) if i < len(pt) and pt[i] else "-"
        bname = str(pt[i]["batch"]) if i < len(pt) and pt[i] else (str(tm[i]["batch"]) if i < len(tm) else "-")
        c = str(tm[i]["loss"]) if i < len(tm) and tm[i] else "-"
        lines.append("| " + bname + " | " + a + " | " + c + " |")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("- **Time-awareness**: baseline ka logit change 0 rehta hai (time input hi nahi). TMU layer wale model ke logits 12h shift par badalte hain - matlab TMU layer model ko actual time-sense deta hai.")
    lines.append("- **Params**: TMU sirf plugin layer train karta hai, base model frozen rehta hai.")
    lines.append("- **Speed**: TMU pace layer-skip se fast generation enable karta hai (thoda quality tradeoff ke saath).")
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
    ap = argparse.ArgumentParser(description="Normal GPT-2 vs GPT-2 + TMU comparison")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--lr-baseline", dest="lr_baseline", type=float, default=3e-4)
    ap.add_argument("--lr-tmu", dest="lr_tmu", type=float, default=1e-3)
    ap.add_argument("--eval-samples", dest="eval_samples", type=int, default=120)
    ap.add_argument("--gen-tokens", dest="gen_tokens", type=int, default=16)
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
    if args.device not in ("auto",):
        device = args.device
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts, stamps = build_records(args.samples, seed=42)
    ev_texts, ev_stamps = build_records(args.eval_samples, seed=7)

    train_ds = TimeTextDataset(texts, stamps, tokenizer, args.seq_len)
    eval_ds = TimeTextDataset(ev_texts, ev_stamps, tokenizer, args.seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    # ---------- Phase 1: shared pretrained base ----------
    print("\n=== Phase 1: pretrain base GPT-2 (shared for both) ===")
    base = build_model(tokenizer, args.seq_len, seed=123).to(device)
    total_params = sum(p.numel() for p in base.parameters())
    pretrain_curve, base_step_ms = train_model(
        base, loader, device, args.epochs, args.lr_baseline,
        use_tmu=False, params=list(base.parameters()), tag="pretrain",
    )
    base_sd = copy.deepcopy(base.state_dict())

    # ---------- Phase 2: baseline eval (normal GPT-2) ----------
    print("\n=== Phase 2: baseline eval (normal GPT-2) ===")
    b_eval = evaluate(base, eval_loader, device, use_tmu=False)
    b_eval_shift = evaluate(base, eval_loader, device, use_tmu=False, time_shift_s=43200.0)
    baseline = {
        "eval_loss": b_eval["eval_loss"],
        "eval_ppl": b_eval["eval_ppl"],
        "logit_change_12h": 0.0,
        "eval_loss_shifted_12h": b_eval_shift["eval_loss"],
        "train_step_ms": base_step_ms,
    }

    # ---------- Phase 3: TMU plugin on the same base ----------
    print("\n=== Phase 3: TMU layer fine-tune (base frozen) ===")
    model2 = build_model(tokenizer, args.seq_len, seed=123).to(device)
    model2.load_state_dict(base_sd)
    plugin = TMUPlugin(
        base_model=model2,
        hidden_dim=int(model2.config.n_embd),
        mode="hooks",
        target_modules=["GPT2Block", "block"],
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
    )
    tmu_params = freeze_base_and_enable_tmu(plugin)
    trainable_tmu = sum(p.numel() for p in plugin.tmu.parameters())
    tmu_curve, tmu_step_ms = train_model(
        plugin, loader, device, args.epochs, args.lr_tmu,
        use_tmu=True, params=list(tmu_params), tag="tmu",
    )
    t_eval = evaluate(plugin, eval_loader, device, use_tmu=True)
    t_eval_shift = evaluate(plugin, eval_loader, device, use_tmu=True, time_shift_s=43200.0)
    tmu = {
        "eval_loss": t_eval["eval_loss"],
        "eval_ppl": t_eval["eval_ppl"],
        "logit_change_12h": t_eval["logit_change_12h"],
        "eval_loss_shifted_12h": t_eval_shift["eval_loss"],
        "train_step_ms": tmu_step_ms,
    }

    # ---------- Phase 4: speed / layer-skip ----------
    print("\n=== Phase 4: generation speed (full vs layer-skip) ===")
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
        },
        "config": {k: getattr(args, k) for k in vars(args)},
        "model": {"total_params": total_params, "tmu_trainable_params": trainable_tmu, "layers": total_layers},
        "curves": {"pretrain": pretrain_curve, "tmu_finetune": tmu_curve},
        "summary": {
            "baseline": baseline,
            "tmu": tmu,
            "trainable_params_baseline": total_params,
            "trainable_params_tmu": trainable_tmu,
            "gen_ms_full_layers": gen_full,
            "gen_ms_skipped_layers": gen_skip,
            "layer_skip_speedup": speedup,
            "skip_layers": skip_layers,
            "total_layers": total_layers,
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
