from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from .plugin import TMUPlugin


def _flatten_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool()
    return logits[mask]


def _compare_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    a = _flatten_valid_logits(logits_a, attention_mask)
    b = _flatten_valid_logits(logits_b, attention_mask)

    if a.numel() == 0 or b.numel() == 0:
        return {"mad": 0.0, "cos": 1.0, "agree": 1.0}

    mad = (a - b).abs().mean().item()
    cos = F.cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1),
        dim=1,
    ).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()

    return {"mad": float(mad), "cos": float(cos), "agree": float(agree)}


@torch.no_grad()
def evaluate_time_coherence(
    plugin: TMUPlugin,
    tokenizer: Any,
    dataset: Any,
    device: str,
    num_samples: int = 2,
    periods: Sequence[float] = (86400.0, 604800.0, 31536000.0),
    cross_period: float = 43200.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    plugin.eval()

    all_periods: List[float] = []
    for p in list(periods) + [cross_period]:
        if p not in all_periods:
            all_periods.append(float(p))

    period_metrics: Dict[float, Dict[str, List[float]]] = {
        p: {"mad": [], "cos": [], "agree": []}
        for p in all_periods
    }

    n = min(int(num_samples), len(dataset))

    for i in range(n):
        sample = dataset[i]

        x = sample["input_ids"].unsqueeze(0).to(device)
        am = sample["attention_mask"].unsqueeze(0).to(device)
        t = sample["t"].to(device)

        valid_ids = x[0][am[0].bool()]
        prompt = tokenizer.decode(valid_ids, skip_special_tokens=True)

        out_t = plugin(
            input_ids=x,
            attention_mask=am,
            time_stamp=t,
            use_cache=False,
            return_dict=True,
        )
        logits_t = out_t.logits

        for p in all_periods:
            t_p = t + float(p)

            out_p = plugin(
                input_ids=x,
                attention_mask=am,
                time_stamp=t_p,
                use_cache=False,
                return_dict=True,
            )
            logits_p = out_p.logits

            comp = _compare_logits(logits_t, logits_p, am)

            period_metrics[p]["mad"].append(comp["mad"])
            period_metrics[p]["cos"].append(comp["cos"])
            period_metrics[p]["agree"].append(comp["agree"])

        if verbose:
            print(f"[Eval sample {i + 1}/{n}] prompt preview: {prompt[:80]}...")

    summary: Dict[str, Any] = {}

    for p in all_periods:
        vals = period_metrics[p]
        if not vals["cos"]:
            continue

        key = f"period_{int(p)}s"
        summary[key] = {
            "mad": float(sum(vals["mad"]) / len(vals["mad"])),
            "cos": float(sum(vals["cos"]) / len(vals["cos"])),
            "agree": float(sum(vals["agree"]) / len(vals["agree"])),
        }

    if "period_86400s" in summary:
        summary["daily_coherence_cos"] = summary["period_86400s"]["cos"]

    if "period_43200s" in summary:
        summary["half_day_sensitivity_1_minus_cos"] = 1.0 - summary["period_43200s"]["cos"]

    if "period_604800s" in summary:
        summary["weekly_coherence_cos"] = summary["period_604800s"]["cos"]

    if "period_31536000s" in summary:
        summary["yearly_coherence_cos"] = summary["period_31536000s"]["cos"]

    return summary
