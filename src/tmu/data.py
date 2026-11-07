from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .time_features import cyclic_time_features
from .utils import clamp01, get_first_field


def parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime):
        return value.timestamp()

    s = str(value).strip()
    if not s:
        raise ValueError("Empty timestamp")

    try:
        return float(s)
    except ValueError:
        pass

    s_iso = s.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(s_iso).timestamp()
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue

    raise ValueError(f"Timestamp parse nahi ho paya: {value}")


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def estimate_text_difficulty(text: str) -> float:
    if not text:
        return 0.0

    t = str(text)
    words = t.split()
    chars = len(t)

    score = 0.0

    score += min(0.30, chars / 1000.0)
    score += min(0.15, len(words) / 150.0)

    if "?" in t:
        score += 0.08

    if any(k in t.lower() for k in ("why", "how", "explain", "compare", "analyze", "reason")):
        score += 0.10

    if any(c.isdigit() for c in t):
        score += 0.06

    if any(c in t for c in "{}[]()<>=+-*/\\"):
        score += 0.10

    letters = [c.lower() for c in t if c.isalpha()]
    if letters:
        uniq_ratio = len(set(letters)) / max(1, len(letters))
        score += 0.08 * clamp01(uniq_ratio)

    return clamp01(score)


def load_timestamped_records(path: str) -> Tuple[List[str], List[float], List[float]]:
    ext = os.path.splitext(path)[1].lower()
    raw: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("records") or data.get("data") or []

        if not isinstance(data, list):
            raise ValueError("JSON file me list of records ya {'records': [...]} format chahiye.")

        raw = data

    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
                raw.append(row)

    else:
        raise ValueError("Sirf .jsonl, .json, .csv supported hain.")

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []

    for rec in raw:
        if not isinstance(rec, dict):
            continue

        text = get_first_field(rec, ["text", "content", "prompt", "message", "sentence"])
        ts = get_first_field(rec, ["timestamp", "time", "datetime", "date", "created_at"])

        if text is None or ts is None:
            continue

        b = get_first_field(rec, ["budget", "time_budget", "budget_norm"])
        b = 0.5 if b is None else float(b)
        b = min(1.0, max(0.05, b))

        texts.append(str(text))
        timestamps.append(parse_timestamp(ts))
        budgets.append(b)

    if not texts:
        raise ValueError("Koi valid record nahi mila.")

    return texts, timestamps, budgets


def load_advanced_records(
    path: str,
) -> Tuple[List[str], List[float], List[float], List[Dict[str, Any]]]:
    ext = os.path.splitext(path)[1].lower()
    raw: List[Dict[str, Any]] = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("records") or data.get("data") or []

        if not isinstance(data, list):
            raise ValueError("JSON me list of records ya {'records': [...]} chahiye.")

        raw = data

    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
                raw.append(row)

    else:
        raise ValueError("Sirf .jsonl, .json, .csv supported hain.")

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []
    metas: List[Dict[str, Any]] = []

    for rec in raw:
        if not isinstance(rec, dict):
            continue

        text = get_first_field(rec, ["text", "content", "prompt", "message", "sentence"])
        ts = get_first_field(rec, ["timestamp", "time", "datetime", "date", "created_at"])

        if text is None or ts is None:
            continue

        b = get_first_field(rec, ["budget", "time_budget", "budget_norm"])
        b = 0.5 if b is None else float(b)
        b = min(1.0, max(0.05, b))

        meta = {
            "latency_ms": get_first_field(rec, ["latency_ms", "response_time_ms", "latency"]),
            "tokens_used": get_first_field(rec, ["tokens_used", "generated_tokens", "completion_tokens"]),
            "desired_layers": get_first_field(rec, ["desired_layers", "active_layers", "layers"]),
            "desired_pace": get_first_field(rec, ["desired_pace", "pace", "speed"]),
            "quality": get_first_field(rec, ["quality", "score", "reward"]),
        }

        texts.append(str(text))
        timestamps.append(parse_timestamp(ts))
        budgets.append(b)
        metas.append(meta)

    if not texts:
        raise ValueError("Koi valid record nahi mila.")

    return texts, timestamps, budgets, metas


def build_demo_records(
    num_samples: int = 192,
    base_time: float = 1_700_000_000.0,
    seed: int = 42,
) -> Tuple[List[str], List[float], List[float]]:
    base_texts = [
        "The morning sun rises and the city starts working.",
        "At noon the office lunch break happens quickly.",
        "In the evening traffic increases on every road.",
        "At night the model should answer shorter messages.",
        "During the day people ask many practical questions.",
        "Early morning thoughts are usually calm and clear.",
        "Late night reasoning can become slower and deeper.",
        "The weekly routine repeats in a predictable rhythm.",
        "Seasons change slowly and affect human plans.",
        "A busy schedule needs faster decisions.",
        "A free schedule allows longer thinking.",
        "Time pressure changes the quality of answers.",
        "The clock controls meetings and deadlines.",
        "Deadlines force compact reasoning.",
        "Open time encourages exploration.",
        "Rhythm helps prediction and coherence.",
    ]

    g = torch.Generator()
    g.manual_seed(int(seed))

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []

    t = float(base_time)

    for i in range(int(num_samples)):
        text = base_texts[i % len(base_texts)] + f" Record {i}."

        if i > 0:
            delta = torch.rand(1, generator=g).item() * 3600.0 + 10.0
            t += delta

        budget = torch.rand(1, generator=g).item() * 0.9 + 0.1

        texts.append(text)
        timestamps.append(t)
        budgets.append(budget)

    return texts, timestamps, budgets


def build_advanced_demo_records(
    num_samples: int = 192,
    num_layers: int = 12,
    base_time: float = 1_700_000_000.0,
    seed: int = 42,
) -> Tuple[List[str], List[float], List[float], List[Dict[str, Any]]]:
    base_texts = [
        "The morning sun rises and the city starts working.",
        "At noon the office lunch break happens quickly.",
        "In the evening traffic increases on every road.",
        "At night the model should answer shorter messages.",
        "During the day people ask many practical questions.",
        "Early morning thoughts are usually calm and clear.",
        "Late night reasoning can become slower and deeper.",
        "The weekly routine repeats in a predictable rhythm.",
        "Seasons change slowly and affect human plans.",
        "A busy schedule needs faster decisions.",
        "A free schedule allows longer thinking.",
        "Time pressure changes the quality of answers.",
        "The clock controls meetings and deadlines.",
        "Deadlines force compact reasoning.",
        "Open time encourages exploration.",
        "Rhythm helps prediction and coherence.",
        "Explain why this code fails and how to fix it.",
        "Compare morning and night productivity in detail.",
        "Analyze traffic patterns and suggest a route.",
        "Give a short answer only.",
    ]

    g = torch.Generator()
    g.manual_seed(int(seed))

    texts: List[str] = []
    timestamps: List[float] = []
    budgets: List[float] = []
    metas: List[Dict[str, Any]] = []

    t = float(base_time)

    for i in range(int(num_samples)):
        text = base_texts[i % len(base_texts)] + f" Record {i}."

        if i > 0:
            delta = torch.rand(1, generator=g).item() * 3600.0 + 10.0
            t += delta

        budget = torch.rand(1, generator=g).item() * 0.9 + 0.1
        difficulty = estimate_text_difficulty(text)

        base_layer_frac = 0.25 + 0.55 * budget + 0.20 * difficulty
        base_layer_frac = clamp01(base_layer_frac)

        desired_layers = int(round(2 + base_layer_frac * max(0, num_layers - 2)))
        desired_layers = max(1, min(num_layers, desired_layers))

        token_frac = clamp01(0.15 + 0.70 * budget + 0.15 * difficulty)
        tokens_used = int(round(4 + token_frac * 60))

        latency_ms = 20.0 + tokens_used * 8.0 + torch.rand(1, generator=g).item() * 120.0
        quality = clamp01(0.55 + 0.25 * budget + 0.20 * (desired_layers / max(1, num_layers)))

        texts.append(text)
        timestamps.append(t)
        budgets.append(budget)
        metas.append({
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "desired_layers": desired_layers,
            "desired_pace": desired_layers / float(num_layers),
            "quality": quality,
        })

    return texts, timestamps, budgets, metas


class GPT2TimestampedDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        timestamps: Sequence[float],
        budgets: Sequence[float],
        tokenizer: Any,
        seq_len: int = 64,
    ):
        super().__init__()

        if len(texts) != len(timestamps) or len(texts) != len(budgets):
            raise ValueError("texts, timestamps, budgets ki length same honi chahiye.")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.samples: List[Dict[str, Any]] = []

        for text, t, b in zip(texts, timestamps, budgets):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            t_tensor = torch.tensor(float(t), dtype=torch.float32)
            budget_tensor = torch.tensor(float(b), dtype=torch.float32)

            y_rhythm = cyclic_time_features(
                torch.tensor([float(t) + 600.0], dtype=torch.float64)
            ).squeeze(0)

            valid_len = int(attention_mask.sum().item())
            difficulty = min(1.0, max(0.0, valid_len / float(max(1, self.seq_len))))

            day_phase = (float(t) % 86400.0) / 86400.0
            time_need = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * day_phase))

            desired_pace = 0.5 * difficulty + 0.5 * time_need
            y_pace_value = min(float(desired_pace), float(b))
            y_pace = torch.tensor(y_pace_value, dtype=torch.float32)

            self.samples.append({
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "t": t_tensor,
                "budget": budget_tensor,
                "y_rhythm": y_rhythm,
                "y_pace": y_pace,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AdvancedTimestampedDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        timestamps: Sequence[float],
        budgets: Sequence[float],
        metas: Sequence[Dict[str, Any]],
        tokenizer: Any,
        seq_len: int = 64,
        num_layers: int = 12,
        max_label_tokens: int = 64,
    ):
        super().__init__()

        if not (len(texts) == len(timestamps) == len(budgets) == len(metas)):
            raise ValueError("texts, timestamps, budgets, metas ki length same honi chahiye.")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.num_layers = int(num_layers)
        self.max_label_tokens = int(max_label_tokens)
        self.samples: List[Dict[str, Any]] = []

        for text, t, b, meta in zip(texts, timestamps, budgets, metas):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            t_tensor = torch.tensor(float(t), dtype=torch.float32)
            budget_tensor = torch.tensor(float(b), dtype=torch.float32)

            y_rhythm = cyclic_time_features(
                torch.tensor([float(t) + 600.0], dtype=torch.float64)
            ).squeeze(0)

            difficulty = estimate_text_difficulty(text)

            desired_pace = _as_float(meta.get("desired_pace"))
            desired_layers = _as_float(meta.get("desired_layers"))
            tokens_used = _as_float(meta.get("tokens_used"))

            if desired_pace is not None:
                target_pace = clamp01(desired_pace)
            elif desired_layers is not None:
                target_pace = clamp01(desired_layers / float(self.num_layers))
            elif tokens_used is not None:
                target_pace = clamp01(tokens_used / float(self.max_label_tokens))
            else:
                valid_len = int(attention_mask.sum().item())
                length_diff = min(1.0, max(0.0, valid_len / float(max(1, self.seq_len))))
                day_phase = (float(t) % 86400.0) / 86400.0
                time_need = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * day_phase))
                target_pace = clamp01(0.5 * length_diff + 0.5 * time_need)

            target_pace = min(target_pace, float(b))

            if desired_layers is not None:
                y_layers = clamp01(desired_layers / float(self.num_layers))
            else:
                y_layers = target_pace

            if tokens_used is not None:
                y_tokens = clamp01(tokens_used / float(self.max_label_tokens))
            else:
                y_tokens = target_pace

            self.samples.append({
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "t": t_tensor,
                "budget": budget_tensor,
                "y_rhythm": y_rhythm,
                "y_pace": torch.tensor(target_pace, dtype=torch.float32),
                "y_layers": torch.tensor(y_layers, dtype=torch.float32),
                "y_tokens": torch.tensor(y_tokens, dtype=torch.float32),
                "difficulty": torch.tensor(difficulty, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]