# TMU-GPT-2: Time Management Unit for GPT-2

[![CI](https://github.com/kapildhinwa2010-dev/TMU/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/TMU/actions/workflows/ci.yml)
[![Train & Compare](https://github.com/kapildhinwa2010-dev/TMU/actions/workflows/train.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/TMU/actions/workflows/train.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

TMU-GPT-2 ek experimental PyTorch/HuggingFace project hai jo GPT-2 model ke andar
ek plugin layer jaisa **Time Management Unit (TMU)** lagata hai.

Ye TMU do tarah ka time samajhta hai:

1. **External / World Time**
   - human time
   - second, minute, hour, day, week, month/year cycles
   - din/raat, weekly rhythm, seasonal rhythm

2. **Internal / AI Compute Time**
   - GPU/CPU processing time
   - milliseconds, microseconds
   - estimated GPU SM cycles
   - per-layer latency
   - traffic load
   - GPU temperature/power/VRAM pressure
   - deadline pressure
   - prompt difficulty
   - confidence/entropy

## Main Features

- GPT-2 ke andar TMU plugin layer
- Kisi bhi PyTorch model me hook-based plugin
- External world-time cyclic features
- Internal AI clock features:
  - CPU nanosecond timer
  - CUDA events
  - GPU SM clock via nvidia-smi
  - estimated GPU cycles
- Per-layer timing
- Traffic/QoS monitor
- GPU thermal/power/memory monitor
- Pace head
- Layer fraction head
- Token fraction head
- Cache-safe static layer skipping
- Dynamic no-cache layer skipping
- Confidence-based early stopping
- Time coherence evaluation
- Real timestamped JSONL/CSV data loading
- Optional true CUDA `clock64()` / `%globaltimer` extension

## Important Reality Check

TMU model ko consciousness ya "samay ka ehsas" nahi deta.

Ye bas itna karta hai:

```text
time measurements -> numerical features -> embeddings -> activation modulation -> pacing decisions
```

"Thinking speed" ka matlab GPU jadu se tez nahi hota.

Iska matlab hai:

- kam pace:
  - kam layers
  - kam tokens
  - compact answer

- zyada pace:
  - zyada layers
  - zyada tokens
  - detailed reasoning

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

## Quick Demo

```bash
python -m tmu.cli --mode advanced --epochs 1 --batch-size 2 --seq-len 32 --samples 64 --max-gen-tokens 16
```

Kam VRAM test:

```bash
python -m tmu.cli --mode basic --epochs 1 --batch-size 2 --seq-len 32 --samples 64
```

Real data ke saath:

```bash
python -m tmu.cli --mode advanced --data examples/data_examples.jsonl --epochs 1
```

Dynamic layer skipping + per-layer timing:

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --dynamic \
  --per-layer-timing \
  --use-gpu-events \
  --max-gen-tokens 12
```

Optional true CUDA cycles:

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --dynamic \
  --enable-cuda-cycles \
  --max-gen-tokens 12
```

## Data Format

### JSONL

```json
{"text": "Morning meeting started.", "timestamp": "2026-01-01T09:30:00", "budget": 0.35}
{"text": "Explain this in detail.", "timestamp": 1767229200, "budget": 0.85, "desired_layers": 10, "tokens_used": 48}
```

### CSV

```csv
text,timestamp,budget,desired_layers,tokens_used
"Morning meeting started.","2026-01-01T09:30:00",0.35,4,12
"Explain this in detail.",1767229200,0.85,10,48
```

Supported timestamp formats:

- Unix seconds: `1767229200`
- ISO: `2026-01-01T09:30:00`
- DateTime: `2026-01-01 09:30:00`
- Date: `2026-01-01`

## Architecture

```text
Human/World Timestamp
        |
        v
External TMU Encoder -----------------\
                                      \
GPU/CPU Internal Clock                 |
Traffic Monitor                        |--> Combined Time Embedding
System Monitor                         |
Per-Layer Timer                        |
Prompt Difficulty                      |
Confidence/Entropy --------------------/
        |
        v
Advanced Internal Adapter
        |
        v
GPT-2 Activation Modulation
        |
        v
Advanced Heads:
- rhythm
- pace
- layer fraction
- token fraction
        |
        v
Advanced Pace Controller
        |
-------------------
|                 |
v                 v
Static KV-cache   Dynamic no-cache
layer skip        layer skip
fast/safe         research/adaptive
```

## Tests

```bash
pytest
```

## License

MIT

## 📁 Repository Structure

```text
TMU/
├── src/tmu/            # Main package (20 modules: clock, pacing, plugin, adapter, ...)
├── tests/              # Pytest suite
├── docs/               # Documentation
├── examples/           # Example data (JSONL)
├── scripts/            # Demo scripts
├── research/           # Research prototypes & experiments (moved from repo root)
├── .github/workflows/  # CI (pytest on push/PR)
├── Dockerfile / Makefile / pyproject.toml
└── README.md / CHANGELOG.md / LICENSE
```

> **Note**: `research/` contains earlier standalone prototypes (GPT-2 + TMU full/advanced scripts, internal-clock experiments). Kept for reference; not part of the installed package.

<!-- TMU-VS-BASELINE:START -->
## 📊 Results: Normal GPT-2 vs GPT-2 + TMU Layer

_Auto-generated by 'Train & Compare' workflow (2026-11-08 00:48:49 UTC, cpu). Source: results/tmu_vs_baseline.json_

### Summary

| Metric | Normal GPT-2 (baseline) | GPT-2 + TMU layer | Note |
|---|---|---|---|
| Eval loss | 2.8371 | 2.8298 | lower = better LM fit |
| Eval perplexity | 17.07 | 16.94 | |
| Eval loss @ 12h time shift | 2.8371 | 2.8389 | TMU expected to react to time shift |
| Logit change on 12h shift | 0.0 | 0.015134 | >0 means model actually uses time |
| Trainable params | 7,231,360 | 34,944 | TMU trains only the plugin |
| Avg train step | 349.6 ms | 232.1 ms | |
| Generation (full layers) | 7.4 ms/token | | |
| Generation (TMU layer-skip, 2/4 layers) | 5.8 ms/token | | 1.28x faster |

### Batch-wise training progress (every 5th batch)

| Batch | Pretrain loss (shared base) | TMU fine-tune loss |
|---|---|---|
| 5 | 10.3417 | 3.0035 |
| 10 | 9.6943 | 2.9864 |
| 15 | 9.3646 | 2.9839 |
| 20 | 8.9973 | 3.0025 |
| 25 | 8.7484 | 2.9712 |
| 30 | 8.4148 | 2.9563 |
| 35 | 8.1037 | 2.9953 |
| 40 | 7.6596 | 2.9938 |
| 45 | 7.4124 | 3.0237 |
| 50 | 7.1194 | 2.9975 |
| 55 | 6.7515 | 2.9624 |
| 60 | 6.4208 | 2.9574 |
| 65 | 6.05 | 2.9774 |
| 70 | 5.6642 | 3.004 |
| 75 | 5.2768 | 2.9745 |
| 80 | 4.9207 | 2.95 |
| 85 | 4.6055 | 2.9673 |
| 90 | 4.3299 | 2.9916 |
| 95 | 4.0217 | 2.9758 |
| 100 | 3.7355 | 3.0074 |
| 105 | 3.4763 | 3.018 |
| 110 | 3.2898 | 2.9963 |

### Interpretation

- **Time-awareness**: baseline ka logit change 0 rehta hai (time input hi nahi). TMU layer wale model ke logits 12h shift par badalte hain - matlab TMU layer model ko actual time-sense deta hai.
- **Params**: TMU sirf plugin layer train karta hai, base model frozen rehta hai.
- **Speed**: TMU pace layer-skip se fast generation enable karta hai (thoda quality tradeoff ke saath).
<!-- TMU-VS-BASELINE:END -->
