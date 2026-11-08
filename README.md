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
- **Additional fusion layer** (TMU Layer v2) — time embedding ke upar gated residual MLP
- **Transplant/export** — trained TMU layer ko plain GPT-2 par mount karke test karna
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
Additional Fusion Layer  <-- NEW (gated residual MLP, identity at init)
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

## 🧬 Additional Fusion Layer (TMU Layer v2)

TMU core me ab ek additional layer hai — `TemporalFusionLayer`:

```text
h     = LayerNorm(x)
u     = GELU(W_in(h))
delta = W_out(u)
gate  = sigmoid(W_gate(u))

y = x + dropout(delta * gate)
```

- `W_out` zero-init hai → layer init pe **exact identity** (pretrained behaviour safe)
- External + internal embedding ab combine hone ke baad fusion stack se guzarte hain
- `--fusion-layers 0` se layer bilkul off ho jati hai (clean ablation)

```bash
python -m tmu.cli --mode advanced --fusion-layers 1 --export-layer artifacts/tmu_layer.pt
```

Details: [docs/fusion_layer.md](docs/fusion_layer.md)

## 🔬 Transplant: TMU layer ko GPT-2 par lana

Trained TMU layer ko export karke kisi bhi plain GPT-2 par mount kar sakte ho:

```bash
# 1. Train + export
python scripts/train_tmu_layer.py --fusion-layers 1 --export artifacts/tmu_layer.pt

# 2. Transplant onto plain GPT-2 aur test
python scripts/test_transplant.py --layer artifacts/tmu_layer.pt \
  --results results/transplant_test.json
```

```python
from tmu import GPT2WithTMU, load_tmu_layer
from transformers import AutoModelForCausalLM

wrapped = GPT2WithTMU(
    base_model=AutoModelForCausalLM.from_pretrained("gpt2"),
    tmu=load_tmu_layer("artifacts/tmu_layer.pt"),
)

ids = wrapped.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    time_stamp=1_700_000_000.0,
    max_new_tokens=16,
)
```

Details: [docs/transplant.md](docs/transplant.md)

## Tests

```bash
pytest
```

## License

MIT

## 📁 Repository Structure

```text
TMU/
├── src/tmu/            # Main package (22 modules: clock, pacing, plugin, adapter, fusion, transplant, ...)
├── tests/              # Pytest suite
├── docs/               # Documentation (incl. fusion_layer.md, transplant.md)
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

_Auto-generated by 'Train & Compare' workflow (2026-11-08 01:16:51 UTC, cpu). Source: results/tmu_vs_baseline.json_

### Summary

| Metric | Normal GPT-2 (baseline) | GPT-2 + TMU layer | Note |
|---|---|---|---|
| Eval loss | 2.8371 | 2.8293 | lower = better LM fit |
| Eval perplexity | 17.07 | 16.93 | |
| Eval loss @ 12h time shift | 2.8371 | 2.8359 | TMU expected to react to time shift |
| Logit change on 12h shift | 0.0 | 0.015912 | >0 means model actually uses time |
| Additional fusion layers | - | 1 | TMU Layer v2 additional layer |
| Trainable params | 7,231,360 | 134,016 | TMU trains only the plugin |
| Avg train step | 1165.0 ms | 936.2 ms | |
| Generation (full layers) | 23.4 ms/token | | |
| Generation (TMU layer-skip, 2/4 layers) | 18.1 ms/token | | 1.3x faster |

### Batch-wise training progress (every 5th batch)

| Batch | Pretrain loss (shared base) | TMU fine-tune loss |
|---|---|---|
| 5 | 10.3417 | 2.9896 |
| 10 | 9.6943 | 2.9811 |
| 15 | 9.3646 | 2.9904 |
| 20 | 8.9973 | 3.0078 |
| 25 | 8.7484 | 2.9508 |
| 30 | 8.4148 | 3.0135 |
| 35 | 8.1037 | 2.981 |
| 40 | 7.6596 | 2.9556 |
| 45 | 7.4124 | 2.9591 |
| 50 | 7.1194 | 2.995 |
| 55 | 6.7515 | 3.0077 |
| 60 | 6.4208 | 3.0119 |
| 65 | 6.05 | 2.9881 |
| 70 | 5.6642 | 2.981 |
| 75 | 5.2768 | 2.9793 |
| 80 | 4.9207 | 2.9806 |
| 85 | 4.6055 | 3.0138 |
| 90 | 4.3299 | 2.9605 |
| 95 | 4.0217 | 2.9705 |
| 100 | 3.7355 | 2.9979 |
| 105 | 3.4763 | 3.0007 |
| 110 | 3.2898 | 2.9407 |

### Interpretation

- **Time-awareness**: baseline ka logit change 0 rehta hai (time input hi nahi). TMU layer wale model ke logits 12h shift par badalte hain - matlab TMU layer model ko actual time-sense deta hai.
- **Params**: TMU sirf plugin layer train karta hai, base model frozen rehta hai.
- **Speed**: TMU pace layer-skip se fast generation enable karta hai (thoda quality tradeoff ke saath).
<!-- TMU-VS-BASELINE:END -->

<!-- TMU-FUSION-ABLATION:START -->
## 🧬 Ablation: Additional Fusion Layer ka Asar

Same benchmark (shared pretrained base), sirf TMU layer ka additional fusion layer
on/off karke:

| Metric | Fusion 0 (purana TMU) | Fusion 1 (additional layer) | Change |
|---|---|---|---|
| TMU eval loss | 2.8298 | 2.8293 | **−0.0005** (better) |
| TMU eval perplexity | 16.94 | 16.93 | **−0.01** |
| Logit change @ 12h shift | 0.015134 | 0.015912 | **+5.1%** time-sensitivity |
| Eval loss @ 12h shift | 2.8389 | 2.8359 | **−0.0030** (better) |
| TMU trainable params | 34,944 | 134,016 | +99,072 (base still frozen) |
| Base GPT-2 eval loss | 2.8371 | 2.8371 | same (shared pretrained base) |

Read: additional layer ne time-sensitivity **~5% badhaayi** aur LM fit thoda improve
kiya, jabki base GPT-2 ke weights bilkul unchanged rahe (sirf TMU layer train hui).

Source: `results/bench_fusion_ablation.json` (`results/bench_fusion0.json`, `results/bench_fusion1.json`)
<!-- TMU-FUSION-ABLATION:END -->

<!-- TMU-TRANSPLANT:START -->
## 🔬 Transplant Test: Plain GPT-2 vs GPT-2 + TMU Layer

_Auto-generated by 'Transplant Test' workflow (n/a). Source: results/transplant_test.json_

| Metric | Baseline GPT-2 | GPT-2 + TMU layer |
|---|---|---|
| Eval loss | 6.4197 | 5.1890 |
| Eval perplexity | 613.80 | 179.29 |

- Samples: 32
- Logit change on 43200s time shift: 1.903747 (time-sensitive)
- TMU trainable params: 4,736,256

### Time coherence (TMU layer ke saath)

| Period | MAD | Cosine | Argmax agree |
|---|---|---|---|
| period_86400s | 1.7376 | 0.99950 | 0.9167 |
| period_43200s | 5.3965 | 0.99524 | 0.7147 |
| period_604800s | 3.3336 | 0.99797 | 0.7949 |
| period_31536000s | 4.6343 | 0.99611 | 0.7532 |

### Interpretation

- **Time sensitivity**: baseline GPT-2 ko time input milta hi nahi, isliye uska logit change 0 hai. TMU layer wale model ke logits time shift par badalte hain — matlab layer time signal ko model tak sach me pahuncha rahi hai.
- **Quality**: eval loss/perplexity compare karke dekho ki layer ne LM fit kitna improve kiya.
- **Cost**: sirf TMU layer train hoti hai, base GPT-2 frozen rehta hai.
<!-- TMU-TRANSPLANT:END -->
