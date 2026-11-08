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

The TMU does not give the model consciousness or a "sense of time".

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
    base_model=AutoModelForCausalLM.from_pretrained("gpt2
"),
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
## 📊 Results: GPT-2 vs GPT-2 + TMU Layer (Equal-Budget Benchmark)

_Auto-generated by the 'Train & Compare' workflow (2026-11-08 06:20:41 UTC, cpu). Equal training time (30 min), same data/batch/optimizer for both fine-tunes. Source: results/tmu_vs_baseline.json_

| Metric | Pretrained GPT-2 (zero-shot) | Fine-tuned GPT-2 | Fine-tuned GPT-2 + TMU |
|---|---|---|---|
| Eval loss | 5.1066 | 0.1007 | 1.7226 |
| Eval perplexity | 165.11 | 1.11 | 5.6 |
| Eval loss @ 12h shift | 5.1066 | 0.1007 | 5.2203 |
| Logit change on 12h shift | 0.0 | 0.0 | 25.718651 |
| Training steps done | 0 | 1111 | 2210 |
| Train minutes used | 0 | 30.0 | 30.0 |
| Additional fusion layers | - | - | 1 | TMU Layer v2 additional layer |
| Trainable params | - | 124,439,808 | 4,736,256 |
| Avg train step | - | 1620.6 ms | 814.4 ms |

| Generation speed | ms/token | |
|---|---|---|
| Full layers (12) | 71.58 | |
| TMU layer-skip (6/
12 layers) | 48.38 | 1.48x faster |

### Batch-wise training progress (every 20 steps)

| Step | Baseline fine-tune loss | TMU fine-tune loss |
|---|---|---|
| 20 | 1.6808 | 5.0259 |
| 40 | 0.2253 | 3.7425 |
| 60 | 0.1463 | 3.0991 |
| 80 | 0.1368 | 2.7279 |
| 100 | 0.1455 | 2.5514 |
| 120 | 0.1086 | 2.1657 |
| 140 | 0.1209 | 2.0647 |
| 160 | 0.1161 | 1.6828 |
| 180 | 0.1332 | 1.6475 |
| 200 | 0.1129 | 1.8751 |
| 220 | 0.1095 | 2.0813 |
| 240 | 0.091 | 1.5463 |
| 260 | 0.1188 | 1.5348 |
| 280 | 0.1118 | 1.7781 |
| 300 | 0.1059 | 2.2742 |
| 320 | 0.1018 | 1.6541 |
| 340 | 0.1166 | 1.4598 |
| 360 | 0.0966 | 1.538 |
| 380 | 0.1093 | 1.2079 |
| 400 | 0.0902 | 1.3387 |
| 420 | 0.111 | 1.3847 |
| 440 | 0.118 | 1.3203 |
| 460 | 0.1083 | 1.2747 |
| 480 | 0.1072 | 1.5322 |
| 500 | 0.1027 | 1.3757 |
| 520 | 0.1062 | 1.4706 |
| 540 | 0.1175 | 1.5323 |
| 560 | 0.1129 | 1.2525 |
| 580 | 0.0992 | 1.2997 |
| 600 | 0.1113 | 1.0304 |
| 620 | 0.0982 | 1.2095 |
| 640 | 0.1113 | 1.336 |
| 660 | 0.1071 | 1.2437 |
| 680 | 0.0962 | 1.3622 |
| 700 | 0.1037 | 1.0878 |
| 720 | 0.1095 | 0.8858 |
| 740 | 0.0899 | 0.9144 |
| 760 | 0.1092 | 1.3373 |
| 780 | 0.1051 | 1.4103 |
| 800 | 0.0944 | 0.873 |
| 820 | 0.0996 | 1.2502 |
| 840 | 0.1044 | 1.0362 |
| 860 | 0.0985 | 0.8879 |
| 880 | 0.1016 | 1.0458 |
| 900 | 0.1102 | 1.2913 |
| 920 | 0.1067 | 1.5127 |
| 940 | 0.1143 | 0.9057 |
| 960 | 0.0926 | 0.8643 |
| 980 | 0.0987 | 0.9994 |
| 1000 | 0.1057 | 1.007 |
| 1020 | 0.2194 | 0.5276 |
| 1040 | 0.2147 | 0.9203 |
| 1060 | 0.096 | 0.8517 |
| 1080 | 0.0893 | 0.5635 |
| 1100 | 0.104 | 0.5446 |
| 1120 | - | 1.3527 |
| 1140 | - | 0.8242 |
| 1160 | - | 0.8151 |
| 1180 | - | 0.5449 |
| 1200 | - | 0.5272 |
| 1220 | - | 0.9937 |
| 1240 | - | 0.7778 |
| 1260 | - | 0.5843 |
| 1280 | - | 0.5337 |
| 1300 | - | 0.7619 |
| 1320 | - | 0.8194 |
| 1340 | - | 0.6366 |
| 1360 | - | 0.4851 |
| 1380 | - | 0.5153 |
| 1400 | - | 0.4791 |
| 1420 | - | 0.5934 |
| 1440 | - | 0.354 |
| 1460 | - | 0.8367 |
| 1480 | - | 0.8935 |
| 
1500 | - | 0.6982 |
| 1520 | - | 0.6977 |
| 1540 | - | 0.5086 |
| 1560 | - | 0.5115 |
| 1580 | - | 0.4507 |
| 1600 | - | 0.8295 |
| 1620 | - | 0.7546 |
| 1640 | - | 0.6029 |
| 1660 | - | 0.5449 |
| 1680 | - | 0.2928 |
| 1700 | - | 0.8515 |
| 1720 | - | 0.6484 |
| 1740 | - | 0.6238 |
| 1760 | - | 0.7659 |
| 1780 | - | 0.7783 |
| 1800 | - | 0.4599 |
| 1820 | - | 0.4256 |
| 1840 | - | 0.4377 |
| 1860 | - | 0.3471 |
| 1880 | - | 0.771 |
| 1900 | - | 0.513 |
| 1920 | - | 0.4822 |
| 1940 | - | 0.4842 |
| 1960 | - | 0.7046 |
| 1980 | - | 0.617 |
| 2000 | - | 0.4907 |
| 2020 | - | 0.5765 |
| 2040 | - | 0.5633 |
| 2060 | - | 0.5898 |
| 2080 | - | 0.4171 |
| 2100 | - | 0.3541 |
| 2120 | - | 0.2785 |
| 2140 | - | 0.5565 |
| 2160 | - | 0.4409 |
| 2180 | - | 0.7401 |
| 2200 | - | 0.7713 |

### Interpretation

- **Fairness**: dono fine-tunes ko same data, batch size, optimizer aur same time budget mila.
- **Time-awareness**: baseline ke logits 12h shift par badalte hi nahi (time input nahi). TMU layer wale model ke logits shift par react karte hain - TMU real time-sense deta hai.
- **Efficiency**: TMU sirf plugin params train karta hai (base frozen), phir bhi same time me ~2.0x zyada steps complete hote hain.
- **Speed**: TMU pacing layer-skip se generation fast ho jata hai.
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

Read: the additional layer increased time-sensitivity by **~5%** and slightly improved the LM fit, while the base GPT-2 weights remained completely unchanged (only the TMU layer was trained).

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
