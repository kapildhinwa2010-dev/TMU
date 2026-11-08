# Usage Guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Basic TMU Demo

Basic TMU sirf external world time use karta hai:

```bash
python -m tmu.cli \
  --mode basic \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 64
```

## 3. Advanced TMU Demo

Advanced TMU external time + internal AI/GPU clock use karta hai:

```bash
python -m tmu.cli \
  --mode advanced \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 64 \
  --max-gen-tokens 16
```

## 4. Use Real Timestamped Data

### JSONL

```json
{"text": "Morning meeting started.", "timestamp": "2026-01-01T09:30:00", "budget": 0.35}
{"text": "Explain this in detail.", "timestamp": 1767229200, "budget": 0.85, "desired_layers": 10, "tokens_used": 48}
```

Run:

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --epochs 1
```

### CSV

```csv
text,timestamp,budget,desired_layers,tokens_used
"Morning meeting started.","2026-01-01T09:30:00",0.35,4,12
"Explain this in detail.",1767229200,0.85,10,48
```

Run:

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.csv \
  --epochs 1
```

## 5. Dynamic Layer Skipping

Dynamic mode har token ke baad internal clock update karta hai aur layers change kar sakta hai.

Warning: ye KV cache use nahi karta, isliye slow ho sakta hai.

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --dynamic \
  --max-gen-tokens 12
```

## 6. Per-Layer Timing

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --dynamic \
  --per-layer-timing \
  --use-gpu-events \
  --max-gen-tokens 12
```

## 7. Optional True CUDA Cycle Timer

Agar CUDA toolkit installed hai:

```bash
python -m tmu.cli \
  --mode advanced \
  --data examples/data_examples.jsonl \
  --dynamic \
  --enable-cuda-cycles \
  --max-gen-tokens 12
```

If CUDA extension compilation fails, the code falls back to CPU/CUDA-event timing.

## 8. Save and Load

Save:

```bash
python -m tmu.cli \
  --mode advanced \
  --epochs 1 \
  --save gpt2_tmu_advanced.pt
```

Load:

```bash
python -m tmu.cli \
  --mode advanced \
  --load gpt2_tmu_advanced.pt
```

## 9. Run Tests

```bash
pytest
```

## 8. Additional Fusion Layer

```bash
python -m tmu.cli \
  --mode advanced \
  --fusion-layers 1 \
  --fusion-expansion 2 \
  --epochs 1
```

## 9. Train + Transplant TMU Layer

```bash
# train + export
python scripts/train_tmu_layer.py \
  --fusion-layers 1 \
  --epochs 1 \
  --samples 96 \
  --export artifacts/tmu_layer.pt \
  --results results/transplant_test.json

# transplant onto plain GPT-2 and test
python scripts/test_transplant.py \
  --layer artifacts/tmu_layer.pt \
  --results results/transplant_test.json
```
