# Advanced Looping v3

This module adds DLR-IA v3:

**Dynamic Layer Refinement with Input Anchoring**

It is add-only and does not require modifying existing TMU files.

## Features

### 1. Dynamic Layer Refinement

Easy prompts use fewer layers and no loops.

Hard prompts can use:

- more active layers
- per-layer loops
- final refinement passes

### 2. Input Anchoring

During loops, the original prompt signal is injected back into hidden states.

This prevents the model from forgetting:

- what the input was
- what task it was solving
- which context matters

### 3. Per-token Anchor Schedule

Anchor strength changes per generation step based on:

- progress
- pressure
- confidence
- prompt difficulty

### 4. Loop Rejection Sampling

If looping reduces confidence, the loop result can be rejected and baseline logits used instead.

### 5. Prompt Difficulty Detection

Difficulty is estimated from:

- text heuristics
- model perplexity
- next-token entropy

### 6. Layer Importance Scoring

Each layer's contribution is tracked using hidden-state delta norms.

Loops are distributed toward more important layers.

### 7. Latency Feedback Loop

Runtime measurements are stored:

- actual latency
- generated tokens
- confidence
- internal features

These can be used to fine-tune latency/pace heads.

### 8. Advanced Traffic Features

Additional signals:

- burstiness
- backlog
- queue growth
- cost per token
- SLA violation
- deadline slack
- latency trend
- throughput trend
- GPU contention
- VRAM fragmentation proxy

## Run Demo

```bash
python scripts/run_loops_advanced_demo.py \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 64 \
  --max-gen-tokens 12
```

## Run with Real Data

```bash
python scripts/run_loops_advanced_demo.py \
  --data examples/data_loops_advanced.jsonl \
  --epochs 1
```

## Enable Prompt Detector

```bash
python scripts/run_loops_advanced_demo.py \
  --prompt-detector \
  --epochs 1
```

## Enable Loop Rejection

```bash
python scripts/run_loops_advanced_demo.py \
  --enable-rejection \
  --rejection-margin 0.02 \
  --epochs 1
```

## Enable Latency Feedback

```bash
python scripts/run_loops_advanced_demo.py \
  --feedback \
  --feedback-epochs 1 \
  --epochs 1
```

## Slow Research Mode

Dynamic no-cache looping is slower:

```bash
python scripts/run_loops_advanced_demo.py \
  --per-layer-timing \
  --use-gpu-events \
  --max-gen-tokens 10 \
  --epochs 1
```

## Tests

```bash
pytest tests/test_looping_advanced.py
```